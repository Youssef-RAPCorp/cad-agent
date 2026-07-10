"""LLM-based refinement for solids that other fitters can't handle.

Sends the solid's geometric description to Gemini Flash and asks for
build123d code that recreates the shape. The returned code is
sandbox-executed and verified against the source via sym-diff before
being accepted. If verification fails, we iterate up to N times
feeding the error back.

This module is OPTIONAL: if `google-genai` is not installed or no
GEMINI_API_KEY is present, try_fit_llm returns FitResult(None, ...)
immediately.

Environment variables:
  GEMINI_API_KEY (or GOOGLE_API_KEY)   required
  GEMINI_CODEGEN_MODEL                 optional; default 'gemini-3.5-flash'
"""
import os
import sys
import math
import textwrap
import traceback
import importlib

from .fitter import FitResult, _verify_fit, _fmt, _FORCE_PRIMITIVES


# ---------------------------------------------------------------------------
# Geometry summarization
# ---------------------------------------------------------------------------

def summarize_solid(solid, max_faces=80):
    """Build a compact text description of a solid that an LLM can reason
    about. Includes bbox, volume, face inventory by type, and per-face
    summaries (normal, area, center) for up to max_faces faces."""
    from build123d import GeomType
    bb = solid.bounding_box()
    dx, dy, dz = bb.max.X - bb.min.X, bb.max.Y - bb.min.Y, bb.max.Z - bb.min.Z
    cx, cy, cz = (bb.max.X + bb.min.X) / 2, (bb.max.Y + bb.min.Y) / 2, (bb.max.Z + bb.min.Z) / 2

    lines = []
    lines.append(f"Volume: {solid.volume:.6f} mm^3")
    lines.append(f"Bbox: X[{bb.min.X:.4f}, {bb.max.X:.4f}] (width {dx:.4f})")
    lines.append(f"      Y[{bb.min.Y:.4f}, {bb.max.Y:.4f}] (width {dy:.4f})")
    lines.append(f"      Z[{bb.min.Z:.4f}, {bb.max.Z:.4f}] (width {dz:.4f})")
    lines.append(f"Bbox center: ({cx:.4f}, {cy:.4f}, {cz:.4f})")
    lines.append(f"Bbox volume: {dx*dy*dz:.6f} mm^3 "
                 f"(fill ratio {100*solid.volume/max(dx*dy*dz,1e-9):.1f}%)")

    faces = list(solid.faces())
    type_counts = {}
    for f in faces:
        t = str(f.geom_type).split('.')[-1]
        type_counts[t] = type_counts.get(t, 0) + 1
    lines.append(f"Total faces: {len(faces)}; by type: {type_counts}")

    # Per-face details
    lines.append(f"Faces (showing up to {max_faces}):")
    for i, f in enumerate(faces[:max_faces]):
        try:
            c = f.center()
            n = f.normal_at(c)
            t = str(f.geom_type).split('.')[-1]
            lines.append(f"  [{i}] {t} area={f.area:.5f} "
                         f"center=({c.X:+.4f},{c.Y:+.4f},{c.Z:+.4f}) "
                         f"normal=({n.X:+.3f},{n.Y:+.3f},{n.Z:+.3f})")
        except Exception:
            lines.append(f"  [{i}] (geometry error)")
    if len(faces) > max_faces:
        lines.append(f"  ... ({len(faces) - max_faces} more faces omitted)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sandbox execution
# ---------------------------------------------------------------------------

EXEC_PRELUDE = """
from build123d import (
    BuildPart, BuildSketch, BuildLine, Polyline,
    Line, RadiusArc, Locations, GridLocations, Plane, Mode, Keep,
    Box, Cylinder, Sphere, Cone, Torus,
    extrude, revolve, loft, sweep, make_face, add, fillet, chamfer,
    Vector, Axis, Location, Rot, GeomType, Align,
)
"""


def execute_recipe(code: str):
    """Execute a build123d code snippet in a fresh namespace.
    Expects the snippet to assign the final result to `_part`.
    Returns (part, error_string). On success: (part, None).
    On failure: (None, error_message)."""
    full = EXEC_PRELUDE + "\n" + code
    ns = {}
    try:
        exec(full, ns)
    except Exception:
        return None, traceback.format_exc()
    part = ns.get("_part", None)
    if part is None:
        return None, "Recipe did not assign to _part"
    try:
        _ = part.volume
    except Exception:
        return None, "Resulting _part has no .volume attribute"
    return part, None


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

from .build123d_reference import FULL_PROMPT_FRAGMENT

PROMPT_TEMPLATE = """You are an expert in build123d (a Python CAD library based on OpenCASCADE).

I have a 3D solid that I need to reconstruct using ONLY pure Python algebraic
build123d primitives - no external file imports, no embedded binary data.

Your job: write a Python snippet that reconstructs this exact solid using
build123d operations. The snippet must end by assigning the final result to
the variable `_part` (which must be a build123d Part / Compound / Solid).

Use Algebra mode (`+`, `-`, `&`, `*` for placement). All build123d names
are already imported via `from build123d import *` in the executing
namespace; do not re-import.

Reference material (the canonical idioms, common patterns, and mistakes
to avoid):

================================================================
""" + FULL_PROMPT_FRAGMENT + """
================================================================

Geometry of the solid to reconstruct:
{geometry}

{previous_attempt}

Target: symmetric difference < 1.0% of source volume.

Respond with ONLY the Python code, no markdown fences, no explanation.
The code must end with `_part = ...` (the final assignment).
"""


def call_llm(geometry_text, previous_attempt=""):
    """Call the configured codegen LLM (Gemini Flash). Returns (code, error).

    Uses gemini-3.5-flash by default. Override with
    GEMINI_CODEGEN_MODEL env var.
    """
    prompt = PROMPT_TEMPLATE.format(
        geometry=geometry_text,
        previous_attempt=previous_attempt,
    )
    from .gemini_codegen import call_gemini_for_code
    return call_gemini_for_code(prompt)


# ---------------------------------------------------------------------------
# Main fitter entry
# ---------------------------------------------------------------------------

def try_fit_llm(solid, tol=0.01, max_iterations=3, target_sym_pct=1.0,
                best_result=None):
    """Try to fit a solid by asking Claude to write build123d code for it.

    Verifies each attempt against the source. Iterates up to max_iterations
    feeding errors/results back. Accepts only if sym-diff < target_sym_pct.

    If `best_result` is given (a FitResult from an earlier tier that
    didn't meet the quality bar), the LLM prompt is framed as a REPAIR
    task: "here is build123d code that produces a close-but-wrong
    reconstruction, fix it to match the source exactly."

    Returns FitResult(code, comp, acc, "llm", details).
    On any error or exhaustion, returns FitResult(None, ...).
    """
    geometry = summarize_solid(solid)
    # Seed the prompt with the best-effort code from an earlier tier,
    # if available. Gives the LLM a starting point instead of a blank page.
    if (best_result is not None and getattr(best_result, "code_body", None)
        and (best_result.completeness > 0 or best_result.accuracy > 0)):
        src_vol = getattr(solid, "volume", 0.0)
        previous = (
            f"\nAN EARLIER FITTER PRODUCED A PARTIAL RECONSTRUCTION\n"
            f"(kind={best_result.kind}, "
            f"comp={best_result.completeness*100:.3f}%, "
            f"acc={best_result.accuracy*100:.3f}%).\n"
            f"  Source volume: {src_vol:.6f}\n"
            f"The reconstruction is close but doesn't hit the 1% sym-diff bar.\n"
            f"Here is its code (you may adapt, replace, or rewrite it):\n"
            f"{best_result.code_body[:2000]}\n"
        )
    else:
        previous = ""
    last_error = "no attempts made"

    for attempt in range(max_iterations):
        code, err = call_llm(geometry, previous)
        if code is None:
            return FitResult(None, 0.0, 0.0, "none", f"llm: {err}")

        part, exec_err = execute_recipe(code)
        if part is None:
            previous = (f"\nPREVIOUS ATTEMPT FAILED with execution error:\n"
                        f"{exec_err[:500]}\n\nPlease fix and try again.\n")
            last_error = f"execution: {exec_err[:200]}"
            continue

        # Verify
        try:
            v = _verify_fit(solid, part, tol)
        except Exception as e:
            previous = (f"\nPREVIOUS ATTEMPT EXECUTED but verification failed:\n"
                        f"{e}\n")
            last_error = f"verify error: {e}"
            continue

        sym_pct = (1 - v.completeness + 1 - v.accuracy) * 100
        if sym_pct < target_sym_pct * 2:  # *2 because comp + acc
            return FitResult(
                code + ("\n" if not code.endswith("\n") else ""),
                v.completeness, v.accuracy, "llm",
                f"LLM refined (attempt {attempt+1}, "
                f"comp={v.completeness*100:.3f}%, "
                f"acc={v.accuracy*100:.3f}%)"
            )

        # Quality not good enough; iterate with feedback
        previous = (f"\nPREVIOUS ATTEMPT EXECUTED but quality insufficient:\n"
                    f"  Source volume: {solid.volume:.6f}\n"
                    f"  Your reconstruction volume: {part.volume:.6f}\n"
                    f"  Completeness: {v.completeness*100:.3f}% (need >99%)\n"
                    f"  Accuracy: {v.accuracy*100:.3f}% (need >99%)\n"
                    f"Please refine your code to better match the source.\n"
                    f"Previous code:\n{code[:1000]}\n")
        last_error = f"quality: comp={v.completeness*100:.2f}% acc={v.accuracy*100:.2f}%"

    return FitResult(None, 0.0, 0.0, "none",
                     f"llm: exhausted {max_iterations} attempts; last: {last_error}")
