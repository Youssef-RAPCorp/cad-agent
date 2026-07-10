"""shape_generator.py — create new build123d shapes from natural language.

Given a text description like "a hex nut with M6 thread hole and 5mm height",
this module calls an LLM to generate build123d code, executes it in a
sandbox, and returns the resulting Part object. Optionally writes the
code + an exported STEP/STL to disk.

Backend: Google Gemini Flash via google-genai SDK (shared with the
image_to_shape vision pass and llm_fitter repair tier — single backend
for simplicity).

This is a NEW-SHAPE path, not a reconstruction path — there is no source
solid to verify against. We do check that the generated code executes
without error and produces a Part with nonzero volume, but the geometric
quality is the LLM's responsibility.

Environment:
  GEMINI_API_KEY (or GOOGLE_API_KEY)   required
  GEMINI_CODEGEN_MODEL                 optional; default 'gemini-3.5-flash'

Usage:
    from cad_agent3.shape_generator import generate_shape

    result = generate_shape("A hex nut with M6 thread hole, 5mm thick")
    if result.part is not None:
        print(f"Generated volume: {result.part.volume:.2f} mm^3")
        result.save_step("/tmp/nut.step")
"""

from __future__ import annotations

import os
import sys
import traceback
import importlib
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

from .build123d_reference import FULL_PROMPT_FRAGMENT

SHAPE_GEN_PROMPT = """You are an expert CAD engineer. Generate build123d \
Python code that constructs the shape described below.

Hard requirements:
1. Use ONLY the build123d library. Import with `from build123d import *`.
2. Assign the final result to a module-level variable named `part`.
3. Use MILLIMETERS as the unit for all dimensions.
4. The code must execute top-to-bottom with no user input.
5. Do NOT import any non-build123d module.
6. Do NOT write files, print to stdout, or access the network.
7. Return ONLY the Python code — no markdown fences, no explanation.
8. Use Algebra mode (with `+`, `-`, `&`, and `*` for placement). Do NOT
   mix in Builder mode (`with BuildPart() as p:`) unless absolutely
   required for an operation algebra mode cannot express.
9. PRECISION PARTS: a mathematically accurate gear helper is already in
   scope (do NOT define or import it):
       involute_gear(module, teeth, thickness=5.0, bore=0.0,
                     pressure_angle=20.0) -> Part
   It returns a true involute spur gear, axis +Z, extruded z=0..thickness,
   tip diameter = module*(teeth+2). ALWAYS use it for gears/cogs — never
   hand-model teeth. MESHING RULE: two meshing gears share the same
   module and their center distance MUST equal module*(teeth1+teeth2)/2;
   lay out gear trains by computing shaft positions from that rule.
   PHASING RULE: also rotate one gear of each meshing pair about its own
   axis so teeth align with the mate's tooth spaces (a half tooth pitch,
   Rot(0, 0, 180/teeth), is the usual correction) — unphased gears
   collide tooth-on-tooth. Gears on the SAME shaft never touch gears on
   another shaft unless meshing; keep non-meshing gears' tip circles
   clear of each other.

Reference material (the canonical idioms, common patterns, and known
mistakes — follow this guide):

================================================================
""" + FULL_PROMPT_FRAGMENT + """
================================================================

Shape description:
{description}

{extra_constraints}
Generate the code now (Algebra mode, MM, named `part`)."""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    description: str
    code: Optional[str]           # the generated build123d source
    part: Optional[object]        # the build123d Part (None on failure)
    error: Optional[str]          # populated when something fails
    attempts: int = 0
    volume: float = 0.0
    bbox: Optional[tuple] = None  # (min, max) tuples as nested tuples

    def save_code(self, path: str) -> bool:
        if self.code is None:
            return False
        try:
            with open(path, "w") as f:
                f.write(self.code)
            return True
        except OSError:
            return False

    def save_step(self, path: str) -> bool:
        if self.part is None:
            return False
        try:
            from build123d import export_step
            export_step(self.part, path)
            return True
        except Exception:
            return False

    def save_stl(self, path: str) -> bool:
        if self.part is None:
            return False
        try:
            from build123d import export_stl
            export_stl(self.part, path)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# LLM call (Codex via Responses API)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LLM call (Gemini Flash via google-genai)
# ---------------------------------------------------------------------------

def _call_codex(prompt: str) -> tuple:
    """Call the configured codegen LLM. Returns (code, error).

    Name kept for backwards compat with internal callers (builder.py).
    Backend is now Gemini Flash; see gemini_codegen.call_gemini_for_code.
    """
    from .gemini_codegen import call_gemini_for_code
    return call_gemini_for_code(prompt)



# ---------------------------------------------------------------------------
# Sandboxed execution
# ---------------------------------------------------------------------------

def _execute_code(code: str) -> tuple[Optional[object], Optional[str]]:
    """Execute generated code in a fresh namespace. Return (part, error)."""
    # Whitelist module lookup: we allow build123d and its submodules only.
    allowed_roots = ("build123d",)
    real_import = importlib.import_module

    def guarded_import(name, *args, **kwargs):
        root = name.split(".")[0]
        if root not in allowed_roots:
            raise ImportError(f"module '{name}' is not permitted in generated code")
        return real_import(name, *args, **kwargs)

    safe_builtins = {
        "__import__": lambda n, g=None, l=None, f=None, lv=0:
            guarded_import(n) if n.split(".")[0] in allowed_roots
            else (_ for _ in ()).throw(ImportError(f"module '{n}' not permitted")),
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "float": float, "int": int, "isinstance": isinstance,
        "len": len, "list": list, "map": map, "max": max, "min": min,
        "range": range, "round": round, "set": set, "sorted": sorted,
        "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    }

    ns = {"__builtins__": safe_builtins}
    # Deterministic precision-part helpers (accurate involute gears
    # etc.) — generated code calls these instead of hand-modeling.
    from .stdparts import SANDBOX_HELPERS
    ns.update(SANDBOX_HELPERS)
    try:
        exec(code, ns, ns)
    except Exception:
        return None, traceback.format_exc(limit=3)

    part = ns.get("part")
    if part is None:
        return None, "code did not define a 'part' variable"
    if not hasattr(part, "volume"):
        return None, f"'part' is a {type(part).__name__}, not a build123d Part"
    try:
        v = part.volume
    except Exception as e:
        return None, f"could not read part.volume: {e}"
    if v is None or v <= 0:
        return None, f"part has non-positive volume: {v}"
    return part, None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_shape(
    description: str,
    extra_constraints: str = "",
    max_iterations: int = 3,
    execute: bool = True,
) -> GenerationResult:
    """Generate a build123d Part from a natural-language description.

    The LLM is asked to write build123d code; we execute it in a sandbox
    and verify it produces a Part with nonzero volume. If it fails, we
    iterate with the error as context, up to max_iterations times.

    Args:
        description: natural-language shape description.
        extra_constraints: optional additional constraints text appended
            to the prompt (e.g. "Target volume around 1000 mm^3").
        max_iterations: retry budget on parse/exec failures.
        execute: if False, return the generated code without running it
            in the sandbox (part will be None, and there are no retries
            since execution errors are what drive them).

    Returns:
        GenerationResult with part (Part or None), code, error, attempts.
    """
    last_err = "no attempts made"
    last_code: Optional[str] = None

    for attempt in range(1, max_iterations + 1):
        # On retry, append the prior error to the prompt.
        if attempt == 1:
            prompt = SHAPE_GEN_PROMPT.format(
                description=description,
                extra_constraints=extra_constraints,
            )
        else:
            retry_note = (
                f"YOUR PREVIOUS ATTEMPT FAILED:\n{last_err[:500]}\n\n"
                f"Previous code:\n{last_code[:1500] if last_code else '<none>'}\n\n"
                f"Fix the issue and try again.\n"
            )
            prompt = SHAPE_GEN_PROMPT.format(
                description=description,
                extra_constraints=extra_constraints + "\n" + retry_note,
            )

        code, err = _call_codex(prompt)
        if code is None:
            return GenerationResult(
                description=description, code=None, part=None,
                error=f"LLM call failed: {err}", attempts=attempt,
            )
        last_code = code

        if not execute:
            return GenerationResult(
                description=description, code=code, part=None,
                error=None, attempts=attempt,
            )

        part, exec_err = _execute_code(code)
        if part is not None:
            bb = None
            try:
                bbv = part.bounding_box()
                bb = ((bbv.min.X, bbv.min.Y, bbv.min.Z),
                      (bbv.max.X, bbv.max.Y, bbv.max.Z))
            except Exception:
                pass
            return GenerationResult(
                description=description, code=code, part=part,
                error=None, attempts=attempt,
                volume=float(part.volume), bbox=bb,
            )
        last_err = exec_err or "unknown execution error"

    return GenerationResult(
        description=description, code=last_code, part=None,
        error=f"exhausted {max_iterations} attempts; last: {last_err}",
        attempts=max_iterations,
    )
