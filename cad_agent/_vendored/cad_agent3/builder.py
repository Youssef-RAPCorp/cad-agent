"""builder.py — central CAD design API.

The Builder wraps every existing cad_agent3 module behind a single
operations interface, holding state across calls. This is the layer
that the orchestrator and design chat sit on top of.

State held by a Builder instance:
    .part           — the current build123d Part being constructed
    .code           — the build123d source code that produced .part
    .history        — list of (operation, kwargs, result) tuples
    .context        — optional ScanResult for "design with context" mode
    .target_envelope — optional (l, w, h) max dimensions in mm

Operations (all mutate self where it makes sense, all return self):
    start_from_text(description, **kw)         — text → Part via shape_generator
    start_from_image(image_path, **kw)          — image → Part via image_to_shape
    start_from_scan(step_or_fcstd_path)         — load context, no Part yet
    add_feature(description)                    — Codex emits new code that
                                                  runs in a namespace where
                                                  `existing_part` is bound
    subtract_feature(description)               — same but result subtracts
    fillet_edges(selector_text, radius)         — apply a fillet
    chamfer_edges(selector_text, length)        — apply a chamfer
    verify_fit_against(other_part)              — sym-diff + intersection check
    decompose_to_primitives()                   — run fitter on current part
    validate(envelope=None, holes=None)         — run validator.py
    render(out_path)                             — render preview
    emit(out_path)                               — write standalone .py recipe
    snapshot()                                   — return a deep-copy state

This module IS the new Builder. The cad_agent3 modules already in the
package (scanner, fitter, engine, verifier, emitter, shape_generator,
image_to_shape) are CALLED from here; nothing is rewritten.

Usage:
    from cad_agent3 import Builder
    b = Builder()
    b.start_from_text("a 50mm × 30mm × 5mm plate")
    b.add_feature("a 5mm-diameter through-hole at the center")
    b.fillet_edges("vertical", 2)
    b.render("/tmp/preview.svg")
    b.emit("/tmp/recipe.py")
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

# We deliberately import lazily inside methods so importing the Builder
# doesn't pay the cost of build123d cold-import at module load.


# ---------------------------------------------------------------------------
# History record
# ---------------------------------------------------------------------------

@dataclass
class HistoryEntry:
    operation: str
    args: dict
    success: bool
    duration_s: float
    error: Optional[str] = None
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class Builder:
    """Central design-time state holder. Wraps existing cad_agent3 modules
    behind a fluent operations API.
    """

    def __init__(self,
                 verbose: bool = True,
                 default_units: str = "mm"):
        self.part = None             # current build123d Part
        self.code: Optional[str] = None  # source code that built .part
        self.history: list[HistoryEntry] = []
        self.context = None          # ScanResult or None
        self.context_path: Optional[str] = None
        self.target_envelope: Optional[tuple] = None
        self.verbose = verbose
        self.default_units = default_units

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _record(self, op: str, args: dict, success: bool,
                duration: float, error: Optional[str] = None,
                notes: Optional[str] = None) -> None:
        entry = HistoryEntry(
            operation=op, args=args, success=success,
            duration_s=duration, error=error, notes=notes)
        self.history.append(entry)
        if self.verbose:
            mark = "✓" if success else "✗"
            tail = f" — {error}" if error else (f" — {notes}" if notes else "")
            print(f"  [builder] {mark} {op} ({duration:.1f}s){tail}",
                  flush=True)

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(f"  [builder] {msg}", flush=True)

    # -------------------------------------------------------------------
    # Starting fresh
    # -------------------------------------------------------------------

    def start_from_text(self, description: str,
                         max_iterations: int = 3,
                         extra_constraints: str = "") -> "Builder":
        """Initialize part from a text description. Calls shape_generator
        which goes to OpenAI Codex."""
        from .shape_generator import generate_shape
        t0 = time.time()
        result = generate_shape(
            description=description,
            extra_constraints=extra_constraints,
            max_iterations=max_iterations,
        )
        dt = time.time() - t0
        if result.part is None:
            self._record("start_from_text", {"description": description},
                          success=False, duration=dt, error=result.error)
            raise RuntimeError(
                f"start_from_text failed: {result.error}")
        self.part = result.part
        self.code = result.code
        self._record("start_from_text", {"description": description},
                      success=True, duration=dt,
                      notes=f"vol={result.volume:.2f} mm³")
        return self

    def start_from_image(self, image_path: str,
                          scale_hint_mm: str = "",
                          max_iterations: int = 3) -> "Builder":
        """Initialize part from an image. Calls image_to_shape (Gemini
        vision + Codex codegen)."""
        from .image_to_shape import image_to_shape
        t0 = time.time()
        result = image_to_shape(
            image_path=image_path,
            scale_hint_mm=scale_hint_mm,
            max_iterations=max_iterations,
        )
        dt = time.time() - t0
        if result.part is None:
            err = (result.vision.error if result.vision.error
                   else (result.generation.error if result.generation
                         else "unknown"))
            self._record("start_from_image", {"image_path": image_path},
                          success=False, duration=dt, error=err)
            raise RuntimeError(f"start_from_image failed: {err}")
        self.part = result.part
        self.code = result.generation.code if result.generation else None
        self._record("start_from_image", {"image_path": image_path},
                      success=True, duration=dt,
                      notes=f"vol={result.generation.volume:.2f} mm³"
                            if result.generation else None)
        return self

    def start_from_scan(self, source_path: str) -> "Builder":
        """Load a STEP/FCStd as design CONTEXT. Does NOT set self.part —
        the context is for the LLM to design AROUND.
        """
        from .scanner import scan_source
        t0 = time.time()
        try:
            self.context = scan_source(source_path)
            self.context_path = source_path
            dt = time.time() - t0
            n = len(self.context.solids_info)
            self._record("start_from_scan", {"source_path": source_path},
                          success=True, duration=dt,
                          notes=f"{n} solids loaded as context")
        except Exception as e:
            dt = time.time() - t0
            self._record("start_from_scan", {"source_path": source_path},
                          success=False, duration=dt,
                          error=f"{type(e).__name__}: {e}")
            raise
        return self

    # -------------------------------------------------------------------
    # Modifying the current part
    # -------------------------------------------------------------------

    def add_feature(self, description: str,
                     mode: str = "add",
                     max_iterations: int = 3) -> "Builder":
        """Ask Codex to extend the current part with a new feature.

        `mode`: "add" (union) or "subtract" (cut).

        The LLM is given the current code AND a short description of the
        addition; it returns a snippet that operates on `existing_part`.
        We exec it and replace self.part with the result.
        """
        if self.part is None:
            raise RuntimeError(
                "no current part — call start_from_text/image/etc first")
        from .shape_generator import _call_codex, _execute_code
        from .build123d_reference import FULL_PROMPT_FRAGMENT

        op_word = {"add": "ADDED to", "subtract": "SUBTRACTED FROM"}.get(
            mode, "added to")
        prompt = f"""You are extending an existing build123d Part.

The current part was built with this code:
```python
{self.code or "# (code unavailable; you will work from the part itself)"}
```

Write a Python snippet that produces a NEW PART representing a feature
to be {op_word} the existing part.

Constraints:
- Use Algebra mode.
- Do NOT redefine `existing_part`.
- Assign the new feature to a variable named `feature`.
- Use millimeters.
- Return ONLY the code, no markdown fences.

Reference material:
{FULL_PROMPT_FRAGMENT}

Feature description:
{description}

Generate the feature code now:"""

        t0 = time.time()
        for attempt in range(1, max_iterations + 1):
            code, err = _call_codex(prompt)
            if code is None:
                dt = time.time() - t0
                self._record("add_feature", {"description": description,
                                                "mode": mode},
                              success=False, duration=dt,
                              error=f"Codex: {err}")
                raise RuntimeError(f"add_feature: Codex call failed: {err}")

            # Execute the snippet in a namespace pre-populated with
            # `existing_part`. Reuse shape_generator's sandbox helpers
            # but with a fresh namespace.
            full_code = (
                "from build123d import *\n"
                f"# existing_part injected by Builder\n"
                + code
                + "\n"
            )
            feature, exec_err = self._exec_with_existing(full_code)
            if feature is None:
                if attempt >= max_iterations:
                    dt = time.time() - t0
                    self._record("add_feature", {"description": description,
                                                  "mode": mode},
                                  success=False, duration=dt,
                                  error=f"exec failed: {exec_err}")
                    raise RuntimeError(
                        f"add_feature: exec failed: {exec_err}")
                # Retry with error feedback
                prompt = (prompt + f"\n\nPREVIOUS ATTEMPT FAILED with:\n"
                          f"{exec_err[:300]}\n\nFix the issue and try again.")
                continue

            try:
                if mode == "subtract":
                    new_part = self.part - feature
                else:
                    new_part = self.part + feature
            except Exception as e:
                dt = time.time() - t0
                self._record("add_feature", {"description": description,
                                              "mode": mode},
                              success=False, duration=dt,
                              error=f"{mode} op failed: {type(e).__name__}: {e}")
                raise RuntimeError(
                    f"add_feature: {mode} op failed: {type(e).__name__}: {e}")

            self.part = new_part
            # Append the snippet to our running code record
            join_op = "+" if mode == "add" else "-"
            self.code = (self.code or "") + (
                f"\n# Feature {mode}: {description!r}\n"
                f"existing_part = part\n"
                f"{code}\n"
                f"part = existing_part {join_op} feature\n"
            )
            dt = time.time() - t0
            try:
                v = self.part.volume
            except Exception:
                v = 0.0
            self._record("add_feature", {"description": description,
                                          "mode": mode},
                          success=True, duration=dt,
                          notes=f"vol now {v:.2f} mm³ (attempts: {attempt})")
            return self

        # exhausted retries
        dt = time.time() - t0
        self._record("add_feature", {"description": description, "mode": mode},
                      success=False, duration=dt,
                      error="max_iterations exhausted")
        raise RuntimeError("add_feature: max_iterations exhausted")

    def subtract_feature(self, description: str,
                           max_iterations: int = 3) -> "Builder":
        """Convenience for add_feature(mode='subtract')."""
        return self.add_feature(description, mode="subtract",
                                  max_iterations=max_iterations)

    def _exec_with_existing(self, code: str) -> tuple:
        """Execute generated code with `existing_part` bound. Returns
        (feature_part_or_None, error_or_None).
        """
        import importlib, traceback
        allowed_roots = ("build123d",)
        real_import = importlib.import_module

        def guarded_import(name, *a, **kw):
            root = name.split(".")[0]
            if root not in allowed_roots:
                raise ImportError(
                    f"module '{name}' not permitted in feature code")
            return real_import(name, *a, **kw)

        safe_builtins = {
            "__import__": lambda n, g=None, l=None, f=None, lv=0:
                guarded_import(n) if n.split(".")[0] in allowed_roots
                else (_ for _ in ()).throw(
                    ImportError(f"module '{n}' not permitted")),
            "abs": abs, "all": all, "any": any, "bool": bool,
            "dict": dict, "enumerate": enumerate, "float": float,
            "int": int, "isinstance": isinstance, "len": len,
            "list": list, "map": map, "max": max, "min": min,
            "range": range, "round": round, "set": set,
            "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
            "zip": zip,
        }

        ns = {"__builtins__": safe_builtins, "existing_part": self.part}
        try:
            exec(code, ns, ns)
        except Exception:
            return None, traceback.format_exc(limit=3)
        feature = ns.get("feature")
        if feature is None:
            return None, "code did not define 'feature' variable"
        if not hasattr(feature, "volume"):
            return None, f"'feature' is a {type(feature).__name__}, not a Part"
        try:
            v = feature.volume
        except Exception as e:
            return None, f"could not read feature.volume: {e}"
        if v is None or v <= 0:
            return None, f"feature has non-positive volume: {v}"
        return feature, None

    # -------------------------------------------------------------------
    # Edge / face operations
    # -------------------------------------------------------------------

    def fillet_edges(self, selector: str = "all",
                      radius: float = 1.0) -> "Builder":
        """Fillet edges of the current part.

        `selector` accepts a few canonical strings:
            "all"      — every edge
            "top"      — top edges (max Z)
            "bottom"   — bottom edges (min Z)
            "vertical" — edges parallel to Z
        """
        if self.part is None:
            raise RuntimeError("no current part to fillet")
        from build123d import fillet, Axis
        t0 = time.time()
        try:
            edges = self._select_edges(selector)
            self.part = fillet(edges, radius)
            dt = time.time() - t0
            self.code = (self.code or "") + (
                f"\n# Fillet {selector!r} edges, r={radius}\n"
                f"part = fillet(part.edges()"
                f"{self._selector_to_code(selector)}, {radius})\n"
            )
            self._record("fillet_edges",
                          {"selector": selector, "radius": radius},
                          success=True, duration=dt)
        except Exception as e:
            dt = time.time() - t0
            self._record("fillet_edges",
                          {"selector": selector, "radius": radius},
                          success=False, duration=dt,
                          error=f"{type(e).__name__}: {e}")
            raise
        return self

    def chamfer_edges(self, selector: str = "all",
                       length: float = 1.0) -> "Builder":
        """Chamfer edges. See fillet_edges for selector vocabulary."""
        if self.part is None:
            raise RuntimeError("no current part to chamfer")
        from build123d import chamfer
        t0 = time.time()
        try:
            edges = self._select_edges(selector)
            self.part = chamfer(edges, length)
            dt = time.time() - t0
            self.code = (self.code or "") + (
                f"\n# Chamfer {selector!r} edges, len={length}\n"
                f"part = chamfer(part.edges()"
                f"{self._selector_to_code(selector)}, {length})\n"
            )
            self._record("chamfer_edges",
                          {"selector": selector, "length": length},
                          success=True, duration=dt)
        except Exception as e:
            dt = time.time() - t0
            self._record("chamfer_edges",
                          {"selector": selector, "length": length},
                          success=False, duration=dt,
                          error=f"{type(e).__name__}: {e}")
            raise
        return self

    def _select_edges(self, selector: str):
        """Translate a selector keyword to an actual ShapeList of edges."""
        from build123d import Axis
        edges = self.part.edges()
        sel = selector.lower().strip()
        if sel in ("all", "*", ""):
            return edges
        if sel == "vertical":
            return edges | Axis.Z
        if sel == "horizontal":
            return edges | Axis.X | edges | Axis.Y
        if sel == "top":
            faces = self.part.faces() >> Axis.Z
            return faces.edges() if hasattr(faces, "edges") else faces[0].edges()
        if sel == "bottom":
            faces = self.part.faces() << Axis.Z
            return faces.edges() if hasattr(faces, "edges") else faces[0].edges()
        # Unknown — fall through to all edges with a warning recorded
        self._say(f"unknown edge selector {sel!r}; using all edges")
        return edges

    def _selector_to_code(self, selector: str) -> str:
        sel = selector.lower().strip()
        if sel in ("all", "*", ""): return ""
        if sel == "vertical":      return " | Axis.Z"
        if sel == "horizontal":    return " | Axis.X | edges | Axis.Y"
        if sel == "top":           return ".faces().sort_by(Axis.Z)[-1:].edges()"
        if sel == "bottom":        return ".faces().sort_by(Axis.Z)[:1].edges()"
        return ""

    # -------------------------------------------------------------------
    # Inspection
    # -------------------------------------------------------------------

    def context_solids(self) -> list:
        """Return the solids loaded as context (from start_from_scan)."""
        if self.context is None:
            return []
        from .scanner import _load_solids
        return _load_solids(self.context_path) if self.context_path else []

    def context_bbox(self) -> Optional[tuple]:
        """Return the overall (xmin, ymin, zmin, xmax, ymax, zmax) of context
        solids — useful as an envelope hint."""
        sols = self.context_solids()
        if not sols:
            return None
        xs_min = min(s.bounding_box().min.X for s in sols)
        ys_min = min(s.bounding_box().min.Y for s in sols)
        zs_min = min(s.bounding_box().min.Z for s in sols)
        xs_max = max(s.bounding_box().max.X for s in sols)
        ys_max = max(s.bounding_box().max.Y for s in sols)
        zs_max = max(s.bounding_box().max.Z for s in sols)
        return (xs_min, ys_min, zs_min, xs_max, ys_max, zs_max)

    # -------------------------------------------------------------------
    # Verification
    # -------------------------------------------------------------------

    def verify_fit_against(self, other_part) -> dict:
        """Check overlap with another part. Returns dict with
        {intersection_volume, sym_diff_volume, fits_inside, fully_contained}.

        Useful for "does my bracket clear this motor body" — pass the
        motor's solid as `other_part`.
        """
        if self.part is None:
            raise RuntimeError("no current part")
        from .verifier import compute_intersection, safe_volume
        v_self = safe_volume(self.part)
        v_other = safe_volume(other_part)
        intersection = compute_intersection(self.part, other_part)
        v_int = safe_volume(intersection) if intersection else 0.0
        # Sym-diff = |A| + |B| - 2|A∩B|
        sym_diff = v_self + v_other - 2 * v_int
        fits_inside = v_int >= v_self * 0.99      # self is mostly inside other
        contains_other = v_int >= v_other * 0.99  # other is mostly inside self
        return {
            "self_volume": v_self,
            "other_volume": v_other,
            "intersection_volume": v_int,
            "sym_diff_volume": sym_diff,
            "self_fits_inside_other": fits_inside,
            "other_fits_inside_self": contains_other,
        }

    def decompose_to_primitives(self,
                                  target_sym_pct: float = 1.0,
                                  enable_llm: bool = False,
                                  workers: int = 1) -> dict:
        """Run the FitEngine on the current part to produce a parametric
        decomposition. Returns a dict with the FitResult + diagnostic.

        Useful for converting an LLM-generated freeform shape into a
        cleaner box/cylinder/extrude composition.
        """
        if self.part is None:
            raise RuntimeError("no current part")
        from .engine import FitEngine
        from .fitter import set_force_primitives
        set_force_primitives(True)
        engine = FitEngine(target_sym_pct=target_sym_pct,
                            enable_llm=enable_llm, verbose=self.verbose)
        # The current part may itself be a Compound of multiple solids
        try:
            solids = list(self.part.solids())
        except Exception:
            solids = [self.part]
        results = []
        for i, s in enumerate(solids):
            fr, diag = engine.fit_solid(s, solid_idx=i)
            results.append({"solid_idx": i, "kind": fr.kind,
                              "completeness": fr.completeness,
                              "accuracy": fr.accuracy,
                              "code_body": fr.code_body,
                              "details": fr.details})
        return {"per_solid": results}

    def validate(self,
                  expected_volume_min: Optional[float] = None,
                  expected_volume_max: Optional[float] = None,
                  envelope: Optional[tuple] = None,
                  expected_hole_count: Optional[int] = None,
                  expected_hole_diameter: Optional[float] = None) -> Any:
        """Run validator.py checks. Returns a ValidationReport."""
        if self.part is None:
            raise RuntimeError("no current part")
        from .validator import validate as _validate
        return _validate(
            self.part,
            expected_volume_min=expected_volume_min,
            expected_volume_max=expected_volume_max,
            envelope=envelope or self.target_envelope,
            expected_hole_count=expected_hole_count,
            expected_hole_diameter=expected_hole_diameter,
        )

    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------

    def render(self, out_path: str,
                width: int = 800, height: int = 600,
                prefer: str = "auto") -> str:
        """Render the current part to PNG (preferred) or SVG. Returns
        the actual output path."""
        if self.part is None:
            raise RuntimeError("no current part to render")
        from .render_preview import render_preview
        result = render_preview(self.part, out_path, width, height, prefer)
        self._say(f"render: {result.backend} → {result.path}")
        return result.path

    def emit(self, out_path: str) -> str:
        """Write the current part as a standalone build123d Python script.
        Returns the path written.

        The emitted script is self-contained: importing build123d, running
        the recorded code, exporting STEP + STL.
        """
        if self.part is None:
            raise RuntimeError("no current part to emit")
        if self.code is None:
            # If we don't have the source code (e.g. context-only mode),
            # serialize the part via STEP + a tiny loader.
            return self._emit_via_step(out_path)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                    exist_ok=True)
        try:
            v = self.part.volume
            bb = self.part.bounding_box()
            sx = bb.max.X - bb.min.X
            sy = bb.max.Y - bb.min.Y
            sz = bb.max.Z - bb.min.Z
        except Exception:
            v, sx, sy, sz = 0.0, 0.0, 0.0, 0.0

        header = (
            '"""Auto-generated build123d design.\n'
            f'\n'
            f'Volume: {v:.3f} mm³\n'
            f'Bounding box: {sx:.1f} × {sy:.1f} × {sz:.1f} mm\n'
            f'Operations: {len(self.history)}\n'
            f'"""\n'
            'import os\n'
            'from build123d import *\n'
            '\n'
            'HERE = os.path.dirname(os.path.abspath(__file__))\n'
            '\n'
        )
        body = self.code
        footer = (
            '\n'
            'if __name__ == "__main__":\n'
            '    step_out = os.path.join(HERE, "design.step")\n'
            '    stl_out  = os.path.join(HERE, "design.stl")\n'
            '    export_step(part, step_out)\n'
            '    export_stl(part, stl_out)\n'
            '    print(f"wrote: {step_out}")\n'
            '    print(f"wrote: {stl_out}")\n'
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(header + body + footer)
        self._say(f"emit: wrote {out_path} ({len(header+body+footer)} chars)")
        return out_path

    def _emit_via_step(self, out_path: str) -> str:
        """Fallback emit: the part is real but we have no source code, so
        serialize via STEP and emit a loader script."""
        import os
        d = os.path.dirname(os.path.abspath(out_path)) or "."
        os.makedirs(d, exist_ok=True)
        step_path = os.path.join(d, "design_geometry.step")
        from build123d import export_step
        export_step(self.part, step_path)
        loader = (
            '"""Auto-generated design loader (no source code available)."""\n'
            'import os\n'
            'from build123d import import_step, export_step, export_stl\n'
            '\n'
            'HERE = os.path.dirname(os.path.abspath(__file__))\n'
            f'part = import_step(os.path.join(HERE, "design_geometry.step"))\n'
            '\n'
            'if __name__ == "__main__":\n'
            '    export_stl(part, os.path.join(HERE, "design.stl"))\n'
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(loader)
        return out_path

    # -------------------------------------------------------------------
    # State / debugging
    # -------------------------------------------------------------------

    def summary(self) -> str:
        if self.part is None:
            return "Builder: (no current part)"
        try:
            v = self.part.volume
            bb = self.part.bounding_box()
            sx = bb.max.X - bb.min.X
            sy = bb.max.Y - bb.min.Y
            sz = bb.max.Z - bb.min.Z
            n_faces = len(list(self.part.faces()))
        except Exception as e:
            return f"Builder: (part unreadable: {e})"
        ctx = ""
        if self.context is not None:
            ctx = (f"\n  context: {len(self.context.solids_info)} solids "
                   f"from {os.path.basename(self.context_path or '?')}")
        ops = len(self.history)
        return (f"Builder: vol={v:.3f} mm³  bbox={sx:.1f}×{sy:.1f}×{sz:.1f} mm"
                f"  faces={n_faces}  ops={ops}{ctx}")

    def history_summary(self) -> str:
        if not self.history:
            return "(no operations)"
        lines = []
        for i, h in enumerate(self.history, 1):
            mark = "✓" if h.success else "✗"
            tail = (f" — {h.error}" if h.error
                    else (f" — {h.notes}" if h.notes else ""))
            lines.append(f"  {i:2d}. {mark} {h.operation} "
                         f"({h.duration_s:.1f}s){tail}")
        return "\n".join(lines)
