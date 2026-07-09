"""cad_agent.drawings — 2D engineering drawings (ASME/ISO-style DXF sheets).

Turn parts into production-style drawing sheets: title blocks, dimensions,
leaders, revision blocks, and collision-aware annotation placement.

Two ways in:

1. Declarative — build a DrawingSpec yourself (full control):

    from cad_agent.drawings import (
        DrawingSpec, DrawingBuilder, TitleBlock, Units,
        Circle, LinearDim, DiameterDim, Ref, Snap,
        render_preview, validate,
    )

    spec = DrawingSpec(
        sheet="A3", units=Units.MILLIMETERS, workflow="mech",
        title_block=TitleBlock(title="WIDGET", drawing_no="RAP-0001"),
        entities=[Circle(id="H1", center=(20, 20), radius=4.0)],
        annotations=[DiameterDim(id="D1", target=Ref(entity_id="H1", snap=Snap.CENTER))],
    )
    builder = DrawingBuilder(spec)
    doc = builder.build()
    builder.save("widget.dxf")
    render_preview(doc, "widget_sheet.png", layout="paperspace")

2. From a generated model — one call from a CADAgent result (or any
   STL/OBJ/PLY/OFF/GLB file) to a third-angle multi-view sheet:

    from cad_agent import CADAgent
    from cad_agent.drawings import draw_multiview

    result = CADAgent().generate("A 50x30x10mm bracket with two M3 holes")
    sheet = draw_multiview(result)
    print(sheet.summary())
    # OK: drawing 'part_4f8e2c19' on A2 at 2:1
    #   DXF:     cad_output/part_4f8e2c19_sheet.dxf
    #   preview: cad_output/part_4f8e2c19_sheet.png

Requires the optional drawing dependencies (from the repo root):

    pip install -e ".[drawings]"

(`draw_multiview` additionally needs `trimesh` for mesh import, which the
`drawings` extra includes.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

try:
    from ._vendored.rapcad_drawings import (
        AngularDim, Annotation, Arc, BuildReport, Circle, DiameterDim,
        DrawingBuilder, DrawingSpec, Ellipse, Entity, Finding, Hatch,
        LinearDim, Line, Mesh3DView, Polyline, RadialDim, Rectangle, Ref,
        RevisionEntry, Snap, TextLabel, TitleBlock, Units, build_dxf,
        render_preview, validate,
    )
except ImportError as exc:  # pragma: no cover - exercised only without extras
    raise ImportError(
        "cad_agent.drawings requires the optional drawing dependencies "
        f"({exc}). Install them with:\n\n"
        "    pip install -e \".[drawings]\"    # from the cad-agent repo root\n"
        "    # or: pip install ezdxf pydantic matplotlib numpy trimesh\n"
    ) from exc

from ._vendored.rapcad_drawings.standards import SHEETS

__all__ = [
    # Sheet-from-model bridge + LLM-assisted generation
    "draw_multiview", "generate_drawing", "SheetResult",
    # Top level
    "DrawingSpec", "DrawingBuilder", "BuildReport", "build_dxf",
    "render_preview", "validate", "Finding",
    # Geometry kinds
    "Line", "Polyline", "Rectangle", "Circle", "Arc", "Ellipse", "Hatch",
    "Mesh3DView", "Entity",
    # Annotations
    "TextLabel", "LinearDim", "RadialDim", "DiameterDim", "AngularDim",
    "Annotation",
    # Misc
    "Ref", "Snap", "Units",
    "TitleBlock", "RevisionEntry",
]


@dataclass
class SheetResult:
    """Outcome of a draw_multiview() call.

    Fields:
      name:       artifact name (stem of the output files)
      sheet:      sheet designation ("A2", "ANSI_B", ...)
      scale:      chosen view scale (2.0 means 2:1 on paper; None when
                  the spec came from an LLM and no numeric scale applies)
      dxf_path:   path to the written DXF
      png_path:   path to the sheet preview PNG (None if preview=False)
      spec:       the DrawingSpec that was built — tweak and rebuild for
                  custom annotations
      report:     the builder's BuildReport (placement warnings etc.)
      findings:   post-build validator findings
    """

    name: str
    sheet: str
    scale: Optional[float]
    dxf_path: Path
    png_path: Optional[Path] = None
    spec: Optional[DrawingSpec] = None
    report: Optional[BuildReport] = None
    findings: List[Finding] = field(default_factory=list)

    def summary(self) -> str:
        scale_txt = f" at {_scale_label(self.scale)}" if self.scale else ""
        lines = [f"OK: drawing '{self.name}' on {self.sheet}{scale_txt}"]
        lines.append(f"  DXF:     {self.dxf_path}")
        if self.png_path:
            lines.append(f"  preview: {self.png_path}")
        errors = [f for f in self.findings if f.severity == "error"]
        if errors:
            lines.append(f"  validator errors: {len(errors)}")
        return "\n".join(lines)


# Standard drawing scales (ISO 5455). The auto-fit picks the largest one
# that still fits all four views on the sheet.
_STANDARD_SCALES = (0.02, 0.05, 0.1, 0.2, 0.25, 0.5,
                    1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 50.0)

# Vertical band reserved above the bottom border for the title block and
# revision block (the title block itself is 180 x 60 mm, bottom-right).
_TITLE_BLOCK_RESERVE = 70.0


def _scale_label(s: float) -> str:
    return f"{s:g}:1" if s >= 1.0 else f"1:{1.0 / s:g}"


def _pick_scale(fit: float) -> float:
    candidates = [s for s in _STANDARD_SCALES if s <= fit]
    return candidates[-1] if candidates else fit


def draw_multiview(
    source,
    *,
    name: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    sheet: str = "A2",
    scale: Optional[float] = None,
    title_block: Optional[TitleBlock] = None,
    revisions: Optional[List[RevisionEntry]] = None,
    spacing: float = 25.0,
    preview: bool = True,
    dpi: int = 180,
) -> SheetResult:
    """Produce a third-angle multi-view drawing sheet from a 3D model.

    Lays out FRONT, TOP, RIGHT and ISO projections of the model on one
    sheet with a title block, writes a DXF (plus an optional PNG
    preview), and returns a SheetResult.

    Args:
        source: a CADResult from CADAgent.generate() (its STL is used),
                or a path to any mesh file trimesh can read (STL, OBJ,
                PLY, OFF, GLB).
        name: artifact name; outputs are `<name>_sheet.dxf` / `.png`.
              Defaults to the CADResult name or the mesh file stem.
        output_dir: where to write outputs. Defaults to the CADResult's
              output_dir, or the mesh file's directory.
        sheet: sheet designation — ISO A0-A4 or ASME ANSI_A-ANSI_E.
        scale: view scale (2.0 = 2:1). Auto-fit to a standard ISO 5455
              scale if omitted.
        title_block: full TitleBlock override. A sensible default is
              built from `name` and the chosen scale if omitted.
        revisions: optional revision-block entries.
        spacing: gap between adjacent views, in sheet mm.
        preview: also render a paperspace PNG of the sheet.
        dpi: preview resolution.

    Returns:
        SheetResult — see its docstring. `result.spec` holds the built
        DrawingSpec; edit `spec.annotations` and rebuild with
        DrawingBuilder for dimensioned sheets.
    """
    # --- Resolve the mesh source ------------------------------------
    stl_path: Optional[Path] = None
    if hasattr(source, "stl_path"):  # CADResult (duck-typed)
        if not source.stl_path:
            raise ValueError(
                "CADResult has no STL artifact (was write_stl disabled, "
                "or did generation fail?) — cannot derive drawing views"
            )
        stl_path = Path(source.stl_path)
        if name is None:
            name = source.metadata.get("name") or stl_path.stem
        if output_dir is None:
            output_dir = source.output_dir
    else:
        stl_path = Path(source)
    if not stl_path.exists():
        raise FileNotFoundError(f"mesh file not found: {stl_path}")
    name = name or stl_path.stem
    outdir = Path(output_dir) if output_dir else stl_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    if sheet not in SHEETS:
        raise ValueError(f"unknown sheet {sheet!r}; choose one of {sorted(SHEETS)}")
    sh = SHEETS[sheet]

    # --- Project the three orthographic views to size the layout ----
    # (model3d lazily imports trimesh and raises a helpful error if
    # it's missing.)
    from ._vendored.rapcad_drawings.model3d import load_mesh, project_mesh

    mesh = load_mesh(str(stl_path))
    front = project_mesh(mesh, view="front", source_path=str(stl_path))
    top = project_mesh(mesh, view="top", source_path=str(stl_path))
    right = project_mesh(mesh, view="right", source_path=str(stl_path))
    iso = project_mesh(mesh, view="iso", source_path=str(stl_path))
    if not front.edges_2d:
        raise ValueError(f"no projectable edges found in {stl_path}")

    # --- Choose a scale that fits front+right / front+top ------------
    usable_w = sh.inside_width
    usable_h = sh.inside_height - _TITLE_BLOCK_RESERVE
    if scale is None:
        fit_w = (usable_w - spacing) / max(front.width + right.width, 1e-9)
        fit_h = (usable_h - spacing) / max(front.height + top.height, 1e-9)
        # 0.85 leaves breathing room for view labels and the iso view.
        scale = _pick_scale(min(fit_w, fit_h) * 0.85)

    # --- Third-angle layout (origins are view centers, sheet mm) -----
    fw, fh = front.width * scale, front.height * scale
    tw, th = top.width * scale, top.height * scale
    rw = right.width * scale
    total_w = fw + spacing + rw
    total_h = fh + spacing + th
    bx = sh.border_left + max((usable_w - total_w) / 2.0, 0.0)
    by = (sh.border_bottom + _TITLE_BLOCK_RESERVE
          + max((usable_h - total_h) / 2.0, 0.0))

    front_c = (bx + fw / 2.0, by + fh / 2.0)
    top_c = (front_c[0], by + fh + spacing + th / 2.0)
    right_c = (bx + fw + spacing + rw / 2.0, front_c[1])
    iso_c = (right_c[0], top_c[1])
    iso_scale = scale * 0.7

    spec = DrawingSpec(
        sheet=sheet,
        units=Units.MILLIMETERS,
        workflow="mech",
        title_block=title_block or TitleBlock(
            title=name.replace("_", " ").upper(),
            subtitle="Multi-view from 3D model",
            drawing_no=f"CAD-{name.upper()}",
            scale=_scale_label(scale),
            notes=["VIEWS DERIVED FROM 3D MODEL"],
        ),
        revisions=revisions or [],
        entities=[
            Mesh3DView(id="V_FRONT", path=str(stl_path), view="front",
                       origin=front_c, scale=scale, label="FRONT VIEW"),
            Mesh3DView(id="V_TOP", path=str(stl_path), view="top",
                       origin=top_c, scale=scale, label="TOP VIEW"),
            Mesh3DView(id="V_RIGHT", path=str(stl_path), view="right",
                       origin=right_c, scale=scale, label="RIGHT VIEW"),
            Mesh3DView(id="V_ISO", path=str(stl_path), view="iso",
                       origin=iso_c, scale=iso_scale, label="ISOMETRIC"),
        ],
        annotations=[],
    )

    builder = DrawingBuilder(spec)
    doc = builder.build()
    findings = validate(builder.index, builder._halo)

    dxf_path = outdir / f"{name}_sheet.dxf"
    builder.save(str(dxf_path))

    png_path: Optional[Path] = None
    if preview:
        png_path = outdir / f"{name}_sheet.png"
        render_preview(doc, str(png_path), layout="paperspace", dpi=dpi)

    return SheetResult(
        name=name,
        sheet=sheet,
        scale=scale,
        dxf_path=dxf_path,
        png_path=png_path,
        spec=spec,
        report=builder.report,
        findings=findings,
    )


# ---------------------------------------------------------------------------
# LLM-assisted drawing generation
# ---------------------------------------------------------------------------
#
# The vendored drawing engine was designed as an LLM target: DrawingSpec
# is "the LLM I/O contract" and BuildReport exists so a model can
# self-correct. This is that missing caller — it asks the configured LLM
# for a DrawingSpec JSON and loops Pydantic-validation / build / validator
# errors back into the prompt until the sheet is clean.

_EXAMPLE_SPEC_JSON = """\
{"sheet": "A3", "units": "mm", "workflow": "mech",
 "title_block": {"title": "SPACER PLATE", "drawing_no": "RAP-0009", "rev": "A",
                 "scale": "1:1", "material": "AL 6061"},
 "entities": [
   {"kind": "rectangle", "id": "BODY", "corner": [0, 0], "width": 80, "height": 40},
   {"kind": "circle", "id": "H1", "center": [20, 20], "radius": 3.2},
   {"kind": "circle", "id": "H2", "center": [60, 20], "radius": 3.2}
 ],
 "annotations": [
   {"kind": "linear_dim", "id": "D_W", "p1": {"entity_id": "BODY", "snap": "vertex", "index": 0},
    "p2": {"entity_id": "BODY", "snap": "vertex", "index": 1}, "side": "below"},
   {"kind": "linear_dim", "id": "D_HH", "p1": {"entity_id": "H1", "snap": "center"},
    "p2": {"entity_id": "H2", "snap": "center"}, "side": "above"},
   {"kind": "diameter_dim", "id": "D_H1", "target": {"entity_id": "H1", "snap": "center"}},
   {"kind": "text", "id": "N1", "text": "2X THRU",
    "target": {"entity_id": "H1", "snap": "center"}, "height": 3.5}
 ]}"""

_DRAWING_PROMPT = """You are an expert mechanical drafter. Produce a DrawingSpec \
JSON object describing a complete, fully dimensioned 2D engineering drawing of \
the part below.

Hard requirements:
1. Output ONLY one JSON object valid against the schema — no markdown fences,
   no commentary before or after.
2. Draw the part's true geometry at real size in millimeters, near the origin.
   The sheet viewport auto-scales; do NOT try to scale coordinates to the sheet.
3. Give every entity and annotation a short unique id.
4. Dimension the part fully: overall width/height, every hole diameter and
   position, radii, and any critical feature spacing. Add text labels for
   notes (thread callouts, finish).
5. Attach annotations to geometry with Ref targets ({{"entity_id": ...,
   "snap": ...}}). Omit explicit offsets/positions for annotation text — the
   engine places text collision-free automatically.
6. Fill in the title block: title, drawing_no, rev, scale, material.
7. Pick the smallest sheet that fits comfortably (A3 for typical parts).

JSON schema for DrawingSpec:
{schema}

Example of a valid spec:
{example}

Part to draw:
{description}
{feedback}
Return the DrawingSpec JSON now."""


def generate_drawing(
    description: str,
    *,
    name: Optional[str] = None,
    output_dir: Union[str, Path] = "./cad_output",
    sheet: Optional[str] = None,
    max_revisions: int = 3,
    preview: bool = True,
    dpi: int = 180,
    verbose: bool = False,
) -> SheetResult:
    """Generate a dimensioned 2D engineering drawing from a text description.

    Asks the configured LLM (Gemini by default; set CAD_AGENT_BACKEND or
    LLM_BACKEND to 'anthropic' for Claude) to emit a DrawingSpec JSON,
    validates it with Pydantic, builds the sheet, and runs the collision
    validator. Validation, build, and collision errors are fed back to
    the LLM for revision, up to `max_revisions` attempts.

    Args:
        description: what to draw, with dimensions — e.g. "A flange
            plate, OD 120mm, with 8 M6 clearance holes on a 95mm bolt
            circle and a 40mm center bore".
        name: artifact name; outputs are `<name>_sheet.dxf` / `.png`.
            Defaults to a slug of the description.
        output_dir: where to write outputs (default ./cad_output).
        sheet: force a sheet size; by default the LLM picks one.
        max_revisions: LLM retry budget on invalid/colliding output.
        preview: also render a paperspace PNG.
        dpi: preview resolution.
        verbose: print per-attempt progress to stderr.

    Returns:
        SheetResult. Raises RuntimeError if no valid drawing is produced
        within max_revisions attempts.
    """
    import json
    import os
    import re
    import sys

    from pydantic import ValidationError

    from ._vendored.cad_agent3 import gemini_codegen

    # Map the cad-agent backend convention onto the vendored caller's.
    if (os.environ.get("CAD_AGENT_BACKEND", "").lower() == "anthropic"):
        os.environ.setdefault("LLM_BACKEND", "anthropic")

    schema_json = json.dumps(DrawingSpec.model_json_schema(),
                             separators=(",", ":"))
    feedback = ""
    last_error = "no attempts made"

    for attempt in range(1, max_revisions + 1):
        if verbose:
            print(f"[cad_agent.drawings] attempt {attempt}/{max_revisions}",
                  file=sys.stderr)
        prompt = _DRAWING_PROMPT.format(
            schema=schema_json,
            example=_EXAMPLE_SPEC_JSON,
            description=description,
            feedback=feedback,
        )
        raw, err = gemini_codegen.call_gemini_for_code(prompt)
        if raw is None:
            raise RuntimeError(f"LLM call failed: {err}")

        def _revise(problem: str) -> str:
            if verbose:
                print(f"[cad_agent.drawings]   revising: {problem[:200]}",
                      file=sys.stderr)
            return (
                f"\nYOUR PREVIOUS ATTEMPT FAILED:\n{problem[:2000]}\n\n"
                f"Previous JSON (truncated):\n{raw[:2000]}\n\n"
                f"Fix these problems and output the corrected JSON.\n"
            )

        try:
            spec = DrawingSpec.model_validate_json(raw)
        except ValidationError as e:
            last_error = f"schema validation failed: {e}"
            feedback = _revise(last_error)
            continue
        if sheet is not None:
            spec.sheet = sheet

        try:
            builder = DrawingBuilder(spec)
            doc = builder.build()
        except Exception as e:
            last_error = f"build failed: {type(e).__name__}: {e}"
            feedback = _revise(last_error)
            continue

        findings = validate(builder.index, builder._halo)
        errors = [f for f in findings if f.severity == "error"]
        if errors:
            last_error = "validator errors: " + "; ".join(
                f"{f.entity_id}: {f.message}" for f in errors[:5])
            feedback = _revise(last_error)
            continue

        if name is None:
            slug = re.sub(r"[^a-z0-9]+", "_",
                          (spec.title_block.title or description).lower())
            name = slug.strip("_")[:40] or "drawing"
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        dxf_path = outdir / f"{name}_sheet.dxf"
        builder.save(str(dxf_path))
        png_path: Optional[Path] = None
        if preview:
            png_path = outdir / f"{name}_sheet.png"
            render_preview(doc, str(png_path), layout="paperspace", dpi=dpi)

        return SheetResult(
            name=name,
            sheet=spec.sheet,
            scale=None,
            dxf_path=dxf_path,
            png_path=png_path,
            spec=spec,
            report=builder.report,
            findings=findings,
        )

    raise RuntimeError(
        f"no valid drawing after {max_revisions} attempts; last: {last_error}"
    )
