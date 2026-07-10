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

import math
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
    "draw_multiview", "generate_drawing", "draft_drawing", "SheetResult",
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

# Candidate sheets for auto-selection, both orientations.
_AUTO_SHEETS = ("A4", "A4P", "A3", "A3P", "A2", "A2P",
                "A1", "A1P", "A0", "A0P")


def _scale_label(s: float) -> str:
    return f"{s:g}:1" if s >= 1.0 else f"1:{1.0 / s:g}"


def _resolve_model(source, name, output_dir):
    """Resolve a CADResult or file path into (stl_path, name, outdir).

    STEP files are tessellated through build123d (trimesh can't read
    BREP); the intermediate STL is kept beside the outputs so specs
    referencing it stay rebuildable.
    """
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

    if stl_path.suffix.lower() in (".step", ".stp"):
        from build123d import import_step, export_stl
        shape = import_step(str(stl_path))
        mesh_path = outdir / f"{name}_tessellated.stl"
        export_stl(shape, str(mesh_path))
        stl_path = mesh_path
    return stl_path, name, outdir


def _pick_scale(fit: float) -> float:
    candidates = [s for s in _STANDARD_SCALES if s <= fit]
    return candidates[-1] if candidates else fit


def draw_multiview(
    source,
    *,
    name: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    sheet: Optional[str] = None,
    scale: Optional[float] = None,
    title_block: Optional[TitleBlock] = None,
    revisions: Optional[List[RevisionEntry]] = None,
    spacing: float = 25.0,
    dimensions: bool = True,
    hidden: bool = True,
    preview: bool = True,
    dpi: int = 180,
    verbose: bool = False,
) -> SheetResult:
    """Produce a third-angle multi-view drawing sheet from a 3D model.

    Lays out FRONT, TOP, RIGHT and ISO projections of the model on one
    sheet with a title block, writes a DXF (plus an optional PNG
    preview), and returns a SheetResult.

    Args:
        source: a CADResult from CADAgent.generate() (its STL is used),
                a path to any mesh file trimesh can read (STL, OBJ,
                PLY, OFF, GLB), or a STEP file (tessellated through
                build123d; the intermediate STL is kept in output_dir).
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
        spacing: minimum gap between adjacent views, in sheet mm. The
              horizontal gaps stretch (up to 3x) to use free sheet width.
        dimensions: add overall dimensions — width and height on the
              front view, depth on the top view. The dim text shows the
              true model size in mm (not the scaled sheet distance).
        hidden: draw occluded edges dashed (ASME hidden lines) on the
              orthographic views. The isometric never shows hidden
              lines. Either way, occluded edges are removed from the
              solid linework — views are true hidden-line-removed
              projections, not wireframes.
        preview: also render a paperspace PNG of the sheet.
        dpi: preview resolution.
        verbose: print pipeline stages to stderr.

    Returns:
        SheetResult — see its docstring. `result.spec` holds the built
        DrawingSpec; edit `spec.annotations` and rebuild with
        DrawingBuilder for dimensioned sheets.
    """
    import sys

    stl_path, name, outdir = _resolve_model(source, name, output_dir)

    if sheet is not None and sheet not in SHEETS:
        raise ValueError(f"unknown sheet {sheet!r}; choose one of {sorted(SHEETS)}")

    def _say(msg):
        if verbose:
            print(f"[cad_agent.drawings] {msg}", file=sys.stderr)

    _say(f"model: {stl_path}")

    # --- Project the three orthographic views to size the layout ----
    # (model3d lazily imports trimesh and raises a helpful error if
    # it's missing.)
    from ._vendored.rapcad_drawings.model3d import load_mesh, project_mesh

    mesh = load_mesh(str(stl_path))
    _say(f"mesh loaded: {len(mesh.faces)} triangles, "
         f"{mesh.body_count} bodies")
    front = project_mesh(mesh, view="front", source_path=str(stl_path))
    top = project_mesh(mesh, view="top", source_path=str(stl_path))
    right = project_mesh(mesh, view="right", source_path=str(stl_path))
    iso = project_mesh(mesh, view="iso", source_path=str(stl_path))
    if not front.edges_2d:
        raise ValueError(f"no projectable edges found in {stl_path}")

    # --- Choose sheet + scale -----------------------------------------
    def _fit_scale(cand) -> float:
        # The iso column (at 0.7x) sits right of the right view, so it
        # counts toward the width budget.
        fw_fit = (cand.inside_width - 2 * spacing) / max(
            front.width + right.width + 0.7 * iso.width, 1e-9)
        fh_fit = ((cand.inside_height - _TITLE_BLOCK_RESERVE - spacing)
                  / max(front.height + top.height, 1e-9))
        # 0.85 leaves breathing room for view labels and dimensions.
        return _pick_scale(min(fw_fit, fh_fit) * 0.85)

    if sheet is None:
        # Pick the sheet (either orientation) the drawing fills best at
        # its standard scale — the weaker of the two axes decides, so a
        # tall part lands on portrait and a flat one on landscape. Ties
        # go to the smaller sheet, then to the scale nearest 1:1.
        def _fill(cand) -> float:
            s = scale or _fit_scale(cand)
            w_need = s * (front.width + right.width + 0.7 * iso.width) + 2 * spacing
            h_need = s * (front.height + top.height) + spacing
            return min(w_need / cand.inside_width,
                       h_need / (cand.inside_height - _TITLE_BLOCK_RESERVE))

        sheet = min(_AUTO_SHEETS,
                    key=lambda nm: (-_fill(SHEETS[nm]),
                                    SHEETS[nm].width_mm * SHEETS[nm].height_mm,
                                    abs(math.log(scale or _fit_scale(SHEETS[nm])))))
    sh = SHEETS[sheet]
    usable_w = sh.inside_width
    usable_h = sh.inside_height - _TITLE_BLOCK_RESERVE
    if scale is None:
        scale = _fit_scale(sh)

    # --- Third-angle layout (origins are view centers, sheet mm) -----
    fw, fh = front.width * scale, front.height * scale
    tw, th = top.width * scale, top.height * scale
    rw = right.width * scale
    iso_scale = scale * 0.7
    iw = iso.width * iso_scale
    # Stretch the horizontal gaps into the leftover width so tall
    # narrow parts don't huddle in one corner of a landscape sheet.
    gap_x = min(3 * spacing,
                max(spacing, (usable_w - (fw + rw + iw)) / 4.0))
    total_w = fw + gap_x + rw + gap_x + iw
    total_h = fh + spacing + th
    bx = sh.border_left + max((usable_w - total_w) / 2.0, 0.0)
    by = (sh.border_bottom + _TITLE_BLOCK_RESERVE
          + max((usable_h - total_h) / 2.0, 0.0))

    _say(f"projected views (hidden-line removed): "
         f"front {front.width:.1f}x{front.height:.1f}, "
         f"top {top.width:.1f}x{top.height:.1f}, "
         f"right {right.width:.1f}x{right.height:.1f} mm")
    _say(f"sheet {sheet} at {_scale_label(scale)}"
         + (" (auto)" if scale else ""))

    front_c = (bx + fw / 2.0, by + fh / 2.0)
    top_c = (front_c[0], by + fh + spacing + th / 2.0)
    right_c = (bx + fw + gap_x + rw / 2.0, front_c[1])
    # Iso in its own column right of the right view, clamped to the sheet.
    iso_x = min(bx + fw + gap_x + rw + gap_x + iw / 2.0,
                sh.width_mm - sh.border_right - iw / 2.0)
    iso_c = (iso_x, top_c[1])

    # --- Overall dimensions -------------------------------------------
    # Anchored to the views' registered bounding boxes (vertices run
    # counterclockwise from the lower-left). The views are drawn scaled,
    # so the dim text is overridden with the true model size.
    annotations: List[Annotation] = []
    if dimensions:
        def _fmt(v: float) -> str:
            return f"{v:.1f}".rstrip("0").rstrip(".")

        def _dim(did, view_id, i1, i2, side, true_mm):
            return LinearDim(
                id=did,
                p1=Ref(entity_id=view_id, snap=Snap.VERTEX, index=i1),
                p2=Ref(entity_id=view_id, snap=Snap.VERTEX, index=i2),
                side=side,
                text_override=_fmt(true_mm),
            )

        annotations = [
            _dim("D_WIDTH", "V_FRONT", 0, 1, "below", front.width),
            _dim("D_HEIGHT", "V_FRONT", 0, 3, "left", front.height),
            _dim("D_DEPTH", "V_TOP", 1, 2, "right", top.height),
        ]

    spec = DrawingSpec(
        sheet=sheet,
        units=Units.MILLIMETERS,
        workflow="mech",
        title_block=title_block or TitleBlock(
            title=name.replace("_", " ").upper(),
            subtitle="Multi-view from 3D model",
            drawing_no=f"CAD-{name.upper()}",
            scale=_scale_label(scale),
            notes=["VIEWS DERIVED FROM 3D MODEL",
                   "OVERALL DIMENSIONS IN MM"] if dimensions
                  else ["VIEWS DERIVED FROM 3D MODEL"],
        ),
        revisions=revisions or [],
        entities=[
            Mesh3DView(id="V_FRONT", path=str(stl_path), view="front",
                       origin=front_c, scale=scale, label="FRONT VIEW",
                       show_hidden=hidden),
            Mesh3DView(id="V_TOP", path=str(stl_path), view="top",
                       origin=top_c, scale=scale, label="TOP VIEW",
                       show_hidden=hidden),
            Mesh3DView(id="V_RIGHT", path=str(stl_path), view="right",
                       origin=right_c, scale=scale, label="RIGHT VIEW",
                       show_hidden=hidden),
            Mesh3DView(id="V_ISO", path=str(stl_path), view="iso",
                       origin=iso_c, scale=iso_scale, label="ISOMETRIC"),
        ],
        annotations=annotations,
    )

    _say("building sheet (placing views, dimensions, title block)...")
    builder = DrawingBuilder(spec)
    doc = builder.build()
    findings = validate(builder.index, builder._halo)
    _say(f"build report: {builder.report}; validator findings: "
         f"{len(findings)}")

    dxf_path = outdir / f"{name}_sheet.dxf"
    builder.save(str(dxf_path))
    _say(f"wrote DXF: {dxf_path}")

    png_path: Optional[Path] = None
    if preview:
        png_path = outdir / f"{name}_sheet.png"
        _say(f"rendering preview at {dpi} dpi...")
        render_preview(doc, str(png_path), layout="paperspace", dpi=dpi)
        _say(f"wrote preview: {png_path}")

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


# ---------------------------------------------------------------------------
# Smart LLM drafting — the model studies the actual 3D shape
# ---------------------------------------------------------------------------
#
# Unlike generate_drawing (text description -> 2D geometry drawn from
# scratch), draft_drawing hands the LLM the REAL model: measured facts,
# classified view images (visible vs hidden ink), and the Mesh3DView
# mechanism so the returned DrawingSpec embeds true projections of the
# 3D shape with dimensions and callouts attached to them.

_SMART_PROMPT = """You are an expert mechanical drafter. You are given a real \
3D CAD model: measured facts below{image_note}. Produce a DrawingSpec JSON for \
a complete engineering drawing of THIS model.

How to embed the real shape: use `mesh3d_view` entities with `"path": "MODEL"` \
(the engine substitutes the actual file). Each mesh3d_view draws the true \
projected geometry of the model:
- `view`: one of front/top/bottom/right/left/back/iso
- `origin`: the CENTER of the view on the sheet, in sheet mm
- `scale`: drawing scale (view size on sheet = model size x scale)
- `show_hidden`: true to draw occluded edges dashed (use on orthographic
  views; leave false on iso)
- `label`: e.g. "FRONT VIEW"
Each mesh3d_view also registers a rectangle named by its `id` whose corners
are vertices 0=lower-left, 1=lower-right, 2=upper-right, 3=upper-left — attach
dimensions to them with Ref targets, e.g.
  {{"kind": "linear_dim", "id": "D_W",
    "p1": {{"entity_id": "V_FRONT", "snap": "vertex", "index": 0}},
    "p2": {{"entity_id": "V_FRONT", "snap": "vertex", "index": 1}},
    "side": "below", "text_override": "<true mm size>"}}
IMPORTANT: views are drawn scaled, so every dimension on a mesh3d_view MUST
set text_override to the true model size in mm (given in the facts).

DETAIL (ZOOM) VIEWS — use these for every fine feature worth studying
(finials, dial/hands, hole groups, joints, moldings): add another
mesh3d_view of the SAME view direction with:
- `region`: [x0, y0, x1, y1] — the crop window in model mm measured from
  that view's LOWER-LEFT corner (view sizes are in the facts; e.g. the
  top 40mm of a 222mm-tall front view is [0, 182, W, 222])
- `scale`: 2x-5x the main scale so the detail reads large
- `frame`: true (draws the standard detail boundary circle)
- `label`: e.g. "DETAIL A (4:1)"
Mark the source area on the parent view: a thin `circle` entity (layer
"VISIBLE") at the feature's sheet position with a `text` label "A"
targeting it. Compute sheet position as parent_origin + (feature_center
- view_center) * parent_scale; its radius should ring just the zoomed
region (~= parent_scale x region_diagonal / 2), never the whole view.

SPACE BUDGET — views may NOT overlap (the engine rejects overlapping
views): a framed detail view occupies a circle of diameter ~=
sqrt(region_w^2 + region_h^2) x detail_scale + 6mm centered on its
origin. Reserve that footprint plus >=15mm clearance from every other
view before choosing origins; shrink the region, lower the detail
scale, or take a bigger sheet if it doesn't fit.

Design the sheet like a drafter — PRODUCTION quality, densely annotated:
1. Choose the views this shape needs (tall parts: front+right+top; flat
   parts: top+front; always one iso at ~0.7x scale in a corner) PLUS
   1-3 detail views of the most intricate regions.
2. Use the suggested sheet/scale/layout from the facts unless you have a
   reason to deviate; go one sheet size UP if needed to fit the detail
   views comfortably. Keep all content inside the border (left 20mm,
   right 10mm, top 10mm) and above the 70mm title-block band at the
   bottom; keep >=15mm clear between views.
3. Dimension thoroughly: overall W/H/D across two views, plus feature
   sizes you can measure from the facts/images (text_override with the
   true mm value); feature callouts (`kind`: "text" with a Ref target on
   a view id and a leader, e.g. "Ø6.4 THRU 2X") for holes/bosses/details
   you identify in the images.
4. Add centerlines through symmetric features: `line` entities on layer
   "CENTER" extending ~3mm past the geometry.
5. Fill the title block completely (title naming what the object IS,
   drawing_no, rev, scale, material if inferable, notes list with
   material/finish/tolerance remarks).
6. Output ONLY one JSON object valid against the schema — no markdown
   fences, no commentary.

JSON schema for DrawingSpec:
{schema}

MODEL FACTS
{facts}
{notes}{feedback}
Return the DrawingSpec JSON now."""


def _render_view_images(mesh, outdir, name, dpi=100):
    """Render classified per-view PNGs (visible black, hidden gray) for
    the multimodal prompt. Returns list of (view_name, path)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ._vendored.rapcad_drawings.model3d import project_mesh

    images = []
    for view in ("front", "top", "right", "iso"):
        pv = project_mesh(mesh, view=view)
        if not pv.edges_2d:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        for a, b in pv.hidden_edges_2d:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="0.7", lw=0.6)
        for a, b in pv.edges_2d:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="k", lw=1.0)
        ax.set_aspect("equal")
        ax.set_title(f"{view.upper()} (gray = hidden)", fontsize=9)
        path = outdir / f"{name}_llmview_{view}.png"
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        images.append((view, path))
    return images


def _call_llm_for_spec(prompt: str, image_paths=(), verbose: bool = False):
    """LLM call for spec drafting. Multimodal Gemini when images are
    given and the SDK/key allow; otherwise the text-only vendored call.
    Returns (text, error)."""
    import os
    import sys

    from ._vendored.cad_agent3 import gemini_codegen

    backend = (os.environ.get("LLM_BACKEND")
               or os.environ.get("CAD_AGENT_BACKEND", "gemini")).lower()
    if verbose:
        model_name = os.environ.get("GEMINI_CODEGEN_MODEL",
                                    "gemini-3.5-flash")
        print(f"[cad_agent.drawings] calling {backend} ({model_name}) "
              f"with {len(image_paths)} view image(s)...", file=sys.stderr)
    if image_paths and backend != "anthropic":
        try:
            from google import genai
            from google.genai import types
            api_key = (os.environ.get("GEMINI_API_KEY")
                       or os.environ.get("GOOGLE_API_KEY"))
            if api_key:
                client = genai.Client(api_key=api_key)
                model = os.environ.get("GEMINI_CODEGEN_MODEL",
                                       "gemini-3.5-flash")
                parts = [types.Part.from_bytes(
                            data=Path(p).read_bytes(), mime_type="image/png")
                         for _, p in image_paths]
                resp = client.models.generate_content(
                    model=model, contents=parts + [prompt])
                text = gemini_codegen._strip_fences(
                    getattr(resp, "text", "") or "")
                if text:
                    return text, None
        except Exception as exc:
            if verbose:
                print(f"[cad_agent.drawings] multimodal call failed "
                      f"({exc}); falling back to text-only", file=sys.stderr)
    return gemini_codegen.call_gemini_for_code(prompt)


def _view_overlaps(spec, builder, tol: float = 5.0):
    """Pairs of Mesh3DView entities whose placed footprints (including
    detail frames) overlap by more than tol mm on both axes."""
    boxes = []
    for ent in spec.entities:
        if isinstance(ent, Mesh3DView):
            ge = builder.index.get(ent.id)
            if ge is not None and ge.aabb is not None:
                boxes.append((ent.id, ge.aabb))
    out = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (ida, a), (idb, b) = boxes[i], boxes[j]
            w = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
            h = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
            if w > tol and h > tol:
                out.append((ida, idb, w, h))
    return out


def draft_drawing(
    source,
    *,
    notes: str = "",
    name: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    sheet: Optional[str] = None,
    max_revisions: int = 3,
    preview: bool = True,
    dpi: int = 180,
    verbose: bool = False,
) -> SheetResult:
    """LLM-drafted engineering drawing of a real 3D model.

    The LLM studies the shape — measured dimensions, per-view sizes, and
    rendered images of the hidden-line-classified views — and composes
    the whole DrawingSpec itself: which views to show, where to place
    them, what to dimension, feature callouts, notes, and title block.
    The real model geometry is embedded via Mesh3DView entities, so the
    sheet shows true projections, not an LLM redrawing.

    Args:
        source: a CADResult (its STL is used) or a mesh/STEP file path.
        notes: extra guidance woven into the prompt, e.g. "material is
            6061; add a section note; imperial title block".
        name/output_dir/sheet/preview/dpi: as draw_multiview.
        max_revisions: LLM retry budget on invalid/colliding output.
        verbose: print per-attempt progress to stderr.

    Returns:
        SheetResult. Raises RuntimeError if no valid drawing emerges
        within max_revisions attempts.
    """
    import json
    import os
    import re
    import sys

    from pydantic import ValidationError

    from ._vendored.rapcad_drawings.model3d import load_mesh, project_mesh

    if (os.environ.get("CAD_AGENT_BACKEND", "").lower() == "anthropic"):
        os.environ.setdefault("LLM_BACKEND", "anthropic")

    def _say(msg):
        if verbose:
            print(f"[cad_agent.drawings] {msg}", file=sys.stderr)

    stl_path, name, outdir = _resolve_model(source, name, output_dir)
    if sheet is not None and sheet not in SHEETS:
        raise ValueError(f"unknown sheet {sheet!r}; choose one of {sorted(SHEETS)}")
    _say(f"model: {stl_path}")

    # --- Study the shape ----------------------------------------------
    mesh = load_mesh(str(stl_path))
    views = {v: project_mesh(mesh, view=v, source_path=str(stl_path))
             for v in ("front", "top", "right", "iso")}
    if not views["front"].edges_2d:
        raise ValueError(f"no projectable edges found in {stl_path}")

    # Reuse the deterministic layout as the LLM's starting suggestion
    # (throwaway artifacts; cleaned up below).
    template = draw_multiview(stl_path, name=f"{name}__layout",
                              output_dir=outdir, sheet=sheet, preview=False)
    try:
        (outdir / f"{name}__layout_sheet.dxf").unlink()
    except OSError:
        pass
    tmpl_views = {e.view: e for e in template.spec.entities}
    sug_sheet = template.sheet
    sug_scale = template.scale
    layout_lines = "\n".join(
        f"  - {v}: origin ({e.origin[0]:.0f}, {e.origin[1]:.0f}), "
        f"scale {e.scale:g}"
        for v, e in tmpl_views.items())

    f, t_, r = views["front"], views["top"], views["right"]
    facts = (
        f"- overall size, mm: width {f.width:.1f} (X), "
        f"depth {t_.height:.1f} (Y), height {f.height:.1f} (Z)\n"
        f"- view sizes, model mm: front {f.width:.1f} x {f.height:.1f}, "
        f"top {t_.width:.1f} x {t_.height:.1f}, "
        f"right {r.width:.1f} x {r.height:.1f}\n"
        f"- mesh: {len(mesh.faces)} triangles, "
        f"{mesh.body_count} bodies\n"
        f"- suggested sheet: {sug_sheet} "
        f"({SHEETS[sug_sheet].width_mm:.0f} x "
        f"{SHEETS[sug_sheet].height_mm:.0f} mm), scale {sug_scale:g}\n"
        f"- suggested view layout (origins are view centers):\n"
        f"{layout_lines}\n"
        f"- sheets available: A4/A3/A2/A1/A0 landscape, A4P-A0P portrait, "
        f"ANSI_A-ANSI_E"
    )

    _say(f"studied shape: {f.width:.1f} x {t_.height:.1f} x "
         f"{f.height:.1f} mm, suggested {sug_sheet} at "
         f"{_scale_label(sug_scale)}")
    images = _render_view_images(mesh, outdir, name)
    _say(f"rendered {len(images)} classified view image(s) for the LLM")
    image_note = (" and rendered images of its views (attached; gray "
                  "lines are hidden/occluded edges)") if images else ""

    schema_json = json.dumps(DrawingSpec.model_json_schema(),
                             separators=(",", ":"))
    notes_txt = f"\nEXTRA GUIDANCE FROM THE USER:\n{notes}\n" if notes else ""
    feedback = ""
    last_error = "no attempts made"

    try:
        for attempt in range(1, max_revisions + 1):
            if verbose:
                print(f"[cad_agent.drawings] smart draft attempt "
                      f"{attempt}/{max_revisions}", file=sys.stderr)
            prompt = _SMART_PROMPT.format(
                schema=schema_json, facts=facts, notes=notes_txt,
                image_note=image_note, feedback=feedback)
            raw, err = _call_llm_for_spec(prompt, images, verbose=verbose)
            if raw is None:
                raise RuntimeError(f"LLM call failed: {err}")

            def _revise(problem: str) -> str:
                if verbose:
                    print(f"[cad_agent.drawings]   revising: "
                          f"{problem[:200]}", file=sys.stderr)
                return (f"\nYOUR PREVIOUS ATTEMPT FAILED:\n{problem[:2000]}"
                        f"\n\nPrevious JSON (truncated):\n{raw[:2000]}\n\n"
                        f"Fix these problems and output the corrected "
                        f"JSON.\n")

            try:
                spec = DrawingSpec.model_validate_json(raw)
            except ValidationError as e:
                last_error = f"schema validation failed: {e}"
                feedback = _revise(last_error)
                continue
            if sheet is not None:
                spec.sheet = sheet

            # The LLM references the model as "MODEL"; substitute the
            # real path (and repair any other path it invented).
            n_views = 0
            for ent in spec.entities:
                if isinstance(ent, Mesh3DView):
                    ent.path = str(stl_path)
                    n_views += 1
            if n_views == 0:
                last_error = ("the spec embeds no mesh3d_view of the "
                              "model; include the real views")
                feedback = _revise(last_error)
                continue

            try:
                builder = DrawingBuilder(spec)
                doc = builder.build()
            except Exception as e:
                last_error = f"build failed: {type(e).__name__}: {e}"
                feedback = _revise(last_error)
                continue

            findings = validate(builder.index, builder._halo)
            errors = [x for x in findings if x.severity == "error"]
            if errors:
                last_error = "validator errors: " + "; ".join(
                    f"{x.entity_id}: {x.message}" for x in errors[:5])
                feedback = _revise(last_error)
                continue

            # Views (including their detail frames) must not overlap
            # each other on the sheet — the general validator only
            # checks annotations, so enforce this here.
            overlaps = _view_overlaps(spec, builder)
            if overlaps:
                last_error = (
                    "views overlap on the sheet: "
                    + "; ".join(f"{a} and {b} overlap by "
                                f"{w:.0f}x{h:.0f}mm" for a, b, w, h
                                in overlaps[:4])
                    + ". Move the views apart (a framed detail view "
                    "occupies a circle of diameter ~= the diagonal of "
                    "region-size x scale, plus 6mm).")
                feedback = _revise(last_error)
                continue
            _say(f"spec accepted: {len(spec.entities)} entities, "
                 f"{len(spec.annotations)} annotations, "
                 f"{len(findings)} validator finding(s)")

            dxf_path = outdir / f"{name}_sheet.dxf"
            builder.save(str(dxf_path))
            png_path: Optional[Path] = None
            if preview:
                png_path = outdir / f"{name}_sheet.png"
                render_preview(doc, str(png_path), layout="paperspace",
                               dpi=dpi)
            return SheetResult(
                name=name, sheet=spec.sheet, scale=None,
                dxf_path=dxf_path, png_path=png_path, spec=spec,
                report=builder.report, findings=findings,
            )
        raise RuntimeError(
            f"no valid drawing after {max_revisions} attempts; "
            f"last: {last_error}")
    finally:
        for _, p in images:
            try:
                p.unlink()
            except OSError:
                pass
