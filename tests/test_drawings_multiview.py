"""
Multi-view test: import a 3D mesh and produce a 4-view engineering
drawing (FRONT, TOP, RIGHT, ISO) on one A2 sheet.

This exercises the Mesh3DView entity end-to-end:
  1. Build a sample STL programmatically (a bracket with a boss and
     mounting holes) using trimesh primitives.
  2. Place four Mesh3DView entities in the spec, each rendering a
     different orthographic / isometric view of the same mesh.
  3. Render a paperspace PNG showing all four views laid out in the
     standard third-angle arrangement.
"""
import math
import os

import pytest

pytest.importorskip("ezdxf")
pytest.importorskip("pydantic")
pytest.importorskip("matplotlib")
np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")

from cad_agent.drawings import (
    DrawingBuilder, DrawingSpec, Mesh3DView, RevisionEntry,
    TextLabel, TitleBlock, Units, Ref, Snap,
    render_preview, validate,
)


# ---------------------------------------------------------------------------
# Build a sample STL (a stepped bracket with a boss + two through-holes)
# ---------------------------------------------------------------------------

def build_sample_stl(path: str) -> str:
    """A small bracket-like part:
        - base block 80 x 40 x 8 mm
        - upright wall 80 x 8 x 30 mm at the back
        - cylindrical boss 20 mm dia x 12 mm tall on top of base
        - two through-holes Ø6 in the base
    All in mm, +Z up, +Y back, +X right.
    """
    parts = []

    # Base block, centered at origin in XY, sitting on z=0
    base = trimesh.creation.box(extents=(80, 40, 8))
    base.apply_translation([0, 0, 4])
    parts.append(base)

    # Back wall: 80 x 8 x 30, sitting on top-back of base
    wall = trimesh.creation.box(extents=(80, 8, 30))
    wall.apply_translation([0, 16, 8 + 15])
    parts.append(wall)

    # Cylindrical boss: r=10, h=12, centered at (0, -8, 8)
    boss = trimesh.creation.cylinder(radius=10, height=12, sections=48)
    boss.apply_translation([0, -8, 8 + 6])
    parts.append(boss)

    # Combine without boolean (trimesh's union needs manifold3d). Just
    # concatenate -- silhouette + feature edges will still be correct
    # for a simple part like this because internal faces are flagged as
    # sharp dihedrals.
    mesh = trimesh.util.concatenate(parts)
    mesh.export(path)
    return path


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

def widget_spec(stl_path: str) -> DrawingSpec:
    # Layout on an A2 landscape sheet (594 x 420 mm), with the four
    # views arranged in the classic third-angle pattern.
    #
    #         (top view above front view)
    #         (right view to the right of front view)
    #         (iso view in the upper-right corner)

    SC = 2.0       # render at 2x scale so the small part fills the sheet

    # Footprint of one view at this scale (rough): bracket spans 80mm
    # widest. At 2x that's 160mm. Use ~180mm pitch to leave room.

    # Place front view in the middle-left area of the drawing
    front_origin = (180, 200)
    top_origin   = (180, 320)
    right_origin = (380, 200)
    iso_origin   = (440, 340)

    return DrawingSpec(
        sheet="A2",
        units=Units.MILLIMETERS,
        workflow="mech",
        title_block=TitleBlock(
            title="STEPPED BRACKET",
            subtitle="Multi-view from STL",
            drawing_no="RAP-MEC-0144",
            rev="A",
            date="2026-05-19",
            drawn_by="Y. EWEIS",
            org="RAPCorp",
            project="Gravity Facility",
            material="STEEL A36",
            scale=f"{SC:g}:1",
            tolerance="ASME Y14.5 +/-0.2 mm",
            notes=["VIEWS DERIVED FROM 3D MODEL", "ALL EDGES VISIBLE"],
        ),
        revisions=[
            RevisionEntry(rev="A", description="INITIAL RELEASE",
                          date="2026-05-19", by="YE"),
        ],
        entities=[
            Mesh3DView(id="V_FRONT", path=stl_path, view="front",
                       origin=front_origin, scale=SC, label="FRONT VIEW"),
            Mesh3DView(id="V_TOP",   path=stl_path, view="top",
                       origin=top_origin,   scale=SC, label="TOP VIEW"),
            Mesh3DView(id="V_RIGHT", path=stl_path, view="right",
                       origin=right_origin, scale=SC, label="RIGHT VIEW"),
            Mesh3DView(id="V_ISO",   path=stl_path, view="iso",
                       origin=iso_origin,   scale=SC * 0.7,
                       label="ISOMETRIC"),
        ],
        annotations=[],
        default_text_height=3.5,
    )


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stl_path = os.path.abspath(os.path.join(out_dir, "widget.stl"))

    print("Generating sample STL...")
    build_sample_stl(stl_path)
    print(f"  Wrote: {stl_path}")

    spec = widget_spec(stl_path)
    builder = DrawingBuilder(spec)
    doc = builder.build()
    findings = validate(builder.index, builder._halo)

    print(f"\n{builder.report}")
    for w in builder.report.warnings[:5]:
        print(f"  WARN: {w}")
    print(f"Validator findings: {len(findings)}")
    for f in findings[:5]:
        print(f"  {f.severity.upper()}: {f.entity_id}: {f.message}")

    dxf_path = os.path.abspath(os.path.join(out_dir, "multiview.dxf"))
    builder.save(dxf_path)
    print(f"\nWrote DXF: {dxf_path}")

    png_paper = os.path.abspath(os.path.join(out_dir, "multiview_sheet.png"))
    render_preview(doc, png_paper, layout="paperspace", dpi=180)
    print(f"Wrote sheet preview: {png_paper}")

    png_model = os.path.abspath(os.path.join(out_dir, "multiview_model.png"))
    render_preview(doc, png_model, layout="modelspace", dpi=180)
    print(f"Wrote model preview: {png_model}")


def test_multiview(tmp_path):
    main(str(tmp_path))
    assert (tmp_path / "multiview.dxf").stat().st_size > 0


if __name__ == "__main__":
    main(os.path.join(os.path.dirname(__file__), "out"))
