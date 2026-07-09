"""
Multi-part test: two distinct parts placed side by side on one sheet,
each with its own full dimension set. Demonstrates that the spec
naturally supports multiple "parts" — anything you put in `entities`
goes into the same modelspace, and the placement engine resolves
collisions across the whole layout.

Layout:
  +-----------+   +-----------+
  | BUSHING   |   | SHAFT     |
  | (Ø dim)   |   | (linear)  |
  +-----------+   +-----------+

The two parts share the same title block + revision block, but are
listed as separate "items" in the drawing — analogous to a parts-
family sheet or a simple assembly callout sheet.
"""
import math
import os

import pytest

pytest.importorskip("ezdxf")
pytest.importorskip("pydantic")
pytest.importorskip("matplotlib")

from cad_agent.drawings import (
    Arc, Circle, DiameterDim, DrawingBuilder, DrawingSpec, LinearDim, Line,
    Polyline, RadialDim, Ref, RevisionEntry, Snap, TextLabel,
    TitleBlock, Units, render_preview, validate,
)


# Part A: Bushing — outer Ø30, inner Ø12, height not shown (2D plan view)
# Centered at origin
BUSHING_CENTER = (0.0, 0.0)
BUSHING_OD = 30.0
BUSHING_ID = 12.0

# Part B: Stepped shaft, viewed from the side. Two diameters along a
# horizontal axis. Drawn as a profile with center line.
# Length: 80mm total. Smaller end (Ø10) for 30mm, larger end (Ø18) for 50mm.
# Place centered around (100, 0)
SHAFT_LEFT = 70.0
SHAFT_AXIS_Y = 0.0
SHAFT_SMALL_R = 5.0     # Ø10
SHAFT_LARGE_R = 9.0     # Ø18
SHAFT_SMALL_LEN = 30.0
SHAFT_LARGE_LEN = 50.0


def spec() -> DrawingSpec:
    entities = [
        # ---- Part A: BUSHING ----
        Circle(id="A_OD",  layer="VISIBLE",
               center=BUSHING_CENTER, radius=BUSHING_OD / 2),
        Circle(id="A_ID",  layer="VISIBLE",
               center=BUSHING_CENTER, radius=BUSHING_ID / 2),
        # Centerlines
        Line(id="A_CX", layer="CENTER",
             start=(-BUSHING_OD / 2 - 4, 0),
             end=  ( BUSHING_OD / 2 + 4, 0)),
        Line(id="A_CY", layer="CENTER",
             start=(0, -BUSHING_OD / 2 - 4),
             end=  (0,  BUSHING_OD / 2 + 4)),

        # ---- Part B: STEPPED SHAFT (side profile) ----
        # Outline as a closed polyline going around the half-section, then
        # mirrored implicitly via centerline.
        # Profile points (CCW from lower-left):
        Polyline(id="B_OUT", layer="VISIBLE", closed=True, points=[
            (SHAFT_LEFT,                            -SHAFT_SMALL_R),
            (SHAFT_LEFT + SHAFT_SMALL_LEN,          -SHAFT_SMALL_R),
            (SHAFT_LEFT + SHAFT_SMALL_LEN,          -SHAFT_LARGE_R),
            (SHAFT_LEFT + SHAFT_SMALL_LEN + SHAFT_LARGE_LEN, -SHAFT_LARGE_R),
            (SHAFT_LEFT + SHAFT_SMALL_LEN + SHAFT_LARGE_LEN,  SHAFT_LARGE_R),
            (SHAFT_LEFT + SHAFT_SMALL_LEN,           SHAFT_LARGE_R),
            (SHAFT_LEFT + SHAFT_SMALL_LEN,           SHAFT_SMALL_R),
            (SHAFT_LEFT,                             SHAFT_SMALL_R),
        ]),
        # Centerline through the shaft axis
        Line(id="B_CL", layer="CENTER",
             start=(SHAFT_LEFT - 5, SHAFT_AXIS_Y),
             end=  (SHAFT_LEFT + SHAFT_SMALL_LEN + SHAFT_LARGE_LEN + 5,
                    SHAFT_AXIS_Y)),
    ]

    annotations = [
        # ---- Part A annotations ----
        DiameterDim(id="A_D_OD",
                    target=Ref(entity_id="A_OD", snap=Snap.CENTER),
                    angle_deg=60),
        DiameterDim(id="A_D_ID",
                    target=Ref(entity_id="A_ID", snap=Snap.CENTER),
                    angle_deg=210),
        TextLabel(id="A_LBL", text="ITEM 1 - BUSHING",
                  target=Ref(entity_id="A_OD", snap=Snap.CENTER),
                  height=3.5, force_leader=False),

        # ---- Part B annotations ----
        # Two diameter dims (one at the small end, one at the large end)
        LinearDim(id="B_D_SMALL",
                  p1=Ref(entity_id="B_OUT", snap=Snap.VERTEX, index=0),
                  p2=Ref(entity_id="B_OUT", snap=Snap.VERTEX, index=7),
                  side="left", angle=90, base_offset=10),
        LinearDim(id="B_D_LARGE",
                  p1=Ref(entity_id="B_OUT", snap=Snap.VERTEX, index=3),
                  p2=Ref(entity_id="B_OUT", snap=Snap.VERTEX, index=4),
                  side="right", angle=90, base_offset=10),
        # Two length dims along the top edge: small step length, large step length
        LinearDim(id="B_L_SMALL",
                  p1=Ref(entity_id="B_OUT", snap=Snap.VERTEX, index=7),
                  p2=Ref(entity_id="B_OUT", snap=Snap.VERTEX, index=6),
                  side="above", base_offset=16),
        LinearDim(id="B_L_LARGE",
                  p1=Ref(entity_id="B_OUT", snap=Snap.VERTEX, index=5),
                  p2=Ref(entity_id="B_OUT", snap=Snap.VERTEX, index=4),
                  side="above", base_offset=16),
        # Overall length (horizontal projection)
        LinearDim(id="B_L_TOTAL",
                  p1=Ref(entity_id="B_OUT", snap=Snap.VERTEX, index=0),
                  p2=Ref(entity_id="B_OUT", snap=Snap.VERTEX, index=3),
                  side="below", angle=0, base_offset=14),
        TextLabel(id="B_LBL", text="ITEM 2 - STEPPED SHAFT",
                  target=Ref(entity_id="B_OUT", snap=Snap.VERTEX, index=4),
                  height=3.5),
    ]

    return DrawingSpec(
        sheet="A3",
        units=Units.MILLIMETERS,
        workflow="mech",
        title_block=TitleBlock(
            title="ASSEMBLY DETAIL",
            subtitle="Bushing + Stepped Shaft",
            drawing_no="RAP-MEC-0145",
            rev="A",
            date="2026-05-19",
            drawn_by="Y. EWEIS",
            checked_by="J. STANLEY",
            org="RAPCorp",
            project="Gravity Facility",
            material="VAR. (see items)",
            scale="2:1",
            tolerance="ASME Y14.5 +/-0.1 mm",
            notes=["ITEM 1: BUSHING - BRASS C36000",
                   "ITEM 2: STEPPED SHAFT - 4140 STEEL"],
        ),
        revisions=[
            RevisionEntry(rev="A", description="INITIAL RELEASE",
                          date="2026-05-19", by="YE"),
        ],
        entities=entities,
        annotations=annotations,
        default_text_height=3.0,
    )


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    s = spec()
    builder = DrawingBuilder(s)
    doc = builder.build()
    findings = validate(builder.index, builder._halo)

    print(f"\n{builder.report}")
    for w in builder.report.warnings[:5]:
        print(f"  WARN: {w}")
    print(f"Validator findings: {len(findings)}")
    for f in findings[:5]:
        print(f"  {f.severity.upper()}: {f.entity_id}: {f.message}")
    if len(findings) > 5:
        print(f"  ... and {len(findings) - 5} more")

    dxf_path = os.path.abspath(os.path.join(out_dir, "multipart.dxf"))
    builder.save(dxf_path)
    print(f"\nWrote DXF: {dxf_path}")

    png_paper = os.path.abspath(os.path.join(out_dir, "multipart_sheet.png"))
    render_preview(doc, png_paper, layout="paperspace", dpi=180)
    print(f"Wrote sheet preview: {png_paper}")

    png_model = os.path.abspath(os.path.join(out_dir, "multipart_model.png"))
    render_preview(doc, png_model, layout="modelspace", dpi=180)
    print(f"Wrote model preview: {png_model}")


def test_multipart(tmp_path):
    main(str(tmp_path))
    assert (tmp_path / "multipart.dxf").stat().st_size > 0


if __name__ == "__main__":
    main(os.path.join(os.path.dirname(__file__), "out"))
