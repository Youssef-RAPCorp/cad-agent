"""
Smoke test: build a mechanical bracket drawing that exercises:
  - multiple geometry kinds (rect, holes, slot, fillet arc, hatch)
  - dimensions that would collide with geometry if not placed carefully
  - text labels that need to avoid both geometry and each other
  - title block + revision block
  - paperspace viewport autoscale
"""
import os

import pytest

pytest.importorskip("ezdxf")
pytest.importorskip("pydantic")
pytest.importorskip("matplotlib")

from cad_agent.drawings import (
    DrawingBuilder, DrawingSpec, RevisionEntry, TitleBlock,
    Line, Polyline, Rectangle, Circle, Arc, Hatch,
    TextLabel, LinearDim, RadialDim, DiameterDim,
    Ref, Snap, Units, validate, render_preview,
)


def bracket_spec() -> DrawingSpec:
    return DrawingSpec(
        sheet="A3",
        units=Units.MILLIMETERS,
        workflow="mech",
        title_block=TitleBlock(
            title="MOUNTING BRACKET",
            subtitle="L-bracket, 6mm steel",
            drawing_no="RAP-MEC-0142",
            rev="B",
            date="2026-05-19",
            drawn_by="Y. EWEIS",
            checked_by="J. STANLEY",
            approved_by="J. STANLEY",
            scale="1:1",
            org="RAPCorp",
            project="Gravity Facility",
            material="STEEL A36, 6mm",
            finish="ZINC PLATE 0.012mm",
            tolerance="ASME Y14.5  +/-0.1 mm UNO",
            notes=["BREAK ALL SHARP EDGES 0.3 MAX",
                   "HOLES TO BE FREE OF BURRS"],
        ),
        revisions=[
            RevisionEntry(rev="A", description="INITIAL RELEASE",
                          date="2026-04-30", by="YE"),
            RevisionEntry(rev="B", description="ADDED CHAMFER NOTE",
                          date="2026-05-19", by="YE"),
        ],
        entities=[
            # Main outline: L-shape, vertices CCW from bottom-left.
            # Indices: 0=BL, 1=BR, 2=mid-R, 3=inside-corner, 4=top-of-narrow, 5=TL
            Polyline(id="OUT", layer="VISIBLE", closed=True, points=[
                (0, 0), (120, 0), (120, 40), (40, 40),
                (40, 80), (0, 80),
            ]),
            # Two mounting holes (Ø8mm)
            Circle(id="H1", layer="VISIBLE", center=(20, 20),  radius=4.0),
            Circle(id="H2", layer="VISIBLE", center=(100, 20), radius=4.0),
            # Slot in the upper flange: 6 wide x 13 long, centered at (20, 65),
            # constructed as two semicircular arcs + two parallel tangent lines.
            Arc(id="SLOT_TOP",    layer="VISIBLE",
                center=(20, 71.5), radius=3.0, start_angle=0,   end_angle=180),
            Arc(id="SLOT_BOTTOM", layer="VISIBLE",
                center=(20, 58.5), radius=3.0, start_angle=180, end_angle=360),
            Line(id="SLOT_L",  layer="VISIBLE", start=(17, 58.5), end=(17, 71.5)),
            Line(id="SLOT_R",  layer="VISIBLE", start=(23, 58.5), end=(23, 71.5)),
            # Inside-corner fillet R5
            Arc(id="F1", layer="VISIBLE", center=(45, 45),
                radius=5.0, start_angle=180, end_angle=270),
            # Center marks for the two holes
            Line(id="CH1H", layer="CENTER", start=(14, 20), end=(26, 20)),
            Line(id="CH1V", layer="CENTER", start=(20, 14), end=(20, 26)),
            Line(id="CH2H", layer="CENTER", start=(94, 20), end=(106, 20)),
            Line(id="CH2V", layer="CENTER", start=(100, 14), end=(100, 26)),
            # Center mark for the slot
            Line(id="CSLOTV", layer="CENTER", start=(20, 55), end=(20, 75)),
        ],
        annotations=[
            # ----- Overall dimensions -----
            LinearDim(id="D_OVERALL_W",
                      p1=Ref(entity_id="OUT", snap=Snap.VERTEX, index=0),
                      p2=Ref(entity_id="OUT", snap=Snap.VERTEX, index=1),
                      side="below", base_offset=14),
            LinearDim(id="D_OVERALL_H",
                      p1=Ref(entity_id="OUT", snap=Snap.VERTEX, index=0),
                      p2=Ref(entity_id="OUT", snap=Snap.VERTEX, index=5),
                      side="left", angle=90, base_offset=14),
            LinearDim(id="D_FLANGE_H",
                      p1=Ref(entity_id="OUT", snap=Snap.VERTEX, index=1),
                      p2=Ref(entity_id="OUT", snap=Snap.VERTEX, index=2),
                      side="right", angle=90, base_offset=14),
            LinearDim(id="D_NARROW_W",
                      p1=Ref(entity_id="OUT", snap=Snap.VERTEX, index=5),
                      p2=Ref(entity_id="OUT", snap=Snap.VERTEX, index=4),
                      side="above", base_offset=14),

            # ----- Hole position dims (horizontal, below the bracket) -----
            LinearDim(id="D_HOLE_X1",
                      p1=Ref(entity_id="OUT", snap=Snap.VERTEX, index=0),
                      p2=Ref(entity_id="H1",  snap=Snap.CENTER),
                      side="below", angle=0, base_offset=28),
            LinearDim(id="D_HOLE_X2",
                      p1=Ref(entity_id="H1", snap=Snap.CENTER),
                      p2=Ref(entity_id="H2", snap=Snap.CENTER),
                      side="below", angle=0, base_offset=28),
            LinearDim(id="D_HOLE_Y",
                      p1=Ref(entity_id="OUT", snap=Snap.VERTEX, index=0),
                      p2=Ref(entity_id="H1",  snap=Snap.CENTER),
                      side="left", angle=90, base_offset=26),

            # ----- Diameter / radius callouts -----
            DiameterDim(id="D_H1",
                        target=Ref(entity_id="H1", snap=Snap.CENTER),
                        angle_deg=135),
            DiameterDim(id="D_H2",
                        target=Ref(entity_id="H2", snap=Snap.CENTER),
                        angle_deg=45),
            RadialDim(id="D_F1",
                      target=Ref(entity_id="F1", snap=Snap.CENTER),
                      angle_deg=225),

            # ----- Text labels -----
            TextLabel(id="L_SLOT", text="6 x 13 SLOT",
                      target=Ref(entity_id="SLOT_TOP", snap=Snap.CENTER),
                      height=3.0),
            TextLabel(id="L_MAT", text="6mm A36 STEEL",
                      target=Ref(entity_id="OUT", snap=Snap.VERTEX, index=5),
                      height=3.0),
        ],
        default_text_height=3.0,
    )


def main(out_dir):
    spec = bracket_spec()
    print(f"Spec validated. {len(spec.entities)} entities, "
          f"{len(spec.annotations)} annotations.")
    os.makedirs(out_dir, exist_ok=True)

    builder = DrawingBuilder(spec)
    doc = builder.build()

    findings = validate(builder.index, builder._halo)
    print(f"\nBuild report: {builder.report}")
    for w in builder.report.warnings:
        print(f"  WARN: {w}")
    print(f"\nValidator findings: {len(findings)}")
    for f in findings:
        print(f"  {f.severity.upper()}: {f.entity_id}: {f.message}")

    dxf_path = os.path.abspath(os.path.join(out_dir, "bracket.dxf"))
    builder.save(dxf_path)
    print(f"\nWrote DXF: {dxf_path}")

    # Renders
    png_paper = os.path.abspath(os.path.join(out_dir, "bracket_sheet.png"))
    render_preview(doc, png_paper, layout="paperspace", dpi=180)
    print(f"Wrote paperspace preview: {png_paper}")

    png_model = os.path.abspath(os.path.join(out_dir, "bracket_model.png"))
    render_preview(doc, png_model, layout="modelspace", dpi=180)
    print(f"Wrote modelspace preview: {png_model}")


def test_bracket(tmp_path):
    main(str(tmp_path))
    assert (tmp_path / "bracket.dxf").stat().st_size > 0


if __name__ == "__main__":
    main(os.path.join(os.path.dirname(__file__), "out"))
