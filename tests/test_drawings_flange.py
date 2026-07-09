"""
Crowded test: a part where the naive label placement would collide with
multiple features. Exercises the placement engine's multi-ring candidate
search and leader fallback.

Geometry: a circular flange with eight bolt holes around the perimeter
plus a central hub. Every hole needs a callout. With holes packed
densely the labels must avoid each other and the central hub.
"""
import os
import math

import pytest

pytest.importorskip("ezdxf")
pytest.importorskip("pydantic")
pytest.importorskip("matplotlib")

from cad_agent.drawings import (
    DrawingBuilder, DrawingSpec, RevisionEntry, TitleBlock,
    Circle, Line, TextLabel, LinearDim, DiameterDim,
    Ref, Snap, Units, validate, render_preview,
)


def flange_spec() -> DrawingSpec:
    # Flange outer radius 60, hub radius 18, 8 bolt holes Ø6 on a 45mm PCD
    OD = 60.0
    HUB = 18.0
    PCD = 45.0
    HOLE_R = 3.0
    N = 8

    entities = [
        Circle(id="OUTER", layer="VISIBLE", center=(0, 0), radius=OD),
        Circle(id="HUB",   layer="VISIBLE", center=(0, 0), radius=HUB),
        Circle(id="PCD",   layer="CENTER",  center=(0, 0), radius=PCD),
        Line(id="CX", layer="CENTER", start=(-OD-5, 0), end=(OD+5, 0)),
        Line(id="CY", layer="CENTER", start=(0, -OD-5), end=(0, OD+5)),
    ]
    annotations = []

    for i in range(N):
        ang = (360.0 / N) * i
        ax = PCD * math.cos(math.radians(ang))
        ay = PCD * math.sin(math.radians(ang))
        hid = f"H{i+1}"
        entities.append(Circle(id=hid, layer="VISIBLE",
                               center=(ax, ay), radius=HOLE_R))
        # Center marks
        entities.append(Line(id=f"{hid}_CX", layer="CENTER",
                             start=(ax - HOLE_R*1.5, ay),
                             end=(ax + HOLE_R*1.5, ay)))
        entities.append(Line(id=f"{hid}_CY", layer="CENTER",
                             start=(ax, ay - HOLE_R*1.5),
                             end=(ax, ay + HOLE_R*1.5)))
        # Each hole gets a label - this will force placement engine to work
        annotations.append(TextLabel(
            id=f"L_{hid}",
            text=f"{i+1}",
            target=Ref(entity_id=hid, snap=Snap.CENTER),
            height=2.5,
        ))

    # One diameter call-out for the outer
    annotations.append(DiameterDim(
        id="D_OUTER",
        target=Ref(entity_id="OUTER", snap=Snap.CENTER),
        angle_deg=210,
    ))
    # Diameter call-out for the hub
    annotations.append(DiameterDim(
        id="D_HUB",
        target=Ref(entity_id="HUB", snap=Snap.CENTER),
        angle_deg=30,
    ))
    # PCD callout
    annotations.append(DiameterDim(
        id="D_PCD",
        target=Ref(entity_id="PCD", snap=Snap.CENTER),
        angle_deg=270,
    ))

    return DrawingSpec(
        sheet="A3",
        units=Units.MILLIMETERS,
        workflow="mech",
        title_block=TitleBlock(
            title="FLANGE PLATE",
            subtitle="8 x M5 on 45 PCD",
            drawing_no="RAP-MEC-0143",
            rev="A",
            date="2026-05-19",
            drawn_by="Y. EWEIS",
            org="RAPCorp",
            project="Gravity Facility",
            material="AL 6061-T6",
            scale="1:1",
        ),
        revisions=[
            RevisionEntry(rev="A", description="INITIAL RELEASE",
                          date="2026-05-19", by="YE"),
        ],
        entities=entities,
        annotations=annotations,
        default_text_height=2.5,
    )


def main(out_dir):
    spec = flange_spec()
    print(f"{len(spec.entities)} entities, {len(spec.annotations)} annotations")
    os.makedirs(out_dir, exist_ok=True)

    builder = DrawingBuilder(spec)
    doc = builder.build()
    findings = validate(builder.index, builder._halo)

    print(f"\n{builder.report}")
    for w in builder.report.warnings:
        print(f"  WARN: {w}")
    print(f"Validator findings: {len(findings)}")
    for f in findings[:10]:
        print(f"  {f.severity.upper()}: {f.entity_id}: {f.message}")
    if len(findings) > 10:
        print(f"  ... and {len(findings) - 10} more")

    dxf_path = os.path.abspath(os.path.join(out_dir, "flange.dxf"))
    builder.save(dxf_path)
    print(f"\nWrote DXF: {dxf_path}")

    png_paper = os.path.abspath(os.path.join(out_dir, "flange_sheet.png"))
    render_preview(doc, png_paper, layout="paperspace", dpi=180)
    print(f"Wrote sheet preview: {png_paper}")

    png_model = os.path.abspath(os.path.join(out_dir, "flange_model.png"))
    render_preview(doc, png_model, layout="modelspace", dpi=180)
    print(f"Wrote model preview: {png_model}")


def test_flange(tmp_path):
    main(str(tmp_path))
    assert (tmp_path / "flange.dxf").stat().st_size > 0


if __name__ == "__main__":
    main(os.path.join(os.path.dirname(__file__), "out"))
