"""Tests for detail (zoom) views: region cropping, magnification, and
the detail boundary frame."""
import pytest

pytest.importorskip("ezdxf")
pytest.importorskip("pydantic")
pytest.importorskip("matplotlib")
trimesh = pytest.importorskip("trimesh")

from cad_agent._vendored.rapcad_drawings.model3d import clip_segments_to_rect
from cad_agent.drawings import DrawingBuilder, DrawingSpec, Mesh3DView, Units


def test_clip_segments_to_rect():
    rect = (0.0, 0.0, 10.0, 10.0)
    inside = ((1.0, 1.0), (9.0, 9.0))
    outside = ((20.0, 20.0), (30.0, 20.0))
    crossing = ((-5.0, 5.0), (15.0, 5.0))
    out = clip_segments_to_rect([inside, outside, crossing], rect)
    assert inside in out
    assert len(out) == 2
    clipped = [s for s in out if s != inside][0]
    assert clipped == ((0.0, 5.0), (10.0, 5.0))


@pytest.fixture
def stl_path(tmp_path):
    """A 40x20x10 base with a small 6x6x6 boss at the top-right corner —
    the boss is the 'fine feature' a detail view would zoom into."""
    base = trimesh.creation.box(extents=(40, 20, 10))
    boss = trimesh.creation.box(extents=(6, 6, 6))
    boss.apply_translation([15, 0, 8])
    path = tmp_path / "bossed.stl"
    trimesh.util.concatenate([base, boss]).export(str(path))
    return path


def _build(stl_path, **mesh_kw):
    spec = DrawingSpec(
        sheet="A3", units=Units.MILLIMETERS, workflow="mech",
        entities=[Mesh3DView(id="V", path=str(stl_path), view="front",
                             origin=(200, 150), **mesh_kw)],
    )
    b = DrawingBuilder(spec)
    b.build()
    return b


def test_region_crops_view(stl_path):
    full = _build(stl_path)
    full_aabb = full.index.get("V").aabb
    # Front view is 40 x 11 (base z -5..5 plus boss to 11). Crop to the
    # boss corner: x 25..40, y 6..16 from the view's lower-left.
    detail = _build(stl_path, region=(25, 6, 40, 16), scale=4.0)
    d_aabb = detail.index.get("V").aabb
    # Cropped content spans at most the region (15 x 5 of real ink)
    # times the 4x scale — far narrower than the full view at 4x.
    assert (d_aabb.xmax - d_aabb.xmin) <= 15 * 4.0 + 1e-6
    assert (full_aabb.xmax - full_aabb.xmin) == pytest.approx(40.0, abs=0.1)


def test_region_with_no_content_warns(stl_path):
    b = _build(stl_path, region=(39.5, 0.1, 39.9, 0.2))
    assert any("region crop left no edges" in w for w in b.report.warnings)


def test_labels_stay_inside_sheet_borders(stl_path):
    """Regression: a label anchored near the right border used to be
    placed past the sheet edge; border guard obstacles now keep all
    placed text inside the drawing area."""
    from cad_agent.drawings import Ref, Snap, TextLabel

    spec = DrawingSpec(
        sheet="A4", units=Units.MILLIMETERS, workflow="mech",
        entities=[Mesh3DView(id="V", path=str(stl_path), view="front",
                             origin=(265, 120), scale=1.0)],
        annotations=[TextLabel(
            id="T_EDGE", text="CLASSIC BASE MOLDING DETAIL",
            target=Ref(entity_id="V", snap=Snap.VERTEX, index=1),
            height=3.5)],
    )
    b = DrawingBuilder(spec)
    b.build()
    aabb = b.index.get("T_EDGE").aabb
    assert aabb.xmax <= 287.01          # A4 right border: 297 - 10
    assert aabb.xmin >= 19.99           # left border
    assert b.index.has("__guard_right")
    assert b.index.has("__guard_titleblock")


def test_guards_do_not_inflate_viewport_bounds(stl_path):
    """Regression: the sheet-margin guard slabs (±500mm) were included
    in overall_bounds(), which drives the paperspace viewport autoscale —
    every drawing rendered tiny in the middle of the sheet."""
    spec = DrawingSpec(
        sheet="A4", units=Units.MILLIMETERS, workflow="mech",
        entities=[Mesh3DView(id="V", path=str(stl_path), view="front",
                             origin=(150, 120), scale=1.0)],
    )
    b = DrawingBuilder(spec)
    b.build()
    bounds = b.index.overall_bounds()
    assert bounds.xmin > -50 and bounds.xmax < 350   # content, not guards
    assert bounds.ymin > -50 and bounds.ymax < 350


def test_frame_draws_detail_circle(stl_path):
    b = _build(stl_path, region=(25, 6, 40, 16), scale=4.0, frame=True)
    circles = [e for e in b.msp if e.dxftype() == "CIRCLE"]
    assert len(circles) == 1
    assert circles[0].dxf.center.x == pytest.approx(200.0)
    assert circles[0].dxf.center.y == pytest.approx(150.0)
    # The frame ring registers as a collision obstacle.
    assert b.index.get("V__frame") is not None

    b_off = _build(stl_path, region=(25, 6, 40, 16), scale=4.0)
    assert not [e for e in b_off.msp if e.dxftype() == "CIRCLE"]
