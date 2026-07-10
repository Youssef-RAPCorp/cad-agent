"""Tests for hidden-line removal in the mesh projection engine.

Canonical cases validated during development: a box's coincident
front/back projections, a fully occluded body, curved-silhouette
tessellation, and the visible/hidden layer routing in the builder.
"""
import pytest

pytest.importorskip("ezdxf")
pytest.importorskip("pydantic")
pytest.importorskip("matplotlib")
trimesh = pytest.importorskip("trimesh")
np = pytest.importorskip("numpy")

from cad_agent._vendored.rapcad_drawings.model3d import project_mesh


def total_len(edges):
    return sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in edges)


@pytest.fixture
def occluded_pair(tmp_path):
    """A 40x20x10 box with a 10x5x5 box fully hidden behind it (front view)."""
    big = trimesh.creation.box(extents=(40, 20, 10))
    small = trimesh.creation.box(extents=(10, 5, 5))
    small.apply_translation([0, 20, 0])
    path = tmp_path / "pair.stl"
    trimesh.util.concatenate([big, small]).export(str(path))
    return path


def test_box_front_merges_to_outline():
    """The 8 coincident front/back outline edges must merge into exactly
    the 4-segment outline; nothing may leak into the hidden class."""
    box = trimesh.creation.box(extents=(40, 20, 10))
    pv = project_mesh(box, view="front")
    assert len(pv.edges_2d) == 4
    assert total_len(pv.edges_2d) == pytest.approx(100.0, abs=0.5)
    assert total_len(pv.hidden_edges_2d) == pytest.approx(0.0, abs=0.5)


def test_fully_occluded_body_is_hidden(occluded_pair):
    mesh = trimesh.load(str(occluded_pair), force="mesh")
    pv = project_mesh(mesh, view="front")
    # Big box outline stays fully visible…
    assert total_len(pv.edges_2d) == pytest.approx(100.0, abs=1.0)
    # …and the small box's projected outline (2*(10+5)) is all hidden.
    assert total_len(pv.hidden_edges_2d) == pytest.approx(30.0, abs=1.0)
    # Hidden ink lies inside the big box outline.
    for a, b in pv.hidden_edges_2d:
        for x, y in (a, b):
            assert -20.5 <= x <= 20.5 and -5.5 <= y <= 5.5


def test_cylinder_silhouette_visible():
    """Curved silhouettes must not self-occlude (slope bias + owner
    exclusion): visible ink = 2 sides + 2 projected rims = ~100."""
    cyl = trimesh.creation.cylinder(radius=10, height=30, sections=72)
    pv = project_mesh(cyl, view="front")
    assert total_len(pv.edges_2d) == pytest.approx(100.0, abs=1.5)


def test_builder_routes_hidden_to_dashed_layer(occluded_pair):
    from cad_agent.drawings import DrawingBuilder, DrawingSpec, Mesh3DView, Units

    def build(show_hidden):
        spec = DrawingSpec(
            sheet="A4", units=Units.MILLIMETERS, workflow="mech",
            entities=[Mesh3DView(id="V", path=str(occluded_pair),
                                 view="front", origin=(150, 120),
                                 show_hidden=show_hidden)],
        )
        builder = DrawingBuilder(spec)
        builder.build()
        return builder

    b = build(True)
    hidden_lines = [e for e in b.msp if e.dxftype() == "LINE"
                    and e.dxf.layer == "HIDDEN"]
    visible_lines = [e for e in b.msp if e.dxftype() == "LINE"
                     and e.dxf.layer == "VISIBLE"]
    assert len(hidden_lines) > 0
    assert len(visible_lines) > 0

    b_off = build(False)
    assert not [e for e in b_off.msp if e.dxftype() == "LINE"
                and e.dxf.layer == "HIDDEN"]


def test_draw_multiview_hidden_flag(occluded_pair):
    from cad_agent.drawings import draw_multiview

    sheet = draw_multiview(occluded_pair, preview=False)
    flags = {e.id: e.show_hidden for e in sheet.spec.entities}
    assert flags["V_FRONT"] and flags["V_TOP"] and flags["V_RIGHT"]
    assert not flags["V_ISO"]          # iso never shows hidden lines

    sheet_off = draw_multiview(occluded_pair, preview=False, hidden=False,
                               name="pair_nohidden")
    assert not any(e.show_hidden for e in sheet_off.spec.entities)
