"""Tests for the cad_agent.drawings facade — the draw_multiview bridge
from a generated 3D model (or any mesh file) to a drawing sheet.
"""
import os

import pytest

pytest.importorskip("ezdxf")
pytest.importorskip("pydantic")
pytest.importorskip("matplotlib")
trimesh = pytest.importorskip("trimesh")

from cad_agent.drawings import SheetResult, TitleBlock, draw_multiview
from cad_agent.results import CADResult


@pytest.fixture
def stl_path(tmp_path):
    """A small stepped block: 40 x 20 x 10 base with a 20 x 20 x 10 riser."""
    base = trimesh.creation.box(extents=(40, 20, 10))
    base.apply_translation([0, 0, 5])
    riser = trimesh.creation.box(extents=(20, 20, 10))
    riser.apply_translation([-10, 0, 15])
    mesh = trimesh.util.concatenate([base, riser])
    path = tmp_path / "step_block.stl"
    mesh.export(str(path))
    return path


def test_draw_multiview_from_path(stl_path, tmp_path):
    sheet = draw_multiview(stl_path, sheet="A3")
    assert isinstance(sheet, SheetResult)
    assert sheet.name == "step_block"
    assert sheet.dxf_path == tmp_path / "step_block_sheet.dxf"
    assert sheet.dxf_path.stat().st_size > 0
    assert sheet.png_path.stat().st_size > 0
    # Auto-fit must land on a standard scale that keeps the widest view
    # pair (front + right) inside an A3 sheet.
    assert sheet.scale > 0
    assert (40 + 20) * sheet.scale < 420
    # All four views should have produced geometry.
    assert not [w for w in sheet.report.warnings if "no edges" in w]
    assert "OK: drawing 'step_block'" in sheet.summary()


def test_draw_multiview_overall_dimensions(stl_path):
    """The 40 x 20 x 20 block gets true-size overall dims regardless of
    the drawn scale."""
    sheet = draw_multiview(stl_path, preview=False)
    dims = {a.id: a for a in sheet.spec.annotations}
    assert set(dims) == {"D_WIDTH", "D_HEIGHT", "D_DEPTH"}
    assert dims["D_WIDTH"].text_override == "40"
    assert dims["D_HEIGHT"].text_override == "20"
    assert dims["D_DEPTH"].text_override == "20"
    assert not [f for f in sheet.findings if f.severity == "error"]


def test_draw_multiview_no_dimensions(stl_path):
    sheet = draw_multiview(stl_path, preview=False, dimensions=False)
    assert sheet.spec.annotations == []


def test_draw_multiview_tall_part_gets_portrait_sheet(tmp_path):
    """A grandfather-clock-shaped part (tall and narrow) should land on
    a portrait sheet at 1:1 instead of huddling on landscape A2."""
    tall = trimesh.creation.box(extents=(40, 25, 220))
    path = tmp_path / "clockish.stl"
    tall.export(str(path))
    sheet = draw_multiview(path, preview=False)
    assert sheet.sheet.endswith("P")
    assert sheet.scale == 1.0
    assert not [f for f in sheet.findings if f.severity == "error"]


def test_draw_multiview_auto_sheet_prefers_near_true_size(stl_path):
    """A 40mm part should land on a small sheet at a modest enlargement,
    not float tiny in a corner of A2. Larger sheets would push the scale
    further from 1:1, so A4 at 2:1 wins."""
    sheet = draw_multiview(stl_path, preview=False)
    assert sheet.sheet == "A4"
    assert sheet.scale == 2.0


def test_draw_multiview_from_cad_result(stl_path, tmp_path):
    result = CADResult(
        spec="a stepped block",
        success=True,
        stl_path=stl_path,
        output_dir=tmp_path,
        metadata={"name": "stepper"},
    )
    out = tmp_path / "sheets"
    sheet = draw_multiview(
        result,
        output_dir=out,
        preview=False,
        title_block=TitleBlock(title="STEPPED BLOCK", drawing_no="RAP-0002"),
    )
    assert sheet.name == "stepper"
    assert sheet.dxf_path == out / "stepper_sheet.dxf"
    assert sheet.dxf_path.stat().st_size > 0
    assert sheet.png_path is None
    assert sheet.spec.title_block.drawing_no == "RAP-0002"


def test_draw_multiview_rejects_result_without_stl(tmp_path):
    result = CADResult(spec="x", success=False, output_dir=tmp_path)
    with pytest.raises(ValueError, match="no STL artifact"):
        draw_multiview(result)


def test_draw_multiview_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        draw_multiview(tmp_path / "nope.stl")


def test_draw_multiview_unknown_sheet(stl_path):
    with pytest.raises(ValueError, match="unknown sheet"):
        draw_multiview(stl_path, sheet="A9")
