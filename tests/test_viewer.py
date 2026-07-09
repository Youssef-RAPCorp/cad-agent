"""Tests for cad_agent.viewer — HTML viewer generation (browser never opened)."""
import pytest

trimesh = pytest.importorskip("trimesh")

from cad_agent.results import CADResult
from cad_agent.viewer import main, view


@pytest.fixture
def stl_path(tmp_path):
    box = trimesh.creation.box(extents=(10, 10, 10))
    path = tmp_path / "box.stl"
    box.export(str(path))
    return path


def test_view_writes_html(stl_path):
    out = view(stl_path, open_browser=False)
    assert out == stl_path.with_name("box_view.html")
    html = out.read_text()
    assert len(html) > 10_000
    assert "<html" in html.lower()


def test_view_from_cad_result(stl_path, tmp_path):
    result = CADResult(spec="a box", success=True, stl_path=stl_path)
    out = view(result, output=tmp_path / "custom.html", open_browser=False)
    assert out == tmp_path / "custom.html"
    assert out.stat().st_size > 10_000


def test_view_rejects_result_without_stl():
    with pytest.raises(ValueError, match="no STL artifact"):
        view(CADResult(spec="x", success=False), open_browser=False)


def test_view_unsupported_suffix(tmp_path):
    bad = tmp_path / "part.dxf"
    bad.write_text("dummy")
    with pytest.raises(ValueError, match="unsupported file type"):
        view(bad, open_browser=False)


def test_view_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        view(tmp_path / "nope.stl", open_browser=False)


def test_cli(stl_path, capsys):
    rc = main([str(stl_path), "--no-open"])
    assert rc == 0
    assert "viewer written:" in capsys.readouterr().out
    assert stl_path.with_name("box_view.html").exists()


def test_cli_error(tmp_path, capsys):
    rc = main([str(tmp_path / "nope.stl"), "--no-open"])
    assert rc == 1
    assert "error:" in capsys.readouterr().err


def test_view_step_file(stl_path, tmp_path):
    build123d = pytest.importorskip("build123d")
    from build123d import Box, export_step

    step = tmp_path / "box.step"
    export_step(Box(10, 10, 10), str(step))
    out = view(step, open_browser=False)
    assert out.stat().st_size > 10_000
