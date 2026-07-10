"""Tests for the smart LLM drafting mode (draft_drawing) — the LLM call
is mocked, so everything runs offline. These verify the plumbing: shape
facts + view images reach the LLM, the "MODEL" path placeholder is
substituted with the real file, the revision loop reacts to bad specs,
and the CLI mode selection (smart / basic / fallback) behaves.
"""
import json

import pytest

pytest.importorskip("ezdxf")
pytest.importorskip("pydantic")
pytest.importorskip("matplotlib")
trimesh = pytest.importorskip("trimesh")

import cad_agent.drawings as drawings
from cad_agent.draw_cli import main
from cad_agent.drawings import Mesh3DView, SheetResult, draft_drawing

SMART_SPEC = json.dumps({
    "sheet": "A3", "units": "mm", "workflow": "mech",
    "title_block": {"title": "SMART BLOCK", "drawing_no": "RAP-0100",
                    "scale": "1:1"},
    "entities": [
        {"kind": "mesh3d_view", "id": "V_FRONT", "path": "MODEL",
         "view": "front", "origin": [150, 200], "scale": 1.0,
         "show_hidden": True, "label": "FRONT VIEW"},
        {"kind": "mesh3d_view", "id": "V_ISO", "path": "MODEL",
         "view": "iso", "origin": [300, 200], "scale": 0.7,
         "label": "ISOMETRIC"},
    ],
    "annotations": [
        {"kind": "linear_dim", "id": "D_W",
         "p1": {"entity_id": "V_FRONT", "snap": "vertex", "index": 0},
         "p2": {"entity_id": "V_FRONT", "snap": "vertex", "index": 1},
         "side": "below", "text_override": "40"},
    ],
})

# A valid spec that embeds no view of the model — must be rejected.
NO_VIEW_SPEC = json.dumps({
    "sheet": "A4", "units": "mm", "workflow": "mech",
    "title_block": {"title": "NOPE"},
    "entities": [{"kind": "circle", "id": "C1", "center": [50, 50],
                  "radius": 10}],
    "annotations": [],
})

# Two views stacked on top of each other — must be rejected and revised.
OVERLAPPING_SPEC = json.dumps({
    "sheet": "A3", "units": "mm", "workflow": "mech",
    "title_block": {"title": "OVERLAP"},
    "entities": [
        {"kind": "mesh3d_view", "id": "V_FRONT", "path": "MODEL",
         "view": "front", "origin": [150, 200], "scale": 1.0},
        {"kind": "mesh3d_view", "id": "V_RIGHT", "path": "MODEL",
         "view": "right", "origin": [155, 200], "scale": 1.0},
    ],
    "annotations": [],
})


@pytest.fixture
def stl_path(tmp_path):
    box = trimesh.creation.box(extents=(40, 20, 10))
    path = tmp_path / "block.stl"
    box.export(str(path))
    return path


@pytest.fixture
def mock_llm(monkeypatch):
    """Replace the spec LLM call with canned responses; records calls."""
    def install(*responses):
        calls = {"n": 0, "prompts": [], "images": []}

        def fake(prompt, image_paths=(), verbose=False):
            i = min(calls["n"], len(responses) - 1)
            calls["n"] += 1
            calls["prompts"].append(prompt)
            calls["images"].append(list(image_paths))
            return responses[i], None

        monkeypatch.setattr(drawings, "_call_llm_for_spec", fake)
        return calls
    return install


def test_draft_drawing_smart(stl_path, tmp_path, mock_llm):
    calls = mock_llm(SMART_SPEC)
    sheet = draft_drawing(stl_path, preview=False)
    assert isinstance(sheet, SheetResult)
    assert calls["n"] == 1
    # The LLM saw the measured facts and the rendered view images.
    assert "MODEL FACTS" in calls["prompts"][0]
    assert "width 40.0 (X)" in calls["prompts"][0]
    assert len(calls["images"][0]) >= 3
    # The MODEL placeholder was substituted with the real mesh path.
    mvs = [e for e in sheet.spec.entities if isinstance(e, Mesh3DView)]
    assert mvs and all(e.path == str(stl_path) for e in mvs)
    assert sheet.dxf_path == tmp_path / "block_sheet.dxf"
    assert sheet.dxf_path.stat().st_size > 0
    assert not [f for f in sheet.findings if f.severity == "error"]
    # Throwaway layout-template artifacts were cleaned up.
    assert not (tmp_path / "block__layout_sheet.dxf").exists()


def test_draft_drawing_retries_until_valid(stl_path, mock_llm):
    calls = mock_llm("this is not json", NO_VIEW_SPEC, SMART_SPEC)
    sheet = draft_drawing(stl_path, preview=False)
    assert calls["n"] == 3
    assert sheet.dxf_path.exists()
    # The revision feedback told the LLM about the missing model views.
    assert "mesh3d_view" in calls["prompts"][2]


def test_draft_drawing_rejects_overlapping_views(stl_path, mock_llm):
    calls = mock_llm(OVERLAPPING_SPEC, SMART_SPEC)
    sheet = draft_drawing(stl_path, preview=False)
    assert calls["n"] == 2
    assert "views overlap on the sheet" in calls["prompts"][1]
    assert sheet.dxf_path.exists()


def test_draft_drawing_exhausts_revisions(stl_path, mock_llm):
    calls = mock_llm("garbage")
    with pytest.raises(RuntimeError, match="no valid drawing after 2"):
        draft_drawing(stl_path, preview=False, max_revisions=2)
    assert calls["n"] == 2


def test_draft_drawing_notes_reach_prompt(stl_path, mock_llm):
    calls = mock_llm(SMART_SPEC)
    draft_drawing(stl_path, preview=False, notes="material is 6061-T6")
    assert "material is 6061-T6" in calls["prompts"][0]


def test_cli_smart_flag(stl_path, mock_llm, capsys):
    mock_llm(SMART_SPEC)
    rc = main([str(stl_path), "--smart", "--no-preview"])
    assert rc == 0
    assert "OK: drawing 'block' on A3" in capsys.readouterr().out


def test_cli_auto_uses_basic_without_key(stl_path, monkeypatch, capsys):
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    called = {"n": 0}
    monkeypatch.setattr(drawings, "_call_llm_for_spec",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1)
                        or ("x", None))
    rc = main([str(stl_path), "--no-preview"])
    assert rc == 0
    assert called["n"] == 0            # never touched the LLM
    assert "OK: drawing 'block'" in capsys.readouterr().out


def test_cli_auto_falls_back_when_llm_fails(stl_path, monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
    monkeypatch.setattr(drawings, "_call_llm_for_spec",
                        lambda *a, **k: (None, "simulated outage"))
    rc = main([str(stl_path), "--no-preview"])
    captured = capsys.readouterr()
    assert rc == 0                     # basic sheet still produced
    assert "falling back" in captured.err
    assert "OK: drawing 'block'" in captured.out


def test_cli_smart_forced_fails_hard(stl_path, monkeypatch, capsys):
    monkeypatch.setattr(drawings, "_call_llm_for_spec",
                        lambda *a, **k: (None, "simulated outage"))
    rc = main([str(stl_path), "--smart", "--no-preview"])
    assert rc == 1
    assert "error:" in capsys.readouterr().err
