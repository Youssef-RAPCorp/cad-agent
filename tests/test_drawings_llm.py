"""Tests for LLM-assisted drawing generation and the cad-agent-draw CLI.

The LLM call is mocked, so everything runs offline with no API key:
these exercise the prompt → JSON → Pydantic → build → validate loop and
its self-correction retries.
"""
import pytest

pytest.importorskip("ezdxf")
pytest.importorskip("pydantic")
pytest.importorskip("matplotlib")

from cad_agent.drawings import _EXAMPLE_SPEC_JSON, SheetResult, generate_drawing
from cad_agent._vendored.cad_agent3 import gemini_codegen
from cad_agent.draw_cli import main


@pytest.fixture
def mock_llm(monkeypatch):
    """Patch the LLM call to replay canned responses; records call count."""
    def install(*responses):
        calls = {"n": 0}

        def fake_call(prompt):
            i = min(calls["n"], len(responses) - 1)
            calls["n"] += 1
            return responses[i], None

        monkeypatch.setattr(gemini_codegen, "call_gemini_for_code", fake_call)
        return calls
    return install


def test_generate_drawing_valid_first_try(mock_llm, tmp_path):
    calls = mock_llm(_EXAMPLE_SPEC_JSON)
    sheet = generate_drawing("a spacer plate", output_dir=tmp_path)
    assert isinstance(sheet, SheetResult)
    assert calls["n"] == 1
    assert sheet.name == "spacer_plate"          # slug of title block title
    assert sheet.sheet == "A3"
    assert sheet.scale is None
    assert sheet.dxf_path.stat().st_size > 0
    assert sheet.png_path.stat().st_size > 0
    assert not [f for f in sheet.findings if f.severity == "error"]
    assert "OK: drawing 'spacer_plate' on A3" in sheet.summary()


def test_generate_drawing_retries_on_invalid_json(mock_llm, tmp_path):
    calls = mock_llm("this is not json", _EXAMPLE_SPEC_JSON)
    sheet = generate_drawing("a spacer plate", output_dir=tmp_path,
                             name="plate", preview=False)
    assert calls["n"] == 2
    assert sheet.dxf_path == tmp_path / "plate_sheet.dxf"
    assert sheet.png_path is None


def test_generate_drawing_sheet_override(mock_llm, tmp_path):
    mock_llm(_EXAMPLE_SPEC_JSON)
    sheet = generate_drawing("a spacer plate", output_dir=tmp_path,
                             sheet="A4", preview=False)
    assert sheet.sheet == "A4"


def test_generate_drawing_exhausts_revisions(mock_llm, tmp_path):
    calls = mock_llm("{}bad")
    with pytest.raises(RuntimeError, match="no valid drawing after 2 attempts"):
        generate_drawing("x", output_dir=tmp_path, max_revisions=2)
    assert calls["n"] == 2


def test_generate_drawing_llm_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(gemini_codegen, "call_gemini_for_code",
                        lambda prompt: (None, "no API key"))
    with pytest.raises(RuntimeError, match="LLM call failed: no API key"):
        generate_drawing("x", output_dir=tmp_path)


def test_cli_description_mode(mock_llm, tmp_path, capsys):
    mock_llm(_EXAMPLE_SPEC_JSON)
    rc = main(["a spacer plate", "-o", str(tmp_path), "--no-preview"])
    assert rc == 0
    assert "OK: drawing" in capsys.readouterr().out
    assert (tmp_path / "spacer_plate_sheet.dxf").exists()


def test_cli_model_mode(tmp_path, capsys):
    trimesh = pytest.importorskip("trimesh")
    box = trimesh.creation.box(extents=(30, 20, 10))
    stl = tmp_path / "block.stl"
    box.export(str(stl))

    rc = main([str(stl), "--sheet", "A3", "--no-preview"])
    assert rc == 0
    assert "OK: drawing 'block' on A3" in capsys.readouterr().out
    assert (tmp_path / "block_sheet.dxf").exists()


def test_cli_missing_model_file(tmp_path, capsys):
    rc = main([str(tmp_path / "nope.stl")])
    assert rc == 1
    assert "not found" in capsys.readouterr().err


def test_cli_step_file_never_goes_to_llm(tmp_path, capsys):
    """Regression: a .step path used to fall through to description mode
    and get sent to the LLM as prose."""
    rc = main([str(tmp_path / "part.step")])
    assert rc == 1  # model mode → missing-file error, NOT an LLM call
    assert "not found" in capsys.readouterr().err


def test_cli_existing_file_with_unknown_suffix(tmp_path, capsys):
    txt = tmp_path / "notes.txt"
    txt.write_text("some notes")
    rc = main([str(txt)])
    assert rc == 1
    assert "not a supported model format" in capsys.readouterr().err


def test_cli_step_model_mode(tmp_path, capsys):
    build123d = pytest.importorskip("build123d")
    pytest.importorskip("trimesh")
    from build123d import Box, export_step

    step = tmp_path / "block.step"
    export_step(Box(30, 20, 10), str(step))
    rc = main([str(step), "--sheet", "A3", "--no-preview"])
    assert rc == 0
    assert "OK: drawing 'block' on A3" in capsys.readouterr().out
    assert (tmp_path / "block_sheet.dxf").exists()
    assert (tmp_path / "block_tessellated.stl").exists()
