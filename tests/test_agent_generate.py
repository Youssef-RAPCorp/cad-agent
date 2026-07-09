"""Regression tests for the CADAgent.generate() facade-to-backend wiring.

The LLM call is mocked out, so these run offline with no API key; they
exercise everything around it — prompt plumbing, sandbox execution,
artifact writing, and the execute/no-execute paths.
"""
import pytest

pytest.importorskip("build123d")

from cad_agent import CADAgent, CADAgentConfig
from cad_agent._vendored.cad_agent3 import shape_generator

CUBE_CODE = "from build123d import *\npart = Box(10, 10, 10)\n"


@pytest.fixture
def mock_llm(monkeypatch):
    monkeypatch.setattr(shape_generator, "_call_codex",
                        lambda prompt: (CUBE_CODE, None))


def make_agent(tmp_path, **overrides):
    cfg = CADAgentConfig(
        api_key="test-key-never-used",
        output_dir=tmp_path,
        use_reasoning_loop=False,
        **overrides,
    )
    return CADAgent(cfg)


def test_generate_executes_and_writes_artifacts(mock_llm, tmp_path):
    result = make_agent(tmp_path).generate("a 10mm cube", name="cube")
    assert result.success, result.error
    assert result.volume_mm3 == pytest.approx(1000.0)
    assert (tmp_path / "cube.py").read_text() == CUBE_CODE
    assert (tmp_path / "cube.step").stat().st_size > 0
    assert (tmp_path / "cube.stl").stat().st_size > 0


def test_generate_no_execute_returns_code_only(mock_llm, tmp_path):
    agent = make_agent(tmp_path, execute_generated_code=False)
    result = agent.generate("a 10mm cube", name="cube")
    assert result.success, result.error
    assert result.script == CUBE_CODE
    assert result.part is None
    assert result.volume_mm3 is None
    assert (tmp_path / "cube.py").exists()
    assert not (tmp_path / "cube.step").exists()
