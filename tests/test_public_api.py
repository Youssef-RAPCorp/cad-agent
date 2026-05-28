"""Tests that don't require an LLM credential or network."""

import os
from pathlib import Path

import pytest

from cad_agent import CADAgent, CADAgentConfig, CADResult


def test_import_surface():
    """Make sure the public API is what we say it is."""
    import cad_agent
    assert hasattr(cad_agent, "CADAgent")
    assert hasattr(cad_agent, "CADAgentConfig")
    assert hasattr(cad_agent, "CADResult")
    assert hasattr(cad_agent, "generate")
    assert cad_agent.__version__


def test_config_from_env_with_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    monkeypatch.delenv("CAD_AGENT_BACKEND", raising=False)
    cfg = CADAgentConfig.from_env()
    assert cfg.backend == "gemini"
    assert cfg.api_key == "test-key-123"
    assert cfg.resolve_api_key() == "test-key-123"


def test_config_backend_override():
    cfg = CADAgentConfig(backend="anthropic", api_key="ak-123")
    assert cfg.backend == "anthropic"
    assert cfg.resolve_api_key() == "ak-123"
    assert cfg.resolve_model() == "claude-opus-4-7"


def test_config_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    cfg = CADAgentConfig(backend="gemini", api_key=None)
    with pytest.raises(RuntimeError, match="No Gemini API key"):
        cfg.resolve_api_key()


def test_config_invalid_backend(monkeypatch):
    monkeypatch.setenv("CAD_AGENT_BACKEND", "openai")
    with pytest.raises(ValueError, match="Unknown backend"):
        CADAgentConfig.from_env()


def test_result_summary_failure():
    r = CADResult(spec="test", success=False, error="boom")
    assert "FAILED" in r.summary()
    assert "boom" in r.summary()
    assert bool(r) is False


def test_result_summary_success(tmp_path):
    r = CADResult(
        spec="A simple cube",
        success=True,
        volume_mm3=1000.0,
        step_path=tmp_path / "x.step",
        stl_path=tmp_path / "x.stl",
    )
    s = r.summary()
    assert "OK" in s
    assert "A simple cube" in s
    assert bool(r) is True


def test_agent_construction_without_execution(monkeypatch, tmp_path):
    """If execute_generated_code=False, we shouldn't need a key to construct."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    cfg = CADAgentConfig(
        execute_generated_code=False,
        output_dir=tmp_path,
    )
    agent = CADAgent(cfg)
    assert agent.config.output_dir == tmp_path
    assert tmp_path.exists()


def test_advanced_lazy_import():
    """`import cad_agent.advanced` should work even if build123d is missing,
    since the import is lazy."""
    import cad_agent.advanced as adv
    # Attribute access triggers the real import; that one CAN fail
    # if build123d isn't installed, but the import itself should not.
    assert adv is not None
