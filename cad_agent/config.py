"""Configuration for the cad_agent package.

A single dataclass holds all runtime knobs: which LLM backend to use,
API credentials, default output paths, retry/timeout policy. Pass an
instance into `CADAgent(...)` or rely on environment variables.

Environment variable defaults:
  GEMINI_API_KEY     / GOOGLE_API_KEY   — Gemini Flash credential
  ANTHROPIC_API_KEY                     — Claude credential
  CAD_AGENT_BACKEND                     — 'gemini' (default) or 'anthropic'
  CAD_AGENT_MODEL                       — model name override
  CAD_AGENT_OUTPUT_DIR                  — where artifacts get written
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


Backend = Literal["gemini", "anthropic"]

DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-7"


@dataclass
class CADAgentConfig:
    """Runtime configuration for the CAD agent.

    All fields have sensible defaults; you only need to set what you
    want to override. Most users will only set `api_key` (or rely on
    the environment variable).
    """

    # --- LLM backend ---
    backend: Backend = "gemini"
    api_key: Optional[str] = None      # if None, read from env
    model: Optional[str] = None        # if None, use backend default

    # --- Output ---
    output_dir: Path = field(default_factory=lambda: Path("./cad_output"))
    write_step: bool = True
    write_stl: bool = True
    write_script: bool = True          # save the generated .py too

    # --- Generation behavior ---
    max_revisions: int = 3             # critic-feedback rounds in reasoning loop
    use_reasoning_loop: bool = True    # if False, single-shot generation
    inject_reference_specs: bool = True  # auto-resolve "NEMA17" → dimensions

    # --- Safety / sandboxing ---
    execute_generated_code: bool = True  # set False to only emit code, not run
    timeout_seconds: int = 120           # per-LLM-call timeout

    # --- Logging ---
    verbose: bool = False
    log_prompts: bool = False            # dump full LLM prompts to stderr

    # ---------------------------------------------------------------
    @classmethod
    def from_env(cls, **overrides) -> "CADAgentConfig":
        """Build a config from environment variables, with overrides."""
        backend = os.environ.get("CAD_AGENT_BACKEND", "gemini").lower()
        if backend not in ("gemini", "anthropic"):
            raise ValueError(f"Unknown backend: {backend!r}")

        key = None
        if backend == "gemini":
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        else:
            key = os.environ.get("ANTHROPIC_API_KEY")

        model = os.environ.get("CAD_AGENT_MODEL")
        out = os.environ.get("CAD_AGENT_OUTPUT_DIR", "./cad_output")

        defaults = dict(
            backend=backend,
            api_key=key,
            model=model,
            output_dir=Path(out),
        )
        defaults.update(overrides)
        return cls(**defaults)

    # ---------------------------------------------------------------
    def resolve_api_key(self) -> str:
        """Return the API key, raising a clear error if missing."""
        if self.api_key:
            return self.api_key
        if self.backend == "gemini":
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if key:
                return key
            raise RuntimeError(
                "No Gemini API key. Set GEMINI_API_KEY in the environment "
                "or pass api_key='...' to CADAgentConfig."
            )
        else:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if key:
                return key
            raise RuntimeError(
                "No Anthropic API key. Set ANTHROPIC_API_KEY in the environment "
                "or pass api_key='...' to CADAgentConfig."
            )

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        return DEFAULT_GEMINI_MODEL if self.backend == "gemini" else DEFAULT_ANTHROPIC_MODEL
