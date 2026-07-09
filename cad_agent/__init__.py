"""cad_agent — LLM-powered CAD model generation.

Quick start:

    from cad_agent import CADAgent

    agent = CADAgent()  # picks up GEMINI_API_KEY from env
    result = agent.generate("A 20mm cube with a 5mm hole through the center")
    print(result.summary())

For finer control, build a config:

    from cad_agent import CADAgent, CADAgentConfig
    from pathlib import Path

    cfg = CADAgentConfig(
        backend="gemini",
        api_key="...",
        output_dir=Path("/tmp/cad"),
        max_revisions=5,
        use_reasoning_loop=True,
    )
    agent = CADAgent(cfg)

Lower-level building blocks (reasoning loop, operation catalog, design
sessions with checkpointing) are exposed via `cad_agent.advanced`.

2D engineering drawings (ASME/ISO-style DXF sheets with title blocks,
dimensions, and collision-aware annotation placement) are exposed via
`cad_agent.drawings` — install with `pip install cad-agent[drawings]`:

    from cad_agent.drawings import draw_multiview

    sheet = draw_multiview(result)   # result from CADAgent.generate()
    print(sheet.summary())
"""

from .agent import CADAgent, generate
from .config import CADAgentConfig
from .results import CADResult

__all__ = [
    "CADAgent",
    "CADAgentConfig",
    "CADResult",
    "generate",
]

__version__ = "0.1.0"
