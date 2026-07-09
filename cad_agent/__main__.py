"""CLI entry point — `cad-agent "A 20mm cube with a hole"`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import CADAgent
from .config import CADAgentConfig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="cad-agent",
        description="Generate CAD models from natural-language specs.",
    )
    p.add_argument("spec", help="Natural-language description of the part")
    p.add_argument("--name", default=None, help="Artifact name (default: random)")
    p.add_argument("--output", "-o", default="./cad_output",
                   help="Output directory (default: ./cad_output)")
    p.add_argument("--backend", default=None, choices=["gemini", "anthropic"],
                   help="LLM backend (default: from env, fallback gemini)")
    p.add_argument("--model", default=None, help="Model name override")
    p.add_argument("--no-reasoning", action="store_true",
                   help="Skip the critic-feedback loop (one-shot generation)")
    p.add_argument("--max-revisions", type=int, default=3,
                   help="Max critic-feedback rounds (default: 3)")
    p.add_argument("--no-execute", action="store_true",
                   help="Emit the script but don't run it")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--extra", default="",
                   help="Extra constraints to append to the prompt")
    p.add_argument("--view", action="store_true",
                   help="Open the generated model in a browser viewer "
                        "(requires trimesh)")
    args = p.parse_args(argv)

    cfg = CADAgentConfig.from_env(
        output_dir=Path(args.output),
        use_reasoning_loop=not args.no_reasoning,
        max_revisions=args.max_revisions,
        execute_generated_code=not args.no_execute,
        verbose=args.verbose,
    )
    if args.backend:
        cfg.backend = args.backend
    if args.model:
        cfg.model = args.model

    agent = CADAgent(cfg)
    result = agent.generate(args.spec, name=args.name, extra_constraints=args.extra)
    print(result.summary())

    if args.view and result.success and result.stl_path:
        from .viewer import view
        try:
            out = view(result)
            print(f"viewer: {out}")
        except ImportError as exc:
            print(f"could not open viewer: {exc}", file=sys.stderr)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
