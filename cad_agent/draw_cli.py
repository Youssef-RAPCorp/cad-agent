"""CLI entry point for drawing sheets — `cad-agent-draw`.

Modes, picked automatically from the argument:

    cad-agent-draw cad_output/part.stl
        The argument is an existing mesh/STEP file. With an LLM API key
        in the environment, the SMART mode runs: the LLM studies the
        shape (measured specs + rendered views) and composes the drawing
        itself, embedding true projections of the model. Without a key
        (or with --basic) the deterministic multi-view template is used
        (offline). --smart forces the LLM path.

    cad-agent-draw "spacer plate 80x40mm with two M3 holes"
        Anything else is a part description → the LLM drafts a fully
        dimensioned DrawingSpec from scratch (needs an API key).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_MODEL_SUFFIXES = {".stl", ".obj", ".ply", ".off", ".glb", ".gltf",
                   ".step", ".stp"}


def _llm_key_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="cad-agent-draw",
        description="Produce an engineering drawing sheet (DXF + PNG) from "
                    "a 3D model file or a natural-language part description.",
    )
    p.add_argument("source",
                   help="Path to an STL/OBJ/PLY/OFF/GLB/STEP file, or a "
                        "text description of the part")
    p.add_argument("--name", default=None, help="Artifact name for outputs")
    p.add_argument("--output", "-o", default=None,
                   help="Output directory (default: next to the model, or "
                        "./cad_output for descriptions)")
    p.add_argument("--sheet", default=None,
                   help="Sheet size: A4-A0, portrait A4P-A0P, or "
                        "ANSI_A-ANSI_E (default: auto)")
    p.add_argument("--scale", type=float, default=None,
                   help="View scale for basic multi-view sheets, e.g. 2 "
                        "for 2:1 (default: auto-fit)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smart", action="store_true",
                      help="Force LLM-drafted drawing of the model (fails "
                           "if the LLM is unavailable)")
    mode.add_argument("--basic", action="store_true",
                      help="Force the offline multi-view template (no LLM)")
    p.add_argument("--notes", default="",
                   help="Extra guidance for the smart mode, e.g. "
                        "'material 6061; callout the bolt circle'")
    p.add_argument("--max-revisions", type=int, default=3,
                   help="LLM retry budget (default: 3)")
    p.add_argument("--no-dims", action="store_true",
                   help="Skip the overall dimensions on basic sheets")
    p.add_argument("--no-hidden", action="store_true",
                   help="Omit dashed hidden lines on basic sheets")
    p.add_argument("--no-preview", action="store_true",
                   help="Skip the PNG preview, write only the DXF")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    try:
        from .drawings import draft_drawing, draw_multiview, generate_drawing
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Mode detection: an existing file (or anything that looks like a
    # model path) is NEVER sent to the LLM as a description.
    src = Path(args.source)
    is_model = src.suffix.lower() in _MODEL_SUFFIXES
    if is_model and not src.exists():
        print(f"error: model file not found: {src}", file=sys.stderr)
        return 1
    if not is_model and src.exists() and src.is_file():
        print(f"error: {src} exists but {src.suffix or 'its type'!r} is not "
              f"a supported model format {sorted(_MODEL_SUFFIXES)}",
              file=sys.stderr)
        return 1

    def _basic():
        return draw_multiview(
            src,
            name=args.name,
            output_dir=args.output,
            sheet=args.sheet,
            scale=args.scale,
            dimensions=not args.no_dims,
            hidden=not args.no_hidden,
            preview=not args.no_preview,
            verbose=args.verbose,
        )

    try:
        if is_model:
            use_smart = args.smart or (not args.basic and _llm_key_available())
            if args.verbose:
                why = ("--smart" if args.smart else "--basic" if args.basic
                       else "API key found" if use_smart else "no API key")
                print(f"[cad-agent-draw] mode: "
                      f"{'smart LLM draft' if use_smart else 'basic multi-view'}"
                      f" ({why})", file=sys.stderr)
            if use_smart:
                try:
                    sheet = draft_drawing(
                        src,
                        notes=args.notes,
                        name=args.name,
                        output_dir=args.output,
                        sheet=args.sheet,
                        max_revisions=args.max_revisions,
                        preview=not args.no_preview,
                        verbose=args.verbose,
                    )
                except Exception as exc:
                    if args.smart:
                        raise
                    print(f"smart draft unavailable ({exc}); falling back "
                          f"to the basic multi-view sheet", file=sys.stderr)
                    sheet = _basic()
            else:
                sheet = _basic()
        else:
            sheet = generate_drawing(
                args.source,
                name=args.name,
                output_dir=args.output or "./cad_output",
                sheet=args.sheet,
                max_revisions=args.max_revisions,
                preview=not args.no_preview,
                verbose=args.verbose,
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(sheet.summary())
    for w in (sheet.report.warnings if sheet.report else [])[:5]:
        print(f"  WARN: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
