"""CLI entry point for drawing sheets — `cad-agent-draw`.

Two modes, picked automatically from the argument:

    cad-agent-draw cad_output/part.stl
        The argument is an existing mesh/STEP file → multi-view sheet
        (FRONT/TOP/RIGHT/ISO). Offline, no API key needed.

    cad-agent-draw "spacer plate 80x40mm with two M3 holes"
        Anything else is a part description → the LLM drafts a fully
        dimensioned DrawingSpec (needs an API key, like `cad-agent`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MODEL_SUFFIXES = {".stl", ".obj", ".ply", ".off", ".glb", ".gltf",
                   ".step", ".stp"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="cad-agent-draw",
        description="Produce an engineering drawing sheet (DXF + PNG) from "
                    "a 3D model file or a natural-language part description.",
    )
    p.add_argument("source",
                   help="Path to an STL/OBJ/PLY/OFF/GLB/STEP file "
                        "(multi-view sheet), or a text description of the "
                        "part (LLM-drafted dimensioned drawing)")
    p.add_argument("--name", default=None, help="Artifact name for outputs")
    p.add_argument("--output", "-o", default=None,
                   help="Output directory (default: next to the model, or "
                        "./cad_output for descriptions)")
    p.add_argument("--sheet", default=None,
                   help="Sheet size: A4-A0 or ANSI_A-ANSI_E (default: "
                        "auto-picked for multi-view; LLM's choice for "
                        "descriptions)")
    p.add_argument("--scale", type=float, default=None,
                   help="View scale for multi-view sheets, e.g. 2 for 2:1 "
                        "(default: auto-fit)")
    p.add_argument("--max-revisions", type=int, default=3,
                   help="LLM retry budget for description mode (default: 3)")
    p.add_argument("--no-dims", action="store_true",
                   help="Skip the overall dimensions on multi-view sheets")
    p.add_argument("--no-preview", action="store_true",
                   help="Skip the PNG preview, write only the DXF")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    try:
        from .drawings import draw_multiview, generate_drawing
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

    try:
        if is_model:
            sheet = draw_multiview(
                src,
                name=args.name,
                output_dir=args.output,
                sheet=args.sheet,
                scale=args.scale,
                dimensions=not args.no_dims,
                preview=not args.no_preview,
            )
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
