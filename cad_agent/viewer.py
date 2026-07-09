"""cad_agent.viewer — quick interactive 3D viewer for generated outputs.

Writes a self-contained HTML file (three.js, orbit controls, works
offline) next to the model and opens it in your browser:

    from cad_agent import CADAgent
    from cad_agent.viewer import view

    result = CADAgent().generate("M6 hex nut, 5mm thick")
    view(result)                       # opens browser

    view("cad_output/part.stl")        # or any mesh / STEP file

Or from the command line:

    cad-agent-view cad_output/part.stl
    cad-agent "M6 hex nut, 5mm thick" --view     # generate then open

Mesh formats (STL, OBJ, PLY, OFF, GLB/GLTF) load via trimesh; STEP
files are tessellated through build123d first. Requires trimesh:

    pip install -e ".[drawings]"    # includes trimesh
    # or: pip install trimesh
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import webbrowser
from pathlib import Path
from typing import Optional, Union

_MESH_SUFFIXES = {".stl", ".obj", ".ply", ".off", ".glb", ".gltf"}
_STEP_SUFFIXES = {".step", ".stp"}


def _load_scene(path: Path):
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "cad_agent.viewer requires trimesh. Install it with:\n\n"
            "    pip install -e \".[drawings]\"    # from the cad-agent repo root\n"
            "    # or: pip install trimesh\n"
        ) from exc

    suffix = path.suffix.lower()
    if suffix in _STEP_SUFFIXES:
        # trimesh can't read BREP; round-trip through build123d's mesher.
        from build123d import import_step, export_stl
        shape = import_step(str(path))
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            export_stl(shape, tmp.name)
            loaded = trimesh.load(tmp.name)
    elif suffix in _MESH_SUFFIXES:
        loaded = trimesh.load(str(path))
    else:
        raise ValueError(
            f"unsupported file type {suffix!r}; expected one of "
            f"{sorted(_MESH_SUFFIXES | _STEP_SUFFIXES)}"
        )

    if isinstance(loaded, trimesh.Scene):
        return loaded
    return trimesh.Scene(loaded)


def view(
    source,
    output: Optional[Union[str, Path]] = None,
    open_browser: bool = True,
) -> Path:
    """Write an interactive HTML viewer for a 3D model and open it.

    Args:
        source: a CADResult from CADAgent.generate() (its STL is used),
            or a path to an STL/OBJ/PLY/OFF/GLB/GLTF/STEP file.
        output: where to write the HTML. Defaults to `<model>_view.html`
            next to the model file.
        open_browser: open the file in the default browser after writing.

    Returns:
        Path to the written HTML file.
    """
    if hasattr(source, "stl_path"):  # CADResult (duck-typed)
        if not source.stl_path:
            raise ValueError(
                "CADResult has no STL artifact (was write_stl disabled, "
                "or did generation fail?) — nothing to view"
            )
        path = Path(source.stl_path)
    else:
        path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {path}")

    scene = _load_scene(path)

    from trimesh.viewer.notebook import scene_to_html
    html = scene_to_html(scene)

    out_path = Path(output) if output else path.with_name(path.stem + "_view.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)

    if open_browser:
        webbrowser.open(out_path.resolve().as_uri())
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="cad-agent-view",
        description="Open a 3D model (STL/OBJ/PLY/OFF/GLB/STEP) in an "
                    "interactive browser viewer.",
    )
    p.add_argument("file", help="Path to the model file")
    p.add_argument("--output", "-o", default=None,
                   help="HTML output path (default: <model>_view.html)")
    p.add_argument("--no-open", action="store_true",
                   help="Write the HTML but don't open the browser")
    args = p.parse_args(argv)

    try:
        out = view(args.file, output=args.output,
                   open_browser=not args.no_open)
    except (ImportError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"viewer written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
