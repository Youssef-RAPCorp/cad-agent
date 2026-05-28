"""render_preview.py — produce an image preview of a build123d Part.

Two backends, in order of preference:

1. **OCP / VTK offscreen render** — produces a real shaded PNG.
   Requires VTK (already a build123d dependency on most installs).
2. **SVG isometric projection** — fallback that always works. Uses
   build123d's project_to_viewport to flatten the part into 2D edges
   and writes them as an SVG.

In either case, the goal is "user can see what they got" between turns
of a design conversation, without launching a full CAD viewer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class PreviewResult:
    path: str
    backend: str       # "vtk" | "svg" | "failed"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Backend 1: VTK offscreen
# ---------------------------------------------------------------------------

def _try_vtk_render(part, out_path: str,
                     width: int = 800, height: int = 600) -> Optional[str]:
    """Render via VTK in a subprocess. Returns error string or None.

    VTK aborts the host process when it can't open an X display. We
    isolate the call in a child process so a hard abort is recoverable.
    """
    try:
        import vtk  # just to detect availability quickly
    except ImportError:
        return "vtk not available"

    # Export the part to STL first (cheap, in-process)
    import tempfile
    try:
        from build123d import export_stl
    except ImportError:
        return "build123d export_stl unavailable"
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        stl_tmp = f.name
    try:
        export_stl(part, stl_tmp)
    except Exception as e:
        try: os.unlink(stl_tmp)
        except OSError: pass
        return f"export_stl failed: {type(e).__name__}: {e}"

    # Spawn the VTK render in a subprocess
    import subprocess, sys
    code = _VTK_SUBPROCESS_SCRIPT.format(
        stl_path=repr(stl_tmp),
        out_path=repr(out_path),
        width=width, height=height,
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        try: os.unlink(stl_tmp)
        except OSError: pass
        return "VTK subprocess timeout"
    except Exception as e:
        try: os.unlink(stl_tmp)
        except OSError: pass
        return f"VTK subprocess error: {type(e).__name__}: {e}"
    finally:
        try: os.unlink(stl_tmp)
        except OSError: pass

    if result.returncode != 0:
        # capture the abort/error nicely for the caller
        err = (result.stderr or result.stdout or "").strip()
        # squash multi-line so the report stays compact
        err_one_line = err.replace("\n", " | ")[:200]
        return f"VTK rc={result.returncode}: {err_one_line}"

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return None
    return "VTK ran but produced no output file"


_VTK_SUBPROCESS_SCRIPT = """
import sys, os
try:
    import vtk
    reader = vtk.vtkSTLReader()
    reader.SetFileName({stl_path})
    reader.Update()
    polydata = reader.GetOutput()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.7, 0.75, 0.85)
    actor.GetProperty().SetAmbient(0.2)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetSpecular(0.3)
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.AddRenderer(renderer)
    window.SetSize({width}, {height})
    cam = renderer.GetActiveCamera()
    cam.SetPosition(1, -1, 1); cam.SetViewUp(0, 0, 1)
    renderer.ResetCamera(); cam.Zoom(1.2)
    window.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(window); w2i.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName({out_path})
    writer.SetInputData(w2i.GetOutput())
    writer.Write()
    sys.exit(0)
except Exception as e:
    sys.stderr.write(f'{{type(e).__name__}}: {{e}}\\n')
    sys.exit(1)
"""


# ---------------------------------------------------------------------------
# Backend 2: SVG isometric (always works)
# ---------------------------------------------------------------------------

def _try_svg_render(part, out_path: str,
                     width: int = 800, height: int = 600) -> Optional[str]:
    """Project the part edges onto an isometric viewport, write SVG.

    Uses build123d's `project_to_viewport` if available; otherwise
    falls back to a manual isometric projection.
    """
    try:
        from build123d import Vector
    except ImportError:
        return "build123d not available"

    try:
        # Try the native projector first
        try:
            from build123d import project_to_viewport
            visible, hidden = project_to_viewport(
                part,
                viewport_origin=(50, -50, 50),
                viewport_up=(0, 0, 1),
                look_at=(0, 0, 0),
            )
            edges_2d = list(visible)
        except Exception:
            edges_2d = None

        if edges_2d is None:
            # Manual isometric: project every edge's endpoints
            edges_2d = []
            for edge in part.edges():
                try:
                    a = edge @ 0
                    b = edge @ 1
                    # Isometric projection: x' = (x - y) * cos30, y' = (x + y) * sin30 - z
                    import math
                    cs = math.cos(math.radians(30))
                    sn = math.sin(math.radians(30))
                    pa = ((a.X - a.Y) * cs, (a.X + a.Y) * sn - a.Z)
                    pb = ((b.X - b.Y) * cs, (b.X + b.Y) * sn - b.Z)
                    edges_2d.append((pa, pb))
                except Exception:
                    continue
            if not edges_2d:
                return "no edges found to project"

            # We have list of ((x1,y1),(x2,y2)) — find bounds
            xs = [p[0] for e in edges_2d for p in e]
            ys = [p[1] for e in edges_2d for p in e]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x_span = x_max - x_min or 1
            y_span = y_max - y_min or 1
            scale = min((width - 40) / x_span, (height - 40) / y_span)
            ox = 20 - x_min * scale
            oy = height - (20 - y_min * scale)  # flip y for SVG

            lines = [f'<svg xmlns="http://www.w3.org/2000/svg" '
                     f'width="{width}" height="{height}" '
                     f'viewBox="0 0 {width} {height}">',
                     f'<rect width="{width}" height="{height}" fill="white"/>',
                     f'<g stroke="black" stroke-width="0.6" fill="none">']
            for (a, b) in edges_2d:
                x1 = ox + a[0] * scale
                y1 = oy - a[1] * scale
                x2 = ox + b[0] * scale
                y2 = oy - b[1] * scale
                lines.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" '
                             f'x2="{x2:.2f}" y2="{y2:.2f}"/>')
            lines.append('</g></svg>')
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            return None

        # If we got here, project_to_viewport worked — write its output
        # as an SVG using ocpsvg if available, else manual
        try:
            from ocpsvg import wires_to_svg
            svg_content = wires_to_svg(visible, hidden=hidden,
                                       width=width, height=height)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(svg_content)
            return None
        except Exception:
            return None  # fallback path already wrote the SVG above
    except Exception as e:
        return f"SVG error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def render_preview(part, out_path: str,
                    width: int = 800, height: int = 600,
                    prefer: str = "auto") -> PreviewResult:
    """Render a preview image of a part.

    Args:
        part: build123d Part / Compound / Solid
        out_path: where to write. Extension determines format:
                  .png → VTK render (best);  .svg → SVG fallback
        prefer: "auto" (PNG if possible, SVG otherwise),
                "png", "svg"

    Returns: PreviewResult with backend name and any error message.
    """
    ext = out_path.lower().rsplit(".", 1)[-1] if "." in out_path else ""
    if prefer == "auto":
        if ext == "svg":
            prefer = "svg"
        else:
            prefer = "png"

    if prefer == "png":
        # Make sure the path has .png
        if not out_path.lower().endswith(".png"):
            out_path = out_path.rsplit(".", 1)[0] + ".png"
        err = _try_vtk_render(part, out_path, width, height)
        if err is None:
            return PreviewResult(path=out_path, backend="vtk")
        # Fall through to SVG with a renamed path
        svg_path = out_path.rsplit(".", 1)[0] + ".svg"
        err2 = _try_svg_render(part, svg_path, width, height)
        if err2 is None:
            return PreviewResult(path=svg_path, backend="svg",
                                 error=f"png failed: {err}; fell back to svg")
        return PreviewResult(path=out_path, backend="failed",
                             error=f"png: {err}; svg: {err2}")
    else:
        if not out_path.lower().endswith(".svg"):
            out_path = out_path.rsplit(".", 1)[0] + ".svg"
        err = _try_svg_render(part, out_path, width, height)
        if err is None:
            return PreviewResult(path=out_path, backend="svg")
        return PreviewResult(path=out_path, backend="failed", error=err)
