"""
Render a DXF (or in-memory document) to PNG / PDF / SVG.

Uses ezdxf's matplotlib drawing add-on, which understands layers,
linetypes, lineweights, dimensions, hatches, and MTEXT.

Two render modes:
  - "modelspace" : render only the modelspace contents (drawing area)
  - "paperspace" : render the active paperspace layout (sheet view with
                   border and title block)

Design choices:
  - The figure aspect matches the actual sheet aspect (paperspace) so
    nothing is squashed.
  - Axis limits are pinned to the sheet boundary in paperspace mode and
    to the geometry's bounding box (with a margin) in modelspace mode —
    no `bbox_inches='tight'`, which silently crops content with thin
    geometry at the edges.
  - All text renders with the same `DejaVu Sans` family the DXF text
    style references, so on-screen text matches the saved DXF.
  - ColorPolicy.BLACK guarantees readable monochrome output regardless
    of layer ACI values — line weight + linetype provide hierarchy.
"""
from __future__ import annotations

import logging
import warnings
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.config import (BackgroundPolicy, ColorPolicy,
                                         Configuration)
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.document import Drawing

from .standards import RAPCAD_FONT_FAMILY, SHEETS


# Reasonable display DPI: 200 = ~3300×2330 px for A3, sharp at typical
# preview sizes.
DEFAULT_DPI = 200


def _sheet_figsize(sheet_name: str) -> tuple[float, float]:
    """Inch dimensions matching the named sheet's aspect."""
    sheet = SHEETS.get(sheet_name)
    if sheet is None:
        return (11.69, 8.27)  # A4 fallback
    return (sheet.width_mm / 25.4, sheet.height_mm / 25.4)


def _sheet_name_from_layout(doc: Drawing) -> str:
    """Best-effort discovery of which sheet a paperspace layout uses."""
    psp = doc.paperspace()
    try:
        layout_dxf = psp.dxf_layout.dxf
        w_mm = layout_dxf.paper_width
        h_mm = layout_dxf.paper_height
        for name, s in SHEETS.items():
            if abs(s.width_mm - w_mm) < 1 and abs(s.height_mm - h_mm) < 1:
                return name
    except Exception:
        pass
    return "A3"


def render_preview(doc: Drawing,
                   output_path: str,
                   layout: Literal["modelspace", "paperspace"] = "paperspace",
                   dpi: int = DEFAULT_DPI,
                   monochrome: bool = True,
                   figsize: tuple | None = None,
                   sheet: str | None = None) -> str:
    """Render `doc` to a raster/vector image.

    Args:
        doc: ezdxf Drawing.
        output_path: target file. Extension chooses format (png/pdf/svg).
        layout: "modelspace" or "paperspace".
        dpi: raster resolution (default 200).
        monochrome: True (default) forces black-on-white. False keeps
            layer ACI colors — useful for screen viewing but harder to
            read in print.
        figsize: override default figure size (inches).
        sheet: override sheet name for figsize auto-detection.
    """
    sheet_name = sheet or _sheet_name_from_layout(doc)
    if figsize is None:
        if layout == "paperspace":
            figsize = _sheet_figsize(sheet_name)
        else:
            figsize = (11.0, 8.5)

    if layout == "modelspace":
        target = doc.modelspace()
    else:
        target = doc.paperspace()

    color_policy = (ColorPolicy.BLACK if monochrome
                    else ColorPolicy.MONOCHROME_LIGHT_BG)

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")

    # Apply the chosen font as matplotlib's default for this figure.
    # ezdxf 1.4+ also resolves fonts from the DXF text style via fontTools.
    saved_family = plt.rcParams.get("font.family")
    saved_sans   = list(plt.rcParams.get("font.sans-serif", []))
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [RAPCAD_FONT_FAMILY] + saved_sans

    cfg = Configuration(
        background_policy=BackgroundPolicy.WHITE,
        color_policy=color_policy,
        lineweight_scaling=1.0,
        min_lineweight=0.25,
    )
    ctx = RenderContext(doc)
    backend = MatplotlibBackend(ax, adjust_figure=False)
    try:
        Frontend(ctx, backend, config=cfg).draw_layout(target, finalize=True)
    finally:
        # Restore prior rcParams so other figures aren't affected.
        plt.rcParams["font.family"] = saved_family
        plt.rcParams["font.sans-serif"] = saved_sans

    # Pin axis limits explicitly so nothing at the edges gets cropped.
    if layout == "paperspace":
        s = SHEETS.get(sheet_name)
        if s is not None:
            ax.set_xlim(0, s.width_mm)
            ax.set_ylim(0, s.height_mm)
        ax.set_aspect("equal", adjustable="box")
    else:
        # Modelspace: keep auto-fit but add 5% margin so dim text / arrows
        # at the boundary don't touch the figure edge. Let data limits
        # adjust to honour aspect='equal' (cleaner than fighting matplotlib).
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        mx = 0.05 * max(1e-6, xhi - xlo)
        my = 0.05 * max(1e-6, yhi - ylo)
        ax.set_xlim(xlo - mx, xhi + mx)
        ax.set_ylim(ylo - my, yhi + my)
        ax.set_aspect("equal", adjustable="datalim")

    ax.axis("off")
    with warnings.catch_warnings():
        # matplotlib emits an "Ignoring fixed x/y limits" UserWarning when
        # the figure aspect doesn't perfectly match data extents — this is
        # expected for modelspace renders and harmless because we use
        # adjustable='datalim'.
        warnings.filterwarnings("ignore",
                                message="Ignoring fixed.*limits.*",
                                category=UserWarning)
        # Same message is also routed through matplotlib's logger.
        mpl_logger = logging.getLogger("matplotlib")
        prev_level = mpl_logger.level
        mpl_logger.setLevel(logging.ERROR)
        try:
            fig.savefig(output_path, dpi=dpi,
                        facecolor="white", edgecolor="none", pad_inches=0)
        finally:
            mpl_logger.setLevel(prev_level)
    plt.close(fig)
    return output_path
