"""
Drawing standards: sheet sizes, layers, dimension styles, linetypes, units.

References:
- ASME Y14.1 / Y14.1M     - Sheet sizes
- ASME Y14.2              - Line conventions
- ASME Y14.5              - Dimensioning and tolerancing
- ISO 5457                - Sheet sizes
- ISO 128 / 129           - Line conventions / dimensioning
- AIA CAD Layer Guidelines / ISO 13567 - Layer naming

All sheet dimensions are in millimetres. The Drawing.units field controls
DXF $INSUNITS and the dimstyle so geometry and dimension display stay
consistent.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

from ezdxf.enums import InsertUnits


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

class Units(str, Enum):
    """Drawing units. Must match the units of all coordinate inputs."""
    MILLIMETERS = "mm"
    CENTIMETERS = "cm"
    METERS      = "m"
    INCHES      = "in"
    FEET        = "ft"

    @property
    def dxf_insunits(self) -> int:
        return {
            Units.MILLIMETERS: InsertUnits.Millimeters,
            Units.CENTIMETERS: InsertUnits.Centimeters,
            Units.METERS:      InsertUnits.Meters,
            Units.INCHES:      InsertUnits.Inches,
            Units.FEET:        InsertUnits.Feet,
        }[self]

    @property
    def suffix(self) -> str:
        return {
            Units.MILLIMETERS: " mm",
            Units.CENTIMETERS: " cm",
            Units.METERS:      " m",
            Units.INCHES:      '"',
            Units.FEET:        "'",
        }[self]

    @property
    def is_metric(self) -> bool:
        return self in (Units.MILLIMETERS, Units.CENTIMETERS, Units.METERS)

    def to_mm(self, value: float) -> float:
        return {
            Units.MILLIMETERS: 1.0,
            Units.CENTIMETERS: 10.0,
            Units.METERS:      1000.0,
            Units.INCHES:      25.4,
            Units.FEET:        304.8,
        }[self] * value


# ---------------------------------------------------------------------------
# Sheet sizes (ISO 5457 + ASME Y14.1, landscape, all in mm)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SheetSize:
    name:          str
    width_mm:      float
    height_mm:     float
    border_left:   float
    border_right:  float
    border_top:    float
    border_bottom: float

    @property
    def inside_width(self) -> float:
        return self.width_mm - self.border_left - self.border_right

    @property
    def inside_height(self) -> float:
        return self.height_mm - self.border_top - self.border_bottom


SHEETS: Dict[str, SheetSize] = {
    # ISO
    "A4": SheetSize("A4", 297.0,  210.0,  20, 10, 10, 10),
    "A3": SheetSize("A3", 420.0,  297.0,  20, 10, 10, 10),
    "A2": SheetSize("A2", 594.0,  420.0,  20, 10, 10, 10),
    "A1": SheetSize("A1", 841.0,  594.0,  20, 10, 10, 10),
    "A0": SheetSize("A0", 1189.0, 841.0,  20, 10, 10, 10),
    # ASME Y14.1 (inch sizes expressed in mm)
    "ANSI_A": SheetSize("ANSI_A", 279.4,  215.9, 12.7,  6.35, 6.35, 6.35),
    "ANSI_B": SheetSize("ANSI_B", 431.8,  279.4, 12.7,  6.35, 6.35, 6.35),
    "ANSI_C": SheetSize("ANSI_C", 558.8,  431.8, 19.05, 12.7, 12.7, 12.7),
    "ANSI_D": SheetSize("ANSI_D", 863.6,  558.8, 19.05, 12.7, 12.7, 12.7),
    "ANSI_E": SheetSize("ANSI_E", 1117.6, 863.6, 19.05, 12.7, 12.7, 12.7),
}


# ISO/BS √2 text-height series in mm, the same family ASME Y14.2 hints
# at. We snap any requested text height to the nearest value in this
# series so drawings produced by different specs still read uniformly.
TEXT_HEIGHT_SERIES_MM = (1.8, 2.5, 3.5, 5.0, 7.0, 10.0, 14.0, 20.0)


def snap_text_height(h_mm: float) -> float:
    """Snap a requested text height to the nearest √2-series value."""
    return min(TEXT_HEIGHT_SERIES_MM, key=lambda v: abs(v - h_mm))


# Recommended *dimension* text height per sheet — the value the dimstyle
# should use so dim text scales sensibly with paper size. These follow
# common practice: ~2.5 mm body / 3.5 mm dim text on A3, 5 mm on A1+.
DIM_TEXT_HEIGHT_FOR_SHEET = {
    "A4":     2.5,
    "A3":     3.5,
    "A2":     3.5,
    "A1":     5.0,
    "A0":     5.0,
    "ANSI_A": 2.5,
    "ANSI_B": 3.5,
    "ANSI_C": 3.5,
    "ANSI_D": 5.0,
    "ANSI_E": 5.0,
}


def dim_text_height_for_sheet(sheet_name: str) -> float:
    return DIM_TEXT_HEIGHT_FOR_SHEET.get(sheet_name, 3.5)


# ---------------------------------------------------------------------------
# Layer scheme
# ---------------------------------------------------------------------------
#
# Color policy: every plottable layer uses ACI 7 (black/white auto). The
# visual hierarchy comes from LINEWEIGHT and LINETYPE — *not* color. This
# matches how technical drawings are actually plotted (monochrome) and
# avoids unreadable yellow/cyan dim text. If a project really needs a
# colored layer, it can be added after register_layers().
#
# Lineweights (×100, mm) follow ASME Y14.2:
#   Visible / object lines        0.60 mm  ->  60
#   Cutting plane / section       0.60 mm  ->  60
#   Border                        0.70 mm  ->  70
#   Hidden, center, dimensions,
#   extension, leaders, hatch     0.30 mm  ->  30
#   Construction / grid           0.18 mm  ->  18

@dataclass(frozen=True)
class LayerDef:
    name:         str
    color:        int           # AutoCAD Color Index (ACI 7 = black/white)
    linetype:     str = "CONTINUOUS"
    lineweight:   int = -3      # -3 = ByLayer default
    description:  str = ""


DEFAULT_LAYERS: Tuple[LayerDef, ...] = (
    # Drawing infrastructure
    LayerDef("BORDER",       7, "CONTINUOUS",  70, "Sheet border (heavy)"),
    LayerDef("TITLEBLOCK",   7, "CONTINUOUS",  50, "Title block geometry"),
    LayerDef("TB_TEXT",      7, "CONTINUOUS",  25, "Title block text"),
    LayerDef("VIEWPORT",   251, "CONTINUOUS",  13, "Paperspace viewports (non-plotting)"),
    # Geometry (ASME Y14.2)
    LayerDef("VISIBLE",      7, "CONTINUOUS",  60, "Visible object lines (thick)"),
    LayerDef("HIDDEN",       7, "DASHED",      30, "Hidden lines"),
    LayerDef("CENTER",       7, "CENTER",      25, "Center lines"),
    LayerDef("PHANTOM",      7, "PHANTOM",     25, "Phantom / alternate position"),
    LayerDef("CONSTRUCTION", 8, "DASHED2",     13, "Construction (non-printing) lines"),
    # Sections
    LayerDef("SECTION",      7, "DASHDOT",     60, "Section / cutting plane"),
    LayerDef("HATCH",        7, "CONTINUOUS",  18, "Section hatching (light)"),
    LayerDef("BREAK",        7, "CONTINUOUS",  30, "Break lines"),
    # Annotation (all black for readability)
    LayerDef("DIMENSIONS",   7, "CONTINUOUS",  25, "Dimensions"),
    LayerDef("TEXT",         7, "CONTINUOUS",  25, "General notes and labels"),
    LayerDef("LEADERS",      7, "CONTINUOUS",  25, "Leader lines and balloons"),
    LayerDef("SYMBOLS",      7, "CONTINUOUS",  25, "GD&T, surface finish, weld"),
    # Architectural / MEP (kept distinct for facility drawings, still printable)
    LayerDef("WALLS",        7, "CONTINUOUS",  60, "Architectural walls"),
    LayerDef("DOORS",        7, "CONTINUOUS",  30, "Doors and swings"),
    LayerDef("WINDOWS",      7, "CONTINUOUS",  30, "Windows"),
    LayerDef("FURNITURE",    8, "CONTINUOUS",  18, "Furniture / equipment (light grey)"),
    LayerDef("GRID",         8, "CENTER",      13, "Grid lines (light grey)"),
    LayerDef("MEP_ELEC",     7, "CONTINUOUS",  30, "Electrical"),
    LayerDef("MEP_HVAC",     7, "CONTINUOUS",  30, "HVAC"),
    LayerDef("MEP_PLUMB",    7, "CONTINUOUS",  30, "Plumbing"),
    LayerDef("REVISION",     7, "CONTINUOUS",  30, "Revision clouds and tags"),
)


# Text style used by every dimstyle and every add_text call.
RAPCAD_TEXT_STYLE = "RAPCAD"
# Font family for the matplotlib renderer (kept in sync with the DXF style).
RAPCAD_FONT_FAMILY = "DejaVu Sans"


def register_layers(doc, layers=DEFAULT_LAYERS) -> None:
    """Idempotent: add any missing layers to the document."""
    _ensure_linetypes(doc)
    for L in layers:
        if L.name in doc.layers:
            continue
        layer = doc.layers.add(L.name, color=L.color, linetype=L.linetype)
        layer.dxf.lineweight = L.lineweight
        if L.description:
            layer.description = L.description


def register_text_style(doc) -> None:
    """Register the RAPCAD text style. Uses a portable open-source sans
    face; the matplotlib renderer is configured to use the same family
    so on-screen text matches the DXF.
    """
    if RAPCAD_TEXT_STYLE in doc.styles:
        return
    doc.styles.new(
        RAPCAD_TEXT_STYLE,
        dxfattribs={
            # ezdxf 1.x will look this up via fontTools; if missing it
            # falls back to a stock SHX face, which is acceptable.
            "font": "DejaVuSans.ttf",
            "height": 0.0,           # 0 = use entity height (TEXT/MTEXT controls it)
            "width": 1.0,
        },
    )


def _ensure_linetypes(doc) -> None:
    from ezdxf.tools.standards import setup_linetypes
    setup_linetypes(doc)


# ---------------------------------------------------------------------------
# Dimension styles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DimStyleDef:
    name:        str
    text_height: float = 3.5      # drawing units (ASME Y14.2 default for A3)
    arrow_size:  float = 3.0
    ext_offset:  float = 0.625    # gap between feature and extension line start
    ext_above:   float = 1.5      # extension above dim line
    dim_line_gap: float = 1.0     # gap around text in dim line
    decimals:    int = 2          # DIMDEC
    unit_format: int = 2          # DIMLUNIT  1=sci 2=dec 3=eng 4=arch 5=frac
    angle_format: int = 0         # DIMAUNIT
    suffix:      str = ""         # DIMPOST
    text_color:  int = 256        # ByLayer


DIMSTYLES_METRIC: Tuple[DimStyleDef, ...] = (
    DimStyleDef("RAPCAD_MECH_MM",   text_height=3.5, arrow_size=3.0,
                decimals=2, unit_format=2, suffix=""),
    DimStyleDef("RAPCAD_ARCH_MM",   text_height=3.5, arrow_size=2.5,
                decimals=0, unit_format=2, suffix=""),
    DimStyleDef("RAPCAD_STRUCT_MM", text_height=3.5, arrow_size=3.5,
                decimals=0, unit_format=2, suffix=""),
)

DIMSTYLES_IMPERIAL: Tuple[DimStyleDef, ...] = (
    DimStyleDef("RAPCAD_MECH_IN", text_height=0.125, arrow_size=0.125,
                ext_offset=0.0625, ext_above=0.125, dim_line_gap=0.09,
                decimals=3, unit_format=2, suffix='"'),
    DimStyleDef("RAPCAD_ARCH_IN", text_height=0.156, arrow_size=0.125,
                ext_offset=0.0625, ext_above=0.125, dim_line_gap=0.09,
                decimals=4, unit_format=4, suffix=""),
)


def register_dimstyles(doc, units: Units, text_height: float | None = None) -> None:
    """Register the RAPCAD dimstyles. If text_height is given, all
    dimstyles override their default text height (and the arrow size is
    scaled proportionally). Otherwise each dimstyle uses its own default.
    """
    register_text_style(doc)
    pool = DIMSTYLES_METRIC if units.is_metric else DIMSTYLES_IMPERIAL
    for ds in pool:
        if ds.name in doc.dimstyles:
            continue
        s = doc.dimstyles.add(ds.name)
        # Per-style defaults
        h_use = ds.text_height if text_height is None else text_height
        scale = h_use / ds.text_height          # proportional scale factor
        s.dxf.dimtxt   = h_use
        s.dxf.dimasz   = ds.arrow_size * scale
        s.dxf.dimexo   = ds.ext_offset * scale
        s.dxf.dimexe   = ds.ext_above * scale
        s.dxf.dimgap   = ds.dim_line_gap * scale
        s.dxf.dimdec   = ds.decimals
        s.dxf.dimlunit = ds.unit_format
        s.dxf.dimaunit = ds.angle_format
        s.dxf.dimpost  = ds.suffix
        s.dxf.dimclrt  = ds.text_color
        s.dxf.dimtad   = 1     # text above dim line
        s.dxf.dimtih   = 1     # text inside extensions horizontal (readable)
        s.dxf.dimtoh   = 1     # text outside extensions horizontal (readable)
        s.dxf.dimzin   = 8     # suppress trailing zeros
        s.dxf.dimscale = 1.0
        s.dxf.dimtxsty = RAPCAD_TEXT_STYLE


def dimstyle_for(workflow: str, units: Units) -> str:
    workflow = workflow.lower()
    if units.is_metric:
        return {"mech":   "RAPCAD_MECH_MM",
                "arch":   "RAPCAD_ARCH_MM",
                "struct": "RAPCAD_STRUCT_MM"}.get(workflow, "RAPCAD_MECH_MM")
    return {"mech":   "RAPCAD_MECH_IN",
            "arch":   "RAPCAD_ARCH_IN",
            "struct": "RAPCAD_MECH_IN"}.get(workflow, "RAPCAD_MECH_IN")
