"""
Title block + revision block, drawn declaratively into paperspace.

Approach
--------
The title block is a fixed grid (5 rows x 4 columns). A flat layout list
maps each field on the TitleBlock dataclass to a cell rectangle defined
by (row, col_start, col_span). The renderer iterates the list, draws
each cell's borders, and places the label + value inside.

This separates *what goes where* (data) from *how to draw it* (code), so
changing the layout is a one-line edit to TITLE_BLOCK_LAYOUT.

Visual grid:

  +-----------+-----------+-----------+-----------+
  | ORG (spans 2 cols)    | PROJECT   | DWG NO    |  row 5 (top)
  +-----------+-----------+-----------+-----------+
  | TITLE (spans 3 cols)              | REV       |  row 4
  +-----------+-----------+-----------+-----------+
  | SUBTITLE (spans 3 cols)           | DATE      |  row 3
  +-----------+-----------+-----------+-----------+
  | SCALE     | UNITS     | MATERIAL  | FINISH    |  row 2
  +-----------+-----------+-----------+-----------+
  | DRAWN     | CHECKED   | APPROVED  | TOLERANCE |  row 1 (bottom)
  +-----------+-----------+-----------+-----------+
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ezdxf.enums import MTextEntityAlignment, TextEntityAlignment

from .schema import RevisionEntry, TitleBlock
from .standards import RAPCAD_TEXT_STYLE, SheetSize


# Character-to-text-height ratio for the renderer. Empirically measured
# for DejaVu Sans rendered via the matplotlib backend, where DXF height
# is passed as fontsize (em-height) and the actual character width
# averages ~0.60 per em. Since ezdxf renders the full em-height where a
# DXF-aware viewer would treat the value as cap-height, the effective
# width per DXF-height-unit lands around 0.85 for mixed-case strings.
# Use 0.85 so the auto-fit shrink kicks in early enough to avoid right-
# edge overflow.
CHAR_W_RATIO_TB = 0.85


def _fit_text_height(text: str, requested_h: float,
                     available_w: float, min_h: float = 1.6) -> float:
    """Return a text height <= requested_h that lets `text` fit in
    `available_w`. Falls back to min_h if even that is too tight."""
    if not text:
        return requested_h
    n = len(text)
    width_at_requested = n * requested_h * CHAR_W_RATIO_TB
    if width_at_requested <= available_w:
        return requested_h
    shrunk = available_w / (n * CHAR_W_RATIO_TB)
    return max(min_h, shrunk)


# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

TB_WIDTH  = 180.0
TB_HEIGHT = 60.0
TB_ROWS   = 5
TB_COLS   = 4
ROW_H     = TB_HEIGHT / TB_ROWS    # 12.0 mm
COL_W     = TB_WIDTH  / TB_COLS    # 45.0 mm

# Text sizes
LABEL_H   = 2.2
VALUE_H   = 3.0
TITLE_H   = 5.0
# Vertical clearances
LABEL_TOP_GAP    = 1.2    # space above label cap, below row divider
VALUE_BOTTOM_GAP = 1.2    # space below value baseline, above row bottom


# ---------------------------------------------------------------------------
# Layout: one entry per cell
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellSpec:
    row:        int          # 1=bottom, TB_ROWS=top
    col_start:  int          # 0..TB_COLS-1
    col_span:   int          # >=1
    label:      str          # small heading, e.g. "TITLE"
    field:      str          # attribute on TitleBlock to read
    value_size: float = VALUE_H
    align:      str = "left"      # "left" | "center"


TITLE_BLOCK_LAYOUT: List[CellSpec] = [
    # Top row: organisation, project, drawing number
    CellSpec(row=5, col_start=0, col_span=2, label="ORG",      field="org",       value_size=TITLE_H, align="left"),
    CellSpec(row=5, col_start=2, col_span=1, label="PROJECT",  field="project"),
    CellSpec(row=5, col_start=3, col_span=1, label="DWG NO",   field="drawing_no"),
    # Title row
    CellSpec(row=4, col_start=0, col_span=3, label="TITLE",    field="title",     value_size=TITLE_H, align="left"),
    CellSpec(row=4, col_start=3, col_span=1, label="REV",      field="rev",       value_size=TITLE_H, align="center"),
    # Subtitle / date
    CellSpec(row=3, col_start=0, col_span=3, label="SUBTITLE", field="subtitle"),
    CellSpec(row=3, col_start=3, col_span=1, label="DATE",     field="date"),
    # Scale / units / material / finish
    CellSpec(row=2, col_start=0, col_span=1, label="SCALE",    field="scale"),
    CellSpec(row=2, col_start=1, col_span=1, label="UNITS",    field="units_label"),
    CellSpec(row=2, col_start=2, col_span=1, label="MATERIAL", field="material"),
    CellSpec(row=2, col_start=3, col_span=1, label="FINISH",   field="finish"),
    # Personnel + tolerance
    CellSpec(row=1, col_start=0, col_span=1, label="DRAWN",    field="drawn_by"),
    CellSpec(row=1, col_start=1, col_span=1, label="CHECKED",  field="checked_by"),
    CellSpec(row=1, col_start=2, col_span=1, label="APPROVED", field="approved_by"),
    CellSpec(row=1, col_start=3, col_span=1, label="TOLERANCE", field="tolerance"),
]


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def draw_border_and_titleblock(psp,
                               sheet: SheetSize,
                               tb: TitleBlock,
                               revisions: List[RevisionEntry]) -> None:
    """Render border, title block, and (if any) revision block."""
    _draw_border(psp, sheet)
    _draw_title_block(psp, sheet, tb)
    if revisions:
        _draw_revision_block(psp, sheet, revisions)
    _draw_notes(psp, sheet, tb)


# ---------------------------------------------------------------------------
# Border + zone tick marks
# ---------------------------------------------------------------------------

def _draw_border(psp, sheet: SheetSize) -> None:
    bl = (sheet.border_left, sheet.border_bottom)
    tr = (sheet.width_mm - sheet.border_right,
          sheet.height_mm - sheet.border_top)
    psp.add_lwpolyline(
        [(bl[0], bl[1]), (tr[0], bl[1]),
         (tr[0], tr[1]), (bl[0], tr[1])],
        close=True,
        dxfattribs={"layer": "BORDER", "lineweight": 50},
    )

    inside_w = sheet.inside_width
    inside_h = sheet.inside_height
    zone_w = inside_w / max(1, int(inside_w // 100))
    zone_h = inside_h / max(1, int(inside_h // 100))
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"

    x = bl[0]
    i = 0
    while x + zone_w <= tr[0] + 1e-6:
        cx = x + zone_w / 2
        if i < len(letters):
            psp.add_text(letters[i],
                         dxfattribs={"layer": "BORDER", "height": 3.0,
                                     "style": RAPCAD_TEXT_STYLE}
                         ).set_placement(
                (cx, tr[1] - 4.0),
                align=TextEntityAlignment.MIDDLE_CENTER)
        x += zone_w
        i += 1

    y = bl[1]
    j = 1
    while y + zone_h <= tr[1] + 1e-6:
        cy = y + zone_h / 2
        psp.add_text(str(j),
                     dxfattribs={"layer": "BORDER", "height": 3.0,
                                 "style": RAPCAD_TEXT_STYLE}
                     ).set_placement(
            (tr[0] - 4.0, cy),
            align=TextEntityAlignment.MIDDLE_CENTER)
        y += zone_h
        j += 1


# ---------------------------------------------------------------------------
# Title block — declarative
# ---------------------------------------------------------------------------

def _draw_title_block(psp, sheet: SheetSize, tb: TitleBlock) -> None:
    right = sheet.width_mm - sheet.border_right
    bot   = sheet.border_bottom
    x0, y0 = right - TB_WIDTH, bot
    x1, y1 = right, bot + TB_HEIGHT

    # Outer box (heavy)
    psp.add_lwpolyline(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        close=True,
        dxfattribs={"layer": "TITLEBLOCK", "lineweight": 50},
    )

    # Horizontal row separators (between rows, not on the outer boundary)
    for i in range(1, TB_ROWS):
        ry = y0 + i * ROW_H
        psp.add_line((x0, ry), (x1, ry),
                     dxfattribs={"layer": "TITLEBLOCK", "lineweight": 18})

    # Draw each cell from the layout list
    for cell in TITLE_BLOCK_LAYOUT:
        _draw_cell(psp, x0, y0, cell, tb)


def _draw_cell(psp, x0: float, y0: float,
               cell: CellSpec, tb: TitleBlock) -> None:
    """Draw one cell: dividers + label + value, with collision-safe spacing.

    The font's actual ascent is slightly larger than the nominal text
    height (DejaVu Sans ascent / cap-height ≈ 1.22). LABEL_TOP_GAP buys
    enough room that the cap top stays well clear of the divider above.
    """
    row_bot = y0 + (cell.row - 1) * ROW_H
    cell_x_left  = x0 + cell.col_start * COL_W
    cell_x_right = cell_x_left + cell.col_span * COL_W
    cell_width   = cell.col_span * COL_W
    avail_w      = cell_width - 3.0      # padding 1.5mm each side

    # Vertical divider to the LEFT of the cell (skip at the block edge)
    if cell.col_start > 0:
        psp.add_line((cell_x_left, row_bot),
                     (cell_x_left, row_bot + ROW_H),
                     dxfattribs={"layer": "TITLEBLOCK", "lineweight": 18})

    # Label: positioned so the cap top sits LABEL_TOP_GAP below the row top
    if cell.label:
        label_baseline_y = row_bot + ROW_H - LABEL_TOP_GAP - LABEL_H
        psp.add_text(cell.label,
                     dxfattribs={"layer": "TB_TEXT", "height": LABEL_H,
                                 "style": RAPCAD_TEXT_STYLE}
                     ).set_placement(
            (cell_x_left + 1.5, label_baseline_y),
            align=TextEntityAlignment.BOTTOM_LEFT)

    # Value: auto-shrink height to fit cell width
    value = getattr(tb, cell.field, "") or ""
    if value:
        h_used = _fit_text_height(value, cell.value_size, avail_w)
        value_baseline_y = row_bot + VALUE_BOTTOM_GAP
        if cell.align == "center":
            cx = cell_x_left + cell_width / 2
            psp.add_text(value,
                         dxfattribs={"layer": "TB_TEXT",
                                     "height": h_used,
                                     "style": RAPCAD_TEXT_STYLE}
                         ).set_placement(
                (cx, value_baseline_y + h_used * 0.5),
                align=TextEntityAlignment.MIDDLE_CENTER)
        else:
            psp.add_text(value,
                         dxfattribs={"layer": "TB_TEXT",
                                     "height": h_used,
                                     "style": RAPCAD_TEXT_STYLE}
                         ).set_placement(
                (cell_x_left + 1.5, value_baseline_y),
                align=TextEntityAlignment.BOTTOM_LEFT)


# ---------------------------------------------------------------------------
# Notes band (above the title block, full sheet width if there's room)
# ---------------------------------------------------------------------------

def _draw_notes(psp, sheet: SheetSize, tb: TitleBlock) -> None:
    if not tb.notes:
        return
    right = sheet.width_mm - sheet.border_right
    left  = sheet.border_left
    bot   = sheet.border_bottom + TB_HEIGHT + 4.0
    width = right - left - 4.0
    text = "NOTES:  " + "   ".join(f"{i+1}. {n}"
                                   for i, n in enumerate(tb.notes))
    psp.add_mtext(text,
                  dxfattribs={"layer": "TB_TEXT",
                              "char_height": 2.5,
                              "width": width,
                              "style": RAPCAD_TEXT_STYLE}
                  ).set_location(
        (left + 2.0, bot),
        attachment_point=MTextEntityAlignment.BOTTOM_LEFT)


# ---------------------------------------------------------------------------
# Revision block (sits above the title block, right-aligned)
# ---------------------------------------------------------------------------

REV_BLOCK_W = 100.0
REV_ROW_H   = 6.0

REV_COLUMNS = [
    ("REV",         8.0,  TextEntityAlignment.MIDDLE_CENTER),
    ("DESCRIPTION", 55.0, TextEntityAlignment.MIDDLE_LEFT),
    ("DATE",        20.0, TextEntityAlignment.MIDDLE_CENTER),
    ("BY",          17.0, TextEntityAlignment.MIDDLE_CENTER),
]


def _draw_revision_block(psp, sheet: SheetSize, revs: List[RevisionEntry]) -> None:
    right = sheet.width_mm - sheet.border_right
    bot   = sheet.border_bottom + TB_HEIGHT
    rows = len(revs) + 1
    x0 = right - REV_BLOCK_W
    y0 = bot
    y1 = bot + rows * REV_ROW_H

    # Outer box
    psp.add_lwpolyline(
        [(x0, y0), (right, y0), (right, y1), (x0, y1)],
        close=True,
        dxfattribs={"layer": "TITLEBLOCK", "lineweight": 25},
    )
    # Header separator
    psp.add_line((x0, y1 - REV_ROW_H), (right, y1 - REV_ROW_H),
                 dxfattribs={"layer": "TITLEBLOCK", "lineweight": 18})

    # Column dividers (cumulative)
    col_x = [x0]
    cum = 0.0
    for _, w, _ in REV_COLUMNS:
        cum += w
        col_x.append(x0 + cum)
    for cx in col_x[1:-1]:
        psp.add_line((cx, y0), (cx, y1),
                     dxfattribs={"layer": "TITLEBLOCK", "lineweight": 13})

    # Header row
    for i, (label, _, align) in enumerate(REV_COLUMNS):
        cell_l, cell_r = col_x[i], col_x[i + 1]
        if align == TextEntityAlignment.MIDDLE_LEFT:
            tx = cell_l + 1.5
        else:
            tx = (cell_l + cell_r) / 2
        psp.add_text(label,
                     dxfattribs={"layer": "TB_TEXT", "height": 2.0,
                                 "style": RAPCAD_TEXT_STYLE}
                     ).set_placement((tx, y1 - REV_ROW_H / 2), align=align)

    # Data rows (newest at top)
    for i, r in enumerate(revs):
        row_top = y1 - REV_ROW_H * (i + 1)
        row_mid_y = row_top - REV_ROW_H / 2
        values = [r.rev, r.description, r.date, r.by]
        for j, ((label, col_w, align), value) in enumerate(zip(REV_COLUMNS, values)):
            if not value:
                continue
            cell_l, cell_r = col_x[j], col_x[j + 1]
            avail_w = col_w - 3.0
            h_used = _fit_text_height(value, 2.4, avail_w, min_h=1.4)
            if align == TextEntityAlignment.MIDDLE_LEFT:
                tx = cell_l + 1.5
            else:
                tx = (cell_l + cell_r) / 2
            psp.add_text(value,
                         dxfattribs={"layer": "TB_TEXT", "height": h_used,
                                     "style": RAPCAD_TEXT_STYLE}
                         ).set_placement((tx, row_mid_y), align=align)
