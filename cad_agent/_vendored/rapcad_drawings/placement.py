"""
Placement engine.

Given a desired annotation (text label, dimension text, leader callout) and
its anchor point on the host geometry, find a non-colliding location:

  1. Try a list of *candidate* positions in priority order. For text labels
     these are the eight compass directions around the anchor at a base
     offset. For linear dimensions the natural side is computed from the
     feature's orientation.
  2. For each candidate, build the text's AABB and query the SpatialIndex.
     If clear (with a halo proportional to text height), accept it.
  3. If none of the close candidates work, expand the search radius and try
     again. Up to `max_rings` rings.
  4. If still no clear spot, return the best candidate with a `leader` flag
     so the caller can draw an explicit leader line from anchor to label.

Text width is estimated from character count × text_height × CHAR_W_RATIO.
This is approximate but consistent with how AutoCAD lays out single-line
TEXT entities at the default font; if exact fitting matters the caller can
override by passing a measured width.

For linear dimensions, this module computes a candidate dimension-line
offset that clears the feature and all upstream geometry, then returns the
final base point. ezdxf's add_linear_dim with `override={"dimtad": 1}`
places the text above the dim line in the usual ASME way.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .geometry import AABB, Point, add, perp_left, scale, sub, unit
from .spatial_index import SpatialIndex


# Mean character-width-to-height ratio for the default AutoCAD font (txt.shx).
# Real ratios vary 0.6-0.85 by glyph; 0.7 is a safe average for AABB padding.
CHAR_W_RATIO = 0.70


@dataclass
class PlacementResult:
    position: Point             # text insert point (lower-left baseline corner of AABB)
    aabb:     AABB              # final AABB occupied by the text
    rotation_deg: float = 0.0
    leader_required: bool = False
    rings_used: int = 0


def linear_dim_aabb(p1: Point, p2: Point, base: Point,
                    angle_deg: float, text_height: float,
                    arrow_size: float | None = None) -> AABB:
    """Tight AABB for a linear dimension's *primary* footprint.

    Covers:
      - the dim line itself, between the feet of p1, p2 projected onto it
      - a small dim-line extension past each foot
      - the text block above the dim line (where DIMTAD=1 puts it)

    Deliberately EXCLUDES the extension lines running from p1, p2 to the
    dim line. Extension lines are thin and crossing them is normal CAD
    practice (e.g. ordinate / baseline / chain dim stacks). Including
    them in collision AABBs would forbid legitimate dim stacks.
    """
    if arrow_size is None:
        arrow_size = text_height
    rad = math.radians(angle_deg)
    u = (math.cos(rad), math.sin(rad))
    nrm = (-u[1], u[0])

    def project(p: Point) -> Point:
        dx, dy = p[0] - base[0], p[1] - base[1]
        t = dx * u[0] + dy * u[1]
        return (base[0] + t * u[0], base[1] + t * u[1])
    f1 = project(p1)
    f2 = project(p2)

    ext = arrow_size * 1.5
    f1_ext = (f1[0] - u[0] * ext, f1[1] - u[1] * ext)
    f2_ext = (f2[0] + u[0] * ext, f2[1] + u[1] * ext)

    # Text sits on the opposite side of nrm from p1 (above the dim line)
    side_p1 = (p1[0] - base[0]) * nrm[0] + (p1[1] - base[1]) * nrm[1]
    side_sign = 1.0 if side_p1 >= 0 else -1.0
    text_band = -side_sign
    text_h = text_height * 2.2

    mid = ((f1[0] + f2[0]) / 2, (f1[1] + f2[1]) / 2)
    text_far = (mid[0] + nrm[0] * text_band * text_h,
                mid[1] + nrm[1] * text_band * text_h)

    # Small strip on the *near* side (arrowheads, tick marks)
    arrow_far = (mid[0] - nrm[0] * text_band * arrow_size * 0.5,
                 mid[1] - nrm[1] * text_band * arrow_size * 0.5)

    pts = [f1, f2, f1_ext, f2_ext, text_far, arrow_far]
    return AABB.from_points(pts)


def text_aabb(insert: Point, text: str, height: float,
              rotation_deg: float = 0.0,
              char_w: float = CHAR_W_RATIO) -> AABB:
    """Estimate the AABB of a single-line text block placed at `insert`
    (left-baseline) and rotated CCW by `rotation_deg`.
    """
    w = max(1, len(text)) * height * char_w
    h = height
    corners_local = [(0, 0), (w, 0), (w, h), (0, h)]
    r = math.radians(rotation_deg)
    cos_r, sin_r = math.cos(r), math.sin(r)
    pts: List[Point] = []
    for x, y in corners_local:
        xr = x * cos_r - y * sin_r + insert[0]
        yr = x * sin_r + y * cos_r + insert[1]
        pts.append((xr, yr))
    return AABB.from_points(pts)


# Eight compass directions (unit vectors) used as candidate offsets for labels.
# Ordered: prefer above, then below, then sides, then diagonals.
COMPASS_8: List[Tuple[str, Point]] = [
    ("N",  (0.0,  1.0)),
    ("S",  (0.0, -1.0)),
    ("E",  (1.0,  0.0)),
    ("W",  (-1.0, 0.0)),
    ("NE", ( 0.7071,  0.7071)),
    ("NW", (-0.7071,  0.7071)),
    ("SE", ( 0.7071, -0.7071)),
    ("SW", (-0.7071, -0.7071)),
]


def place_label(index: SpatialIndex,
                anchor: Point,
                text: str,
                height: float,
                preferred_dir: Optional[Point] = None,
                base_gap: Optional[float] = None,
                clearance: Optional[float] = None,
                max_rings: int = 5,
                exclude_ids: Optional[List[str]] = None) -> PlacementResult:
    """Place a text label near `anchor`.

    base_gap   - initial offset from anchor to nearest edge of the text box.
                 Default: 1.5 * height.
    clearance  - additional padding around the text box treated as occupied.
                 Default: 0.5 * height. This is the "halo" — minimum air
                 between the text and any other geometry.
    """
    if base_gap is None:
        base_gap = 1.5 * height
    if clearance is None:
        clearance = 0.5 * height

    # Build the candidate direction list: preferred_dir first, then compass.
    candidates: List[Tuple[str, Point]] = []
    if preferred_dir is not None:
        u = unit(preferred_dir)
        if u != (0.0, 0.0):
            candidates.append(("PREF", u))
    candidates.extend(COMPASS_8)

    text_w = max(1, len(text)) * height * CHAR_W_RATIO
    text_h = height

    # Try each direction at increasing radii.
    best: Optional[PlacementResult] = None
    for ring in range(max_rings):
        gap = base_gap * (1.0 + 0.85 * ring)
        for label, d in candidates:
            insert = _insert_for_direction(anchor, d, gap, text_w, text_h)
            ab = text_aabb(insert, text, height)
            if index.is_clear(ab, clearance=clearance, exclude_ids=exclude_ids):
                return PlacementResult(position=insert, aabb=ab,
                                       rotation_deg=0.0,
                                       leader_required=False,
                                       rings_used=ring)
            # Track best-so-far by least overlap for leader fallback
            if best is None:
                best = PlacementResult(position=insert, aabb=ab,
                                       rotation_deg=0.0,
                                       leader_required=True,
                                       rings_used=ring)

    # No clear spot found in any ring. Push out farther and require a leader.
    far_gap = base_gap * (1.0 + 0.85 * max_rings) * 1.4
    insert = _insert_for_direction(anchor, candidates[0][1], far_gap, text_w, text_h)
    ab = text_aabb(insert, text, height)
    return PlacementResult(position=insert, aabb=ab,
                           rotation_deg=0.0,
                           leader_required=True,
                           rings_used=max_rings)


def _insert_for_direction(anchor: Point, d: Point, gap: float,
                          text_w: float, text_h: float) -> Point:
    """Compute the lower-left insert point so the text box sits on the
    `anchor` side of direction `d` with a `gap` air space between them.

    The strategy: place the text-box center at anchor + d * (gap + half_extent),
    then convert to lower-left insert.
    """
    dx, dy = d
    # Half extent of the text box along d. For an axis-aligned box, this is
    # the L^inf distance from center to edge in direction d.
    half_extent = abs(dx) * (text_w / 2.0) + abs(dy) * (text_h / 2.0)
    cx = anchor[0] + dx * (gap + half_extent)
    cy = anchor[1] + dy * (gap + half_extent)
    return (cx - text_w / 2.0, cy - text_h / 2.0)


# ---------------------------------------------------------------------------
# Dimension line placement
# ---------------------------------------------------------------------------

def place_linear_dim(index: SpatialIndex,
                     p1: Point,
                     p2: Point,
                     side: str = "auto",
                     base_offset: float = 10.0,
                     text_height: float = 2.5,
                     max_rings: int = 6,
                     exclude_ids: Optional[List[str]] = None) -> Tuple[Point, float]:
    """Compute a base point for an ezdxf linear dimension that clears all
    upstream geometry. Returns (base_point, angle_deg).

    The dimension line passes through `base_point` perpendicular to (p2-p1).
    """
    direction = sub(p2, p1)
    n = math.hypot(*direction)
    if n == 0:
        return p1, 0.0
    u = (direction[0] / n, direction[1] / n)        # along the feature
    nrm = perp_left(u)                               # 90 deg CCW
    angle = math.degrees(math.atan2(u[1], u[0]))

    # Pick the side. "auto" picks the side that's least obstructed.
    midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    sides = []
    if side in ("auto", "left", "above"):
        sides.append(("+", nrm))
    if side in ("auto", "right", "below"):
        sides.append(("-", (-nrm[0], -nrm[1])))
    if not sides:
        sides = [("+", nrm)]

    # Try increasing offsets along each side; pick the first clear one.
    dim_text_w = 6 * text_height * CHAR_W_RATIO     # assume up to 6 chars
    text_band = AABB(-dim_text_w/2, -text_height/2,
                     dim_text_w/2, text_height*2.0)  # text + dim line band

    best_offset: Optional[Tuple[Point, float]] = None
    for ring in range(max_rings):
        offset = base_offset * (1.0 + 0.6 * ring)
        for tag, d in sides:
            candidate_mid = (midpoint[0] + d[0] * offset,
                             midpoint[1] + d[1] * offset)
            # Build a swept AABB for the dim line + text along the feature.
            # The dim line is parallel to (p2-p1), length matches the feature
            # plus a margin for extensions.
            margin = 2 * text_height
            half_len = n / 2 + margin
            # AABB of the dim band in world coords:
            corners: List[Point] = []
            for s in (-half_len, +half_len):
                for h in (-text_height * 0.6, +text_height * 2.0):
                    px = candidate_mid[0] + s * u[0] + h * d[0]
                    py = candidate_mid[1] + s * u[1] + h * d[1]
                    corners.append((px, py))
            cand_aabb = AABB.from_points(corners)
            if index.is_clear(cand_aabb,
                              clearance=text_height * 0.5,
                              exclude_ids=exclude_ids):
                return candidate_mid, angle
            if best_offset is None:
                best_offset = (candidate_mid, angle)

    # Fall back to the farthest tried.
    return best_offset if best_offset is not None else (
        (midpoint[0] + nrm[0] * base_offset,
         midpoint[1] + nrm[1] * base_offset), angle)
