"""
Geometric primitives, reference resolution, and AABB math.

Every annotation references geometry by an *entity id* + a snap kind
(START / END / MID / CENTER / QUADRANT / VERTEX:n), not by raw coordinates.
This keeps text and dimensions locked to the features even after edits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# AABB
# ---------------------------------------------------------------------------

@dataclass
class AABB:
    """Axis-aligned bounding box. Open intervals on either side mean empty."""
    xmin: float = math.inf
    ymin: float = math.inf
    xmax: float = -math.inf
    ymax: float = -math.inf

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def is_empty(self) -> bool:
        return self.xmax < self.xmin or self.ymax < self.ymin

    @property
    def center(self) -> Point:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def expanded(self, pad: float) -> "AABB":
        return AABB(self.xmin - pad, self.ymin - pad,
                    self.xmax + pad, self.ymax + pad)

    def contains_point(self, p: Point) -> bool:
        return self.xmin <= p[0] <= self.xmax and self.ymin <= p[1] <= self.ymax

    def intersects(self, other: "AABB", pad: float = 0.0) -> bool:
        if self.is_empty or other.is_empty:
            return False
        return not (self.xmax + pad < other.xmin
                    or other.xmax + pad < self.xmin
                    or self.ymax + pad < other.ymin
                    or other.ymax + pad < self.ymin)

    @classmethod
    def from_points(cls, points: List[Point]) -> "AABB":
        if not points:
            return cls()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return cls(min(xs), min(ys), max(xs), max(ys))

    @classmethod
    def from_circle(cls, center: Point, radius: float) -> "AABB":
        cx, cy = center
        return cls(cx - radius, cy - radius, cx + radius, cy + radius)

    @classmethod
    def from_arc(cls, center: Point, radius: float,
                 start_deg: float, end_deg: float) -> "AABB":
        """Tight AABB for a CCW arc from start_deg to end_deg."""
        cx, cy = center
        pts: List[Point] = []
        # Endpoints
        for a in (start_deg, end_deg):
            r = math.radians(a)
            pts.append((cx + radius * math.cos(r), cy + radius * math.sin(r)))
        # Quadrant points that lie inside the sweep
        s = start_deg % 360.0
        e = end_deg % 360.0
        # Normalize to a CCW sweep from 0..delta
        delta = (end_deg - start_deg) % 360.0
        for q in (0.0, 90.0, 180.0, 270.0):
            offset = (q - start_deg) % 360.0
            if offset <= delta + 1e-9:
                r = math.radians(q)
                pts.append((cx + radius * math.cos(r), cy + radius * math.sin(r)))
        return cls.from_points(pts)


# ---------------------------------------------------------------------------
# Geometric reference (entity_id + snap kind)
# ---------------------------------------------------------------------------

class Snap(str, Enum):
    START    = "start"
    END      = "end"
    MID      = "mid"
    CENTER   = "center"
    Q_E      = "quadrant_east"
    Q_N      = "quadrant_north"
    Q_W      = "quadrant_west"
    Q_S      = "quadrant_south"
    VERTEX   = "vertex"     # use index field
    NEAREST  = "nearest"    # use parameter t ∈ [0,1] along path


@dataclass(frozen=True)
class GeomRef:
    """Reference to a snap point on a registered entity."""
    entity_id: str
    snap:      Snap = Snap.MID
    index:     int = 0       # for VERTEX
    t:         float = 0.5   # for NEAREST (path parameter)


# ---------------------------------------------------------------------------
# Entity-level geometric records (used by spatial index + reference solver)
# ---------------------------------------------------------------------------

@dataclass
class GeomEntity:
    """Lightweight record of a placed primitive — feeds the spatial index."""
    entity_id: str
    kind:      str                        # "line", "polyline", "circle", "arc", "text", "dimension", "hatch", "block"
    points:    List[Point] = field(default_factory=list)
    center:    Optional[Point] = None
    radius:    float = 0.0
    start_angle: float = 0.0              # degrees, CCW from +x
    end_angle:   float = 0.0
    closed:    bool = False
    aabb:      AABB = field(default_factory=AABB)
    # Whether this entity should be considered an obstacle for annotation
    # placement. Annotations themselves set this False.
    obstacle:  bool = True
    # For annotations: ids of the geometry entities this annotation is
    # anchored to. The validator excludes these from collision checks so
    # that a dim against a circle doesn't get flagged for touching that
    # circle (which it must, by definition).
    host_ids:  List[str] = field(default_factory=list)


def aabb_of(entity: GeomEntity) -> AABB:
    if entity.kind == "circle":
        return AABB.from_circle(entity.center, entity.radius)
    if entity.kind == "arc":
        return AABB.from_arc(entity.center, entity.radius,
                             entity.start_angle, entity.end_angle)
    return AABB.from_points(entity.points)


def resolve_point(entity: GeomEntity, ref: GeomRef) -> Point:
    """Resolve a GeomRef against a stored GeomEntity. Raises on mismatch."""
    s = ref.snap
    if entity.kind == "line":
        a, b = entity.points[0], entity.points[1]
        if s == Snap.START:  return a
        if s == Snap.END:    return b
        if s == Snap.MID:    return ((a[0]+b[0])/2, (a[1]+b[1])/2)
        if s == Snap.NEAREST:
            t = max(0.0, min(1.0, ref.t))
            return (a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t)
        raise ValueError(f"snap {s} not valid for line")

    if entity.kind == "polyline":
        pts = entity.points
        if s == Snap.START:  return pts[0]
        if s == Snap.END:    return pts[-1]
        if s == Snap.VERTEX:
            i = ref.index
            if not (0 <= i < len(pts)):
                raise IndexError(f"vertex {i} out of range for polyline {ref.entity_id}")
            return pts[i]
        if s == Snap.MID:
            # midpoint by cumulative arclength
            return _polyline_at_param(pts, 0.5, closed=entity.closed)
        if s == Snap.NEAREST:
            return _polyline_at_param(pts, max(0.0, min(1.0, ref.t)), closed=entity.closed)
        if s == Snap.CENTER:
            return AABB.from_points(pts).center
        raise ValueError(f"snap {s} not valid for polyline")

    if entity.kind in ("circle", "arc"):
        cx, cy = entity.center
        r = entity.radius
        if s == Snap.CENTER: return (cx, cy)
        if s == Snap.Q_E:    return (cx + r, cy)
        if s == Snap.Q_N:    return (cx, cy + r)
        if s == Snap.Q_W:    return (cx - r, cy)
        if s == Snap.Q_S:    return (cx, cy - r)
        if entity.kind == "arc":
            if s == Snap.START:
                a = math.radians(entity.start_angle)
                return (cx + r*math.cos(a), cy + r*math.sin(a))
            if s == Snap.END:
                a = math.radians(entity.end_angle)
                return (cx + r*math.cos(a), cy + r*math.sin(a))
            if s == Snap.MID:
                a = math.radians((entity.start_angle + entity.end_angle) / 2)
                return (cx + r*math.cos(a), cy + r*math.sin(a))
        raise ValueError(f"snap {s} not valid for {entity.kind}")

    raise ValueError(f"resolve_point: unsupported kind {entity.kind}")


def _polyline_at_param(pts: List[Point], t: float, closed: bool) -> Point:
    """Point at parameter t∈[0,1] along a polyline by arclength."""
    seq = list(pts) + ([pts[0]] if closed else [])
    if len(seq) < 2:
        return seq[0]
    segs = []
    total = 0.0
    for i in range(len(seq) - 1):
        d = math.dist(seq[i], seq[i+1])
        segs.append(d)
        total += d
    if total == 0.0:
        return seq[0]
    target = t * total
    acc = 0.0
    for i, d in enumerate(segs):
        if acc + d >= target:
            local = 0.0 if d == 0 else (target - acc) / d
            a, b = seq[i], seq[i+1]
            return (a[0]+(b[0]-a[0])*local, a[1]+(b[1]-a[1])*local)
        acc += d
    return seq[-1]


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def sub(a: Point, b: Point) -> Point:
    return (a[0]-b[0], a[1]-b[1])

def add(a: Point, b: Point) -> Point:
    return (a[0]+b[0], a[1]+b[1])

def scale(a: Point, k: float) -> Point:
    return (a[0]*k, a[1]*k)

def norm(a: Point) -> float:
    return math.hypot(a[0], a[1])

def unit(a: Point) -> Point:
    n = norm(a)
    return (0.0, 0.0) if n == 0 else (a[0]/n, a[1]/n)

def perp_left(a: Point) -> Point:
    """Rotate 90 deg CCW."""
    return (-a[1], a[0])

def angle_deg(a: Point, b: Point) -> float:
    return math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]))
