"""
Spatial index for AABB-based collision queries.

For most entities, an entity's "ink footprint" matches its bounding box.
Polylines (especially closed ones) are an exception: their AABB covers
the *enclosed* area, but the actual ink only exists along the edges. To
avoid spurious collisions between annotations placed inside a closed
polyline and the polyline itself, the index stores a list of
fine-grained "ink AABBs" per entity — one per segment for polylines,
one per arc for arcs, and a single AABB for everything else.

A candidate collides with an entity only if it intersects at least one
of the entity's ink AABBs (with optional clearance).

Implementation is a flat list + linear scan. For genuine large-N use
(10k+ entities), swap in shapely.strtree.STRtree.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .geometry import AABB, GeomEntity


def _ink_aabbs(e: GeomEntity) -> List[AABB]:
    """Decompose an entity into one or more ink-footprint AABBs.

    For closed/curved shapes (polyline, circle, arc), the geometric AABB
    covers the *enclosed* area but the actual ink is only on the boundary.
    Decomposing into short segment AABBs eliminates false collisions for
    anything that sits inside the boundary but doesn't cross the ink.
    """
    if e.kind == "polyline":
        pts = list(e.points)
        if e.closed and pts:
            pts.append(pts[0])
        if len(pts) < 2:
            return [e.aabb]
        return [AABB.from_points([pts[i], pts[i + 1]])
                for i in range(len(pts) - 1)]

    if e.kind == "circle":
        cx, cy = e.center
        r = e.radius
        N = 16
        out: List[AABB] = []
        prev = (cx + r, cy)
        for i in range(1, N + 1):
            ang = (2 * 3.141592653589793 * i) / N
            cur = (cx + r * _math_cos(ang), cy + r * _math_sin(ang))
            out.append(AABB.from_points([prev, cur]))
            prev = cur
        return out

    if e.kind == "arc":
        cx, cy = e.center
        r = e.radius
        sa, ea = e.start_angle, e.end_angle
        sweep = (ea - sa) % 360.0
        if sweep == 0:
            sweep = 360.0
        N = max(4, int(sweep / 22.5))
        out2: List[AABB] = []
        prev = None
        for i in range(N + 1):
            a = (sa + sweep * i / N) * (3.141592653589793 / 180.0)
            cur = (cx + r * _math_cos(a), cy + r * _math_sin(a))
            if prev is not None:
                out2.append(AABB.from_points([prev, cur]))
            prev = cur
        return out2 or [e.aabb]

    return [e.aabb]


# Local imports to avoid top-level math overhead in tight scans
from math import cos as _math_cos, sin as _math_sin


class SpatialIndex:
    def __init__(self):
        self._by_id: Dict[str, GeomEntity] = {}
        # Cache ink-AABBs per entity. Recomputed when entities are added.
        self._ink_cache: Dict[str, List[AABB]] = {}

    # -- registration --

    def add(self, entity: GeomEntity) -> None:
        if entity.entity_id in self._by_id:
            raise KeyError(f"duplicate entity_id: {entity.entity_id}")
        self._by_id[entity.entity_id] = entity
        self._ink_cache[entity.entity_id] = _ink_aabbs(entity)

    def get(self, entity_id: str) -> GeomEntity:
        try:
            return self._by_id[entity_id]
        except KeyError:
            raise KeyError(f"unknown entity_id: {entity_id}") from None

    def has(self, entity_id: str) -> bool:
        return entity_id in self._by_id

    def all(self) -> Iterable[GeomEntity]:
        return self._by_id.values()

    def __len__(self) -> int:
        return len(self._by_id)

    # -- queries --

    def overall_bounds(self) -> AABB:
        """Union AABB of every obstacle entity."""
        out = AABB()
        for e in self._by_id.values():
            if not e.obstacle:
                continue
            b = e.aabb
            if b.is_empty:
                continue
            out.xmin = min(out.xmin, b.xmin)
            out.ymin = min(out.ymin, b.ymin)
            out.xmax = max(out.xmax, b.xmax)
            out.ymax = max(out.ymax, b.ymax)
        return out

    def collisions(self, candidate: AABB,
                   clearance: float = 0.0,
                   exclude_ids: Optional[Iterable[str]] = None,
                   obstacles_only: bool = True) -> List[GeomEntity]:
        """Return entities whose INK overlaps `candidate` (with clearance)."""
        skip = set(exclude_ids or ())
        hits: List[GeomEntity] = []
        for eid, e in self._by_id.items():
            if eid in skip:
                continue
            if obstacles_only and not e.obstacle:
                continue
            # Coarse reject on full AABB first
            if e.aabb.is_empty or not candidate.intersects(e.aabb,
                                                           pad=clearance):
                continue
            # Fine check against ink AABBs
            for ink in self._ink_cache.get(eid, [e.aabb]):
                if candidate.intersects(ink, pad=clearance):
                    hits.append(e)
                    break
        return hits

    def is_clear(self, candidate: AABB,
                 clearance: float = 0.0,
                 exclude_ids: Optional[Iterable[str]] = None) -> bool:
        return not self.collisions(candidate, clearance, exclude_ids)
