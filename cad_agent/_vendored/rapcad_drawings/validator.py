"""
Post-build validator. Pydantic catches schema issues; this catches
*drawing* issues that only become apparent after geometry is placed:

  - degenerate geometry (zero-length lines, near-coincident polyline vertices)
  - text or dimensions that significantly overlap unrelated geometry's
    ink footprint (per-segment for polylines, full-AABB for everything else)
  - annotations whose AABBs significantly overlap one another

The placement engine already leaves a halo of air around every label,
so an AABB that grazes another by <halo is acceptable. Only overlaps
deeper than `halo` are reported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Set, Tuple

from .geometry import AABB
from .spatial_index import SpatialIndex, _ink_aabbs


@dataclass
class Finding:
    severity: str
    entity_id: str
    message:   str


def _significant_overlap(a: AABB, b: AABB, halo: float) -> bool:
    if a.is_empty or b.is_empty:
        return False
    ox = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    oy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    return ox > halo and oy > halo


def _significant_vs_ink(ann_aabb: AABB, other, halo: float) -> bool:
    """True if ann_aabb significantly overlaps any ink-AABB of `other`."""
    for ink in _ink_aabbs(other):
        if _significant_overlap(ann_aabb, ink, halo):
            return True
    return False


def validate(index: SpatialIndex, halo: float) -> List[Finding]:
    out: List[Finding] = []
    seen: Set[Tuple[str, str]] = set()

    # 1. Degenerate geometry
    for e in index.all():
        if e.kind == "line" and len(e.points) >= 2:
            if math.dist(e.points[0], e.points[1]) < 1e-9:
                out.append(Finding("error", e.entity_id, "zero-length line"))
        elif e.kind == "polyline" and len(e.points) >= 2:
            dups = sum(1 for a, b in zip(e.points, e.points[1:])
                       if math.dist(a, b) < 1e-9)
            if dups > 0:
                out.append(Finding("warning", e.entity_id,
                                   f"{dups} coincident polyline vertices"))

    # 2. Annotation overlaps
    annotations = [e for e in index.all() if e.kind in ("text", "dimension")]
    geometry    = [e for e in index.all() if e.obstacle and
                                              e.kind not in ("text", "dimension")]

    for ann in annotations:
        host_set = set(ann.host_ids)
        for other in geometry:
            if other.entity_id in host_set:
                continue
            key = (ann.entity_id, other.entity_id)
            if key in seen:
                continue
            if _significant_vs_ink(ann.aabb, other, halo):
                seen.add(key)
                out.append(Finding("warning", ann.entity_id,
                    f"annotation overlaps geometry {other.entity_id}"))
        for other in annotations:
            if ann.entity_id >= other.entity_id:
                continue
            key = (ann.entity_id, other.entity_id)
            if key in seen:
                continue
            if _significant_overlap(ann.aabb, other.aabb, halo):
                seen.add(key)
                out.append(Finding("warning", ann.entity_id,
                    f"annotation overlaps annotation {other.entity_id}"))

    return out
