"""
Solid-aware scanner.

For each solid in the source STEP, compute a shape signature and cluster
solids into congruence groups. Within each group, detect whether the
members form an arithmetic grid along some axis.

This replaces voxel-based structure detection entirely. It works because
STEP files produced by CAD tools and component libraries almost always
already encode design structure in their solid decomposition -- a 7-pin
connector has 7 identical pin solids in the STEP, not one fused blob.

The only cases where this fails are:
  - Single-solid STEPs where the whole part is one fused TopoDS_Solid.
    For these the voxel fallback is appropriate.
  - STEPs where repeated features are genuinely merged. Rare in
    manufacturing CAD.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SolidInfo:
    """Per-solid metadata extracted from the source."""
    index: int
    signature: tuple              # (sx, sy, sz, vol) rounded
    center: tuple                 # (cx, cy, cz) bbox center
    volume: float
    bbox_min: tuple
    bbox_max: tuple


@dataclass
class GridPattern:
    """A detected 1D arithmetic grid within a congruence group."""
    axis: int                     # 0=X, 1=Y, 2=Z
    pitch: float                  # spacing between centers
    count: int
    start_index: int              # index of the "canonical" member
    start_center: tuple           # center of the first member in world coords
    member_indices: list[int]     # all indices in grid order (low to high)

    @property
    def axis_name(self) -> str:
        return "XYZ"[self.axis]


@dataclass
class CongruenceGroup:
    """A set of solids with matching shape signatures."""
    signature: tuple
    member_indices: list[int]
    grid: Optional[GridPattern] = None


@dataclass
class ScanResult:
    """Full scan output."""
    solids_info: list[SolidInfo]
    groups: list[CongruenceGroup]
    total_volume: float
    source_path: str

    def pretty(self) -> str:
        lines = [f"Scan of {self.source_path}"]
        lines.append(f"  {len(self.solids_info)} solids, "
                     f"total volume {self.total_volume:.4f} mm^3")
        lines.append(f"  {len(self.groups)} congruence groups:")
        for g in self.groups:
            status = ""
            if g.grid is not None:
                status = (f" [grid: axis={g.grid.axis_name}, "
                          f"pitch={g.grid.pitch:.4f}, count={g.grid.count}]")
            lines.append(f"    sig={g.signature} x{len(g.member_indices)}"
                         f"{status}")
        return "\n".join(lines)


def solid_signature(solid, precision: int = 6) -> tuple:
    """Shape signature: bbox dimensions + volume, rounded to `precision`
    decimals. Two solids with identical signatures have the same bounding
    box and volume -- strong evidence they're geometrically congruent up
    to rigid transformation. False positives possible but rare in CAD.
    """
    bb = solid.bounding_box()
    sx = round(bb.max.X - bb.min.X, precision)
    sy = round(bb.max.Y - bb.min.Y, precision)
    sz = round(bb.max.Z - bb.min.Z, precision)
    vol = round(solid.volume, precision)
    return (sx, sy, sz, vol)


def _extract_solid_info(solids) -> list[SolidInfo]:
    infos = []
    for i, s in enumerate(solids):
        bb = s.bounding_box()
        sig = solid_signature(s)
        center = ((bb.min.X + bb.max.X) / 2.0,
                  (bb.min.Y + bb.max.Y) / 2.0,
                  (bb.min.Z + bb.max.Z) / 2.0)
        infos.append(SolidInfo(
            index=i,
            signature=sig,
            center=center,
            volume=s.volume,
            bbox_min=(bb.min.X, bb.min.Y, bb.min.Z),
            bbox_max=(bb.max.X, bb.max.Y, bb.max.Z),
        ))
    return infos


def _detect_1d_grid_in_group(infos: list[SolidInfo],
                              group_indices: list[int],
                              tol: float = 1e-3) -> Optional[GridPattern]:
    """Check if the solids in `group_indices` form a 1D arithmetic grid
    along some axis. Returns the grid or None.

    NOTE: this only confirms bbox center alignment. It does NOT verify
    that the solids are true translations of each other. Two solids
    with matching bbox + volume but different internal shape (e.g.
    mirror images, rotated copies) will pass this check but fail
    reconstruction. The caller should use `verify_grid_translation`
    with the actual Solid objects to confirm.
    """
    if len(group_indices) < 2:
        return None

    group_infos = [infos[i] for i in group_indices]

    for axis in range(3):
        sorted_members = sorted(group_infos, key=lambda m: m.center[axis])
        coords = [m.center[axis] for m in sorted_members]
        diffs = [coords[i+1] - coords[i] for i in range(len(coords)-1)]
        if not diffs:
            continue
        d0 = diffs[0]
        if abs(d0) < tol:
            continue
        if not all(abs(d - d0) < tol for d in diffs):
            continue
        other_axes = [i for i in range(3) if i != axis]
        others_const = True
        for oax in other_axes:
            vals = [m.center[oax] for m in sorted_members]
            if max(vals) - min(vals) > tol:
                others_const = False
                break
        if not others_const:
            continue
        first = sorted_members[0]
        return GridPattern(
            axis=axis,
            pitch=d0,
            count=len(sorted_members),
            start_index=first.index,
            start_center=first.center,
            member_indices=[m.index for m in sorted_members],
        )
    return None


def verify_grid_translation(grid: GridPattern, solids,
                             min_match_fraction: float = 0.99) -> bool:
    """Verify that the solids in `grid` are genuine translations of each
    other by translating the canonical to each other position and
    checking boolean intersection covers the target solid.

    Returns True if every non-canonical member intersects the translated
    canonical by at least `min_match_fraction` of its own volume.

    IMPORTANT: uses .moved() rather than .located(). .located() ADDS
    the new location on top of the existing one; .moved() REPLACES
    the location absolutely. For raw STEP-loaded solids whose existing
    location is identity these are equivalent, but we use .moved() for
    clarity and for the two-step "to origin, then to target" workflow
    where composition would double-translate.
    """
    from build123d import Location, Vector
    canonical = solids[grid.start_index]
    canonical_center = grid.start_center
    # Move canonical to origin. Raw STEP solids have identity location
    # and geometry in world coords, so .moved() to the negative center
    # here is effectively a translation.
    canonical_at_origin = canonical.moved(
        Location(Vector(-canonical_center[0],
                        -canonical_center[1],
                        -canonical_center[2])))

    for k, member_idx in enumerate(grid.member_indices):
        if member_idx == grid.start_index:
            continue
        target = solids[member_idx]
        expected_center = list(canonical_center)
        expected_center[grid.axis] += k * grid.pitch
        # Translate the at-origin canonical to expected_center. .moved()
        # replaces the location, so this gives absolute position.
        translated = canonical_at_origin.moved(
            Location(Vector(*expected_center)))
        inter = _intersect_volume(translated, target)
        target_vol = target.volume
        if target_vol <= 0:
            continue
        if inter / target_vol < min_match_fraction:
            return False
    return True


def _intersect_volume(a, b) -> float:
    """Minimal piecewise intersection volume (verifier module not
    available at scanner layer)."""
    try:
        inter = a & b
    except Exception:
        return 0.0
    if inter is None:
        return 0.0
    v = getattr(inter, "volume", None)
    if isinstance(v, (int, float)):
        return float(v)
    if hasattr(inter, "__iter__"):
        try:
            return float(sum(getattr(s, "volume", 0.0) for s in inter))
        except Exception:
            return 0.0
    return 0.0


def _load_solids_from_fcstd(source_path: str):
    """Read all solid shapes out of a FreeCAD .FCStd file.

    FCStd is a zip that contains `Document.xml` and one or more
    `PartShape*.brp` files — OpenCascade's native BREP format. We
    extract the .brp files, load each via OCCT's BRepTools, wrap as
    the appropriate build123d type (Solid or Compound), and return
    the flat list of contained Solids.
    """
    import zipfile
    import tempfile
    import os as _os
    from OCP.BRepTools import BRepTools
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_Compound
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from build123d import Solid, Compound

    solids = []
    with zipfile.ZipFile(source_path, "r") as zf:
        names = [n for n in zf.namelist()
                 if n.lower().endswith((".brp", ".brep"))]
        # Sort so output order is reproducible: PartShape.brp,
        # PartShape1.brp, ..., PartShape10.brp, ...
        def _sort_key(name):
            stem = name.rsplit(".", 1)[0]
            digits = ""
            for c in reversed(stem):
                if c.isdigit(): digits = c + digits
                else: break
            return (int(digits) if digits else -1, name)
        names.sort(key=_sort_key)
        if not names:
            raise ValueError(f"No .brp shapes inside {source_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            for name in names:
                zf.extract(name, tmpdir)
                path = _os.path.join(tmpdir, name)
                topo_shape = TopoDS_Shape()
                builder = BRep_Builder()
                try:
                    ok = BRepTools.Read_s(topo_shape, path, builder)
                except Exception:
                    continue
                if not ok or topo_shape.IsNull():
                    continue
                # Walk the topology and extract every SOLID.
                # Works for all cases: single solid, compound of solids,
                # compsolid, or nested compounds.
                exp = TopExp_Explorer(topo_shape, TopAbs_SOLID)
                while exp.More():
                    topo_solid = exp.Current()
                    # Downcast to TopoDS_Solid explicitly so the
                    # build123d wrapper picks the Solid class.
                    try:
                        ts = TopoDS_Solid()
                        # TopExp_Explorer returns TopoDS_Shape; we
                        # can construct a TopoDS_Solid from it via
                        # the .TShape()/.Location()/.Orientation()
                        # triple. Easier: directly pass to Solid().
                        s = Solid(topo_solid.TShape() if False else topo_solid)
                    except Exception:
                        # Fallback: wrap via the Shape→Solid promotion
                        try:
                            s = Solid(exp.Current())
                        except Exception:
                            exp.Next()
                            continue
                    try:
                        v = s.volume
                    except Exception:
                        v = 0.0
                    if v > 1e-9:
                        solids.append(s)
                    exp.Next()
    if not solids:
        raise ValueError(f"No solid geometry found in {source_path}")
    return solids


def _load_solids(source_path: str):
    """Dispatch by extension. Returns list of build123d solids."""
    lower = source_path.lower()
    if lower.endswith((".fcstd", ".fcstd1")):
        return _load_solids_from_fcstd(source_path)
    # STEP path (default for .step / .stp / .STEP / .STP)
    from build123d import import_step
    obj = import_step(source_path)
    return list(obj.solids())


def scan_source(source_path: str) -> ScanResult:
    """Load a STEP or FCStd and produce a full scan result."""
    solids = _load_solids(source_path)
    if not solids:
        raise ValueError(f"No solids in {source_path}")

    infos = _extract_solid_info(solids)
    total_vol = sum(s.volume for s in solids)

    # Cluster by signature
    sig_to_indices = defaultdict(list)
    for info in infos:
        sig_to_indices[info.signature].append(info.index)

    groups = []
    for sig, indices in sig_to_indices.items():
        g = CongruenceGroup(signature=sig, member_indices=list(indices))
        if len(indices) >= 2:
            candidate_grid = _detect_1d_grid_in_group(infos, indices)
            if candidate_grid is not None:
                # Verify the solids are genuine translations, not mirrors
                # or rotations. Mirror images have matching bbox + volume
                # but the tiled reconstruction will miss material.
                if verify_grid_translation(candidate_grid, solids):
                    g.grid = candidate_grid
                # else: grid rejected; members emit individually
        groups.append(g)

    # Sort groups so big groups come first (more interesting)
    groups.sort(key=lambda g: -len(g.member_indices))

    return ScanResult(
        solids_info=infos,
        groups=groups,
        total_volume=total_vol,
        source_path=source_path,
    )
