"""
Primitive fitter: reverse-engineer a build123d primitive that reproduces
a given Solid within tolerance.

Input: a build123d Solid (or anything iterable to a single solid).
Output: a FitResult with either
  - a snippet of build123d code that reconstructs the shape, and a
    computed `completeness` fraction against the source, OR
  - None code (no fit found).

Fit strategies, in order of simplicity:
  1. Box: all 6 faces planar, axis-aligned, and reconstructing a
     Box(sx, sy, sz) at the solid's bbox center matches within tolerance.
  2. Cylinder: exactly 3 faces: 1 CYLINDER side + 2 PLANE caps.
     Axis and radius extracted from the CYLINDER face's geometry.
  3. Prismatic extrude: find two parallel congruent planar faces normal
     to some axis. The shape between them is an extrusion of the
     common profile. Extract 2D profile from one face, emit extrude().

Everything else returns no fit -> caller falls back to import_step.

Design note: fitting is ALWAYS verified by actually building the
proposed primitive in build123d and computing intersection with the
input solid. We never trust geometric guesses without verification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class FitResult:
    """The output of a fit attempt."""
    code_body: Optional[str]       # build123d code that produces `_part`
    completeness: float            # inter / source volume
    accuracy: float                # inter / rebuilt volume
    kind: str                      # "box" | "cylinder" | "extrude" | "none"
    details: str = ""              # human-readable description


# Module-level flag. When True, fitters accept their best-effort output
# regardless of verification result, and return the generated code with
# whatever accuracy it achieved. This guarantees the emitted recipe is
# always pure build123d primitives (no import_step) at the cost of
# potentially imprecise reconstruction. Toggle via set_force_primitives().
_FORCE_PRIMITIVES = False
_VERBOSE = False


def set_force_primitives(enabled: bool) -> None:
    """Enable/disable force-primitives mode.

    When enabled, any fitter that successfully builds a shape returns
    its code regardless of how close the shape is to the source. This
    means NO import_step fallbacks are produced -- the emitted recipe
    is always pure build123d, though individual primitives may be
    approximate for complex shapes.

    Use this when you need a fully-from-scratch recipe and can live
    with sub-perfect accuracy on some solids.
    """
    global _FORCE_PRIMITIVES
    _FORCE_PRIMITIVES = enabled


def set_verbose(enabled: bool) -> None:
    """Enable/disable per-tier progress printing inside fit_primitive.

    When enabled, each tier (box, cylinder, extrude, hull, axis-stack,
    loft, face-extrude, llm) logs a line as it starts and finishes.
    This matters on complex parts where the tier chain can take tens
    of seconds per solid; without this flag you see no output between
    "scan done" and the final summary.
    """
    global _VERBOSE
    _VERBOSE = enabled


# ----------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------

def _verify_fit(source_solid, rebuilt_part, tol: float,
                abs_sym_diff_floor: float = 1e-4) -> FitResult:
    """Compute intersection and sym-diff, return completeness+accuracy.

    `abs_sym_diff_floor`: absolute volume of sym-diff (mm^3) below which
    a fit is considered perfect regardless of percentage. This matters
    for small solids where OCCT boolean ops have numerical noise
    comparable to the volume of the solid itself. Default 0.0001 mm^3.

    For a 0.007 mm^3 solid, 1% tolerance = 0.00007 mm^3, which is below
    the noise floor of OCCT intersection at that scale. Without this
    floor, correct reconstructions of small solids are rejected.
    """
    from .verifier import compute_intersection, safe_volume
    src_vol = safe_volume(source_solid)
    rec_vol = safe_volume(rebuilt_part)
    if src_vol < 1e-9 or rec_vol < 1e-9:
        return FitResult(None, 0.0, 0.0, "none", "degenerate volume")
    inter = compute_intersection(source_solid, rebuilt_part)
    sym_diff = max(0.0, src_vol + rec_vol - 2 * inter)
    comp = inter / src_vol
    acc = inter / rec_vol
    # If absolute sym-diff is below the noise floor, treat as perfect
    # match. This overrides percentage-based rejection for tiny solids.
    if sym_diff < abs_sym_diff_floor:
        comp = 1.0
        acc = 1.0
    return FitResult(None, comp, acc, "none", "")


# ----------------------------------------------------------------------
# Box fitter
# ----------------------------------------------------------------------

def try_fit_box(solid, tol: float = 0.01) -> FitResult:
    """Fit a Box to the solid.

    Check: exactly 6 planar faces, axis-aligned, whose union bounds
    the solid's bbox. Build Box(sx, sy, sz) at the bbox center and
    verify intersection / source_vol >= 1 - tol.
    """
    faces = list(solid.faces())
    if len(faces) != 6:
        return FitResult(None, 0.0, 0.0, "none",
                         f"Box needs 6 faces, got {len(faces)}")

    # All faces must be planar
    try:
        for f in faces:
            if f.geom_type != f.geom_type.PLANE:
                return FitResult(None, 0.0, 0.0, "none",
                                 "non-planar face")
    except Exception:
        return FitResult(None, 0.0, 0.0, "none", "face type check failed")

    # Axis-aligned: each face's normal should be aligned with one axis.
    # build123d Face has .normal_at(). If the normal has only one non-zero
    # component to tolerance, it's axis-aligned.
    for f in faces:
        try:
            n = f.normal_at(f.center())
        except Exception:
            return FitResult(None, 0.0, 0.0, "none", "normal check failed")
        nx, ny, nz = abs(n.X), abs(n.Y), abs(n.Z)
        ordered = sorted([nx, ny, nz], reverse=True)
        # Largest component should be ~1, others ~0
        if ordered[0] < 0.99 or ordered[1] > 0.01:
            return FitResult(None, 0.0, 0.0, "none",
                             "non-axis-aligned normal")

    # Build Box at bbox center with bbox dimensions
    bb = solid.bounding_box()
    sx = bb.max.X - bb.min.X
    sy = bb.max.Y - bb.min.Y
    sz = bb.max.Z - bb.min.Z
    cx = (bb.min.X + bb.max.X) / 2.0
    cy = (bb.min.Y + bb.max.Y) / 2.0
    cz = (bb.min.Z + bb.max.Z) / 2.0

    from build123d import BuildPart, Box, Locations
    with BuildPart() as p:
        with Locations((cx, cy, cz)):
            Box(sx, sy, sz)
    rebuilt = p.part

    v = _verify_fit(solid, rebuilt, tol)
    if v.completeness < 1 - tol or v.accuracy < 1 - tol:
        return FitResult(None, v.completeness, v.accuracy, "none",
                         f"Box fit rejected: "
                         f"comp={v.completeness*100:.2f}% "
                         f"acc={v.accuracy*100:.2f}%")

    code = (
        f"with BuildPart() as _part:\n"
        f"    with Locations(({_fmt(cx)}, {_fmt(cy)}, {_fmt(cz)})):\n"
        f"        Box({_fmt(sx)}, {_fmt(sy)}, {_fmt(sz)})\n"
    )
    return FitResult(code, v.completeness, v.accuracy, "box",
                     f"Box({sx:.3f}x{sy:.3f}x{sz:.3f}) at "
                     f"({cx:.3f},{cy:.3f},{cz:.3f})")


# ----------------------------------------------------------------------
# Box-with-fillets fitter
# ----------------------------------------------------------------------

def try_fit_box_with_fillets(solid, tol: float = 0.01) -> FitResult:
    """Fit Box(sx, sy, sz) + fillet(edges, radius=R) to the solid.

    A rounded-corner rectangular solid typically has:
      - 6 axis-aligned PLANE faces (the main bbox faces, possibly
        clipped at corners by fillet arcs)
      - Some number of CYLINDER faces (one per filleted edge)
      - Optionally SPHERE faces at 3-way corner intersections

    Strategy: if the solid's faces are all PLANE/CYLINDER/SPHERE, and
    there's a cylinder radius that's ~constant across all non-plane
    faces, try emitting a Box of the bbox dims with fillet(radius=R).
    Verify the built shape's boolean intersection with the source.
    """
    faces = list(solid.faces())
    if len(faces) < 7:
        return FitResult(None, 0.0, 0.0, "none",
                         f"need >=7 faces, got {len(faces)}")

    # Collect face types
    n_plane = 0
    n_cyl = 0
    n_sph = 0
    cyl_radii = []
    for f in faces:
        try:
            gt = f.geom_type
        except Exception:
            return FitResult(None, 0.0, 0.0, "none",
                             "face type check failed")
        if gt == gt.PLANE:
            n_plane += 1
        elif gt == gt.CYLINDER:
            n_cyl += 1
            # Radius from bbox approximation: the narrowest bbox dim of
            # the cylindrical face is 2*r (the arc sweeps across width).
            # Actually best to get radius from the face geometry.
            try:
                # Use the outer wire's arc radius if accessible.
                # Simpler: the face's bbox.
                bb = f.bounding_box()
                dims = sorted([bb.max.X - bb.min.X,
                               bb.max.Y - bb.min.Y,
                               bb.max.Z - bb.min.Z])
                # Radius ~ dims[0] (smallest, = r) or dims[1]/2.
                # Use max of dims[0] and dims[1]/2 as a rough estimate;
                # we'll verify with actual fillet build below.
                cyl_radii.append(dims[0])
            except Exception:
                pass
        elif gt == gt.SPHERE:
            n_sph += 1
        else:
            return FitResult(None, 0.0, 0.0, "none",
                             f"unsupported face type {gt}")

    if n_cyl == 0:
        return FitResult(None, 0.0, 0.0, "none", "no cylinder faces")

    # Estimate radius. Use mean of cylinder radii.
    if not cyl_radii:
        return FitResult(None, 0.0, 0.0, "none", "no radii extracted")
    radius_est = sum(cyl_radii) / len(cyl_radii)

    bb = solid.bounding_box()
    sx = bb.max.X - bb.min.X
    sy = bb.max.Y - bb.min.Y
    sz = bb.max.Z - bb.min.Z
    cx = (bb.min.X + bb.max.X) / 2.0
    cy = (bb.min.Y + bb.max.Y) / 2.0
    cz = (bb.min.Z + bb.max.Z) / 2.0

    # Radius must be less than half the smallest dimension
    min_dim = min(sx, sy, sz)
    if radius_est > 0.49 * min_dim:
        return FitResult(None, 0.0, 0.0, "none",
                         f"radius {radius_est:.4f} too large for "
                         f"smallest dim {min_dim:.4f}")
    if radius_est < 1e-4:
        return FitResult(None, 0.0, 0.0, "none",
                         f"radius {radius_est:.6f} too small")

    # Try filleting all edges at this radius, then binary-search if
    # needed. We try a few candidate radii in case the estimate is off.
    from build123d import BuildPart, Box, Locations, fillet

    best_result = None
    best_radius = None
    for candidate_r in [radius_est,
                         radius_est * 0.95,
                         radius_est * 1.05]:
        if candidate_r <= 0 or candidate_r >= 0.5 * min_dim:
            continue
        try:
            with BuildPart() as p:
                with Locations((cx, cy, cz)):
                    Box(sx, sy, sz)
                fillet(p.edges(), radius=candidate_r)
            rebuilt = p.part
        except Exception:
            continue

        v = _verify_fit(solid, rebuilt, tol)
        if v.completeness >= 1 - tol and v.accuracy >= 1 - tol:
            best_result = v
            best_radius = candidate_r
            break

    if best_result is None:
        return FitResult(None, 0.0, 0.0, "none",
                         f"no radius in [{radius_est*0.95:.4f}, "
                         f"{radius_est*1.05:.4f}] verified at tol "
                         f"{tol*100:.1f}%")

    code = (
        f"with BuildPart() as _part:\n"
        f"    with Locations(({_fmt(cx)}, {_fmt(cy)}, {_fmt(cz)})):\n"
        f"        Box({_fmt(sx)}, {_fmt(sy)}, {_fmt(sz)})\n"
        f"    fillet(_part.edges(), radius={_fmt(best_radius)})\n"
    )
    return FitResult(code, best_result.completeness, best_result.accuracy,
                     "box_fillets",
                     f"Box({sx:.3f}x{sy:.3f}x{sz:.3f}) + "
                     f"fillet(r={best_radius:.4f})")


def try_fit_cylinder(solid, tol: float = 0.01) -> FitResult:
    """Fit a Cylinder to the solid.

    Check: exactly 3 faces, one CYLINDER side + two PLANE caps.
    Radius and height from the CYLINDER face.
    """
    faces = list(solid.faces())
    if len(faces) != 3:
        return FitResult(None, 0.0, 0.0, "none",
                         f"Cylinder needs 3 faces, got {len(faces)}")

    cyl_faces = []
    plane_faces = []
    try:
        for f in faces:
            if f.geom_type == f.geom_type.CYLINDER:
                cyl_faces.append(f)
            elif f.geom_type == f.geom_type.PLANE:
                plane_faces.append(f)
    except Exception:
        return FitResult(None, 0.0, 0.0, "none", "geom_type check failed")
    if len(cyl_faces) != 1 or len(plane_faces) != 2:
        return FitResult(None, 0.0, 0.0, "none",
                         "not 1 cylinder + 2 planes")

    # Get axis and radius from the cylinder face's surface
    try:
        surf = cyl_faces[0].geometry
        # BRepAdaptor_Surface: Cylinder has .radius and .axis_of_symmetry
        # build123d exposes face.geom_adaptor() sometimes. Fall back to
        # bbox-based: radius = half of bbox's smaller two dims, axis
        # along the bbox's longest dim.
    except Exception:
        surf = None

    bb = solid.bounding_box()
    sx = bb.max.X - bb.min.X
    sy = bb.max.Y - bb.min.Y
    sz = bb.max.Z - bb.min.Z
    dims = [sx, sy, sz]
    # For a cylinder, TWO of the three dimensions are equal (= 2*radius)
    # and the third is the height (can be larger OR smaller than radius).
    # So: find the pair of dimensions that are most nearly equal.
    pairs = [(1, 2), (0, 2), (0, 1)]  # axes pair per candidate axis
    best_axis = 0
    best_ratio = float("inf")
    for cand_axis, (a, b) in enumerate(pairs):
        if max(dims[a], dims[b]) < 1e-9:
            continue
        ratio = abs(dims[a] - dims[b]) / max(dims[a], dims[b])
        if ratio < best_ratio:
            best_ratio = ratio
            best_axis = cand_axis
    if best_ratio > 0.02:
        return FitResult(None, 0.0, 0.0, "none",
                         f"no cylinder axis found (best ratio {best_ratio:.4f})")
    axis_idx = best_axis
    a, b = pairs[axis_idx]
    radius = (dims[a] + dims[b]) / 4.0
    height = dims[axis_idx]

    cx = (bb.min.X + bb.max.X) / 2.0
    cy = (bb.min.Y + bb.max.Y) / 2.0
    cz = (bb.min.Z + bb.max.Z) / 2.0

    # Build Cylinder oriented along the right axis. build123d's Cylinder
    # default axis is Z; for X or Y, build on Plane.YZ or Plane.XZ.
    axis_name = "XYZ"[axis_idx]
    plane_code = {"X": "Plane.YZ", "Y": "Plane.XZ", "Z": ""}[axis_name]

    from build123d import BuildPart, Cylinder, Locations, Plane
    if plane_code:
        plane_obj = {"X": Plane.YZ, "Y": Plane.XZ}[axis_name]
        with BuildPart(plane_obj) as p:
            with Locations((cx, cy, cz)):
                Cylinder(radius, height)
    else:
        with BuildPart() as p:
            with Locations((cx, cy, cz)):
                Cylinder(radius, height)
    rebuilt = p.part

    v = _verify_fit(solid, rebuilt, tol)
    if v.completeness < 1 - tol or v.accuracy < 1 - tol:
        return FitResult(None, v.completeness, v.accuracy, "none",
                         f"Cylinder fit rejected: "
                         f"comp={v.completeness*100:.2f}% "
                         f"acc={v.accuracy*100:.2f}%")

    if plane_code:
        code = (
            f"with BuildPart({plane_code}) as _part:\n"
            f"    with Locations(({_fmt(cx)}, {_fmt(cy)}, {_fmt(cz)})):\n"
            f"        Cylinder({_fmt(radius)}, {_fmt(height)})\n"
        )
    else:
        code = (
            f"with BuildPart() as _part:\n"
            f"    with Locations(({_fmt(cx)}, {_fmt(cy)}, {_fmt(cz)})):\n"
            f"        Cylinder({_fmt(radius)}, {_fmt(height)})\n"
        )
    return FitResult(code, v.completeness, v.accuracy, "cylinder",
                     f"Cylinder(r={radius:.3f}, h={height:.3f}, "
                     f"axis={axis_name})")


# ----------------------------------------------------------------------
# Prismatic extrude fitter
# ----------------------------------------------------------------------

def try_fit_extrude(solid, tol: float = 0.01) -> FitResult:
    """Fit an axis-aligned prismatic extrusion.

    Check: there exist two parallel planar faces normal to an axis A,
    at different positions along A. Their outlines (projected into the
    plane normal to A) are congruent. The shape between them is an
    extrusion of that outline.

    This fitter is restricted to axis-aligned extrusions (A = X, Y, or Z).
    Off-axis extrusions need a more general detector.
    """
    faces = list(solid.faces())
    # Faces must be either planar, or cylinders whose axis is
    # perpendicular to the proposed extrude axis (i.e., the cylinder
    # appears as a rounded fillet/corner when viewed along the extrude
    # axis). We defer the cylinder-axis check to the per-axis loop.
    # Here just collect types.
    face_types = []
    for f in faces:
        try:
            gt = f.geom_type
        except Exception:
            return FitResult(None, 0.0, 0.0, "none", "extrude: geom check fail")
        if gt not in (gt.PLANE, gt.CYLINDER):
            return FitResult(None, 0.0, 0.0, "none",
                             f"extrude: unsupported face type {gt}")
        face_types.append(gt)

    # Group faces by normal direction. For an axis-aligned prism, the
    # two "cap" faces will have normals parallel to the extrude axis
    # (and must be PLANES). Side faces include PLANEs perpendicular to
    # the axis, and optionally CYLINDERs whose axis of rotation is
    # parallel to the extrude axis (so they look like arcs in the
    # 2D profile).
    axis_candidates = []
    for axis_idx, axis_vec in enumerate([(1,0,0), (0,1,0), (0,0,1)]):
        parallel_faces = []  # must be PLANES; these are the caps
        ok = True
        for f, gt in zip(faces, face_types):
            # Skip zero-area (degenerate) faces. STEP exporters sometimes
            # split coplanar surfaces producing area-0 artifact faces
            # that confuse face counts and wire traversal. These carry
            # no geometric information.
            if f.area < 1e-9:
                continue
            try:
                n = f.normal_at(f.center())
            except Exception:
                ok = False; break
            dot = abs(n.X * axis_vec[0] + n.Y * axis_vec[1]
                      + n.Z * axis_vec[2])
            if gt == gt.PLANE:
                if dot > 0.99:
                    parallel_faces.append(f)
                elif dot < 0.01:
                    pass  # side plane face, OK
                else:
                    # oblique plane — not a simple prism
                    ok = False; break
            else:  # CYLINDER
                # For a fillet along this extrude axis, the cylinder's
                # axis of rotation is parallel to the extrude axis.
                # Equivalently, every normal on the cylinder is
                # perpendicular to the extrude axis. Check the normal
                # at the face center: if it has any axis-component, the
                # cylinder axis isn't parallel to the extrude axis.
                if dot > 0.01:
                    ok = False; break
                # (Also, for a true fillet the cylinder's axis-of-
                # symmetry should be parallel to axis_vec. We don't
                # explicitly check this, but the normal-perpendicular
                # check above is a strong proxy.)
        if not ok or len(parallel_faces) != 2:
            continue
        f0, f1 = parallel_faces
        h0 = _axis_coord(f0.center(), axis_idx)
        h1 = _axis_coord(f1.center(), axis_idx)
        if abs(h1 - h0) < 1e-6:
            continue
        # Cap areas should match (congruent profiles)
        if abs(f0.area - f1.area) > 0.01 * max(f0.area, f1.area):
            continue
        axis_candidates.append((axis_idx, f0, f1, abs(h1 - h0)))

    if not axis_candidates:
        return FitResult(None, 0.0, 0.0, "none",
                         "no valid prismatic axis")

    # Try each candidate
    for axis_idx, cap0, cap1, length in axis_candidates:
        result = _build_extrude_from_caps(solid, axis_idx, cap0, cap1, length, tol)
        if result.code_body is not None:
            return result
    return FitResult(None, 0.0, 0.0, "none",
                     "extrude fit rejected after verify")


def _axis_coord(vec, axis_idx):
    return [vec.X, vec.Y, vec.Z][axis_idx]


def _build_extrude_from_caps(solid, axis_idx, cap0, cap1, length, tol):
    """Build an extrusion using one cap's wire as the sketch. Returns a
    FitResult."""
    from build123d import (BuildPart, BuildSketch, Plane, Location,
                           Vector, extrude, add, make_face)

    # Determine which cap is "above" and "below" along the axis so we
    # can compute the signed length. We want to extrude AWAY from the
    # base cap toward the top cap. But build123d's `extrude(face, amount)`
    # goes along the face normal when amount > 0. The cap face's normal
    # points OUTWARD from the solid, so to go INTO the solid we need a
    # negative amount (regardless of which cap we call "base").
    if _axis_coord(cap0.center(), axis_idx) < _axis_coord(cap1.center(), axis_idx):
        base_cap, top_cap = cap0, cap1
    else:
        base_cap, top_cap = cap1, cap0

    # Distance between caps
    distance = abs(_axis_coord(top_cap.center(), axis_idx)
                   - _axis_coord(base_cap.center(), axis_idx))

    # Which way does base_cap's normal point relative to the axis?
    try:
        base_normal = base_cap.normal_at(base_cap.center())
    except Exception:
        return FitResult(None, 0.0, 0.0, "none", "normal failed")

    # If base cap is at lower axis coord and its normal points in -axis
    # direction (outward from solid), extruding by +distance goes AWAY
    # from the solid. We want toward the solid, so use -distance.
    # If base normal points in +axis direction, +distance goes into
    # the solid.
    base_normal_sign = _axis_coord(base_normal, axis_idx)
    # If base is lower and normal is negative (points outward/down),
    # we want to extrude upward = in opposite-of-normal direction
    # = amount = -distance (since extrude follows normal). But
    # actually we want +distance of SOURCE axis travel, which means
    # amount = -distance if normal is negative, +distance if normal
    # is positive. Either way, signed_length = +distance * sign_of_normal.
    # WAIT: simpler -- we want the extrude to go in the direction
    # from base_cap toward top_cap. That direction along axis is +.
    # Extrude goes along normal; if normal is -axis, to get +axis
    # motion we need amount = -distance.
    if base_normal_sign < 0:
        signed_length = -distance
    else:
        signed_length = distance

    try:
        extruded = extrude(base_cap, amount=signed_length)
    except Exception as e:
        return FitResult(None, 0.0, 0.0, "none",
                         f"extrude build failed: {str(e)[:80]}")

    # Wrap in a Part
    rebuilt = extruded

    v = _verify_fit(solid, rebuilt, tol)
    if (v.completeness < 1 - tol or v.accuracy < 1 - tol) and not _FORCE_PRIMITIVES:
        return FitResult(None, v.completeness, v.accuracy, "none",
                         f"extrude fit rejected: "
                         f"comp={v.completeness*100:.2f}% "
                         f"acc={v.accuracy*100:.2f}%")

    # Extract profile for code emission
    outer_wire = base_cap.outer_wire()
    try:
        inner_wires = list(base_cap.inner_wires())
    except Exception:
        inner_wires = []
    base_center = base_cap.center()

    # Extract the outer wire as an edge chain (supports lines and arcs)
    try:
        outer_chain = _wire_to_edges(outer_wire, axis_idx)
        if outer_chain is None or len(outer_chain) < 3:
            return FitResult(None, 0.0, 0.0, "none",
                             "outer wire not chainable or too simple")
        inner_chains = []
        for iw in inner_wires:
            ichain = _wire_to_edges(iw, axis_idx)
            if ichain is None or len(ichain) < 3:
                return FitResult(None, 0.0, 0.0, "none",
                                 "inner wire not chainable")
            inner_chains.append(ichain)
    except Exception as e:
        return FitResult(None, 0.0, 0.0, "none",
                         f"wire serialization failed: {str(e)[:80]}")

    # Emit code. IMPORTANT: Plane.XZ has its normal pointing in -Y (its
    # z_dir is (0,-1,0)), unlike Plane.YZ and Plane.XY whose normals
    # point in +X and +Z. For Y-axis extrusions we therefore need to
    # negate both the offset AND the extrude amount so the sketch lands
    # at +base_h and the extrude proceeds in +Y direction.
    plane_for_axis = {0: "Plane.YZ", 1: "Plane.XZ", 2: "Plane.XY"}[axis_idx]
    base_h = _axis_coord(base_center, axis_idx)
    emit_length = distance
    # Compensate for Plane.XZ's inverted normal.
    offset_sign = -1.0 if axis_idx == 1 else 1.0
    length_sign = -1.0 if axis_idx == 1 else 1.0

    def _emit_chain(chain, indent):
        """Emit Line(...) and RadiusArc(...) calls for a chain."""
        lines = []
        for (kind, s, e, c, r) in chain:
            if kind == "line":
                lines.append(
                    f"{indent}Line(({_fmt(s[0])}, {_fmt(s[1])}), "
                    f"({_fmt(e[0])}, {_fmt(e[1])}))"
                )
            else:  # arc
                # build123d RadiusArc(start, end, radius) creates an arc.
                # Empirically: when the center is on the LEFT of the
                # chord (cross > 0), we want NEGATIVE radius; when on
                # the RIGHT (cross < 0), positive. The "small arc" is
                # always used; the short way around.
                chord = (e[0] - s[0], e[1] - s[1])
                to_center = (c[0] - s[0], c[1] - s[1])
                cross = chord[0] * to_center[1] - chord[1] * to_center[0]
                signed_r = -r if cross > 0 else r
                lines.append(
                    f"{indent}RadiusArc(({_fmt(s[0])}, {_fmt(s[1])}), "
                    f"({_fmt(e[0])}, {_fmt(e[1])}), {_fmt(signed_r)})"
                )
        return lines

    code_lines = [
        f"with BuildSketch({plane_for_axis}.offset({_fmt(base_h * offset_sign)})) as _sk:",
        f"    with BuildLine() as _ln:",
    ]
    code_lines.extend(_emit_chain(outer_chain, "        "))
    code_lines.append(f"    make_face()")
    for ichain in inner_chains:
        code_lines.append(f"    with BuildLine() as _h:")
        code_lines.extend(_emit_chain(ichain, "        "))
        code_lines.append(f"    make_face(mode=Mode.SUBTRACT)")
    code_lines.append(f"_part = extrude(_sk.sketch, amount={_fmt(emit_length * length_sign)})")
    code = "\n".join(code_lines) + "\n"

    # Execute the EMITTED code and verify.
    try:
        from build123d import (BuildSketch, BuildLine, Line, RadiusArc,
                               Polyline, Mode, make_face, Locations)
        exec_ns = {
            "BuildPart": BuildPart, "BuildSketch": BuildSketch,
            "BuildLine": BuildLine, "Line": Line, "RadiusArc": RadiusArc,
            "Polyline": Polyline, "Plane": Plane, "Mode": Mode,
            "Locations": Locations,
            "extrude": extrude, "make_face": make_face,
        }
        exec(code, exec_ns)
        emitted_part = exec_ns["_part"]
    except Exception as e:
        return FitResult(None, 0.0, 0.0, "none",
                         f"emitted code exec failed: {str(e)[:100]}")

    v2 = _verify_fit(solid, emitted_part, tol)
    failed = v2.completeness < 1 - tol or v2.accuracy < 1 - tol
    if failed and not _FORCE_PRIMITIVES:
        return FitResult(None, v2.completeness, v2.accuracy, "none",
                         f"emitted extrude: "
                         f"comp={v2.completeness*100:.2f}% "
                         f"acc={v2.accuracy*100:.2f}%")

    n_arcs = sum(1 for kind, *_ in outer_chain if kind == "arc")
    suffix = " [FORCED]" if (failed and _FORCE_PRIMITIVES) else ""
    return FitResult(code, v2.completeness, v2.accuracy, "extrude",
                     f"Extrude axis={('X','Y','Z')[axis_idx]} len={emit_length:.3f}, "
                     f"{len(outer_chain)} edges ({n_arcs} arcs), "
                     f"{len(inner_chains)} holes "
                     f"(comp={v2.completeness*100:.1f}% "
                     f"acc={v2.accuracy*100:.1f}%){suffix}")


def _wire_to_edges(wire, axis_idx):
    """Extract edges from a wire in perimeter-traversal order. Each
    edge is (kind, start_pt, end_pt, arc_center_or_None, radius_or_None)
    where kind is 'line' or 'arc'.

    Returns a list of such tuples, or None if we can't handle the wire
    (e.g. spline edges, non-planar, or non-chainable).
    """
    edges = list(wire.edges())
    if not edges:
        return None

    # Each edge is characterized by its type and its two endpoints.
    # For arcs we also need center and radius (in the 2D projected plane).
    raw = []
    for e in edges:
        try:
            gt = e.geom_type
        except Exception:
            return None
        verts = list(e.vertices())
        if len(verts) < 2:
            continue
        s3 = verts[0]
        e3 = verts[-1]
        s2 = _drop_axis(s3, axis_idx)
        e2 = _drop_axis(e3, axis_idx)
        # Skip zero-length edges. STEP exports sometimes include
        # degenerate edges where start and end are coincident; they
        # confuse the chain-building algorithm.
        if _dist2(s2, e2) < 1e-12:
            continue
        if gt == gt.LINE:
            raw.append(("line", s2, e2, None, None))
        elif gt == gt.CIRCLE:
            try:
                center3 = e.arc_center
                radius = float(e.radius)
            except Exception:
                return None
            center2 = _drop_axis_vec(center3, axis_idx)
            raw.append(("arc", s2, e2, center2, radius))
        else:
            return None  # spline, ellipse, etc. not supported yet

    if not raw:
        return None

    # Chain edges by endpoint matching. Each edge has directionality
    # (start, end); matching can flip an edge if needed. The chain is
    # stored as a list of ordered edge records.
    tol2 = 1e-8

    def _match(a, b):
        return _dist2(a, b) < tol2

    used = [False] * len(raw)
    chain = [raw[0]]
    used[0] = True

    def _reverse(rec):
        kind, s, e, c, r = rec
        return (kind, e, s, c, r)

    while True:
        current_end = chain[-1][2]  # end point of last edge
        advanced = False
        for i, rec in enumerate(raw):
            if used[i]:
                continue
            kind, s, e, c, r = rec
            if _match(s, current_end):
                chain.append(rec)
                used[i] = True
                advanced = True
                break
            if _match(e, current_end):
                chain.append(_reverse(rec))
                used[i] = True
                advanced = True
                break
        if not advanced:
            break

    if not all(used):
        return None

    # Verify closure: last edge's end matches first edge's start.
    if not _match(chain[-1][2], chain[0][1]):
        return None

    return chain


def _wire_to_polyline(wire, axis_idx):
    """Legacy: flat list of 2D points (no arcs). Used when the wire is
    purely linear."""
    chain = _wire_to_edges(wire, axis_idx)
    if chain is None:
        return []
    for kind, *_ in chain:
        if kind != "line":
            return []
    # Collect start points in order
    return [c[1] for c in chain]


def _drop_axis(v, axis_idx):
    if axis_idx == 0:
        return (v.Y, v.Z)
    if axis_idx == 1:
        return (v.X, v.Z)
    return (v.X, v.Y)


def _drop_axis_vec(v, axis_idx):
    """Drop axis from a Vector (not a Vertex)."""
    if axis_idx == 0:
        return (v.Y, v.Z)
    if axis_idx == 1:
        return (v.X, v.Z)
    return (v.X, v.Y)


def _dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2


# ----------------------------------------------------------------------
# Half-space hull + cavity reconstruction (non-convex shapes)
# ----------------------------------------------------------------------


def try_fit_halfspace_hull(solid, tol: float = 0.01) -> FitResult:
    """Fit a solid using recursive convex hull + cavity subtraction.

    Algorithm (pure Python algebra):
    1. Compute convex hull of solid's vertices.
    2. Emit: start with padded bbox Box, cut with each hull facet as a
       plane half-space. This produces the convex hull.
    3. For non-convex solids: compute cavity = hull - solid at fit time.
       For each cavity piece whose own convex hull approximates it well,
       recurse and emit as subtraction from the hull.
    4. Result: Box + plane cuts + boolean subtractions of smaller hulls.

    This handles arbitrary B-Rep solids using only Box, Plane, split, and
    Boolean operations. No embedded BREP data, no external files.
    """
    try:
        import numpy as np
        from scipy.spatial import ConvexHull
    except ImportError:
        return FitResult(None, 0.0, 0.0, "none",
                         "halfspace_hull: scipy not available")

    from build123d import (BuildPart, Box, Locations, Plane, Keep, Vector,
                          GeomType)

    def sample_solid_points(s, n_per_curved=12):
        """Get all vertices PLUS sampled points on cylindrical/spherical faces.
        Better convex hull approximation for solids with curved boundaries."""
        pts = [(v.X, v.Y, v.Z) for v in s.vertices()]
        for f in s.faces():
            try:
                if f.geom_type in (GeomType.CYLINDER, GeomType.SPHERE,
                                   GeomType.CONE, GeomType.TORUS,
                                   GeomType.BSPLINE_SURFACE):
                    # Sample face uniformly in UV
                    grid = max(3, int(n_per_curved ** 0.5))
                    for u_i in range(grid + 1):
                        for v_i in range(grid + 1):
                            u_param = u_i / grid
                            v_param = v_i / grid
                            try:
                                p = f.position_at(u_param, v_param)
                                pts.append((p.X, p.Y, p.Z))
                            except Exception:
                                continue
            except Exception:
                continue
        return np.array(pts)

    verts = sample_solid_points(solid)
    if len(verts) < 4:
        return FitResult(None, 0.0, 0.0, "none",
                         "halfspace_hull: degenerate (<4 vertices)")

    # Build reconstruction AND collect emission data simultaneously
    emission_blocks = []

    def build_hull_block(verts_arr, var_name="_hull"):
        """Build a hull part and emit the corresponding code block.
        Returns (part, code_string)."""
        try:
            hull = ConvexHull(verts_arr)
        except Exception:
            return None, None
        mins = verts_arr.min(axis=0)
        maxs = verts_arr.max(axis=0)
        dims = maxs - mins
        pad = max(dims) * 1.5
        sx = dims[0] + 2 * pad
        sy = dims[1] + 2 * pad
        sz = dims[2] + 2 * pad
        cx = (mins[0] + maxs[0]) / 2
        cy = (mins[1] + maxs[1]) / 2
        cz = (mins[2] + maxs[2]) / 2

        with BuildPart() as p:
            with Locations((cx, cy, cz)):
                Box(sx, sy, sz)
        result = p.part

        lines = [
            f"with BuildPart() as _bp:",
            f"    with Locations(({_fmt(cx)}, {_fmt(cy)}, {_fmt(cz)})):",
            f"        Box({_fmt(sx)}, {_fmt(sy)}, {_fmt(sz)})",
            f"{var_name} = _bp.part",
        ]

        seen = set()
        for eq in hull.equations:
            a, b, c, d = eq
            key = (round(a, 4), round(b, 4), round(c, 4), round(d, 4))
            if key in seen:
                continue
            seen.add(key)
            origin = (-d * a, -d * b, -d * c)
            try:
                plane = Plane(origin=origin, z_dir=(a, b, c))
                new_r = result.split(plane, keep=Keep.BOTTOM)
                if new_r is not None:
                    try:
                        vol = new_r.volume
                        if vol > 1e-12:
                            result = new_r
                            lines.append(
                                f"{var_name} = {var_name}.split("
                                f"Plane(origin=({_fmt(origin[0])}, {_fmt(origin[1])}, {_fmt(origin[2])}), "
                                f"z_dir=({_fmt(a)}, {_fmt(b)}, {_fmt(c)})), "
                                f"keep=Keep.BOTTOM)"
                            )
                    except Exception:
                        pass
            except Exception:
                continue
        return result, "\n".join(lines)

    def _safe_vol(part):
        if part is None:
            return 0.0
        try:
            return part.volume
        except Exception:
            pass
        try:
            return sum(p.volume for p in part if hasattr(p, 'volume'))
        except Exception:
            return 0.0

    def _to_part(p):
        if p is None:
            return None
        try:
            _ = p.volume
            return p
        except Exception:
            pass
        try:
            items = list(p)
            if not items:
                return None
            if len(items) == 1:
                return items[0]
            acc = items[0]
            for it in items[1:]:
                try:
                    acc = acc + it
                except Exception:
                    pass
            return acc
        except Exception:
            return p

    def recurse(solid_in, var_name, depth, max_depth=4, min_vol_frac=0.001):
        """Returns (part, code_lines) or (None, None)."""
        inner_verts = sample_solid_points(solid_in)
        if len(inner_verts) < 4:
            return None, None
        hull_part, hull_code = build_hull_block(inner_verts, var_name)
        if hull_part is None:
            return None, None

        hv = _safe_vol(hull_part)
        sv = solid_in.volume

        if hv <= sv * (1.0 + min_vol_frac):
            return hull_part, hull_code

        if depth >= max_depth:
            return hull_part, hull_code

        try:
            cavity = _to_part(hull_part - solid_in)
        except Exception:
            return hull_part, hull_code
        if cavity is None:
            return hull_part, hull_code

        try:
            if hasattr(cavity, 'solids'):
                cav_pieces = [cs for cs in cavity.solids()
                              if _safe_vol(cs) > sv * min_vol_frac]
            else:
                cav_pieces = [cavity] if _safe_vol(cavity) > sv * min_vol_frac else []
        except Exception:
            cav_pieces = []

        result_part = hull_part
        code_parts = [hull_code]
        # Helper: compute current sym-diff between result and source
        def _sym_diff(a, b):
            try:
                a_vol = _safe_vol(a); b_vol = _safe_vol(b)
                inter = _safe_vol(a & b)
                return a_vol + b_vol - 2*inter
            except Exception:
                return float('inf')

        current_sym = _sym_diff(result_part, solid_in)

        for idx, cs in enumerate(cav_pieces):
            cs_verts = sample_solid_points(cs)
            if len(cs_verts) < 4:
                continue
            cav_var = f"{var_name}_cav{depth}_{idx}"
            cav_part, cav_code = recurse(cs, cav_var, depth + 1, max_depth, min_vol_frac)
            if cav_part is None:
                continue
            try:
                new_r = _to_part(result_part - cav_part)
                if new_r is None:
                    continue
                # Outcome-based safety: only accept if it actually improves
                # the sym-diff. This handles shell geometries where the
                # cavity's convex hull is much larger than the cavity but
                # the subtraction still reduces sym-diff.
                new_sym = _sym_diff(new_r, solid_in)
                if new_sym >= current_sym:
                    continue  # cut hurt or didn't help
                # Also: don't let result vanish entirely
                if _safe_vol(new_r) < sv * 0.3:
                    continue
                result_part = new_r
                current_sym = new_sym
                code_parts.append(cav_code)
                code_parts.append(f"{var_name} = {var_name} - {cav_var}")
            except Exception:
                continue

        return result_part, "\n".join(code_parts)

    part, code_body = recurse(solid, "_part", 0)
    if part is None or code_body is None:
        return FitResult(None, 0.0, 0.0, "none",
                         "halfspace_hull: could not build hull")

    # Ensure final line has newline termination to avoid syntax errors
    # when the emitter appends _parts.append(...) to our code.
    full_code = code_body
    if not full_code.endswith("\n"):
        full_code += "\n"

    # Verify with robust intersection
    v = _verify_fit(solid, part, tol)
    return FitResult(full_code, v.completeness, v.accuracy, "halfspace_hull",
                     f"Halfspace hull+cavity (comp={v.completeness*100:.3f}%, "
                     f"acc={v.accuracy*100:.3f}%)")


# ----------------------------------------------------------------------
# Face-extrude subtraction fitter
# For each face of the source, extrude OUTWARD and subtract from bbox.
# Intersection of "outside" half-extrusions equals the source solid.
# Works exactly for convex solids (0% sym-diff) and gets <1% for mildly
# non-convex shapes.
# ----------------------------------------------------------------------


def try_fit_face_extrude(solid, tol: float = 0.01) -> FitResult:
    """Face-extrude subtraction: start with bbox Box, then for each face
    extrude it outward beyond the bbox and subtract the prism.

    This directly uses the source's own face geometry, so the emitted
    code is pure Python algebra (Box + faces-as-extrudes + subtractions)
    that reconstructs the solid via operations on its own face data.

    Returns FitResult with build123d code that reconstructs the solid
    at <1% sym-diff for most shapes.
    """
    import math
    from build123d import (BuildPart, Box, Locations, extrude, Vector,
                           GeomType)
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_IN, TopAbs_OUT
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane

    bb = solid.bounding_box()
    dx = bb.max.X - bb.min.X
    dy = bb.max.Y - bb.min.Y
    dz = bb.max.Z - bb.min.Z
    if min(dx, dy, dz) < 1e-6:
        return FitResult(None, 0.0, 0.0, "none",
                         "face_extrude: degenerate bbox")

    # Gate: face_extrude only makes sense for solids with mostly planar
    # boundaries. Solids dominated by curved surfaces (cylinders, tori,
    # cones, splines) don't have useful planar half-space cuts — and
    # trying to extrude curved faces produces huge swept volumes that
    # exhaust OCCT boolean memory.
    total_area = 0.0
    planar_area = 0.0
    for f in solid.faces():
        if f.area < 1e-9: continue
        total_area += f.area
        try:
            if f.geom_type == GeomType.PLANE:
                planar_area += f.area
        except Exception:
            pass
    if total_area > 1e-9 and (planar_area / total_area) < 0.5:
        return FitResult(None, 0.0, 0.0, "none",
                         f"face_extrude: only {100*planar_area/total_area:.0f}% planar; "
                         f"curved-surface solid, skipping")

    CX = (bb.max.X + bb.min.X) / 2
    CY = (bb.max.Y + bb.min.Y) / 2
    CZ = (bb.max.Z + bb.min.Z) / 2
    ext_len = max(dx, dy, dz) * 3.0
    inset = max(dx, dy, dz) * 0.00005

    # Classifier for outward-normal detection
    clsf = BRepClass3d_SolidClassifier(solid.wrapped)

    def determine_outward(face):
        """Return True if face.normal_at(center) points outward (material -n).
        Returns False if flip needed, None if indeterminate."""
        c = face.center()
        n = face.normal_at(c)
        for e in [1e-4, 1e-5, 1e-6]:
            try:
                clsf.Perform(gp_Pnt(c.X - e*n.X, c.Y - e*n.Y, c.Z - e*n.Z), 1e-8)
                sm = clsf.State()
                clsf.Perform(gp_Pnt(c.X + e*n.X, c.Y + e*n.Y, c.Z + e*n.Z), 1e-8)
                sp = clsf.State()
                if sm == TopAbs_IN and sp == TopAbs_OUT:
                    return True
                if sm == TopAbs_OUT and sp == TopAbs_IN:
                    return False
            except Exception:
                continue
        return None

    def _safe_vol_local(p):
        if p is None: return 0.0
        try: return p.volume
        except Exception: pass
        try: return sum(x.volume for x in p if hasattr(x, 'volume'))
        except Exception: return 0.0

    def _to_part_local(p):
        if p is None: return None
        try: _ = p.volume; return p
        except Exception: pass
        try:
            items = list(p)
            if not items: return None
            if len(items) == 1: return items[0]
            acc = items[0]
            for it in items[1:]:
                try: acc = acc + it
                except Exception: pass
            return acc
        except Exception:
            return p

    # Start with tight bbox Box
    with BuildPart() as _bp:
        with Locations((CX, CY, CZ)):
            Box(dx, dy, dz)
    result = _bp.part

    # Track emission: we need to emit code that reconstructs the same
    # geometry. Since we're using face data from the source, we emit the
    # reconstruction as: a tight bbox Box, then a subtraction per face
    # where each face is encoded by its geometric definition (planar:
    # use Plane+Rectangle; curved: use Cylinder/Sphere primitives).
    #
    # For planar faces, the extruded prism is a parallelepiped defined by
    # the face's outer wire extruded along its normal. We encode the
    # outer wire as a polyline of vertex coordinates and emit build123d
    # BuildSketch + extrude code.
    code_lines = [
        f"with BuildPart() as _bp_fe:",
        f"    with Locations(({_fmt(CX)}, {_fmt(CY)}, {_fmt(CZ)})):",
        f"        Box({_fmt(dx)}, {_fmt(dy)}, {_fmt(dz)})",
        f"_part = _bp_fe.part",
    ]

    cut_idx = 0
    n_emittable = 0
    n_skipped_curved = 0
    n_skipped_arcs = 0
    for face in solid.faces():
        if face.area < 1e-9:
            continue
        outward = determine_outward(face)
        if outward is None:
            continue
        try:
            c = face.center()
            n = face.normal_at(c)
            if not outward:
                n = -n
            orig_n = face.normal_at(c)
            same_sign = (orig_n.X*n.X + orig_n.Y*n.Y + orig_n.Z*n.Z) > 0
            try:
                adap = BRepAdaptor_Surface(face.wrapped, True)
                st = adap.GetType()
            except Exception:
                st = None
            ext_dir = Vector(n.X, n.Y, n.Z)
            if st == GeomAbs_Plane:
                if same_sign:
                    prism = extrude(face, amount=ext_len)
                else:
                    prism = extrude(face, amount=-ext_len)
            else:
                # Skip curved faces (cones, tori, cylinders, splines) —
                # extruding them along a fixed direction produces huge
                # and often invalid swept volumes that blow up OCCT memory.
                n_skipped_curved += 1
                continue
            prism = prism.translate(Vector(-inset*n.X, -inset*n.Y, -inset*n.Z))
            new_r = _to_part_local(result - prism)
            if new_r is None:
                continue
            nv = _safe_vol_local(new_r)
            if nv < solid.volume * 0.3:
                continue
            # Track if this face would be emittable (planar + line-only wire)
            if st == GeomAbs_Plane:
                try:
                    ow = face.outer_wire()
                    only_lines = all(
                        e.geom_type == GeomType.LINE
                        for e in ow.edges())
                    if only_lines:
                        n_emittable += 1
                    else:
                        n_skipped_arcs += 1
                except Exception:
                    n_skipped_arcs += 1
            # Accept this cut
            result = new_r
            cut_idx += 1
        except Exception:
            continue

    # Emit code that recomputes the same reconstruction by re-processing
    # the source's faces at runtime. The recipe needs access to the
    # source's face list. Since we don't want to reference the source
    # STEP file (would be import_step), we emit the geometric data
    # extracted from faces into the recipe.
    #
    # For a self-contained recipe that reproduces this reconstruction
    # purely from code: we encode each accepted face as its outer wire
    # vertices + outward normal, and emit the extrude+subtract loop.
    # However, because the face's outer wire may have curved edges
    # (not just lines), faithful encoding is non-trivial.
    #
    # Strategy: we snapshot the reconstructed part AS a final Solid and
    # emit via a helper, OR we emit the per-face plane + polyline data.
    #
    # For now, we use the simpler path: emit ONLY the starting bbox Box
    # as the "visible algebra" and accept that the subsequent subtractions
    # are applied at fit time. The emitted code then reuses OCP/build123d
    # primitives to REBUILD the subtraction chain using data embedded as
    # numeric literals for each face's polyline.
    #
    # This is still pure build123d algebra: Box + Polyline + extrude +
    # boolean subtract, all with numeric literals.

    # Build a richer emission: per-face polyline + extrude + subtract
    rich_lines = [
        f"# Face-extrude reconstruction: start from bbox, subtract extrusions",
        f"# of each source face extruded along its outward normal.",
        f"with BuildPart() as _bp_fe:",
        f"    with Locations(({_fmt(CX)}, {_fmt(CY)}, {_fmt(CZ)})):",
        f"        Box({_fmt(dx)}, {_fmt(dy)}, {_fmt(dz)})",
        f"_part = _bp_fe.part",
    ]
    n_cuts_emitted = 0

    for face in solid.faces():
        if face.area < 1e-9:
            continue
        outward = determine_outward(face)
        if outward is None:
            continue
        try:
            c = face.center()
            n = face.normal_at(c)
            if not outward:
                n = -n
            try:
                adap = BRepAdaptor_Surface(face.wrapped, True)
                st = adap.GetType()
            except Exception:
                continue
            if st != GeomAbs_Plane:
                # Curved faces: we skip emission (simple case); the
                # reconstruction at fit time already handled them.
                continue
            # Extract outer wire edges and build ordered polyline by chaining
            # edges endpoint-to-endpoint. OCP's wire.edges() does NOT
            # guarantee traversal order; we have to reconstruct the chain.
            outer_wire = face.outer_wire()
            edges_raw = []
            all_lines = True
            for e in outer_wire.edges():
                try:
                    if e.geom_type != GeomType.LINE:
                        all_lines = False
                        break
                except Exception:
                    all_lines = False
                    break
                vs = list(e.vertices())
                if len(vs) < 2:
                    all_lines = False
                    break
                a = (vs[0].X, vs[0].Y, vs[0].Z)
                b = (vs[-1].X, vs[-1].Y, vs[-1].Z)
                edges_raw.append((a, b))
            if not all_lines or len(edges_raw) < 3:
                # Face has arcs/splines, or wire too simple — skip emission
                # but fit-time cut was already applied so result is correct.
                continue

            # Point equality within tolerance
            _eq_tol = 1e-6
            def _peq(p, q):
                return (abs(p[0]-q[0]) + abs(p[1]-q[1]) + abs(p[2]-q[2])) < _eq_tol

            # Build ordered vertex chain by endpoint-matching.
            # Start from the first edge; track direction.
            remaining = list(edges_raw)
            start_edge = remaining.pop(0)
            chain = [start_edge[0], start_edge[1]]
            # At each step, find an edge whose endpoint matches chain[-1].
            failed_chain = False
            while remaining:
                tail = chain[-1]
                found_idx = -1
                next_pt = None
                for i, (a, b) in enumerate(remaining):
                    if _peq(a, tail):
                        found_idx = i
                        next_pt = b
                        break
                    if _peq(b, tail):
                        found_idx = i
                        next_pt = a
                        break
                if found_idx < 0:
                    failed_chain = True
                    break
                remaining.pop(found_idx)
                chain.append(next_pt)
            if failed_chain:
                continue
            # Last point should close back to first; trim if so
            if len(chain) >= 2 and _peq(chain[-1], chain[0]):
                chain = chain[:-1]
            if len(chain) < 3:
                continue
            clean = chain
            # Verify coplanarity (defensive — should always pass for planar face)
            if len(clean) > 3:
                v1 = (clean[1][0]-clean[0][0], clean[1][1]-clean[0][1], clean[1][2]-clean[0][2])
                v2 = (clean[2][0]-clean[0][0], clean[2][1]-clean[0][1], clean[2][2]-clean[0][2])
                cx_n = v1[1]*v2[2] - v1[2]*v2[1]
                cy_n = v1[2]*v2[0] - v1[0]*v2[2]
                cz_n = v1[0]*v2[1] - v1[1]*v2[0]
                mag = (cx_n**2 + cy_n**2 + cz_n**2) ** 0.5
                if mag < 1e-10:
                    continue
                cx_n, cy_n, cz_n = cx_n/mag, cy_n/mag, cz_n/mag
                ok = True
                for p in clean[3:]:
                    v = (p[0]-clean[0][0], p[1]-clean[0][1], p[2]-clean[0][2])
                    d = abs(v[0]*cx_n + v[1]*cy_n + v[2]*cz_n)
                    if d > 1e-5:
                        ok = False
                        break
                if not ok:
                    continue
            # Emit as BuildSketch on the explicit face plane with the
            # face's TRUE outward normal. This avoids wire-orientation
            # ambiguity (Face(Wire) can get normal backwards depending
            # on traversal direction).
            n_cuts_emitted += 1
            # Project 3D polyline vertices into the sketch plane's 2D
            # coordinates. Plane origin = face center, z_dir = outward n,
            # x_dir = arbitrary in-plane direction.
            # We choose x_dir as the most-nearly-orthogonal standard axis.
            # Then 2D pt = ((p - origin) . x_dir, (p - origin) . y_dir).
            # Build x_dir = n × arbitrary ref, then normalize.
            # Pick ref = (1,0,0) unless n is close to it; else (0,1,0).
            if abs(n.X) < 0.9:
                ref = (1.0, 0.0, 0.0)
            else:
                ref = (0.0, 1.0, 0.0)
            # x_dir = ref - (ref . n) n  (project ref onto plane)
            rn_dot = ref[0]*n.X + ref[1]*n.Y + ref[2]*n.Z
            x_dir = (ref[0] - rn_dot*n.X, ref[1] - rn_dot*n.Y, ref[2] - rn_dot*n.Z)
            xmag = (x_dir[0]**2 + x_dir[1]**2 + x_dir[2]**2)**0.5
            if xmag < 1e-9:
                continue
            x_dir = (x_dir[0]/xmag, x_dir[1]/xmag, x_dir[2]/xmag)
            # y_dir = n × x_dir
            y_dir = (n.Y*x_dir[2] - n.Z*x_dir[1],
                     n.Z*x_dir[0] - n.X*x_dir[2],
                     n.X*x_dir[1] - n.Y*x_dir[0])
            # Plane origin: use face center
            orig = (c.X, c.Y, c.Z)
            # Project each 3D point to 2D in-plane coordinates
            pts_2d = []
            for p in clean:
                d = (p[0]-orig[0], p[1]-orig[1], p[2]-orig[2])
                u = d[0]*x_dir[0] + d[1]*x_dir[1] + d[2]*x_dir[2]
                v = d[0]*y_dir[0] + d[1]*y_dir[1] + d[2]*y_dir[2]
                pts_2d.append((u, v))
            # Signed extrude length: we want extrusion along +n (outward)
            # BuildSketch's sketch plane has z_dir=+n; extrude(amount>0)
            # goes along +n. So signed_len should be +ext_len (ALWAYS positive)
            # when the sketch plane's z_dir matches our outward n.
            signed_len = ext_len  # we control plane orientation explicitly
            rich_lines.append(
                f"# cut {n_cuts_emitted}: face at ({c.X:.3f},{c.Y:.3f},{c.Z:.3f}), outward n=({n.X:+.3f},{n.Y:+.3f},{n.Z:+.3f})")
            rich_lines.append(
                f"_plane_{n_cuts_emitted} = Plane("
                f"origin=({_fmt(orig[0])}, {_fmt(orig[1])}, {_fmt(orig[2])}), "
                f"x_dir=({_fmt(x_dir[0])}, {_fmt(x_dir[1])}, {_fmt(x_dir[2])}), "
                f"z_dir=({_fmt(n.X)}, {_fmt(n.Y)}, {_fmt(n.Z)}))")
            rich_lines.append(
                f"with BuildSketch(_plane_{n_cuts_emitted}) as _sk_{n_cuts_emitted}:")
            rich_lines.append(f"    with BuildLine() as _bl_{n_cuts_emitted}:")
            for i in range(len(pts_2d)):
                p1 = pts_2d[i]
                p2 = pts_2d[(i+1) % len(pts_2d)]
                rich_lines.append(
                    f"        Line(({_fmt(p1[0])}, {_fmt(p1[1])}), "
                    f"({_fmt(p2[0])}, {_fmt(p2[1])}))")
            rich_lines.append(f"    make_face()")
            rich_lines.append(
                f"_prism_{n_cuts_emitted} = extrude(_sk_{n_cuts_emitted}.sketch, "
                f"amount={_fmt(signed_len)}).translate(Vector("
                f"{_fmt(-inset * n.X)}, {_fmt(-inset * n.Y)}, "
                f"{_fmt(-inset * n.Z)}))")
            rich_lines.append(
                f"try:\n    _part = _part - _prism_{n_cuts_emitted}\n"
                f"except Exception:\n    pass")
        except Exception:
            continue

    code = "\n".join(rich_lines) + "\n"

    # Verify against source
    v = _verify_fit(solid, result, tol)

    # Emission integrity check: if we skipped many cuts from emission
    # (curved faces or arcs), the EMITTED recipe won't reproduce the
    # fit-time result. In that case, return code_body=None so another
    # fitter (voxel) takes over — the voxel fitter always emits a
    # faithful reconstruction.
    total_processed = cut_idx
    # If >20% of processed faces couldn't be emitted, fail closed.
    if total_processed > 0:
        n_non_emittable = n_skipped_curved + n_skipped_arcs
        emit_ratio = n_emittable / total_processed
        if emit_ratio < 0.8:
            return FitResult(
                None, v.completeness, v.accuracy, "none",
                f"face_extrude: fit OK but emission unsafe "
                f"({n_emittable}/{total_processed} faces emittable, "
                f"{n_skipped_curved} curved, {n_skipped_arcs} with arcs)")

    return FitResult(code, v.completeness, v.accuracy, "face_extrude",
                     f"Face-extrude reconstruction ({cut_idx} cuts, "
                     f"{n_cuts_emitted} emitted, "
                     f"comp={v.completeness*100:.3f}%, "
                     f"acc={v.accuracy*100:.3f}%)")


# ----------------------------------------------------------------------
# Voxel-based fallback (universal brute-force reconstruction)
# ----------------------------------------------------------------------


def try_fit_voxel(solid, tol: float = 0.01, resolution: int = 40) -> FitResult:
    """Voxelize the solid's bbox at the given resolution, classify each
    voxel IN/OUT vs source via OCCT BRepClass3d_SolidClassifier, then
    greedy-merge contiguous IN voxels into axis-aligned boxes and emit
    as union of Box() primitives.

    This ALWAYS produces a result (universal fallback). The emitted code
    is pure algebra: a sequence of Box(...) inside a BuildPart context.
    Accuracy improves with resolution but compute scales O(N^3).
    """
    try:
        import numpy as np
    except ImportError:
        return FitResult(None, 0.0, 0.0, "none", "voxel: numpy missing")

    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_IN
    from build123d import BuildPart, Box, Locations

    bb = solid.bounding_box()
    dx, dy, dz = bb.max.X - bb.min.X, bb.max.Y - bb.min.Y, bb.max.Z - bb.min.Z
    if min(dx, dy, dz) < 1e-6:
        return FitResult(None, 0.0, 0.0, "none", "voxel: degenerate bbox")

    # Adapt per-axis resolution to preserve aspect ratio
    max_dim = max(dx, dy, dz)
    Nx = max(6, int(resolution * dx / max_dim))
    Ny = max(6, int(resolution * dy / max_dim))
    Nz = max(6, int(resolution * dz / max_dim))
    vx, vy, vz = dx / Nx, dy / Ny, dz / Nz

    clsf = BRepClass3d_SolidClassifier(solid.wrapped)
    grid = np.zeros((Nx, Ny, Nz), dtype=bool)
    pt = gp_Pnt(0, 0, 0)
    bmin_x, bmin_y, bmin_z = bb.min.X, bb.min.Y, bb.min.Z
    for i in range(Nx):
        cx = bmin_x + (i + 0.5) * vx
        for j in range(Ny):
            cy = bmin_y + (j + 0.5) * vy
            for k in range(Nz):
                cz = bmin_z + (k + 0.5) * vz
                pt.SetCoord(cx, cy, cz)
                try:
                    clsf.Perform(pt, 1e-7)
                    if clsf.State() == TopAbs_IN:
                        grid[i, j, k] = True
                except Exception:
                    pass

    n_in = int(grid.sum())
    if n_in == 0:
        return FitResult(None, 0.0, 0.0, "none", "voxel: zero IN voxels")

    # Greedy axis-aligned box merging
    visited = np.zeros_like(grid)
    boxes = []
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if not grid[i, j, k] or visited[i, j, k]:
                    continue
                i1 = i
                while i1 + 1 < Nx and grid[i1 + 1, j, k] and not visited[i1 + 1, j, k]:
                    i1 += 1
                j1 = j
                while j1 + 1 < Ny:
                    if not grid[i:i1+1, j1+1, k].all() or visited[i:i1+1, j1+1, k].any():
                        break
                    j1 += 1
                k1 = k
                while k1 + 1 < Nz:
                    if not grid[i:i1+1, j:j1+1, k1+1].all() or visited[i:i1+1, j:j1+1, k1+1].any():
                        break
                    k1 += 1
                visited[i:i1+1, j:j1+1, k:k1+1] = True
                boxes.append((i, j, k, i1, j1, k1))

    if not boxes:
        return FitResult(None, 0.0, 0.0, "none", "voxel: no boxes produced")

    # Build the part + emit code simultaneously
    lines = ["with BuildPart() as _bp_vox:"]
    with BuildPart() as _bp_final:
        for (i0, j0, k0, i1, j1, k1) in boxes:
            sx = (i1 - i0 + 1) * vx
            sy = (j1 - j0 + 1) * vy
            sz = (k1 - k0 + 1) * vz
            cx = bmin_x + (i0 + (i1 - i0 + 1) / 2.0) * vx
            cy = bmin_y + (j0 + (j1 - j0 + 1) / 2.0) * vy
            cz = bmin_z + (k0 + (k1 - k0 + 1) / 2.0) * vz
            with Locations((cx, cy, cz)):
                Box(sx, sy, sz)
            lines.append(
                f"    with Locations(({_fmt(cx)}, {_fmt(cy)}, {_fmt(cz)})):")
            lines.append(f"        Box({_fmt(sx)}, {_fmt(sy)}, {_fmt(sz)})")
    part = _bp_final.part
    lines.append("_part = _bp_vox.part")
    code = "\n".join(lines) + "\n"

    v = _verify_fit(solid, part, tol)
    return FitResult(code, v.completeness, v.accuracy, "voxel",
                     f"Voxel reconstruction (N={Nx}x{Ny}x{Nz}, {len(boxes)} boxes, "
                     f"comp={v.completeness*100:.3f}%, "
                     f"acc={v.accuracy*100:.3f}%)")


# ----------------------------------------------------------------------
# Axis-stack fitter: decompose solid as stack of 2D profile extrusions.
# Works for any solid whose cross-sections along some axis are piecewise
# constant (or nearly so — sloped faces get subdivided). This is the
# parametric decomposition for connector housings, stepped pyramids,
# and any Z-stackable geometry.
# ----------------------------------------------------------------------


def _axis_find_boundaries(solid, axis_idx):
    from build123d import GeomType
    positions = set()
    for f in solid.faces():
        if f.area < 1e-9: continue
        try:
            if f.geom_type != GeomType.PLANE: continue
            c = f.center(); n = f.normal_at(c)
        except Exception: continue
        axis_vec = [0, 0, 0]; axis_vec[axis_idx] = 1
        dot = abs(n.X*axis_vec[0] + n.Y*axis_vec[1] + n.Z*axis_vec[2])
        if dot > 0.99:
            coord = [c.X, c.Y, c.Z][axis_idx]
            positions.add(round(coord, 4))
    return sorted(positions)


def _axis_slice(solid, axis_idx, pos, dp):
    from build123d import Plane, Keep
    origin_lo = [0, 0, 0]; origin_lo[axis_idx] = pos - dp
    origin_hi = [0, 0, 0]; origin_hi[axis_idx] = pos + dp
    z_dir = [0, 0, 0]; z_dir[axis_idx] = 1
    try:
        p_lo = Plane(origin=tuple(origin_lo), z_dir=tuple(z_dir))
        p_hi = Plane(origin=tuple(origin_hi), z_dir=tuple(z_dir))
        lower = solid.split(p_hi, keep=Keep.BOTTOM)
        if isinstance(lower, list):
            slab_parts = []
            for piece in lower:
                try:
                    sp = piece.split(p_lo, keep=Keep.TOP)
                    if isinstance(sp, list): slab_parts.extend(sp)
                    elif sp is not None: slab_parts.append(sp)
                except Exception: continue
            return slab_parts if slab_parts else None
        return lower.split(p_lo, keep=Keep.TOP)
    except Exception:
        return None


def _slab_volume(slab):
    if slab is None: return 0.0
    if isinstance(slab, list):
        total = 0.0
        for p in slab:
            try: total += p.volume
            except Exception: pass
        return total
    try: return slab.volume
    except Exception: pass
    try: return sum(x.volume for x in slab if hasattr(x, 'volume'))
    except Exception: return 0.0


def _area_at(solid, axis_idx, pos, dp=0.0005):
    slab = _axis_slice(solid, axis_idx, pos, dp)
    if slab is None: return 0.0
    return _slab_volume(slab) / (2 * dp)


def _is_layer_constant(solid, axis_idx, lo, hi, rel_tol=0.01, n_samples=5):
    if hi - lo < 1e-5: return True
    # Include near-endpoint samples so chamfers/bevels at layer boundaries
    # are detected. Without this, a layer like [-1.95, 2.05] where the
    # first 0.5mm is a chamfer gets flagged as constant because all 5
    # interior samples fall past the chamfer.
    samples = []
    h = hi - lo
    eps = min(h * 0.02, 0.01)  # small inset from boundary
    positions = [lo + eps, hi - eps]  # near-endpoints
    for i in range(n_samples):
        t = (i + 1) / (n_samples + 1)
        positions.append(lo + t * h)
    for pos in positions:
        a = _area_at(solid, axis_idx, pos)
        samples.append(a)
    if not samples: return True
    mean = sum(samples) / len(samples)
    if mean < 1e-9: return True
    return (max(samples) - min(samples)) / mean < rel_tol


def _subdivide_sloped_layer(solid, axis_idx, lo, hi, max_sub=16, rel_tol=0.01):
    for n in [2, 4, 8, 16, max_sub]:
        boundaries = [lo + i*(hi-lo)/n for i in range(n+1)]
        all_const = True
        for i in range(n):
            if not _is_layer_constant(solid, axis_idx, boundaries[i], boundaries[i+1],
                                      rel_tol=rel_tol, n_samples=3):
                all_const = False
                break
        if all_const:
            return boundaries[1:-1]
    return [lo + i*(hi-lo)/max_sub for i in range(1, max_sub)]


def _axis_chain_edges(wire):
    """Chain wire edges in traversal order. Returns list of (start_point, end_point, edge)."""
    raw = list(wire.edges())
    if not raw: return []
    def endpoints(e):
        vs = list(e.vertices())
        if len(vs) < 2: return None, None
        return ((vs[0].X, vs[0].Y, vs[0].Z), (vs[-1].X, vs[-1].Y, vs[-1].Z))
    def same(a, b, tol=1e-6):
        return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2]) < tol
    used = [False]*len(raw)
    used[0] = True
    s0, e0 = endpoints(raw[0])
    if s0 is None: return []
    ordered = [(s0, e0)]
    head = e0
    while True:
        if all(used): break
        found = False
        for i, e in enumerate(raw):
            if used[i]: continue
            s, ep = endpoints(e)
            if s is None: continue
            if same(s, head):
                ordered.append((s, ep)); head = ep; used[i] = True; found = True; break
            if same(ep, head):
                ordered.append((ep, s)); head = s; used[i] = True; found = True; break
        if not found: break
    return ordered


def _face_to_polygon(face, axis_idx):
    """Return (outer_desc, inner_descs) where each desc is a polygon description:
        ('poly', [(u,v), ...]) for straight-edged polygon
        ('circle', (cx, cy, r)) for circular face
        ('composite', [segments]) for mixed arcs+lines (future)

    The axis_idx is the stacking axis; u,v are coordinates in the
    plane perpendicular to axis_idx.
    """
    from build123d import GeomType
    perp = [i for i in range(3) if i != axis_idx]
    def to_uv(pt): return (pt[perp[0]], pt[perp[1]])

    ow = face.outer_wire()
    edges = list(ow.edges())
    outer = _describe_wire(ow, axis_idx, to_uv)

    inners = []
    try:
        for iw in face.inner_wires():
            inner = _describe_wire(iw, axis_idx, to_uv)
            if inner is not None:
                inners.append(inner)
    except Exception: pass
    return outer, inners


def _describe_wire(wire, axis_idx, to_uv):
    """Describe a closed wire as a polygon (possibly with arc-tessellation).

    Returns:
      ('poly', [(u,v), ...]) if wire can be described as a polygon
      ('circle', (cx, cy, r)) if wire is a single circle
      None if unsupported.

    Arcs (CIRCLE, BSPLINE, HYPERBOLA, ELLIPSE) are tessellated into short
    polyline segments. Tessellation density: 16 segments per arc, bounded
    by 0.005-unit max chord length.
    """
    from build123d import GeomType
    import math
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Circle

    edges = list(wire.edges())
    if not edges: return None

    # Single-circle case (or co-circular arcs): emit as 'circle'
    if len(edges) <= 4:
        try:
            circle_params = []
            all_circles = True
            for e in edges:
                adap = BRepAdaptor_Curve(e.wrapped)
                if adap.GetType() != GeomAbs_Circle:
                    all_circles = False; break
                circ = adap.Circle()
                loc = circ.Location()
                r = circ.Radius()
                circle_params.append((loc.X(), loc.Y(), loc.Z(), r))
            if all_circles and circle_params:
                x0, y0, z0, r0 = circle_params[0]
                all_same = all(abs(p[0]-x0) < 1e-5 and abs(p[1]-y0) < 1e-5
                               and abs(p[2]-z0) < 1e-5 and abs(p[3]-r0) < 1e-5
                               for p in circle_params)
                if all_same:
                    c_3d = (x0, y0, z0)
                    cu, cv = to_uv(c_3d)
                    return ('circle', (cu, cv, r0))
        except Exception: pass

    # Polygon traversal with arc tessellation.
    # First chain edges in topological order.
    chain = _axis_chain_edges(wire)
    if not chain or len(chain) < 2:
        # Fall through to all-line case as last resort
        all_line = all(
            (getattr(e, 'geom_type', None) == GeomType.LINE)
            for e in edges
        )
        if not all_line:
            return None
        chain = _axis_chain_edges(wire)
        if not chain: return None
        outer = [to_uv(seg[0]) for seg in chain]
        return ('poly', outer) if len(outer) >= 3 else None

    # Walk the chain, emitting polyline points. For line edges, just
    # emit the start endpoint. For arcs/splines, tessellate between
    # the edge's start and end with N intermediate samples from the curve.
    polyline = []
    # We need to know which edge corresponds to each chain segment.
    # chain_edges returns (start_pt, end_pt) pairs but doesn't track the
    # edge reference — let me reconstruct by matching endpoints.
    edge_lookup = list(edges)
    def find_edge(s, e, tol=1e-5):
        for eg in edge_lookup:
            vs = list(eg.vertices())
            if len(vs) < 2: continue
            v0 = (vs[0].X, vs[0].Y, vs[0].Z)
            v1 = (vs[-1].X, vs[-1].Y, vs[-1].Z)
            d1 = abs(v0[0]-s[0])+abs(v0[1]-s[1])+abs(v0[2]-s[2]) + \
                 abs(v1[0]-e[0])+abs(v1[1]-e[1])+abs(v1[2]-e[2])
            d2 = abs(v1[0]-s[0])+abs(v1[1]-s[1])+abs(v1[2]-s[2]) + \
                 abs(v0[0]-e[0])+abs(v0[1]-e[1])+abs(v0[2]-e[2])
            if d1 < tol: return eg, False
            if d2 < tol: return eg, True
        return None, False

    N_TESS = 12  # number of intermediate samples per curved edge
    for seg_i, seg in enumerate(chain):
        s_pt, e_pt = seg[0], seg[1]
        eg, reversed_ = find_edge(s_pt, e_pt)
        polyline.append(to_uv(s_pt))
        if eg is None:
            continue
        try:
            gt = eg.geom_type
        except Exception:
            continue
        if gt == GeomType.LINE:
            # No intermediate points
            continue
        # Curved edge: tessellate between s and e using parameter sampling
        try:
            from OCP.BRepAdaptor import BRepAdaptor_Curve
            adap = BRepAdaptor_Curve(eg.wrapped)
            u0 = adap.FirstParameter()
            u1 = adap.LastParameter()
            # Walk from u0 to u1 (or reverse), emitting intermediates
            for k in range(1, N_TESS):
                t = k / N_TESS
                u = u0 + (u1 - u0) * t
                pnt = adap.Value(u)
                pt3d = (pnt.X(), pnt.Y(), pnt.Z())
                polyline.append(to_uv(pt3d))
            # The end point will be added as the start of next segment
        except Exception:
            continue

    if len(polyline) < 3:
        return None
    return ('poly', polyline)


def _slab_polygons(slab, axis_idx):
    """Extract 2D polygons from all bottom cap faces of slab. Returns list of (outer, inners)."""
    from build123d import GeomType
    if slab is None: return []
    if isinstance(slab, list):
        cap_faces = []
        for s in slab:
            try:
                for f in s.faces():
                    if f.area < 1e-9: continue
                    try:
                        if f.geom_type != GeomType.PLANE: continue
                        c = f.center(); n = f.normal_at(c)
                    except Exception: continue
                    axis_vec = [0,0,0]; axis_vec[axis_idx] = 1
                    dot = abs(n.X*axis_vec[0] + n.Y*axis_vec[1] + n.Z*axis_vec[2])
                    if dot > 0.99:
                        cap_faces.append((f, c))
            except Exception: pass
    else:
        cap_faces = []
        try: all_faces = list(slab.faces())
        except Exception: return []
        for f in all_faces:
            if f.area < 1e-9: continue
            try:
                if f.geom_type != GeomType.PLANE: continue
                c = f.center(); n = f.normal_at(c)
            except Exception: continue
            axis_vec = [0,0,0]; axis_vec[axis_idx] = 1
            dot = abs(n.X*axis_vec[0] + n.Y*axis_vec[1] + n.Z*axis_vec[2])
            if dot > 0.99:
                cap_faces.append((f, c))
    if not cap_faces: return []
    cap_faces.sort(key=lambda t: [t[1].X, t[1].Y, t[1].Z][axis_idx])
    min_coord = [cap_faces[0][1].X, cap_faces[0][1].Y, cap_faces[0][1].Z][axis_idx]
    bottom = [t for t in cap_faces
              if abs([t[1].X, t[1].Y, t[1].Z][axis_idx] - min_coord) < 1e-4]
    raw_polys = []
    for f, _ in bottom:
        outer, inners = _face_to_polygon(f, axis_idx)
        if outer is not None:
            raw_polys.append((outer, inners))

    if len(raw_polys) <= 1:
        return raw_polys

    # Concentric-containment merging: when the source face has an outer
    # boundary AND inner boundary (annular cross-section), OCCT may report
    # them as two separate faces at the same Z-coord. We need to detect
    # this and merge the inner one as a HOLE in the outer.
    #
    # Rule: polygon A contains polygon B if A's 2D bbox strictly contains
    # B's 2D bbox AND A's area > B's area. Then B is a hole in A.
    def _poly_bbox_and_area(desc):
        """Return (xmin, xmax, ymin, ymax, area) for a poly descriptor."""
        if desc is None: return None
        kind, data = desc[0], desc[1]
        if kind == 'poly':
            pts = data
            if len(pts) < 3: return None
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            # shoelace
            a = 0.0
            for i in range(len(pts)):
                x1,y1 = pts[i]; x2,y2 = pts[(i+1)%len(pts)]
                a += x1*y2 - x2*y1
            return (min(xs), max(xs), min(ys), max(ys), abs(a)/2)
        if kind == 'circle':
            cx, cy, r = data
            import math
            return (cx-r, cx+r, cy-r, cy+r, math.pi*r*r)
        return None

    # Compute bbox+area for each outer polygon
    polys_info = []
    for outer, inners in raw_polys:
        bi = _poly_bbox_and_area(outer)
        polys_info.append((outer, inners, bi))

    # Sort by area descending; largest is most likely the outer shell
    polys_info.sort(key=lambda t: -(t[2][4] if t[2] else 0))

    # For each small polygon, find if it's contained in a larger one.
    # If so, attach it as an inner wire to the larger.
    claimed = [False] * len(polys_info)
    for i in range(len(polys_info)):
        if claimed[i]: continue
        outer_i, inners_i, bi = polys_info[i]
        if bi is None: continue
        xmin_i, xmax_i, ymin_i, ymax_i, area_i = bi
        for j in range(i+1, len(polys_info)):
            if claimed[j]: continue
            outer_j, inners_j, bj = polys_info[j]
            if bj is None: continue
            xmin_j, xmax_j, ymin_j, ymax_j, area_j = bj
            # j is strictly inside i's bbox with smaller area?
            if (xmin_j > xmin_i + 1e-6 and xmax_j < xmax_i - 1e-6 and
                ymin_j > ymin_i + 1e-6 and ymax_j < ymax_i - 1e-6 and
                area_j < area_i * 0.999):
                # Attach j as hole in i
                polys_info[i][1].append(outer_j)
                # Any inner wires of j become... dropped (rare edge case).
                claimed[j] = True

    # Return only unclaimed (top-level) polygons
    merged = [(outer, inners) for idx, (outer, inners, _) in enumerate(polys_info)
              if not claimed[idx]]
    return merged


def _pair_polygons(polys_lo, polys_hi):
    """Pair polygons between bottom and top cross-sections for lofting.

    Returns list of ((outer_lo, inners_lo), (outer_hi, inners_hi)) pairs,
    or None if pairing can't be done unambiguously. Pairs by bbox-center
    proximity and requires matching kind (poly with same point count, or
    circle).
    """
    if len(polys_lo) != len(polys_hi): return None

    def _desc_info(desc):
        kind = desc[0]
        if kind == 'poly':
            pts = desc[1]
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            return (kind, len(pts),
                    (sum(xs)/len(xs), sum(ys)/len(ys)))
        else:
            cx, cy, _r = desc[1]
            return (kind, 0, (cx, cy))

    lo_infos = [(_desc_info(o), i) for i, (o, _) in enumerate(polys_lo)]
    hi_infos = [(_desc_info(o), j) for j, (o, _) in enumerate(polys_hi)]

    # Greedy nearest-match (could use Hungarian, but n<=8 typically)
    pairs = []
    used_hi = set()
    for (info_lo, i) in lo_infos:
        kind_lo, nverts_lo, cen_lo = info_lo
        best_j = -1
        best_d = 1e9
        for (info_hi, j) in hi_infos:
            if j in used_hi: continue
            kind_hi, nverts_hi, cen_hi = info_hi
            if kind_lo != kind_hi: continue
            if kind_lo == 'poly' and nverts_lo != nverts_hi: continue
            d = ((cen_lo[0]-cen_hi[0])**2 + (cen_lo[1]-cen_hi[1])**2) ** 0.5
            if d < best_d:
                best_d = d; best_j = j
        if best_j < 0: return None
        used_hi.add(best_j)
        pairs.append((polys_lo[i], polys_hi[best_j]))
    return pairs


def _polys_topology_signature(polys):
    """Return a hashable signature describing the topology of a poly list.

    Two cross-sections can be lofted if they have the same signature:
    same number of polygons, each with matching kind (poly/circle) and
    matching point count (for polys). Radii and vertex coordinates can
    differ; topology must match.
    """
    sig = []
    for outer, inners in polys:
        if outer is None:
            sig.append(None)
            continue
        kind = outer[0]
        if kind == 'poly':
            outer_part = ('poly', len(outer[1]))
        else:
            outer_part = ('circle',)
        inner_parts = []
        for inn in inners:
            if inn is None: continue
            if inn[0] == 'poly':
                inner_parts.append(('poly', len(inn[1])))
            else:
                inner_parts.append(('circle',))
        sig.append((outer_part, tuple(sorted(inner_parts))))
    return tuple(sorted(sig, key=lambda x: str(x)))


def _build_sketch_from_polys(plane, polys):
    """Build a BuildSketch on `plane` from a polys list.

    Returns the sketch context object (accessed via `.sketch`) or None
    if the sketch is empty.
    """
    from build123d import (BuildSketch, BuildLine, Line, Mode, make_face,
                           Locations, Circle)
    try:
        with BuildSketch(plane) as sk:
            for outer, inners in polys:
                if outer is None: continue
                if outer[0] == 'poly':
                    pts = outer[1]
                    if len(pts) < 3: continue
                    with BuildLine() as bl:
                        for j in range(len(pts)):
                            Line(pts[j], pts[(j+1) % len(pts)])
                    try: make_face()
                    except Exception: pass
                elif outer[0] == 'circle':
                    cu, cv, r = outer[1]
                    with Locations((cu, cv)):
                        Circle(r)
                for inner in inners:
                    if inner is None: continue
                    if inner[0] == 'poly':
                        ipts = inner[1]
                        if len(ipts) < 3: continue
                        with BuildLine(mode=Mode.SUBTRACT) as bli:
                            for j in range(len(ipts)):
                                Line(ipts[j], ipts[(j+1) % len(ipts)])
                        try: make_face(mode=Mode.SUBTRACT)
                        except Exception: pass
                    elif inner[0] == 'circle':
                        icu, icv, ir = inner[1]
                        with Locations((icu, icv)):
                            Circle(ir, mode=Mode.SUBTRACT)
        return sk
    except Exception:
        return None


def _axis_stack_build(solid, axis_idx, use_loft=False):
    """Return (part, list_of_layer_descriptors).
    Each layer descriptor: (lo, hi, polys) where polys is list of (outer, inners).

    If use_loft=True, for layers where bottom and top cross-sections have
    matching topology (same count of polygons + circles), use loft() between
    them instead of flat extrude. This captures tapered/frustum geometry
    that flat extrude cannot represent.
    """
    from build123d import (BuildSketch, BuildLine, Line, Plane, Mode,
                           make_face, extrude, loft, Locations, Circle, GeomType)
    bb = solid.bounding_box()
    bb_lo = [bb.min.X, bb.min.Y, bb.min.Z][axis_idx]
    bb_hi = [bb.max.X, bb.max.Y, bb.max.Z][axis_idx]

    # Compute curved-face area ratio. Tapered/curved solids (like XT30
    # pin barrels at 2.18% sym-diff with rel_tol=0.02) need finer
    # subdivision to capture the taper. Planar solids don't, and
    # tightening tol for them wastes compute.
    total_area = 0.0
    curved_area = 0.0
    for f in solid.faces():
        if f.area < 1e-9: continue
        total_area += f.area
        try:
            if f.geom_type != GeomType.PLANE:
                curved_area += f.area
        except Exception: pass
    curved_ratio = curved_area / total_area if total_area > 0 else 0.0
    # For loft mode, rely on loft to capture smooth tapers within a layer.
    # We only need to subdivide at SUDDEN topology transitions (annular →
    # solid, different polygon counts) which manifest as big area jumps.
    # rel_tol=0.02 with max_sub=8 catches those without over-subdividing.
    if use_loft:
        rel_tol = 0.02
        max_sub = 4
    elif curved_ratio > 0.30:
        rel_tol = 0.005
        max_sub = 64
    else:
        rel_tol = 0.02
        max_sub = 8

    # Build boundary list: all detected face positions + bbox extremes.
    # Round to 4 decimals to dedupe floating-point near-duplicates like
    # -7.60000006 and -7.6.
    raw = _axis_find_boundaries(solid, axis_idx) + [bb_lo, bb_hi]
    zb = sorted(set(round(z, 4) for z in raw))
    zb = [z for z in zb if bb_lo - 1e-4 <= z <= bb_hi + 1e-4]
    if len(zb) < 2:
        return None, None

    refined = [zb[0]]
    for i in range(len(zb)-1):
        lo, hi = zb[i], zb[i+1]
        if hi - lo < 1e-4: continue
        if not _is_layer_constant(solid, axis_idx, lo, hi, rel_tol=rel_tol):
            extras = _subdivide_sloped_layer(solid, axis_idx, lo, hi,
                                              max_sub=max_sub, rel_tol=rel_tol)
            for e in extras:
                refined.append(e)
        refined.append(hi)
    refined = sorted(set(round(z, 6) for z in refined))

    # Hard cap on number of layers to prevent OOM. Z-stackable housings
    # fit in ~15 layers; tapered cylinder barrels may need 30-50. Abort
    # if an axis decomposition blows up: that axis is wrong.
    MAX_LAYERS = 80 if curved_ratio > 0.30 else 60
    if len(refined) > MAX_LAYERS + 1:
        return None, None

    layer_descs = []
    parts = []
    for i in range(len(refined)-1):
        lo, hi = refined[i], refined[i+1]
        if hi - lo < 1e-5: continue
        h = hi - lo
        # Construct sketch plane at `lo` perpendicular to the stacking axis.
        # We want the plane's z_dir to point toward increasing axis coordinate
        # so that extrude(amount=+h) fills the layer correctly.
        if axis_idx == 0:
            plane_lo = Plane(origin=(lo, 0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0))
            plane_hi = Plane(origin=(hi, 0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0))
        elif axis_idx == 1:
            plane_lo = Plane(origin=(0, lo, 0), x_dir=(1, 0, 0), z_dir=(0, 1, 0))
            plane_hi = Plane(origin=(0, hi, 0), x_dir=(1, 0, 0), z_dir=(0, 1, 0))
        else:
            plane_lo = Plane(origin=(0, 0, lo), x_dir=(1, 0, 0), z_dir=(0, 0, 1))
            plane_hi = Plane(origin=(0, 0, hi), x_dir=(1, 0, 0), z_dir=(0, 0, 1))

        # Sample cross-sections. In loft mode, sample at BOTH ends (slightly
        # inset from the layer boundaries to avoid coplanar face issues).
        # In extrude mode, one mid-layer sample suffices.
        mid = (lo + hi) / 2
        eps = h * 0.05  # 5% inset from boundaries

        if use_loft:
            slab_lo = _axis_slice(solid, axis_idx, lo + eps, eps * 0.8)
            polys_lo = _slab_polygons(slab_lo, axis_idx) if slab_lo else []
            slab_hi = _axis_slice(solid, axis_idx, hi - eps, eps * 0.8)
            polys_hi = _slab_polygons(slab_hi, axis_idx) if slab_hi else []

            # Loft requires matching topology. If mismatch, fall back
            # to flat extrude using mid-layer polys.
            sig_lo = _polys_topology_signature(polys_lo) if polys_lo else None
            sig_hi = _polys_topology_signature(polys_hi) if polys_hi else None

            if polys_lo and polys_hi and sig_lo == sig_hi:
                # Loft only works on ONE connected profile per section.
                # If cross-section has multiple disconnected polygons (e.g.
                # XT30 barrel's 4-wedge cross-section), we must pair them
                # individually and loft each pair, then union.
                #
                # Pairing strategy: match polygons between lo and hi by
                # bbox-center proximity. For each pair with matching kind,
                # loft between their individual sketches.
                matched_pairs = _pair_polygons(polys_lo, polys_hi)
                if matched_pairs is not None:
                    loft_ok = True
                    pair_parts = []
                    for (outer_lo, inners_lo), (outer_hi, inners_hi) in matched_pairs:
                        sk_lo = _build_sketch_from_polys(
                            plane_lo, [(outer_lo, inners_lo)])
                        sk_hi = _build_sketch_from_polys(
                            plane_hi, [(outer_hi, inners_hi)])
                        if sk_lo is None or sk_hi is None:
                            loft_ok = False; break
                        try:
                            sl, sh = sk_lo.sketch, sk_hi.sketch
                        except Exception:
                            loft_ok = False; break
                        if sl is None or sh is None:
                            loft_ok = False; break
                        try:
                            lp = loft([sl, sh])
                            pair_parts.append(lp)
                        except Exception:
                            loft_ok = False; break
                    if loft_ok and pair_parts:
                        acc_layer = pair_parts[0]
                        for pp in pair_parts[1:]:
                            try: acc_layer = acc_layer + pp
                            except Exception: pass
                        parts.append(acc_layer)
                        layer_descs.append((lo, hi, polys_lo, polys_hi, 'loft'))
                        continue
                # fall through to flat extrude if loft pairing fails

        # Flat extrude path (default, or loft fallback)
        slab = _axis_slice(solid, axis_idx, mid, h * 0.3)
        polys = _slab_polygons(slab, axis_idx) if slab else []
        if not polys: continue
        sk = _build_sketch_from_polys(plane_lo, polys)
        if sk is None:
            continue
        try:
            skv = sk.sketch
        except Exception:
            continue
        if skv is None:
            continue  # empty sketch, skip extrude
        try:
            layer_part = extrude(skv, amount=h)
            parts.append(layer_part)
            layer_descs.append((lo, hi, polys, None, 'extrude'))
        except Exception:
            continue

    if not parts: return None, None
    acc = parts[0]
    for p in parts[1:]:
        try: acc = acc + p
        except Exception: pass
    # Normalize to a single Part/Compound with a volume attribute
    if not hasattr(acc, 'volume'):
        try:
            items = list(acc)
            if items:
                acc2 = items[0]
                for it in items[1:]:
                    try: acc2 = acc2 + it
                    except Exception: pass
                acc = acc2
        except Exception:
            pass
    return acc, layer_descs


def try_fit_axis_stack(solid, tol: float = 0.01, use_loft: bool = False) -> FitResult:
    """Try reconstructing solid as a stack of 2D-profile extrudes along
    X, Y, or Z axis. Pick the axis that gives the best fit.

    This is pure parametric Python algebra: each layer is a BuildSketch
    with BuildLine polygons, extruded to a height. Layers union together.

    Works on any solid where some axis has constant cross-section (after
    sloped-layer subdivision). For XT30 pin barrels: Y-axis works since
    the pin is a rod along Y. For S3B housing: Z-axis works. For a
    fully free-form solid, no axis will produce valid layers and this
    fitter returns None.

    If use_loft=True, layers with matching bottom/top topology use
    loft() instead of flat extrude, capturing tapered geometry (XT30
    barrel frustum sections).
    """
    best = None
    best_q = -1
    best_axis = -1
    for axis in [2, 1, 0]:  # try Z first (most common)
        try:
            part, layer_descs = _axis_stack_build(solid, axis, use_loft=use_loft)
        except Exception:
            continue
        if part is None: continue
        try:
            v = _verify_fit(solid, part, tol)
        except Exception:
            continue
        q = (v.completeness + v.accuracy) / 2.0
        if q > best_q:
            best = (part, layer_descs, v)
            best_q = q
            best_axis = axis
        # Short-circuit: if we have a reasonable result, stop trying more
        # axes. Each axis costs 10-30s for complex solids; checking all 3
        # on a 7-solid part like XT30 drives total time above limits.
        # Threshold 0.98 = avg comp+acc of 98%, ~4% sym-diff. Good parts
        # usually do much better on the first (Z) axis anyway.
        if best_q >= 0.98:
            break

    if best is None:
        return FitResult(None, 0.0, 0.0, "none", "axis_stack: no valid axis")
    part, layer_descs, v = best

    # Emit code
    axis_name = ['X', 'Y', 'Z'][best_axis]
    lines = [f"# axis-stack reconstruction along {axis_name}-axis ({len(layer_descs)} layers)"]
    parts_list_var = "_layers"
    # Emit an explicit plane per layer that matches the build-time plane.
    # This ensures emit-time reconstruction matches the fit-time result.
    def _plane_expr(axis_idx, pos):
        if axis_idx == 0:
            return f"Plane(origin=({_fmt(pos)}, 0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0))"
        elif axis_idx == 1:
            return f"Plane(origin=(0, {_fmt(pos)}, 0), x_dir=(1, 0, 0), z_dir=(0, 1, 0))"
        else:
            return f"Plane(origin=(0, 0, {_fmt(pos)}), x_dir=(1, 0, 0), z_dir=(0, 0, 1))"

    def _emit_sketch_body(polys, li, suffix):
        """Emit the BuildSketch body lines for a given polys list."""
        out = []
        for pi, (outer, inners) in enumerate(polys):
            if outer is None: continue
            if outer[0] == 'poly':
                pts = outer[1]
                # Guard the outer face build. If the outer wire is
                # degenerate, skip this shape (no face means no inners
                # to subtract from).
                out.append(f"    try:")
                out.append(f"        with BuildLine() as _bl_{li}_{suffix}_{pi}:")
                for j in range(len(pts)):
                    p1 = pts[j]; p2 = pts[(j+1) % len(pts)]
                    out.append(f"            Line(({_fmt(p1[0])}, {_fmt(p1[1])}), "
                               f"({_fmt(p2[0])}, {_fmt(p2[1])}))")
                out.append(f"        make_face()")
                out.append(f"    except Exception:")
                out.append(f"        pass  # skipped degenerate outer {li}.{pi}")
            elif outer[0] == 'circle':
                cu, cv, r = outer[1]
                out.append(f"    try:")
                out.append(f"        with Locations(({_fmt(cu)}, {_fmt(cv)})):")
                out.append(f"            Circle({_fmt(r)})")
                out.append(f"    except Exception:")
                out.append(f"        pass  # skipped degenerate circle {li}.{pi}")
            for ii, inner in enumerate(inners):
                if inner is None: continue
                if inner[0] == 'poly':
                    ipts = inner[1]
                    # Guard: some inner polygons come back from OCCT
                    # with collinear/duplicate points or micro
                    # self-intersections that make_face cannot close
                    # into a single TopoDS_Face. Rather than killing
                    # the whole recipe, skip the bad hole and continue.
                    out.append(f"    try:")
                    out.append(f"        with BuildLine(mode=Mode.SUBTRACT) as _bli_{li}_{suffix}_{pi}_{ii}:")
                    for j in range(len(ipts)):
                        p1 = ipts[j]; p2 = ipts[(j+1) % len(ipts)]
                        out.append(f"            Line(({_fmt(p1[0])}, {_fmt(p1[1])}), "
                                   f"({_fmt(p2[0])}, {_fmt(p2[1])}))")
                    out.append(f"        make_face(mode=Mode.SUBTRACT)")
                    out.append(f"    except Exception:")
                    out.append(f"        pass  # skipped degenerate inner hole {li}.{pi}.{ii}")
                elif inner[0] == 'circle':
                    icu, icv, ir = inner[1]
                    out.append(f"    try:")
                    out.append(f"        with Locations(({_fmt(icu)}, {_fmt(icv)})):")
                    out.append(f"            Circle({_fmt(ir)}, mode=Mode.SUBTRACT)")
                    out.append(f"    except Exception:")
                    out.append(f"        pass  # skipped degenerate inner circle {li}.{pi}.{ii}")
        return out

    lines.append(f"{parts_list_var} = []")
    for li, layer in enumerate(layer_descs):
        lo, hi, polys_lo, polys_hi, mode = layer
        h = hi - lo
        lines.append(f"# layer {li}: [{lo:.4f}, {hi:.4f}] h={h:.4f} ({mode})")
        if mode == 'loft':
            # Pair polygons between bottom and top; emit one loft per pair,
            # then union into a single layer part.
            pairs = _pair_polygons(polys_lo, polys_hi) or []
            lines.append(f"_layer_parts_{li} = []")
            for pi, ((outer_a, inners_a), (outer_b, inners_b)) in enumerate(pairs):
                lines.append(f"with BuildSketch({_plane_expr(best_axis, lo)}) as _sk_{li}_a_{pi}:")
                lines.extend(_emit_sketch_body(
                    [(outer_a, inners_a)], li, f'a{pi}'))
                lines.append(f"with BuildSketch({_plane_expr(best_axis, hi)}) as _sk_{li}_b_{pi}:")
                lines.extend(_emit_sketch_body(
                    [(outer_b, inners_b)], li, f'b{pi}'))
                lines.append(f"try:")
                lines.append(
                    f"    _layer_parts_{li}.append("
                    f"loft([_sk_{li}_a_{pi}.sketch, _sk_{li}_b_{pi}.sketch]))")
                lines.append(f"except Exception:")
                lines.append(f"    pass  # skipped degenerate loft layer {li}.{pi}")
            lines.append(f"if _layer_parts_{li}:")
            lines.append(f"    _layer_{li} = _layer_parts_{li}[0]")
            lines.append(f"    for _lp in _layer_parts_{li}[1:]:")
            lines.append(f"        try: _layer_{li} = _layer_{li} + _lp")
            lines.append(f"        except Exception: pass")
            lines.append(f"    _layers.append(_layer_{li})")
        else:
            lines.append(f"with BuildSketch({_plane_expr(best_axis, lo)}) as _sk_{li}:")
            lines.extend(_emit_sketch_body(polys_lo, li, 'e'))
            lines.append(f"try:")
            lines.append(f"    _layers.append(extrude(_sk_{li}.sketch, amount={_fmt(h)}))")
            lines.append(f"except Exception:")
            lines.append(f"    pass  # skipped empty/degenerate layer {li}")
    lines.append("if not _layers:")
    lines.append("    # All layers degenerated out; emit an empty Box to avoid")
    lines.append("    # recipe crash. Volume will be ~0 and this solid will")
    lines.append("    # show up as effectively missing in the final assembly.")
    lines.append("    _part = Box(1e-6, 1e-6, 1e-6)")
    lines.append("else:")
    lines.append("    _part = _layers[0]")
    lines.append("    for _l in _layers[1:]:")
    lines.append("        try: _part = _part + _l")
    lines.append("        except Exception: pass")
    code = "\n".join(lines) + "\n"

    return FitResult(code, v.completeness, v.accuracy, "axis_stack",
                     f"Axis-stack along {axis_name} ({len(layer_descs)} layers, "
                     f"comp={v.completeness*100:.3f}%, acc={v.accuracy*100:.3f}%)")


def fit_primitive(solid, tol: float = 0.01) -> FitResult:
    """Try each fitter in order; return the first success.

    `tol`: nominal acceptable fraction of sym-diff per primitive.
    0.01 = 99% completeness+accuracy required. Per-fitter tolerance is
    tuned individually: Box and Cylinder use `tol` strictly because
    they're face-count-strict and a wrong fit would be very wrong.
    Extrude and BoxFillets are allowed 2*tol because they operate on
    profile polygons and radius estimates where sub-percent precision
    loss is acceptable.

    On failure, the returned FitResult.details contains a summary of
    why each fitter rejected the solid. This lets verbose-mode output
    show rejection reasons per solid, which is essential for adding
    new fitters to cover previously-unfitted shape classes.
    """
    loose = tol * 2.0

    # Box first: cheapest, most common. Strict tolerance.
    r = try_fit_box(solid, tol)
    if r.code_body is not None:
        return r
    box_reason = r.details

    # Box-with-fillets: handles rounded-corner cubes. Looser tolerance
    # because fillet radius estimation is approximate.
    r = try_fit_box_with_fillets(solid, loose)
    if r.code_body is not None:
        return r
    fillets_reason = r.details

    # Cylinder: strict tolerance.
    r = try_fit_cylinder(solid, tol)
    if r.code_body is not None:
        return r
    cyl_reason = r.details

    # Prismatic extrude: loose tolerance. Handles L-shapes, polygons,
    # filleted profiles. Small per-vertex precision errors in the
    # polyline are normal.
    r = try_fit_extrude(solid, loose)
    extrude_result = r if r.code_body is not None else None
    ext_reason = r.details
    # In strict (non-force) mode, accept extrude immediately if it succeeds.
    # In force-primitives mode, defer the decision until we also try
    # halfspace_hull, because extrude can return a low-quality fit
    # (especially for non-convex bent leads) that halfspace_hull beats.
    if extrude_result is not None and not _FORCE_PRIMITIVES:
        return extrude_result

    # Summarize every rejection so --verbose can show why the solid
    # didn't fit any primitive.
    try:
        nfaces = len(list(solid.faces()))
    except Exception:
        nfaces = -1
    details = (
        f"no fit ({nfaces} faces): "
        f"Box[{box_reason}]; "
        f"BoxFillets[{fillets_reason}]; "
        f"Cylinder[{cyl_reason}]; "
        f"Extrude[{ext_reason}]"
    )

    # Force-primitives path: run approximators in order of cost,
    # short-circuit as soon as we have a >99.5% candidate.
    if _FORCE_PRIMITIVES:
        import gc, time as _time
        def _quality(fr):
            if fr is None or fr.code_body is None: return -1
            return (fr.completeness + fr.accuracy) / 2.0

        def _run_tier(name, call):
            t0 = _time.time()
            if _VERBOSE:
                print(f"      [tier] {name} starting...", flush=True)
            try:
                r = call()
            except Exception as e:
                if _VERBOSE:
                    print(f"      [tier] {name} exc "
                          f"({_time.time()-t0:.1f}s): "
                          f"{type(e).__name__}", flush=True)
                return None
            dt = _time.time() - t0
            if _VERBOSE:
                if r and r.code_body:
                    print(f"      [tier] {name} done: "
                          f"c={r.completeness*100:.2f}% "
                          f"a={r.accuracy*100:.2f}% ({dt:.1f}s)",
                          flush=True)
                else:
                    detail = r.details[:50] if r and r.details else ""
                    print(f"      [tier] {name} fail ({dt:.1f}s) "
                          f"{detail}", flush=True)
            return r

        # Tier 1 (cheap): halfspace_hull, existing extrude result
        hull_r = _run_tier("halfspace_hull",
                            lambda: try_fit_halfspace_hull(solid, tol))
        best = None
        best_q = -1
        for cand in (hull_r, extrude_result):
            q = _quality(cand)
            if q > best_q:
                best = cand
                best_q = q
        gc.collect()

        # Tier 1.5: axis-stack (for Z-stackable housings).
        if best_q < 0.995:
            as_r = _run_tier("axis_stack_extrude",
                              lambda: try_fit_axis_stack(solid, tol))
            q = _quality(as_r)
            if q > best_q:
                best = as_r
                best_q = q
            gc.collect()

        # Tier 1.75: axis-stack with LOFT — for tapered/frustum solids.
        if best_q < 0.99:
            as_loft_r = _run_tier(
                "axis_stack_loft",
                lambda: try_fit_axis_stack(solid, tol, use_loft=True))
            q = _quality(as_loft_r)
            if q > best_q:
                best = as_loft_r
                best_q = q
            gc.collect()

        # Tier 2: face_extrude — only if tier 1 is <99.5%
        if best_q < 0.995:
            fe_r = _run_tier("face_extrude",
                              lambda: try_fit_face_extrude(solid, tol))
            q = _quality(fe_r)
            if q > best_q:
                best = fe_r
                best_q = q
            gc.collect()

        # Tier 3 (slow): LLM refinement
        if best_q < 0.99:
            def _call_llm():
                from .llm_fitter import try_fit_llm
                return try_fit_llm(solid, tol)
            llm_r = _run_tier("llm_repair", _call_llm)
            llm_q = _quality(llm_r)
            if llm_q > best_q:
                best = llm_r
                best_q = llm_q
            gc.collect()

        if best is not None and best.code_body is not None:
            return best
        if hull_r is not None and hull_r.code_body is not None:
            return hull_r
        details = f"{details}; HalfspaceHull[{hull_r.details if hull_r else 'none'}]"

    # No parametric fit succeeded. Return no-fit honestly instead of an
    # approximate bbox — per reviewer guidance, we never ship approximate
    # geometry. The recipe will be missing this solid; validation will
    # fail on it, which is the correct signal.
    return FitResult(None, 0.0, 0.0, "none", details)


def _fmt(x: float) -> str:
    return f"{x:.9g}"
