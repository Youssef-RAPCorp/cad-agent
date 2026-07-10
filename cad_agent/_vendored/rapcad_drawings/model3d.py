"""
3D model import + 2D orthographic projection.

Loads a triangulated 3D model (STL, OBJ, PLY, OFF, GLB) and produces a
list of 2D line segments suitable for drawing in modelspace. The
output is the silhouette + sharp feature edges projected onto the chosen
view plane.

Pipeline:

  1. Load the mesh via trimesh.
  2. Compute, per unique edge:
       * `is_silhouette` — the two adjacent faces straddle the view
         direction (one points toward the camera, the other away).
       * `is_feature`    — the dihedral angle between the faces exceeds
         `angle_threshold_deg` (default 30°), i.e. a "sharp" edge a
         human would draw on a technical sketch.
       * `is_boundary`   — the edge belongs to only one face (open mesh
         boundaries are always drawn).
  3. For every edge that passes any of those tests, project both
     vertices onto the view plane.
  4. Return as a list of `(p0, p1)` 2D tuples in modelspace units.

Conventions (third-angle projection, ASME Y14.3):

  +Z = up, +Y = forward, +X = right
  FRONT  view: camera at -Y, project (x, z)
  TOP    view: camera above (+Z), project (x, y)
  RIGHT  view: camera at +X, project (y, z)
  LEFT   view: camera at -X, project (-y, z)
  BACK   view: camera at +Y, project (-x, z)
  BOTTOM view: camera below (-Z), project (x, -y)
  ISO    view: standard isometric (30° down, 30° around Z)

The output is centered on (0, 0) by default; callers translate to the
desired modelspace origin.

Hidden-line removal is implemented with a rasterized depth buffer:
every candidate edge is sampled against the mesh's own depth image and
split into visible and hidden sub-segments. Visible edges land in
``edges_2d``; occluded ones in ``hidden_edges_2d`` (drawn dashed on the
HIDDEN layer when Mesh3DView.show_hidden is set). Collinear overlapping
segments are merged so stacked front/back features emit single strokes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

try:
    import trimesh
except ImportError as e:
    raise ImportError(
        "rapcad_drawings.model3d requires the 'trimesh' package. "
        "Install with: pip install trimesh"
    ) from e


# ---------------------------------------------------------------------------
# View definitions
# ---------------------------------------------------------------------------

ViewName = Literal["front", "top", "bottom", "right", "left", "back", "iso"]

# 3x3 rotation matrices that bring "look-along-Z" to align with each named
# view. After applying the matrix, dropping the Z coordinate gives the 2D
# projection.
#
# We use the convention: the matrix R transforms a 3D point such that the
# camera looks along the -Z axis of the rotated frame. So R * world_pt
# expresses the point in camera coordinates; the camera image is its
# (x, y) components.

def _rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


# View matrices: world coordinates → camera coordinates. Camera always
# looks along the camera-frame -Z axis; the 2D drawing is the camera's
# (x, y).
#
# Mechanical convention used here: model with +Z up, +Y forward, +X right.
#
#   FRONT  — camera at (0, -inf, 0) looking +Y. Project (x, z) → drawing.
#            Bring world +Y → camera +Z (so camera looks at +Y by looking
#            along its own -Z is wrong — we want world Y to map to
#            camera -Z so camera "sees" along +Y). After mapping, world
#            +X → camera +X, world +Z → camera +Y.
#
# Easiest formulation: rotate model so the desired view direction
# becomes the camera's -Z, and the world up becomes camera +Y.

VIEW_MATRICES = {
    # Front: camera at -Y looking +Y. Camera frame:
    #   cam +X = world +X, cam +Y = world +Z, cam +Z = world -Y
    "front":  np.array([[ 1,  0,  0],
                        [ 0,  0,  1],
                        [ 0, -1,  0]], dtype=float),
    # Top: camera at +Z looking -Z.
    #   cam +X = world +X, cam +Y = world +Y, cam +Z = world +Z
    "top":    np.eye(3, dtype=float),
    # Bottom: camera at -Z looking +Z (third-angle flips Y).
    "bottom": np.array([[ 1,  0,  0],
                        [ 0, -1,  0],
                        [ 0,  0, -1]], dtype=float),
    # Right: camera at +X looking -X.
    #   cam +X = world +Y, cam +Y = world +Z, cam +Z = world +X
    "right":  np.array([[ 0,  1,  0],
                        [ 0,  0,  1],
                        [ 1,  0,  0]], dtype=float),
    # Left: camera at -X looking +X (mirror of right).
    "left":   np.array([[ 0, -1,  0],
                        [ 0,  0,  1],
                        [-1,  0,  0]], dtype=float),
    # Back: camera at +Y looking -Y.
    "back":   np.array([[-1,  0,  0],
                        [ 0,  0,  1],
                        [ 0,  1,  0]], dtype=float),
    # Isometric: 30° down from +Z, then 45° around Z. Standard ASME iso.
    "iso":    _rot_x(-math.radians(35.264)) @ _rot_z(-math.radians(45.0)),
}


@dataclass
class ProjectedView:
    """Result of projecting a 3D mesh to a 2D view.

    ``edges_2d`` holds the visible ink; ``hidden_edges_2d`` the occluded
    edges (same coordinate frame), for callers that draw hidden detail
    dashed per ASME Y14.2.
    """
    view:        str
    edges_2d:    List[Tuple[Tuple[float, float], Tuple[float, float]]]
    bounds_2d:   Tuple[float, float, float, float]    # (xmin, ymin, xmax, ymax)
    source_path: Optional[str] = None
    hidden_edges_2d: List[Tuple[Tuple[float, float],
                                Tuple[float, float]]] = field(default_factory=list)

    @property
    def width(self) -> float:
        return self.bounds_2d[2] - self.bounds_2d[0]

    @property
    def height(self) -> float:
        return self.bounds_2d[3] - self.bounds_2d[1]

    @property
    def center(self) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.bounds_2d
        return ((x0 + x1) / 2, (y0 + y1) / 2)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_mesh(path_or_data) -> trimesh.Trimesh:
    """Load a mesh from a file path or a tuple of (vertices, faces).

    Mesh formats supported by trimesh include STL (binary + ascii), OBJ,
    PLY, OFF, GLB. STEP/BREP require OCP and aren't supported here.
    """
    if isinstance(path_or_data, tuple) and len(path_or_data) == 2:
        verts, faces = path_or_data
        return trimesh.Trimesh(vertices=np.asarray(verts, dtype=float),
                               faces=np.asarray(faces, dtype=int),
                               process=False)
    obj = trimesh.load(path_or_data, force="mesh")
    if isinstance(obj, trimesh.Scene):
        # Concatenate all sub-geometries into one mesh
        meshes = [g for g in obj.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {path_or_data}")
        return trimesh.util.concatenate(meshes)
    return obj


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

# Hidden-line tuning. Validated on canonical parts (box, occluded box,
# 72-segment cylinder) and real multi-body meshes; see the docstrings of
# the helpers for what each guards against.
_HLR_RES_LONG = 1024        # depth-buffer pixels on the longer XY axis
_HLR_EPS_REL = 3e-3         # depth tolerance, fraction of bbox diagonal
_HLR_SLOPE_BIAS_PX = 4.0    # slope-scaled depth bias (shadow-map style)
_HLR_SAMPLES_PER_PX = 0.5   # one visibility sample per 2 px of edge
_HLR_N_MIN, _HLR_N_MAX = 5, 256

# Segment post-processing. All length tolerances are RELATIVE to the
# mesh bounding-box diagonal so models in meters/inches behave like
# millimeter models (for a typical 230mm part these reproduce
# offset 0.02mm / gap+cull 0.06mm).
_MERGE_ANGLE_DEG = 0.2      # angular bucket for collinearity grouping
_MERGE_OFFSET_REL = 1e-4    # perpendicular "same line" tol, x diagonal
_MERGE_GAP_REL = 2.5e-4     # join collinear runs within, x diagonal
_MIN_SEG_REL = 2.5e-4       # cull fragments shorter than, x diagonal


def _candidate_edges(mesh: trimesh.Trimesh, R: np.ndarray,
                     cam_verts: np.ndarray, threshold: float):
    """Silhouette, feature, and open-boundary edges in camera coords.

    Returns (edges, edge_verts): (E, 2, 3) camera-frame endpoints and the
    (E, 2) source vertex ids (needed for owner-face exclusion). Edges
    that project to a point are dropped. Among exact 2D duplicates, a
    copy is dropped only when another copy is at least as close to the
    camera at BOTH endpoints — a dominated duplicate is either equally
    hidden or subtracted from the hidden class later. Copies whose depth
    order crosses along the edge are all kept (each may be the frontmost
    over part of the span).
    """
    facing = (mesh.face_normals @ R.T)[:, 2]      # >0 = toward camera
    pairs: List[Tuple[int, int]] = []

    if len(mesh.face_adjacency) > 0:
        fa = mesh.face_adjacency
        fae = mesh.face_adjacency_edges
        try:
            faa = mesh.face_adjacency_angles
        except Exception:
            n0 = mesh.face_normals[fa[:, 0]]
            n1 = mesh.face_normals[fa[:, 1]]
            faa = np.arccos(np.clip(np.sum(n0 * n1, axis=1), -1.0, 1.0))
        # Silhouette: facing flips sign across the edge. A strict
        # product < 0 misses fillets/arcs where one tessellation strip
        # lands EXACTLY perpendicular to the view (facing == 0.0 — the
        # product is 0 on both sides of the crest and the whole
        # silhouette line vanishes). Treat zero-facing paired with a
        # non-zero face as a silhouette too; zero-zero pairs (interior
        # of an edge-on wall) stay excluded.
        f0, f1 = facing[fa[:, 0]], facing[fa[:, 1]]
        zero0 = np.abs(f0) < 1e-9
        zero1 = np.abs(f1) < 1e-9
        sil = ((f0 * f1) < 0.0) | (zero0 ^ zero1)
        keep = sil | (faa > threshold)
        pairs.extend((int(v0), int(v1)) for v0, v1 in fae[keep])

    # Open-boundary edges: referenced by exactly one face. (Replaces
    # mesh.outline(), which returns nothing on many real meshes.)
    try:
        from trimesh.grouping import group_rows
        for ei in group_rows(mesh.edges_sorted, require_count=1):
            v0, v1 = mesh.edges_sorted[ei]
            pairs.append((int(v0), int(v1)))
    except Exception:
        pass

    if not pairs:
        return (np.zeros((0, 2, 3)), np.zeros((0, 2), dtype=np.int64))

    vpairs = np.array(pairs, dtype=np.int64)
    p0 = cam_verts[vpairs[:, 0]]
    p1 = cam_verts[vpairs[:, 1]]
    ok = ~((np.abs(p0[:, 0] - p1[:, 0]) < 1e-9)
           & (np.abs(p0[:, 1] - p1[:, 1]) < 1e-9))
    vpairs, p0, p1 = vpairs[ok], p0[ok], p1[ok]

    kept: Dict[tuple, list] = {}      # key -> [(za, zb, i), ...] Pareto set
    for i in range(len(vpairs)):
        a = (round(float(p0[i, 0]), 6), round(float(p0[i, 1]), 6))
        b = (round(float(p1[i, 0]), 6), round(float(p1[i, 1]), 6))
        if a <= b:
            key, za, zb = (a, b), float(p0[i, 2]), float(p1[i, 2])
        else:
            key, za, zb = (b, a), float(p1[i, 2]), float(p0[i, 2])
        copies = kept.setdefault(key, [])
        if any(ka >= za - 1e-9 and kb >= zb - 1e-9 for ka, kb, _ in copies):
            continue                              # dominated: drop
        copies[:] = [(ka, kb, j) for ka, kb, j in copies
                     if not (za >= ka - 1e-9 and zb >= kb - 1e-9)]
        copies.append((za, zb, i))
    idx = np.array(sorted(j for copies in kept.values()
                          for _, _, j in copies), dtype=np.int64)
    edges = np.stack([p0[idx], p1[idx]], axis=1)
    return edges, vpairs[idx]


def _classify_hidden(mesh: trimesh.Trimesh, cam_verts: np.ndarray,
                     edges: np.ndarray, edge_verts: np.ndarray):
    """Split candidate edges into visible / hidden 2D sub-segments.

    Depth-buffer approach (camera looks along -Z; larger Z = closer):

    1. Rasterize every triangle into a max-Z buffer plus a face-id
       "owner" buffer. Each triangle writes ``z - bias`` where bias is
       slope-scaled (shadow-mapping): a steeply sloped occluder must not
       occlude within its own per-pixel depth variation — without this,
       sub-pixel facet slivers at curved silhouettes shadow their own
       silhouette edges.
    2. Depth-test edge samples against a covered-only 3x3 min-filter of
       the buffer (farthest covered neighbor; background ignored so
       edges on the outline still get occluded by covered neighbors).
    3. Owner-face exclusion, two-tier: a pixel owned by a face ADJACENT
       to the edge (sharing both endpoints) never occludes it — it IS
       that surface. A face sharing only ONE endpoint (quad-mates along
       curved silhouettes) is exempted only while its depth stays within
       a small margin of the edge — otherwise a top face that merely
       touches a hidden edge's corner would exempt the whole edge and
       print solid stubs at silhouette vertices.
    4. Samples classify at one per 2 px; isolated single-sample flips
       are smoothed; edges split into sub-segments at run boundaries.
    """
    faces_arr = np.asarray(mesh.faces)
    if len(faces_arr) == 0:
        return ([(tuple(e[0, :2]), tuple(e[1, :2])) for e in edges], [])

    # ---- rasterize depth + owner buffers ----------------------------
    tris = cam_verts[faces_arr]                     # (T, 3, 3)
    xy = tris[:, :, :2].reshape(-1, 2)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = hi - lo
    pixel = max(float(span[0]), float(span[1])) / _HLR_RES_LONG
    if pixel <= 0:
        pixel = 1.0
    margin = 2
    origin = lo - margin * pixel
    W = int(math.ceil(span[0] / pixel)) + 2 * margin + 1
    H = int(math.ceil(span[1] / pixel)) + 2 * margin + 1
    zbuf = np.full((H, W), -np.inf)
    owner = np.full((H, W), -1, dtype=np.int32)

    P = (tris[:, :, :2] - origin) / pixel           # pixel space
    Z = tris[:, :, 2]
    e1, e2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    denom = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    ok = np.abs(denom) > 1e-9                       # skip edge-on tris
    dz1, dz2 = Z[:, 1] - Z[:, 0], Z[:, 2] - Z[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        gx = (dz1 * e2[:, 1] - dz2 * e1[:, 1]) / denom
        gy = (dz2 * e1[:, 0] - dz1 * e2[:, 0]) / denom
    slope = np.hypot(gx, gy)                        # |dz| per pixel step
    bias = _HLR_SLOPE_BIAS_PX * np.where(np.isfinite(slope), slope, 0.0)
    ix0 = np.clip(np.floor(P[:, :, 0].min(axis=1)).astype(int) - 1, 0, W - 1)
    ix1 = np.clip(np.ceil(P[:, :, 0].max(axis=1)).astype(int) + 1, 0, W - 1)
    iy0 = np.clip(np.floor(P[:, :, 1].min(axis=1)).astype(int) - 1, 0, H - 1)
    iy1 = np.clip(np.ceil(P[:, :, 1].max(axis=1)).astype(int) + 1, 0, H - 1)
    for t in np.nonzero(ok)[0]:
        gxs = np.arange(ix0[t], ix1[t] + 1) + 0.5
        gys = np.arange(iy0[t], iy1[t] + 1) + 0.5
        if gxs.size == 0 or gys.size == 0:
            continue
        px = gxs[None, :] - P[t, 0, 0]
        py = gys[:, None] - P[t, 0, 1]
        w1 = (px * e2[t, 1] - py * e2[t, 0]) / denom[t]
        w2 = (py * e1[t, 0] - px * e1[t, 1]) / denom[t]
        w0 = 1.0 - w1 - w2
        mask = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not mask.any():
            continue
        z = np.where(mask,
                     w0 * Z[t, 0] + w1 * Z[t, 1] + w2 * Z[t, 2] - bias[t],
                     -np.inf)
        sub = zbuf[iy0[t]:iy1[t] + 1, ix0[t]:ix1[t] + 1]
        subo = owner[iy0[t]:iy1[t] + 1, ix0[t]:ix1[t] + 1]
        better = z > sub
        sub[better] = z[better]
        subo[better] = t

    # ---- covered-only 3x3 min filter --------------------------------
    work = np.where(np.isinf(zbuf), np.inf, zbuf)
    pad = np.pad(work, 1, mode="constant", constant_values=np.inf)
    zmin = work.copy()
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            np.minimum(zmin, pad[dy:dy + H, dx:dx + W], out=zmin)
    zmin[np.isposinf(zmin)] = -np.inf

    # ---- sample edges, classify, split into runs ---------------------
    eps = _HLR_EPS_REL * float(np.linalg.norm(mesh.extents))
    p0, p1 = edges[:, 0], edges[:, 1]
    len_px = np.linalg.norm((p1 - p0)[:, :2], axis=1) / pixel
    Ns = np.clip(np.ceil(len_px * _HLR_SAMPLES_PER_PX).astype(int),
                 _HLR_N_MIN, _HLR_N_MAX)
    nmax = int(Ns.max())
    idx = np.arange(nmax)
    tt = np.where(idx[None, :] < Ns[:, None],
                  (idx[None, :] + 0.5) / Ns[:, None], 0.5)
    pts = p0[:, None, :] + tt[..., None] * (p1 - p0)[:, None, :]
    ix = np.clip(((pts[..., 0] - origin[0]) / pixel).astype(int), 0, W - 1)
    iy = np.clip(((pts[..., 1] - origin[1]) / pixel).astype(int), 0, H - 1)
    depth_over = zmin[iy, ix] - pts[..., 2]
    occluded = depth_over > eps

    own_full = np.zeros(occluded.shape, dtype=bool)
    own_part = np.zeros(occluded.shape, dtype=bool)
    v0 = edge_verts[:, 0][:, None, None]
    v1 = edge_verts[:, 1][:, None, None]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            o = owner[np.clip(iy + dy, 0, H - 1), np.clip(ix + dx, 0, W - 1)]
            ov = faces_arr[np.where(o >= 0, o, 0)]
            hit0 = ((ov == v0).any(axis=2)) & (o >= 0)
            hit1 = ((ov == v1).any(axis=2)) & (o >= 0)
            own_full |= hit0 & hit1
            own_part |= hit0 ^ hit1
    # Single-vertex owners exempt only while genuinely tangent (their
    # depth within a few eps of the edge), not when properly in front.
    own_part &= depth_over <= 3.0 * eps
    hidden_smp = occluded & ~own_full & ~own_part

    visible: List[tuple] = []
    hidden: List[tuple] = []
    stub_len = 3.0 * eps
    for e in range(len(edges)):
        n = int(Ns[e])
        h = hidden_smp[e, :n].copy()
        if n >= 3:                                  # kill 1-sample flicker
            iso = (h[:-2] == h[2:]) & (h[1:-1] != h[:-2])
            h[1:-1] = np.where(iso, h[:-2], h[1:-1])
        ts = (np.arange(n) + 0.5) / n
        change = np.nonzero(h[1:] != h[:-1])[0]
        t_cuts = np.concatenate(([0.0], (ts[change] + ts[change + 1]) / 2,
                                 [1.0]))
        cls = list(h[np.concatenate((change, [n - 1]))])
        a2, b2 = p0[e, :2], p1[e, :2]
        elen = float(np.hypot(*(b2 - a2)))
        # Where an occluder merely touches an endpoint (a face over a
        # hidden edge's corner), depth_over stays inside the tolerance
        # band for the first ~eps of the edge and a short solid stub
        # leaks out. Flip terminal visible runs shorter than a few eps
        # when their inward neighbor is hidden.
        if len(cls) >= 2:
            if (not cls[0] and cls[1]
                    and (t_cuts[1] - t_cuts[0]) * elen < stub_len):
                cls[0] = True
            if (not cls[-1] and cls[-2]
                    and (t_cuts[-1] - t_cuts[-2]) * elen < stub_len):
                cls[-1] = True
        for k in range(len(t_cuts) - 1):
            qa = tuple((a2 + t_cuts[k] * (b2 - a2)).tolist())
            qb = tuple((a2 + t_cuts[k + 1] * (b2 - a2)).tolist())
            (hidden if cls[k] else visible).append((qa, qb))
    return visible, hidden


def _merge_segments(segs, subtract=None, *, offset_tol, gap_tol, min_len):
    """Merge collinear overlapping segments; optionally subtract others.

    Tessellated meshes stack many coincident/collinear fragments on the
    same supporting line (front/back features of a prismatic part, facet
    fragments of curves seen edge-on). Segments are grouped by quantized
    (angle, perpendicular offset); each group unions its 1D intervals
    along a frame taken from the group's longest member (NOT the
    quantized bucket angle — that would rotate long lines by up to the
    bucket width). A member is only merged if it actually lies on the
    frame line (within 2x offset_tol) — near-parallel lines far from
    their crossing point can share a bucket without being collinear, and
    snapping those would displace ink; they are emitted unmerged
    instead. With ``subtract`` (the visible ink, when merging hidden),
    any interval already covered there is removed so dashed lines never
    overdraw solid ones. Fragments below min_len are culled. All
    tolerances are in drawing units, pre-scaled by the caller to the
    model's size.
    """
    if not segs:
        return []
    angle_q = math.radians(_MERGE_ANGLE_DEG)
    n_bins = max(1, int(round(math.pi / angle_q)))

    def _canon(dx, dy, length):
        """Direction canonicalization stable near the axes: fix the sign
        of the DOMINANT component, so fp noise in the tiny component
        (e.g. dx = ±1e-9 on a vertical line) can't flip the frame."""
        ux, uy = dx / length, dy / length
        if (abs(ux) >= abs(uy) and ux < 0) or (abs(uy) > abs(ux) and uy < 0):
            ux, uy = -ux, -uy
        return ux, uy

    def _bucketize(items):
        buckets: Dict[tuple, list] = {}
        for a, b in items:
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dy)
            if length < 1e-12:
                continue
            ux, uy = _canon(dx, dy, length)
            th = math.atan2(uy, ux) % math.pi
            k_th = int(round(th / angle_q)) % n_bins
            c = -a[0] * uy + a[1] * ux              # perpendicular offset
            key = (k_th, int(round(c / offset_tol)))
            buckets.setdefault(key, []).append((a, b, length, ux, uy, c))
        return buckets

    def _interval_union(ivals, gap):
        ivals.sort()
        out = [list(ivals[0])]
        for t0, t1 in ivals[1:]:
            if t0 <= out[-1][1] + gap:
                out[-1][1] = max(out[-1][1], t1)
            else:
                out.append([t0, t1])
        return out

    def _off_frame(a, b, ux, uy, c):
        """Max perpendicular deviation of a segment from the frame line."""
        da = abs((-a[0] * uy + a[1] * ux) - c)
        db = abs((-b[0] * uy + b[1] * ux) - c)
        return max(da, db)

    buckets = _bucketize(segs)
    sub_buckets = _bucketize(subtract) if subtract else {}
    merged = []
    for key, members in buckets.items():
        # Frame from the longest member: direction u, offset c.
        _, _, _, ux, uy, c = max(members, key=lambda m: m[2])
        ivals = []
        for a, b, length, _, _, _ in members:
            if _off_frame(a, b, ux, uy, c) > 2 * offset_tol:
                # Shares the bucket but not the line: emit unmerged.
                if length >= min_len:
                    merged.append((a, b))
                continue
            t0 = a[0] * ux + a[1] * uy
            t1 = b[0] * ux + b[1] * uy
            ivals.append((min(t0, t1), max(t0, t1)))
        if not ivals:
            continue
        spans = _interval_union(ivals, gap_tol)

        if sub_buckets:
            cover = []
            for dth in (-1, 0, 1):                  # angle-bin straddle
                for dk in (-1, 0, 1):               # offset-bin straddle
                    skey = ((key[0] + dth) % n_bins, key[1] + dk)
                    for a, b, _, _, _, _ in sub_buckets.get(skey, []):
                        if _off_frame(a, b, ux, uy, c) > 2 * offset_tol:
                            continue
                        t0 = a[0] * ux + a[1] * uy
                        t1 = b[0] * ux + b[1] * uy
                        cover.append((min(t0, t1) - gap_tol,
                                      max(t0, t1) + gap_tol))
            if cover:
                cover = _interval_union(cover, 0.0)
                remaining = []
                for s0, s1 in spans:
                    cur = s0
                    for c0, c1 in cover:
                        if c1 <= cur or c0 >= s1:
                            continue
                        if c0 > cur:
                            remaining.append([cur, min(c0, s1)])
                        cur = max(cur, c1)
                        if cur >= s1:
                            break
                    if cur < s1:
                        remaining.append([cur, s1])
                spans = remaining

        nx, ny = -uy, ux
        for t0, t1 in spans:
            if t1 - t0 < min_len:
                continue
            merged.append(((c * nx + t0 * ux, c * ny + t0 * uy),
                           (c * nx + t1 * ux, c * ny + t1 * uy)))
    return merged


def project_mesh(mesh: trimesh.Trimesh,
                 view: ViewName = "front",
                 angle_threshold_deg: float = 30.0,
                 source_path: Optional[str] = None) -> ProjectedView:
    """Compute the 2D projection of a mesh for the named view.

    Candidate silhouette / feature / boundary edges are classified by a
    depth-buffer hidden-line pass: ``edges_2d`` holds only the VISIBLE
    ink; occluded edges land in ``hidden_edges_2d`` (draw dashed, or
    ignore). Collinear overlapping fragments are merged, so the lists
    approximate the minimal ink of the view. Coordinates are NOT
    translated — the caller decides where to place the view.
    """
    R = VIEW_MATRICES[view]                              # (3, 3)
    cam_verts = mesh.vertices @ R.T                      # (N, 3) cam frame
    edges, edge_verts = _candidate_edges(
        mesh, R, cam_verts, math.radians(angle_threshold_deg))

    if len(edges) == 0:
        return ProjectedView(view=view, edges_2d=[],
                             bounds_2d=(0.0, 0.0, 0.0, 0.0),
                             source_path=source_path)

    visible_raw, hidden_raw = _classify_hidden(mesh, cam_verts,
                                               edges, edge_verts)
    # Merge tolerances scale with the model so meter/inch-unit meshes
    # behave like millimeter ones.
    diag = float(np.linalg.norm(mesh.extents)) or 1.0
    tols = dict(offset_tol=max(_MERGE_OFFSET_REL * diag, 1e-12),
                gap_tol=_MERGE_GAP_REL * diag,
                min_len=_MIN_SEG_REL * diag)
    visible = _merge_segments(visible_raw, **tols)
    hidden = _merge_segments(hidden_raw, subtract=visible, **tols)

    pts = ([p for e in visible for p in e]
           + [p for e in hidden for p in e])
    if pts:
        arr = np.array(pts)
        xmin, ymin = arr.min(axis=0)
        xmax, ymax = arr.max(axis=0)
    else:
        xmin = ymin = xmax = ymax = 0.0

    return ProjectedView(
        view=view,
        edges_2d=visible,
        hidden_edges_2d=hidden,
        bounds_2d=(float(xmin), float(ymin),
                   float(xmax), float(ymax)),
        source_path=source_path,
    )


def project_file(path: str,
                 view: ViewName = "front",
                 angle_threshold_deg: float = 30.0) -> ProjectedView:
    """Convenience: load a mesh file and project it in one call."""
    mesh = load_mesh(path)
    return project_mesh(mesh, view=view,
                        angle_threshold_deg=angle_threshold_deg,
                        source_path=path)


def clip_segments_to_rect(segments, rect):
    """Clip 2D segments to an axis-aligned rectangle (Liang-Barsky).

    ``rect`` is (xmin, ymin, xmax, ymax). Segments fully outside are
    dropped; crossing segments are shortened to the window. Used for
    detail (zoom) views that show only a region of a projection.
    """
    x0r, y0r, x1r, y1r = rect
    out = []
    for (ax, ay), (bx, by) in segments:
        dx, dy = bx - ax, by - ay
        t0, t1 = 0.0, 1.0
        ok = True
        for p, q in ((-dx, ax - x0r), (dx, x1r - ax),
                     (-dy, ay - y0r), (dy, y1r - ay)):
            if abs(p) < 1e-12:
                if q < 0:
                    ok = False
                    break
            else:
                t = q / p
                if p < 0:
                    if t > t1:
                        ok = False
                        break
                    if t > t0:
                        t0 = t
                else:
                    if t < t0:
                        ok = False
                        break
                    if t < t1:
                        t1 = t
        if ok and t1 > t0:
            out.append(((ax + t0 * dx, ay + t0 * dy),
                        (ax + t1 * dx, ay + t1 * dy)))
    return out


# ---------------------------------------------------------------------------
# Multi-view layout helper
# ---------------------------------------------------------------------------
#
# Third-angle multi-view convention:
#
#       +--------+
#       |  TOP   |
#       +--------+
#       | FRONT  |---RIGHT---
#       +--------+
#
# Plus an isometric in the upper-right corner. The caller specifies a
# spacing between views; this helper returns the (origin_x, origin_y)
# for each view, given a primary view's footprint.

def multi_view_layout(front_view: ProjectedView,
                      top_view:   Optional[ProjectedView] = None,
                      right_view: Optional[ProjectedView] = None,
                      iso_view:   Optional[ProjectedView] = None,
                      spacing:    float = 25.0,
                      anchor:     Tuple[float, float] = (0.0, 0.0)
                      ) -> dict:
    """Compute placement origins for a third-angle multi-view layout.

    Each returned origin is the (lower-left) corner of where the view's
    bounds_2d should be translated to. The caller subtracts the view's
    own (xmin, ymin) from each placed point.
    """
    ax, ay = anchor
    fw = front_view.width
    fh = front_view.height
    origins = {"front": (ax, ay)}

    if top_view is not None:
        # Above the front view
        origins["top"] = (ax + (fw - top_view.width) / 2,
                          ay + fh + spacing)
    if right_view is not None:
        # To the right of the front view
        origins["right"] = (ax + fw + spacing,
                            ay + (fh - right_view.height) / 2)
    if iso_view is not None:
        # Upper-right corner: above the right view (or where it would be)
        ox = ax + fw + spacing
        oy = ay + fh + spacing
        if right_view is not None:
            # Sit it diagonally up-right from front + right
            oy = ay + fh + spacing
        origins["iso"] = (ox, oy)

    return origins
