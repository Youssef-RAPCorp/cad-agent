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

Hidden-line removal is *not* implemented; only visible silhouette and
feature edges are returned. For most mechanical parts that is enough
to produce a recognisable view.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

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
    """Result of projecting a 3D mesh to a 2D view."""
    view:        str
    edges_2d:    List[Tuple[Tuple[float, float], Tuple[float, float]]]
    bounds_2d:   Tuple[float, float, float, float]    # (xmin, ymin, xmax, ymax)
    source_path: Optional[str] = None

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

def project_mesh(mesh: trimesh.Trimesh,
                 view: ViewName = "front",
                 angle_threshold_deg: float = 30.0,
                 source_path: Optional[str] = None) -> ProjectedView:
    """Compute the 2D projection of a mesh for the named view.

    Returns a ProjectedView whose ``edges_2d`` list contains every
    silhouette / feature / boundary edge projected to the camera plane.
    Coordinates are NOT translated — the caller decides where to place
    the view in modelspace.
    """
    R = VIEW_MATRICES[view]                              # (3, 3)
    cam_verts   = mesh.vertices @ R.T                    # (N, 3) in cam frame
    cam_normals = mesh.face_normals @ R.T                # (M, 3)
    # In the camera frame, the camera looks along its own -Z. A face's
    # outward normal pointing in +Z faces the camera (front-facing).
    facing = cam_normals[:, 2]                            # >0 = toward camera

    threshold = math.radians(angle_threshold_deg)
    edges_kept: List[Tuple[Tuple[float, float],
                           Tuple[float, float]]] = []

    # 1. Silhouette + feature edges (interior edges of the mesh)
    if len(mesh.face_adjacency) > 0:
        fa  = mesh.face_adjacency               # (K, 2) face indices
        fae = mesh.face_adjacency_edges         # (K, 2) vertex indices
        # face_adjacency_angles may not be available on older trimesh
        try:
            faa = mesh.face_adjacency_angles    # (K,) radians
        except Exception:
            # Fallback: compute from normals
            n0 = mesh.face_normals[fa[:, 0]]
            n1 = mesh.face_normals[fa[:, 1]]
            dot = np.clip(np.sum(n0 * n1, axis=1), -1.0, 1.0)
            faa = np.arccos(dot)

        is_silhouette = (facing[fa[:, 0]] * facing[fa[:, 1]]) < 0.0
        is_feature    = faa > threshold
        keep_mask     = is_silhouette | is_feature
        for v0, v1 in fae[keep_mask]:
            p0 = cam_verts[v0]
            p1 = cam_verts[v1]
            # Skip edges that collapse to a point in the projection
            # (i.e. edges parallel to the view direction).
            if abs(p0[0] - p1[0]) < 1e-9 and abs(p0[1] - p1[1]) < 1e-9:
                continue
            edges_kept.append(((float(p0[0]), float(p0[1])),
                               (float(p1[0]), float(p1[1]))))

    # 2. Boundary edges (open meshes / unmatched edges) — always drawn
    try:
        # trimesh exposes "outline" which gives the open boundary
        boundary = mesh.outline()
        if boundary is not None:
            for entity in boundary.entities:
                pts_3d = boundary.vertices[entity.points]
                pts_cam = pts_3d @ R.T
                for i in range(len(pts_cam) - 1):
                    a = pts_cam[i]; b = pts_cam[i + 1]
                    edges_kept.append(((float(a[0]), float(a[1])),
                                       (float(b[0]), float(b[1]))))
    except Exception:
        # mesh.outline() can fail on degenerate inputs; ignore — we
        # already have silhouette + feature edges which cover most cases.
        pass

    # 3. Compute bounds and deduplicate
    # The projection can produce duplicate edges (e.g. same 3D edge from
    # different mesh traversals). Drop exact duplicates regardless of
    # endpoint order.
    seen = set()
    deduped = []
    for a, b in edges_kept:
        # Canonicalise by sorting endpoints
        if a <= b:
            key = (a, b)
        else:
            key = (b, a)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((a, b))
    edges_kept = deduped

    if edges_kept:
        all_pts = np.array([p for e in edges_kept for p in e])
        xmin, ymin = all_pts.min(axis=0)
        xmax, ymax = all_pts.max(axis=0)
    else:
        xmin = ymin = xmax = ymax = 0.0

    return ProjectedView(
        view=view,
        edges_2d=edges_kept,
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
