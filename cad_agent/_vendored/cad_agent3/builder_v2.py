"""builder_v2.py — high-level Builder API on top of session + operations.

The Phase 2 Builder is a thin orchestration layer. Instead of holding
inline geometry code for every operation, it dispatches to the registered
operation catalog. The session handles state, log, undo.

Existing builder.py is preserved for backward compatibility — code that
calls Builder() still works. New code should prefer BuilderV2 / DesignSession.

Usage:
    b = BuilderV2()
    b.start_from_box(50, 40, 20)
    b.drill_hole(diameter=3.4, position=(0, 0, 10))
    b.fillet_edges(radius=1.0, axis="Z")
    b.export_step("/tmp/out.step")
"""
from __future__ import annotations
from typing import Any, Optional

from .session import DesignSession


class BuilderV2:
    """High-level builder backed by a DesignSession and the operation catalog.

    Convenience methods wrap the most common ops with friendly arg names.
    Anything not covered by a method can be reached via .session.apply(op_name, **inputs).
    """

    def __init__(self, verbose: bool = False):
        self.session = DesignSession(verbose=verbose)

    # ---- starting state ----

    def start_from_box(self, x_mm: float, y_mm: float, z_mm: float):
        from build123d import Box
        return self.session.start_with(Box(x_mm, y_mm, z_mm))

    def start_from_cylinder(self, radius_mm: float, height_mm: float):
        from build123d import Cylinder
        return self.session.start_with(Cylinder(radius_mm, height_mm))

    def start_from_step(self, path: str):
        from build123d import import_step
        return self.session.start_with(import_step(path))

    def start_with_part(self, part: Any):
        return self.session.start_with(part)

    # ---- features ----

    def drill_hole(self, diameter, position, through=True, depth=None):
        return self.session.apply("hole", diameter_mm=diameter,
                                    position=position, through=through,
                                    depth_mm=depth)

    def counterbore(self, thru_diameter, cb_diameter, cb_depth, position):
        return self.session.apply("counterbore",
                                    thru_diameter_mm=thru_diameter,
                                    cb_diameter_mm=cb_diameter,
                                    cb_depth_mm=cb_depth, position=position)

    def countersink(self, thru_diameter, cs_diameter, position, cone_angle=90):
        return self.session.apply("countersink",
                                    thru_diameter_mm=thru_diameter,
                                    cs_diameter_mm=cs_diameter,
                                    position=position,
                                    cone_angle_deg=cone_angle)

    def cut_pocket(self, width, length, depth, position):
        return self.session.apply("pocket", width_mm=width, length_mm=length,
                                    depth_mm=depth, position=position)

    def cut_slot(self, length, width, position, rotation_deg=0,
                  through=True, depth=None):
        return self.session.apply("slot", length_mm=length, width_mm=width,
                                    position=position, rotation_deg=rotation_deg,
                                    through=through, depth_mm=depth)

    def add_boss(self, diameter, height, position, pilot_hole=None):
        return self.session.apply("boss", diameter_mm=diameter,
                                    height_mm=height, position=position,
                                    pilot_hole_mm=pilot_hole)

    def add_rib(self, start, end, height, thickness):
        return self.session.apply("rib", start=start, end=end,
                                    height_mm=height, thickness_mm=thickness)

    def fillet_edges(self, radius, axis=None, edge_selector=None):
        return self.session.apply("fillet", radius_mm=radius,
                                    axis_filter=axis,
                                    edge_selector=edge_selector)

    def chamfer_edges(self, distance, axis=None, edge_selector=None):
        return self.session.apply("chamfer", distance_mm=distance,
                                    axis_filter=axis,
                                    edge_selector=edge_selector)

    def shell_part(self, thickness, open_faces=None):
        return self.session.apply("shell", thickness_mm=thickness,
                                    open_faces=open_faces)

    def cut_score_line(self, start, end, depth, width=1.0):
        return self.session.apply("score_line", start=start, end=end,
                                    depth_mm=depth, width_mm=width)

    # ---- transforms ----

    def translate(self, dx, dy, dz):
        return self.session.apply("translate", vector=(dx, dy, dz))

    def rotate(self, axis, angle):
        return self.session.apply("rotate", axis=axis, angle_deg=angle)

    def mirror(self, plane, combine=False):
        return self.session.apply("mirror", plane=plane, combine=combine)

    def pattern_linear(self, count, vector, combine=True):
        return self.session.apply("pattern_linear", count=count,
                                    vector=vector, combine=combine)

    def pattern_circular(self, count, axis="Z", total_angle=360, combine=True):
        return self.session.apply("pattern_circular", count=count, axis=axis,
                                    total_angle_deg=total_angle, combine=combine)

    # ---- analysis ----

    def get_volume(self):
        return self.session.apply("volume")

    def get_bbox(self):
        return self.session.apply("bbox")

    def estimate_mass(self, material, density=None):
        return self.session.apply("mass", material=material,
                                    density_g_per_cc=density)

    def check_manifold(self):
        return self.session.apply("manifold_check")

    # ---- repair ----

    def heal_geometry(self, tolerance=0.01):
        return self.session.apply("shape_fix", tolerance_mm=tolerance)

    def simplify(self, tolerance=0.001):
        return self.session.apply("simplify", tolerance_mm=tolerance)

    # ---- session controls ----

    def checkpoint(self, label):
        return self.session.checkpoint(label)

    def rollback(self, label):
        return self.session.rollback_to(label)

    def undo(self):
        return self.session.undo_last()

    @property
    def part(self):
        return self.session.part

    # ---- export ----

    def export_step(self, path: str):
        from build123d import export_step
        if self.part is None:
            raise ValueError("no part to export")
        export_step(self.part, path)
        return path

    def export_stl(self, path: str):
        from build123d import export_stl
        if self.part is None:
            raise ValueError("no part to export")
        export_stl(self.part, path)
        return path

    def summary(self):
        return self.session.summary()
