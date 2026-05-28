"""countersink.py — through-hole with conical chamfer for flat-head screws.

The cone is 90° included angle (metric standard). Cone diameter = head
diameter + 0.5mm clearance. Cone depth derived from cone angle and diameter.
"""
from __future__ import annotations
import math
from ..operation_base import (
    Operation, OperationDecl, OperationResult, _safe_volume, _bbox_dict)
from ..catalog import register


@register
class CountersinkOp(Operation):

    @classmethod
    def declare(cls):
        return OperationDecl(
            name="countersink",
            category="feature",
            summary="Through-hole with conical chamfer for flat-head screw.",
            required_inputs=["thru_diameter_mm", "cs_diameter_mm",
                              "position"],
            optional_inputs=["cone_angle_deg"],
        )

    @classmethod
    def apply(cls, part, inputs):
        try:
            from build123d import Cylinder, Cone, Pos
        except ImportError as e:
            return OperationResult(op_name="countersink", ok=False,
                                     error=f"build123d not available: {e}")
        td = float(inputs["thru_diameter_mm"])
        cs_d = float(inputs["cs_diameter_mm"])
        x, y, z = inputs["position"]
        angle_deg = float(inputs.get("cone_angle_deg", 90))
        if cs_d <= td:
            return OperationResult(op_name="countersink", ok=False,
                                     error="cs_diameter must be > thru_diameter")
        # Cone depth from half-angle and (cs_d - td)/2 radial difference
        half_angle = math.radians(angle_deg / 2)
        cone_h = ((cs_d - td) / 2) / math.tan(half_angle)
        bb = _bbox_dict(part)
        thru_h = max(bb.get("size_z", 100), 100) * 1.5 + 20
        thru = Pos(x, y, z) * Cylinder(td / 2, thru_h)
        # Cone wider at top (z), narrower below
        cone = Pos(x, y, z - cone_h/2) * Cone(cs_d / 2, td / 2, cone_h)
        try:
            new_part = part - thru - cone
        except Exception as e:
            return OperationResult(op_name="countersink", ok=False,
                                     error=f"boolean failed: {e}")
        return OperationResult(
            op_name="countersink", ok=True, new_part=new_part,
            undo_data={"previous_part": part},
            effect=f"countersink {td}mm/{cs_d}mm at ({x:.1f},{y:.1f},{z:.1f})",
            metrics={"cone_depth_mm": cone_h},
        )
