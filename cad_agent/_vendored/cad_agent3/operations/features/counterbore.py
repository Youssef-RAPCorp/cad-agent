"""counterbore.py — hole with a coaxial cylindrical recess for socket-head screws.

Composite feature: a through-hole + a larger-diameter blind cylinder at
the top, sized to hide a socket-head cap screw flush.
"""
from __future__ import annotations
from typing import Any, Dict

from ..operation_base import (
    Operation, OperationDecl, OperationResult, _safe_volume, _bbox_dict)
from ..catalog import register


@register
class CounterboreOp(Operation):

    @classmethod
    def declare(cls):
        return OperationDecl(
            name="counterbore",
            category="feature",
            summary="Through-hole with cylindrical recess for socket-head screw.",
            required_inputs=["thru_diameter_mm", "cb_diameter_mm",
                              "cb_depth_mm", "position"],
            optional_inputs=[],
        )

    @classmethod
    def apply(cls, part, inputs):
        try:
            from build123d import Cylinder, Pos
        except ImportError as e:
            return OperationResult(op_name="counterbore", ok=False,
                                     error=f"build123d not available: {e}")
        td = float(inputs["thru_diameter_mm"])
        cd = float(inputs["cb_diameter_mm"])
        cdepth = float(inputs["cb_depth_mm"])
        x, y, z = inputs["position"]
        if cd <= td:
            return OperationResult(op_name="counterbore", ok=False,
                                     error="cb_diameter must be > thru_diameter")
        bb = _bbox_dict(part)
        thru_h = max(bb.get("size_z", 100), 100) * 1.5 + 20
        thru = Pos(x, y, z) * Cylinder(td / 2, thru_h)
        # Counterbore sits in the top, depth `cdepth` below z
        cb = Pos(x, y, z - cdepth/2 + 0.05) * Cylinder(cd / 2, cdepth)
        try:
            new_part = part - thru - cb
        except Exception as e:
            return OperationResult(op_name="counterbore", ok=False,
                                     error=f"boolean failed: {e}")
        v_before = _safe_volume(part); v_after = _safe_volume(new_part)
        return OperationResult(
            op_name="counterbore", ok=True, new_part=new_part,
            undo_data={"previous_part": part},
            effect=(f"counterbore: {td}mm thru + {cd}mm × {cdepth}mm recess at "
                     f"({x:.1f},{y:.1f},{z:.1f})"),
            metrics={"volume_removed_mm3": v_before - v_after},
        )
