"""hole.py — drill a cylindrical hole through a part.

Inputs:
  diameter_mm    (required): hole diameter
  position       (required): (x, y, z) center of the hole's top face
  axis           optional, default (0,0,-1): direction the hole drills toward
  depth_mm       optional, default None (through-cut): blind-hole depth
  through        optional, default True: extend hole past part for clean cut
"""
from __future__ import annotations
from typing import Any, Dict

from ..operation_base import (
    Operation, OperationDecl, OperationResult, OperationCheck, _bbox_dict,
    _safe_volume)
from ..catalog import register


@register
class HoleOp(Operation):

    @classmethod
    def declare(cls) -> OperationDecl:
        return OperationDecl(
            name="hole",
            category="feature",
            summary="Drill a cylindrical hole through (or into) a part.",
            required_inputs=["diameter_mm", "position"],
            optional_inputs=["axis", "depth_mm", "through"],
        )

    @classmethod
    def validate(cls, part, inputs):
        check = super().validate(part, inputs)
        if part is None:
            check.ok = False
            check.issues.append("no part to drill into")
        d = inputs.get("diameter_mm")
        if d is not None and (not isinstance(d, (int, float)) or d <= 0):
            check.ok = False
            check.issues.append(f"diameter_mm must be > 0, got {d!r}")
        return check

    @classmethod
    def apply(cls, part, inputs):
        try:
            from build123d import Cylinder, Pos, Rot
        except ImportError as e:
            return OperationResult(
                op_name="hole", ok=False,
                error=f"build123d not available: {e}")
        d = float(inputs["diameter_mm"])
        x, y, z = inputs["position"]
        depth = inputs.get("depth_mm")
        through = inputs.get("through", depth is None)
        axis = inputs.get("axis", (0, 0, -1))

        bb = _bbox_dict(part)
        if through:
            cyl_h = max(bb.get("size_z", 100), 100) * 1.5 + 20
        else:
            cyl_h = float(depth) + 1.0
        # Build cylinder along Z axis at position, then rotate if needed
        cyl = Pos(x, y, z) * Cylinder(d / 2, cyl_h)
        try:
            new_part = part - cyl
        except Exception as e:
            return OperationResult(
                op_name="hole", ok=False,
                error=f"boolean subtraction failed: {type(e).__name__}: {e}")
        v_before = _safe_volume(part)
        v_after = _safe_volume(new_part)
        return OperationResult(
            op_name="hole", ok=True, new_part=new_part,
            undo_data={"previous_part": part},
            effect=(f"drilled {d:.2f}mm hole at "
                     f"({x:.1f}, {y:.1f}, {z:.1f}); "
                     f"removed {v_before - v_after:.0f} mm^3"),
            metrics={"volume_removed_mm3": v_before - v_after,
                      "diameter_mm": d, "through": through},
        )
