"""boss.py — add a cylindrical raised feature on top of a part.

Used for screw bosses (with optional inner pilot hole), mounting
posts, alignment pins.
"""
from __future__ import annotations
from ..operation_base import (
    Operation, OperationDecl, OperationResult, _safe_volume)
from ..catalog import register


@register
class BossOp(Operation):

    @classmethod
    def declare(cls):
        return OperationDecl(
            name="boss",
            category="feature",
            summary="Add a cylindrical raised boss to a part.",
            required_inputs=["diameter_mm", "height_mm", "position"],
            optional_inputs=["pilot_hole_mm"],
        )

    @classmethod
    def apply(cls, part, inputs):
        try:
            from build123d import Cylinder, Pos
        except ImportError as e:
            return OperationResult(op_name="boss", ok=False,
                                     error=f"build123d not available: {e}")
        d = float(inputs["diameter_mm"])
        h = float(inputs["height_mm"])
        x, y, z = inputs["position"]
        pilot = inputs.get("pilot_hole_mm")

        boss = Pos(x, y, z + h/2) * Cylinder(d / 2, h)
        if pilot is not None:
            pilot_d = float(pilot)
            boss = boss - Pos(x, y, z + h/2) * Cylinder(pilot_d / 2, h + 1)
        try:
            new_part = part + boss
        except Exception as e:
            return OperationResult(op_name="boss", ok=False,
                                     error=f"boolean failed: {e}")
        v_before = _safe_volume(part); v_after = _safe_volume(new_part)
        return OperationResult(
            op_name="boss", ok=True, new_part=new_part,
            undo_data={"previous_part": part},
            effect=(f"boss {d}mm × {h}mm at ({x:.1f},{y:.1f},{z:.1f})"
                     + (f" with {pilot}mm pilot" if pilot else "")),
            metrics={"volume_added_mm3": v_after - v_before},
        )
