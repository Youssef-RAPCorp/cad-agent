"""rib.py — add a thin reinforcing rib between two surfaces.

Specified by start/end points + height + thickness.
"""
from __future__ import annotations
import math
from ..operation_base import (
    Operation, OperationDecl, OperationResult, _safe_volume)
from ..catalog import register


@register
class RibOp(Operation):

    @classmethod
    def declare(cls):
        return OperationDecl(
            name="rib",
            category="feature",
            summary="Add a reinforcing rib between two points.",
            required_inputs=["start", "end", "height_mm", "thickness_mm"],
            optional_inputs=[],
        )

    @classmethod
    def apply(cls, part, inputs):
        try:
            from build123d import Box, Pos, Rot
        except ImportError as e:
            return OperationResult(op_name="rib", ok=False,
                                     error=f"build123d not available: {e}")
        x0, y0, z0 = inputs["start"]
        x1, y1, z1 = inputs["end"]
        h = float(inputs["height_mm"])
        t = float(inputs["thickness_mm"])
        dx = x1 - x0; dy = y1 - y0
        L = math.hypot(dx, dy)
        if L < 0.001:
            return OperationResult(op_name="rib", ok=False,
                                     error="rib length is zero")
        cx = (x0 + x1) / 2; cy = (y0 + y1) / 2; cz = (z0 + z1) / 2
        ang = math.degrees(math.atan2(dy, dx))
        bar = Box(L, t, h)
        rib = Pos(cx, cy, cz + h / 2) * Rot(0, 0, ang) * bar
        try:
            new_part = part + rib
        except Exception as e:
            return OperationResult(op_name="rib", ok=False,
                                     error=f"boolean failed: {e}")
        v_before = _safe_volume(part); v_after = _safe_volume(new_part)
        return OperationResult(
            op_name="rib", ok=True, new_part=new_part,
            undo_data={"previous_part": part},
            effect=f"rib {L:.1f}mm long, {t}mm thick × {h}mm tall",
            metrics={"volume_added_mm3": v_after - v_before, "length_mm": L},
        )
