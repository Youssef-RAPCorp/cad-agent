"""slot.py — elongated through-cut (rounded ends).

A slot has length, width (= rounded end diameter), and depth.
Common feature for tslot brackets, sliding adjustment, etc.
"""
from __future__ import annotations
from ..operation_base import (
    Operation, OperationDecl, OperationResult, _bbox_dict, _safe_volume)
from ..catalog import register


@register
class SlotOp(Operation):

    @classmethod
    def declare(cls):
        return OperationDecl(
            name="slot",
            category="feature",
            summary="Cut an elongated slot (rounded-end rectangle) into a part.",
            required_inputs=["length_mm", "width_mm", "position"],
            optional_inputs=["depth_mm", "through", "rotation_deg"],
        )

    @classmethod
    def apply(cls, part, inputs):
        try:
            from build123d import Box, Cylinder, Pos, Rot
        except ImportError as e:
            return OperationResult(op_name="slot", ok=False,
                                     error=f"build123d not available: {e}")
        L = float(inputs["length_mm"])    # along X by default
        W = float(inputs["width_mm"])     # along Y; also = rounded-end dia
        x, y, z = inputs["position"]
        depth = inputs.get("depth_mm")
        through = inputs.get("through", depth is None)
        rot = float(inputs.get("rotation_deg", 0))

        bb = _bbox_dict(part)
        h = (max(bb.get("size_z", 100), 100) * 1.5 + 20) if through else float(depth) + 1.0
        # Cutter: rectangle + 2 cylinders at the ends, rotated then placed
        rect = Box(L - W, W, h)  # rectangle between the rounded ends
        end_l = Pos(-(L - W) / 2, 0, 0) * Cylinder(W / 2, h)
        end_r = Pos((L - W) / 2, 0, 0) * Cylinder(W / 2, h)
        cutter = rect + end_l + end_r
        cutter = Rot(0, 0, rot) * cutter
        cutter = Pos(x, y, z) * cutter
        try:
            new_part = part - cutter
        except Exception as e:
            return OperationResult(op_name="slot", ok=False,
                                     error=f"boolean failed: {e}")
        v_before = _safe_volume(part); v_after = _safe_volume(new_part)
        return OperationResult(
            op_name="slot", ok=True, new_part=new_part,
            undo_data={"previous_part": part},
            effect=f"slot {L}×{W}mm @ ({x:.1f},{y:.1f},{z:.1f}) rot {rot}°",
            metrics={"volume_removed_mm3": v_before - v_after},
        )
