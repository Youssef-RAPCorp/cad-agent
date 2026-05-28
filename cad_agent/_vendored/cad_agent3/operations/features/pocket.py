"""pocket.py — rectangular pocket cut into a part.

Inputs:
  width_mm, length_mm, depth_mm     (required)
  position                          (required): center of pocket on top face
"""
from __future__ import annotations
from ..operation_base import (
    Operation, OperationDecl, OperationResult, _safe_volume)
from ..catalog import register


@register
class PocketOp(Operation):

    @classmethod
    def declare(cls):
        return OperationDecl(
            name="pocket",
            category="feature",
            summary="Cut a rectangular pocket into a part.",
            required_inputs=["width_mm", "length_mm", "depth_mm", "position"],
            optional_inputs=[],
        )

    @classmethod
    def apply(cls, part, inputs):
        try:
            from build123d import Box, Pos
        except ImportError as e:
            return OperationResult(op_name="pocket", ok=False,
                                     error=f"build123d not available: {e}")
        w = float(inputs["width_mm"])
        l = float(inputs["length_mm"])
        d = float(inputs["depth_mm"])
        x, y, z = inputs["position"]
        cutter = Pos(x, y, z - d/2 + 0.05) * Box(w, l, d)
        try:
            new_part = part - cutter
        except Exception as e:
            return OperationResult(op_name="pocket", ok=False,
                                     error=f"boolean failed: {e}")
        v_before = _safe_volume(part); v_after = _safe_volume(new_part)
        return OperationResult(
            op_name="pocket", ok=True, new_part=new_part,
            undo_data={"previous_part": part},
            effect=f"pocket {w}×{l}×{d}mm at ({x:.1f},{y:.1f},{z:.1f})",
            metrics={"volume_removed_mm3": v_before - v_after},
        )
