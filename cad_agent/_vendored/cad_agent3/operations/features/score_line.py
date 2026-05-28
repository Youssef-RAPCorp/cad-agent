"""score_line.py — cut a partial-depth groove (living hinge for cardboard)."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class ScoreLineOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="score_line", category="feature",
            summary="Cut a partial-depth groove (living hinge for foldable cardboard).",
            required_inputs=["start", "end", "depth_mm"],
            optional_inputs=["width_mm"])
    @classmethod
    def apply(cls, part, inputs):
        try: from build123d import Box, Pos, Rot
        except ImportError as e:
            return OperationResult(op_name="score_line", ok=False, error=str(e))
        import math
        x0, y0, z0 = inputs["start"]; x1, y1, z1 = inputs["end"]
        d = float(inputs["depth_mm"]); w = float(inputs.get("width_mm", 1.0))
        dx = x1 - x0; dy = y1 - y0
        L = math.hypot(dx, dy)
        if L < 0.001:
            return OperationResult(op_name="score_line", ok=False, error="zero-length score")
        cx, cy = (x0 + x1)/2, (y0 + y1)/2
        ang = math.degrees(math.atan2(dy, dx))
        groove = Box(L + 5, w, d + 0.1)
        groove = Pos(cx, cy, z0 - d/2 + 0.05) * Rot(0, 0, ang) * groove
        try: new_part = part - groove
        except Exception as e:
            return OperationResult(op_name="score_line", ok=False, error=str(e))
        return OperationResult(op_name="score_line", ok=True, new_part=new_part,
            undo_data={"previous_part": part},
            effect=f"score line {L:.0f}mm long, {d}mm deep")
