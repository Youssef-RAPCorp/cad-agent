"""scale.py — uniform or non-uniform scaling of a part."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class ScaleOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="scale", category="transform",
            summary="Scale a part uniformly or by axis-specific factors.",
            required_inputs=["factor"], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        try:
            from build123d import scale, Axis
        except ImportError as e:
            return OperationResult(op_name="scale", ok=False, error=str(e))
        f = inputs["factor"]
        try:
            if isinstance(f, (int, float)):
                new_part = scale(part, by=float(f))
            else:
                new_part = scale(part, by=tuple(float(x) for x in f))
        except Exception as e:
            return OperationResult(op_name="scale", ok=False, error=str(e))
        return OperationResult(op_name="scale", ok=True, new_part=new_part,
            undo_data={"previous_part": part}, effect=f"scaled by {f}")
