"""pattern_linear.py — duplicate a part along a vector at fixed spacing."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class PatternLinearOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="pattern_linear", category="transform",
            summary="Duplicate a feature N times along a vector.",
            required_inputs=["count", "vector"], optional_inputs=["combine"])
    @classmethod
    def apply(cls, part, inputs):
        try: from build123d import Pos
        except ImportError as e:
            return OperationResult(op_name="pattern_linear", ok=False, error=str(e))
        n = int(inputs["count"])
        dx, dy, dz = inputs["vector"]
        copies = [Pos(dx*i, dy*i, dz*i) * part for i in range(n)]
        if inputs.get("combine", True):
            new_part = copies[0]
            for c in copies[1:]: new_part = new_part + c
        else:
            new_part = copies
        return OperationResult(op_name="pattern_linear", ok=True, new_part=new_part,
            undo_data={"previous_part": part}, effect=f"linear pattern: {n} copies",
            metrics={"count": n})
