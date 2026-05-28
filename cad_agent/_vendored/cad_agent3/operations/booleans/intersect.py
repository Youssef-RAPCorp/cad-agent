"""intersect.py — boolean intersection."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class IntersectOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="intersect", category="boolean",
            summary="Boolean intersection of part with another shape.",
            required_inputs=["other"], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        try: new_part = part & inputs["other"]
        except Exception as e:
            return OperationResult(op_name="intersect", ok=False, error=str(e))
        return OperationResult(op_name="intersect", ok=True, new_part=new_part,
            undo_data={"previous_part": part}, effect="boolean intersection applied")
