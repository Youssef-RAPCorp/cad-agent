"""union.py — boolean union (combine) of two parts."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class UnionOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="union", category="boolean",
            summary="Boolean union of part with another shape.",
            required_inputs=["other"], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        try: new_part = part + inputs["other"]
        except Exception as e:
            return OperationResult(op_name="union", ok=False, error=str(e))
        return OperationResult(op_name="union", ok=True, new_part=new_part,
            undo_data={"previous_part": part}, effect="boolean union applied")
