"""cut.py — boolean subtraction (cut)."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class CutOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="cut", category="boolean",
            summary="Boolean cut: remove `other` from `part`.",
            required_inputs=["other"], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        try: new_part = part - inputs["other"]
        except Exception as e:
            return OperationResult(op_name="cut", ok=False, error=str(e))
        return OperationResult(op_name="cut", ok=True, new_part=new_part,
            undo_data={"previous_part": part}, effect="boolean cut applied")
