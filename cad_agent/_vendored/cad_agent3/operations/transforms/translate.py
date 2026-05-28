"""translate.py — move a part by a vector."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class TranslateOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="translate", category="transform",
            summary="Translate a part by a (dx, dy, dz) vector.",
            required_inputs=["vector"], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        try: from build123d import Pos
        except ImportError as e:
            return OperationResult(op_name="translate", ok=False, error=str(e))
        dx, dy, dz = inputs["vector"]
        new_part = Pos(dx, dy, dz) * part
        return OperationResult(op_name="translate", ok=True, new_part=new_part,
            undo_data={"previous_part": part, "vector": (-dx, -dy, -dz)},
            effect=f"translated by ({dx}, {dy}, {dz})")
