"""rotate.py — rotate a part around the X, Y, or Z axis."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class RotateOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="rotate", category="transform",
            summary="Rotate a part around X, Y, or Z axis by an angle.",
            required_inputs=["axis", "angle_deg"], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        try: from build123d import Rot
        except ImportError as e:
            return OperationResult(op_name="rotate", ok=False, error=str(e))
        ax = str(inputs["axis"]).upper()
        a = float(inputs["angle_deg"])
        rx, ry, rz = (a, 0, 0) if ax == "X" else (0, a, 0) if ax == "Y" else (0, 0, a)
        new_part = Rot(rx, ry, rz) * part
        return OperationResult(op_name="rotate", ok=True, new_part=new_part,
            undo_data={"previous_part": part, "axis": ax, "angle_deg": -a},
            effect=f"rotated {a}° around {ax}")
