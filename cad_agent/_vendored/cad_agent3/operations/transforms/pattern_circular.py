"""pattern_circular.py — duplicate a part around an axis."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class PatternCircularOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="pattern_circular", category="transform",
            summary="Duplicate a feature N times around an axis.",
            required_inputs=["count"], optional_inputs=["axis", "total_angle_deg", "combine"])
    @classmethod
    def apply(cls, part, inputs):
        try: from build123d import Rot
        except ImportError as e:
            return OperationResult(op_name="pattern_circular", ok=False, error=str(e))
        n = int(inputs["count"])
        ax = str(inputs.get("axis", "Z")).upper()
        total = float(inputs.get("total_angle_deg", 360))
        step = total / n if total < 360 else total / n
        copies = []
        for i in range(n):
            a = step * i
            rx, ry, rz = (a,0,0) if ax=="X" else (0,a,0) if ax=="Y" else (0,0,a)
            copies.append(Rot(rx, ry, rz) * part)
        if inputs.get("combine", True):
            new_part = copies[0]
            for c in copies[1:]: new_part = new_part + c
        else:
            new_part = copies
        return OperationResult(op_name="pattern_circular", ok=True, new_part=new_part,
            undo_data={"previous_part": part},
            effect=f"circular pattern: {n} copies around {ax}",
            metrics={"count": n})
