"""interference.py — check whether two parts physically intersect."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult, _safe_volume
from ..catalog import register

@register
class InterferenceOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="interference", category="analysis",
            summary="Check whether two parts intersect (overlap volume > tolerance).",
            required_inputs=["part_a", "part_b"], optional_inputs=["tolerance_mm3"])
    @classmethod
    def apply(cls, part, inputs):
        a, b = inputs["part_a"], inputs["part_b"]
        tol = float(inputs.get("tolerance_mm3", 0.01))
        try:
            inter = a & b
            v = _safe_volume(inter)
        except Exception as e:
            return OperationResult(op_name="interference", ok=False, error=str(e))
        clear = v <= tol
        return OperationResult(op_name="interference", ok=True, new_part=part,
            undo_data={},
            effect=("clear" if clear else f"interference: {v:.2f}mm³ overlap"),
            metrics={"overlap_mm3": v, "clear": clear})
