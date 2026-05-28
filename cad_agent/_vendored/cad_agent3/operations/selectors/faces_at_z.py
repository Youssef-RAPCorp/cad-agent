"""faces_at_z.py — select faces whose center is near a given Z coordinate."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class FacesAtZOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="faces_at_z", category="selector",
            summary="Select faces whose center Z is within a tolerance of a target.",
            required_inputs=["z_mm"], optional_inputs=["tolerance_mm"])
    @classmethod
    def apply(cls, part, inputs):
        z = float(inputs["z_mm"]); tol = float(inputs.get("tolerance_mm", 0.5))
        matched = []
        for f in part.faces():
            try:
                if abs(f.center().Z - z) <= tol:
                    matched.append(f)
            except Exception:
                continue
        return OperationResult(op_name="faces_at_z", ok=True, new_part=part,
            undo_data={}, effect=f"{len(matched)} faces at z={z}±{tol}mm",
            metrics={"selected_count": len(matched)})
