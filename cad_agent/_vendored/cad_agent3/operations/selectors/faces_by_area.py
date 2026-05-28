"""faces_by_area.py — select faces by area range."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class FacesByAreaOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="faces_by_area", category="selector",
            summary="Select faces whose area is within a given range.",
            required_inputs=["min_mm2", "max_mm2"], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        lo = float(inputs["min_mm2"]); hi = float(inputs["max_mm2"])
        matched = [f for f in part.faces() if lo <= f.area <= hi]
        return OperationResult(op_name="faces_by_area", ok=True, new_part=part,
            undo_data={}, effect=f"selected {len(matched)} faces in [{lo},{hi}]mm²",
            metrics={"selected_count": len(matched)})
