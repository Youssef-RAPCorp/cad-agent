"""edges_by_length.py — select edges within a length range."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class EdgesByLengthOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="edges_by_length", category="selector",
            summary="Select edges whose length is within a given range.",
            required_inputs=["min_mm", "max_mm"], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        lo = float(inputs["min_mm"]); hi = float(inputs["max_mm"])
        matched = [e for e in part.edges() if lo <= e.length <= hi]
        return OperationResult(op_name="edges_by_length", ok=True, new_part=part,
            undo_data={}, effect=f"selected {len(matched)} edges in [{lo}, {hi}]mm",
            metrics={"selected_count": len(matched)})
