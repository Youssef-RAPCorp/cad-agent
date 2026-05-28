"""centroid.py — compute the centroid (center of mass for uniform density)."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class CentroidOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="centroid", category="analysis",
            summary="Compute the centroid (center of mass for uniform density).",
            required_inputs=[], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        try: c = part.center()
        except Exception as e:
            return OperationResult(op_name="centroid", ok=False, error=str(e))
        return OperationResult(op_name="centroid", ok=True, new_part=part,
            undo_data={}, effect=f"centroid: ({c.X:.2f}, {c.Y:.2f}, {c.Z:.2f})",
            metrics={"x_mm": c.X, "y_mm": c.Y, "z_mm": c.Z})
