"""surface_area.py — total surface area in mm²."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class SurfaceAreaOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="surface_area", category="analysis",
            summary="Total surface area of a part in mm².",
            required_inputs=[], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        try:
            a = sum(f.area for f in part.faces())
        except Exception as e:
            return OperationResult(op_name="surface_area", ok=False, error=str(e))
        return OperationResult(op_name="surface_area", ok=True, new_part=part,
            undo_data={}, effect=f"surface area = {a:.0f} mm² ({a/100:.2f} cm²)",
            metrics={"surface_area_mm2": a})
