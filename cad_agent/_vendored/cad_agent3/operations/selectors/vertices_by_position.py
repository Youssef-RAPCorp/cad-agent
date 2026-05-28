"""vertices_by_position.py — select vertices in a 3D bounding box."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class VerticesByPositionOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="vertices_by_position", category="selector",
            summary="Select vertices inside a given 3D bounding box.",
            required_inputs=["min_xyz", "max_xyz"], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        x0, y0, z0 = inputs["min_xyz"]
        x1, y1, z1 = inputs["max_xyz"]
        matched = []
        for v in part.vertices():
            try:
                p = v.center() if hasattr(v, "center") else v
                if x0 <= p.X <= x1 and y0 <= p.Y <= y1 and z0 <= p.Z <= z1:
                    matched.append(v)
            except Exception:
                continue
        return OperationResult(op_name="vertices_by_position", ok=True, new_part=part,
            undo_data={}, effect=f"selected {len(matched)} vertices in box",
            metrics={"selected_count": len(matched)})
