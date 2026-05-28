"""faces_by_normal.py — find faces whose normal is parallel/anti-parallel to a vector."""
from __future__ import annotations
import math
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class FacesByNormalOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="faces_by_normal", category="selector",
            summary="Select faces whose normal is parallel to a given direction.",
            required_inputs=["direction"], optional_inputs=["tolerance_deg"])
    @classmethod
    def apply(cls, part, inputs):
        try:
            from build123d import Vector
        except ImportError as e:
            return OperationResult(op_name="faces_by_normal", ok=False, error=str(e))
        dx, dy, dz = inputs["direction"]
        tol = float(inputs.get("tolerance_deg", 1.0))
        target = Vector(dx, dy, dz).normalized()
        cos_tol = math.cos(math.radians(tol))
        matched = []
        for f in part.faces():
            try:
                n = f.normal_at(f.center()).normalized()
                if abs(n.dot(target)) >= cos_tol:
                    matched.append(f)
            except Exception:
                continue
        return OperationResult(op_name="faces_by_normal", ok=True, new_part=part,
            undo_data={}, effect=f"selected {len(matched)} faces normal to {inputs['direction']}",
            metrics={"selected_count": len(matched)},
        )
