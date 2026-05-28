"""simplify.py — merge coplanar faces and collinear edges."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class SimplifyOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="simplify", category="repair",
            summary="Merge coplanar faces and collinear edges (ShapeUpgrade).",
            required_inputs=[], optional_inputs=["tolerance_mm"])
    @classmethod
    def apply(cls, part, inputs):
        try:
            from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
            from build123d import Compound
            shape = part.wrapped if hasattr(part, "wrapped") else part
            unifier = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
            tol = float(inputs.get("tolerance_mm", 0.001))
            unifier.SetLinearTolerance(tol); unifier.SetAngularTolerance(0.001)
            unifier.Build()
            new_shape = unifier.Shape()
        except Exception as e:
            return OperationResult(op_name="simplify", ok=False, error=str(e))
        return OperationResult(op_name="simplify", ok=True, new_part=Compound(new_shape),
            undo_data={"previous_part": part}, effect="merged coplanar faces & collinear edges")
