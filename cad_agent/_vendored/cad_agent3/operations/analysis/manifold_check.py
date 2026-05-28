"""manifold_check.py — verify a part is closed/manifold using OCCT BRepCheck."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class ManifoldCheckOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="manifold_check", category="analysis",
            summary="Verify the part is a valid manifold solid (BRepCheck).",
            required_inputs=[], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        try:
            from OCP.BRepCheck import BRepCheck_Analyzer
            shape = part.wrapped if hasattr(part, "wrapped") else part
            analyzer = BRepCheck_Analyzer(shape)
            valid = bool(analyzer.IsValid())
        except Exception as e:
            return OperationResult(op_name="manifold_check", ok=False, error=str(e))
        return OperationResult(op_name="manifold_check", ok=True, new_part=part,
            undo_data={}, effect=("manifold OK" if valid else "INVALID — see ShapeFix"),
            metrics={"valid": valid})
