"""sew_shells.py — sew adjacent faces into a closed shell."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class SewShellsOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="sew_shells", category="repair",
            summary="Sew adjacent faces into a closed shell (BRepBuilderAPI_Sewing).",
            required_inputs=[], optional_inputs=["tolerance_mm"])
    @classmethod
    def apply(cls, part, inputs):
        try:
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
            from build123d import Solid, Compound
            tol = float(inputs.get("tolerance_mm", 0.01))
            sewer = BRepBuilderAPI_Sewing(tol)
            shape = part.wrapped if hasattr(part, "wrapped") else part
            sewer.Add(shape); sewer.Perform()
            new_shape = sewer.SewedShape()
        except Exception as e:
            return OperationResult(op_name="sew_shells", ok=False, error=str(e))
        return OperationResult(op_name="sew_shells", ok=True, new_part=Compound(new_shape),
            undo_data={"previous_part": part},
            effect="sewed adjacent faces", metrics={"tolerance_mm": tol})
