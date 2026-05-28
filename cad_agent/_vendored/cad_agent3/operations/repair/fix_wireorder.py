"""fix_wireorder.py — repair wires whose edges are in the wrong direction."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class FixWireOrderOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="fix_wireorder", category="repair",
            summary="Reorder wire edges so they form a valid loop (ShapeFix_Wire).",
            required_inputs=[], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        try:
            from OCP.ShapeFix import ShapeFix_Wireframe
            from build123d import Compound
            sf = ShapeFix_Wireframe()
            shape = part.wrapped if hasattr(part, "wrapped") else part
            sf.SetShape(shape); sf.FixWireGaps()
            new_shape = sf.Shape()
        except Exception as e:
            return OperationResult(op_name="fix_wireorder", ok=False, error=str(e))
        return OperationResult(op_name="fix_wireorder", ok=True, new_part=Compound(new_shape),
            undo_data={"previous_part": part}, effect="wire edges reordered")
