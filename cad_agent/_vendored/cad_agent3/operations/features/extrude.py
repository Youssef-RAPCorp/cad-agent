"""extrude.py — extrude a 2D sketch into 3D."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class ExtrudeOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="extrude", category="feature",
            summary="Extrude a 2D sketch (or face) into a 3D solid.",
            required_inputs=["sketch", "amount_mm"],
            optional_inputs=["both_directions"])
    @classmethod
    def apply(cls, part, inputs):
        try: from build123d import extrude
        except ImportError as e: return OperationResult(op_name="extrude", ok=False, error=str(e))
        try:
            new_geom = extrude(inputs["sketch"], amount=float(inputs["amount_mm"]),
                                 both=bool(inputs.get("both_directions", False)))
            new_part = (part + new_geom) if part is not None else new_geom
        except Exception as e:
            return OperationResult(op_name="extrude", ok=False, error=f"extrude failed: {e}")
        return OperationResult(op_name="extrude", ok=True, new_part=new_part,
            undo_data={"previous_part": part},
            effect=f"extruded sketch by {inputs['amount_mm']}mm")
