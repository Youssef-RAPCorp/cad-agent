"""loft.py — loft between sketches at different positions."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class LoftOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="loft", category="feature",
            summary="Loft a solid between two or more cross-section sketches.",
            required_inputs=["sketches"],
            optional_inputs=["ruled"])
    @classmethod
    def apply(cls, part, inputs):
        try: from build123d import loft
        except ImportError as e: return OperationResult(op_name="loft", ok=False, error=str(e))
        try:
            new_geom = loft(inputs["sketches"], ruled=bool(inputs.get("ruled", False)))
            new_part = (part + new_geom) if part is not None else new_geom
        except Exception as e:
            return OperationResult(op_name="loft", ok=False, error=f"loft failed: {e}")
        return OperationResult(op_name="loft", ok=True, new_part=new_part,
            undo_data={"previous_part": part},
            effect=f"lofted between {len(inputs['sketches'])} sketches")
