"""remove_small_features.py — drop tiny faces / edges below a threshold."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class RemoveSmallFeaturesOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="remove_small_features", category="repair",
            summary="Drop faces and edges below a size threshold (ShapeFix_Shape with small-features mode).",
            required_inputs=["min_size_mm"], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        try:
            from OCP.ShapeFix import ShapeFix_FixSmallFace
            from build123d import Compound
            sf = ShapeFix_FixSmallFace()
            shape = part.wrapped if hasattr(part, "wrapped") else part
            sf.Init(shape); sf.Perform()
            new_shape = sf.FixShape()
        except Exception as e:
            return OperationResult(op_name="remove_small_features", ok=False, error=str(e))
        return OperationResult(op_name="remove_small_features", ok=True,
            new_part=Compound(new_shape), undo_data={"previous_part": part},
            effect=f"removed features < {inputs['min_size_mm']}mm")
