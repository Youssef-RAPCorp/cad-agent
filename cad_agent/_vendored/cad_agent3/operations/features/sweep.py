"""sweep.py — sweep a profile along a path."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class SweepOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="sweep", category="feature",
            summary="Sweep a 2D profile along a 3D path to form a solid.",
            required_inputs=["profile", "path"], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        try: from build123d import sweep
        except ImportError as e: return OperationResult(op_name="sweep", ok=False, error=str(e))
        try:
            new_geom = sweep(inputs["profile"], path=inputs["path"])
            new_part = (part + new_geom) if part is not None else new_geom
        except Exception as e:
            return OperationResult(op_name="sweep", ok=False, error=f"sweep failed: {e}")
        return OperationResult(op_name="sweep", ok=True, new_part=new_part,
            undo_data={"previous_part": part},
            effect="swept profile along path")
