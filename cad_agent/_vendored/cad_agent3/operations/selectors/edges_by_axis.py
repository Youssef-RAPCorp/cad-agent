"""edges_by_axis.py — select edges parallel to an axis."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class EdgesByAxisOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="edges_by_axis", category="selector",
            summary="Select edges aligned with X, Y, or Z axis.",
            required_inputs=["axis"], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        try: from build123d import Axis
        except ImportError as e:
            return OperationResult(op_name="edges_by_axis", ok=False, error=str(e))
        ax_name = str(inputs["axis"]).upper()
        ax = {"X": Axis.X, "Y": Axis.Y, "Z": Axis.Z}.get(ax_name)
        if ax is None:
            return OperationResult(op_name="edges_by_axis", ok=False,
                error=f"axis must be X/Y/Z, got {ax_name!r}")
        try: edges = part.edges() | ax
        except Exception as e:
            return OperationResult(op_name="edges_by_axis", ok=False, error=str(e))
        return OperationResult(op_name="edges_by_axis", ok=True, new_part=part,
            undo_data={}, effect=f"selected {len(edges)} edges along {ax_name}",
            metrics={"selected_count": len(edges)})
