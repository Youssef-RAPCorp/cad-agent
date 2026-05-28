"""fillet.py — round selected edges of a part.

Inputs:
  radius_mm (required)
  edge_selector (optional): a callable that takes a build123d Part and
                            returns a ShapeList of edges. If absent,
                            fillets ALL edges.
"""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register


@register
class FilletOp(Operation):

    @classmethod
    def declare(cls):
        return OperationDecl(
            name="fillet",
            category="feature",
            summary="Fillet (round) selected edges of a part.",
            required_inputs=["radius_mm"],
            optional_inputs=["edge_selector", "axis_filter"],
        )

    @classmethod
    def apply(cls, part, inputs):
        try:
            from build123d import fillet, Axis
        except ImportError as e:
            return OperationResult(op_name="fillet", ok=False,
                                     error=f"build123d not available: {e}")
        r = float(inputs["radius_mm"])
        sel = inputs.get("edge_selector")
        axis_filter = inputs.get("axis_filter")
        try:
            edges = part.edges()
            if sel is not None:
                edges = sel(part)
            elif axis_filter is not None:
                axis_map = {"X": Axis.X, "Y": Axis.Y, "Z": Axis.Z}
                ax = axis_map.get(axis_filter.upper())
                if ax is not None:
                    edges = edges | ax
            new_part = fillet(edges, radius=r)
        except Exception as e:
            return OperationResult(op_name="fillet", ok=False,
                                     error=f"fillet failed: {type(e).__name__}: {e}")
        return OperationResult(
            op_name="fillet", ok=True, new_part=new_part,
            undo_data={"previous_part": part},
            effect=f"filleted {len(edges)} edges with r={r}mm",
            metrics={"radius_mm": r, "edge_count": len(edges)},
        )
