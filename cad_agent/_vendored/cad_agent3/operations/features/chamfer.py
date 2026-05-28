"""chamfer.py — bevel selected edges of a part."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register


@register
class ChamferOp(Operation):

    @classmethod
    def declare(cls):
        return OperationDecl(
            name="chamfer",
            category="feature",
            summary="Chamfer (bevel) selected edges of a part.",
            required_inputs=["distance_mm"],
            optional_inputs=["edge_selector", "axis_filter", "distance2_mm"],
        )

    @classmethod
    def apply(cls, part, inputs):
        try:
            from build123d import chamfer, Axis
        except ImportError as e:
            return OperationResult(op_name="chamfer", ok=False,
                                     error=f"build123d not available: {e}")
        d = float(inputs["distance_mm"])
        d2 = inputs.get("distance2_mm")
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
            if d2 is not None:
                new_part = chamfer(edges, length=d, length2=float(d2))
            else:
                new_part = chamfer(edges, length=d)
        except Exception as e:
            return OperationResult(op_name="chamfer", ok=False,
                                     error=f"chamfer failed: {e}")
        return OperationResult(
            op_name="chamfer", ok=True, new_part=new_part,
            undo_data={"previous_part": part},
            effect=f"chamfered {len(edges)} edges (d={d}mm)",
            metrics={"distance_mm": d, "edge_count": len(edges)},
        )
