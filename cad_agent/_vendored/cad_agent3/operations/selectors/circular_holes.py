"""circular_holes.py — find existing cylindrical holes by diameter."""
from __future__ import annotations
import math
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class CircularHolesOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="circular_holes", category="selector",
            summary="Locate existing cylindrical holes; optionally filter by diameter.",
            required_inputs=[], optional_inputs=["min_diameter_mm", "max_diameter_mm"])
    @classmethod
    def apply(cls, part, inputs):
        # Find faces that are cylindrical surfaces (geometry type "Cylinder")
        try: from build123d import GeomType
        except ImportError as e:
            return OperationResult(op_name="circular_holes", ok=False, error=str(e))
        lo = float(inputs.get("min_diameter_mm", 0))
        hi = float(inputs.get("max_diameter_mm", 1e6))
        holes = []
        for f in part.faces():
            try:
                if f.geom_type == GeomType.CYLINDER:
                    # Diameter from circumference / pi
                    edges = f.edges()
                    if edges:
                        circ_edge = edges[0]
                        d = circ_edge.length / math.pi
                        if lo <= d <= hi:
                            holes.append({"face": f, "diameter_mm": d,
                                          "center": f.center()})
            except Exception:
                continue
        return OperationResult(op_name="circular_holes", ok=True, new_part=part,
            undo_data={}, effect=f"found {len(holes)} circular holes",
            metrics={"hole_count": len(holes), "holes": holes})
