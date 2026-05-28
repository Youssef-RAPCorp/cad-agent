"""planar_faces.py — select all planar faces."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class PlanarFacesOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="planar_faces", category="selector",
            summary="Select all planar faces of a part.",
            required_inputs=[], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        try: from build123d import GeomType
        except ImportError as e:
            return OperationResult(op_name="planar_faces", ok=False, error=str(e))
        matched = [f for f in part.faces() if f.geom_type == GeomType.PLANE]
        return OperationResult(op_name="planar_faces", ok=True, new_part=part,
            undo_data={}, effect=f"selected {len(matched)} planar faces",
            metrics={"selected_count": len(matched)})
