"""face_count.py — count faces, edges, vertices of a part."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class FaceCountOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="face_count", category="analysis",
            summary="Count faces, edges, and vertices of a part.",
            required_inputs=[], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        try:
            nf = len(part.faces()); ne = len(part.edges()); nv = len(part.vertices())
        except Exception as e:
            return OperationResult(op_name="face_count", ok=False, error=str(e))
        return OperationResult(op_name="face_count", ok=True, new_part=part,
            undo_data={}, effect=f"faces={nf}, edges={ne}, vertices={nv}",
            metrics={"face_count": nf, "edge_count": ne, "vertex_count": nv})
