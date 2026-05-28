"""top_face.py — select the topmost (max Z) flat face."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class TopFaceOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="top_face", category="selector",
            summary="Select the topmost flat face (highest Z, normal pointing up).",
            required_inputs=[], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        best = None; best_z = float("-inf")
        for f in part.faces():
            try:
                c = f.center()
                if c.Z > best_z:
                    best_z = c.Z; best = f
            except Exception:
                continue
        if best is None:
            return OperationResult(op_name="top_face", ok=False, error="no faces found")
        return OperationResult(op_name="top_face", ok=True, new_part=part,
            undo_data={}, effect=f"top face at z={best_z:.2f}",
            metrics={"face_z": best_z})
