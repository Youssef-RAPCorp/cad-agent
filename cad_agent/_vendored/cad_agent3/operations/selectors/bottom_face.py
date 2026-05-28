"""bottom_face.py — select the bottommost (min Z) flat face."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class BottomFaceOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="bottom_face", category="selector",
            summary="Select the bottommost face (lowest Z).",
            required_inputs=[], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        best = None; best_z = float("inf")
        for f in part.faces():
            try:
                c = f.center()
                if c.Z < best_z:
                    best_z = c.Z; best = f
            except Exception:
                continue
        if best is None:
            return OperationResult(op_name="bottom_face", ok=False, error="no faces found")
        return OperationResult(op_name="bottom_face", ok=True, new_part=part,
            undo_data={}, effect=f"bottom face at z={best_z:.2f}",
            metrics={"face_z": best_z})
