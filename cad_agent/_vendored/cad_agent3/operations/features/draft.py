"""draft.py — apply a draft angle to selected faces (for moldability)."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class DraftOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="draft", category="feature",
            summary="Apply draft angle to faces (for injection-molded parts).",
            required_inputs=["angle_deg"],
            optional_inputs=["face_selector"])
    @classmethod
    def apply(cls, part, inputs):
        # build123d 0.10 doesn't have a built-in draft op — we'd need to
        # use offset on selected faces. This is a placeholder that warns
        # the user.
        return OperationResult(op_name="draft", ok=False,
            error="draft is not yet implemented; use chamfer or rebuild geometry")
