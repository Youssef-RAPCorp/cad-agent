"""shell.py — hollow out a part to a given wall thickness.

Optional list of faces to remove (open the shell). If no faces given,
the result is a fully closed hollow shell.
"""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register


@register
class ShellOp(Operation):

    @classmethod
    def declare(cls):
        return OperationDecl(
            name="shell",
            category="feature",
            summary="Hollow a part to a given wall thickness.",
            required_inputs=["thickness_mm"],
            optional_inputs=["open_faces"],
        )

    @classmethod
    def apply(cls, part, inputs):
        try:
            from build123d import offset, Kind
        except ImportError as e:
            return OperationResult(op_name="shell", ok=False,
                                     error=f"build123d not available: {e}")
        t = float(inputs["thickness_mm"])
        open_faces = inputs.get("open_faces")
        try:
            if open_faces:
                new_part = offset(part, amount=-t,
                                    openings=open_faces, kind=Kind.INTERSECTION)
            else:
                new_part = offset(part, amount=-t, kind=Kind.INTERSECTION)
        except Exception as e:
            return OperationResult(op_name="shell", ok=False,
                                     error=f"shell failed: {e}")
        return OperationResult(
            op_name="shell", ok=True, new_part=new_part,
            undo_data={"previous_part": part},
            effect=f"shelled with wall thickness {t}mm",
            metrics={"thickness_mm": t},
        )
