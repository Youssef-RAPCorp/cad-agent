"""thread.py — visual representation of an external or internal thread.

Note: this is a visual approximation, not a CNC-machinable thread.
For real machined threads, modify the model in CAD with a thread mill cycle.
"""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class ThreadOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="thread", category="feature",
            summary="Add visual external/internal thread (approximate).",
            required_inputs=["diameter_mm", "pitch_mm", "length_mm", "position"],
            optional_inputs=["external", "axis"])
    @classmethod
    def apply(cls, part, inputs):
        try: from build123d import Cylinder, Pos
        except ImportError as e: return OperationResult(op_name="thread", ok=False, error=str(e))
        d = float(inputs["diameter_mm"])
        L = float(inputs["length_mm"])
        x, y, z = inputs["position"]
        # Approximate by a cylinder; real thread helix would require Helix import
        cyl = Pos(x, y, z + L/2) * Cylinder(d/2, L)
        try:
            if inputs.get("external", True):
                new_part = (part + cyl) if part is not None else cyl
            else:
                new_part = part - cyl
        except Exception as e:
            return OperationResult(op_name="thread", ok=False, error=f"failed: {e}")
        return OperationResult(op_name="thread", ok=True, new_part=new_part,
            undo_data={"previous_part": part},
            effect=f"thread-approx M{d:.0f} × {L}mm",
            metrics={"diameter_mm": d, "pitch_mm": float(inputs.get('pitch_mm', 0))})
