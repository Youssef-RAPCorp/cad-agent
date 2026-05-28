"""revolve.py — revolve a 2D sketch around an axis to form a solid of revolution."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class RevolveOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="revolve", category="feature",
            summary="Revolve a 2D sketch around an axis to make a solid of revolution.",
            required_inputs=["sketch"],
            optional_inputs=["axis", "angle_deg"])
    @classmethod
    def apply(cls, part, inputs):
        try: from build123d import revolve, Axis
        except ImportError as e: return OperationResult(op_name="revolve", ok=False, error=str(e))
        axis = inputs.get("axis", "Z")
        ax = {"X": Axis.X, "Y": Axis.Y, "Z": Axis.Z}.get(axis.upper(), Axis.Z) \
             if isinstance(axis, str) else axis
        ang = float(inputs.get("angle_deg", 360))
        try:
            new_geom = revolve(inputs["sketch"], axis=ax, revolution_arc=ang)
            new_part = (part + new_geom) if part is not None else new_geom
        except Exception as e:
            return OperationResult(op_name="revolve", ok=False, error=f"revolve failed: {e}")
        return OperationResult(op_name="revolve", ok=True, new_part=new_part,
            undo_data={"previous_part": part},
            effect=f"revolved by {ang}° around {axis}")
