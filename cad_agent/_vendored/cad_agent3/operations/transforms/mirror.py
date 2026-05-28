"""mirror.py — mirror a part across a plane (XY, XZ, or YZ)."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class MirrorOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="mirror", category="transform",
            summary="Mirror a part across XY, XZ, or YZ plane.",
            required_inputs=["plane"], optional_inputs=["combine"])
    @classmethod
    def apply(cls, part, inputs):
        try: from build123d import mirror, Plane
        except ImportError as e:
            return OperationResult(op_name="mirror", ok=False, error=str(e))
        pl = str(inputs["plane"]).upper()
        plane = {"XY": Plane.XY, "XZ": Plane.XZ, "YZ": Plane.YZ}.get(pl)
        if plane is None:
            return OperationResult(op_name="mirror", ok=False, error=f"unknown plane {pl}")
        try:
            mirrored = mirror(part, about=plane)
            new_part = (part + mirrored) if inputs.get("combine", False) else mirrored
        except Exception as e:
            return OperationResult(op_name="mirror", ok=False, error=str(e))
        return OperationResult(op_name="mirror", ok=True, new_part=new_part,
            undo_data={"previous_part": part}, effect=f"mirrored about {pl}")
