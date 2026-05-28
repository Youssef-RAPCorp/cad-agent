"""split.py — split a part by a plane."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class SplitOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="split", category="boolean",
            summary="Split a part by a plane (returns one or both halves).",
            required_inputs=["plane"], optional_inputs=["keep"])
    @classmethod
    def apply(cls, part, inputs):
        try: from build123d import split, Plane, Keep
        except ImportError as e:
            return OperationResult(op_name="split", ok=False, error=str(e))
        pl = str(inputs["plane"]).upper()
        plane = {"XY": Plane.XY, "XZ": Plane.XZ, "YZ": Plane.YZ}.get(pl)
        if plane is None:
            return OperationResult(op_name="split", ok=False, error=f"unknown plane {pl}")
        keep_str = str(inputs.get("keep", "top")).lower()
        keep = {"top": Keep.TOP, "bottom": Keep.BOTTOM, "both": Keep.BOTH}.get(keep_str, Keep.TOP)
        try: new_part = split(part, bisect_by=plane, keep=keep)
        except Exception as e:
            return OperationResult(op_name="split", ok=False, error=str(e))
        return OperationResult(op_name="split", ok=True, new_part=new_part,
            undo_data={"previous_part": part}, effect=f"split by {pl} plane (keep={keep_str})")
