"""tab_and_slot.py — interlocking tab + matching slot pair.

Adds a tab feature on one part position and the matching slot on another.
For laser-cut sheet construction.
"""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class TabAndSlotOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="tab_and_slot", category="feature",
            summary="Add a tab on a part with a matching slot on another panel.",
            required_inputs=["tab_position", "slot_position", "width_mm",
                              "height_mm", "thickness_mm"],
            optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        try: from build123d import Box, Pos
        except ImportError as e:
            return OperationResult(op_name="tab_and_slot", ok=False, error=str(e))
        w = float(inputs["width_mm"]); h = float(inputs["height_mm"])
        t = float(inputs["thickness_mm"])
        tx, ty, tz = inputs["tab_position"]; sx, sy, sz = inputs["slot_position"]
        tab = Pos(tx, ty, tz + t/2) * Box(w, h, t)
        slot = Pos(sx, sy, sz) * Box(w + 0.2, h + 0.2, t * 3)  # slight clearance
        try:
            new_part = part + tab - slot
        except Exception as e:
            return OperationResult(op_name="tab_and_slot", ok=False, error=str(e))
        return OperationResult(op_name="tab_and_slot", ok=True, new_part=new_part,
            undo_data={"previous_part": part},
            effect=f"tab+slot pair, w={w}×h={h}mm")
