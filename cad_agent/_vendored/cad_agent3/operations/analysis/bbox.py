"""bbox.py — compute the axis-aligned bounding box of a part."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult, _bbox_dict
from ..catalog import register

@register
class BboxOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="bbox", category="analysis",
            summary="Axis-aligned bounding box (min_xyz, max_xyz, size_xyz).",
            required_inputs=[], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        bb = _bbox_dict(part)
        return OperationResult(op_name="bbox", ok=True, new_part=part,
            undo_data={},
            effect=(f"bbox: {bb.get('size_x',0):.0f} × {bb.get('size_y',0):.0f} × "
                    f"{bb.get('size_z',0):.0f} mm"),
            metrics=bb)
