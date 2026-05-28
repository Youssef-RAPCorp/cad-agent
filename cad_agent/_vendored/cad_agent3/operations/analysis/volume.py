"""volume.py — compute the volume of a part in mm³."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult, _safe_volume
from ..catalog import register

@register
class VolumeOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="volume", category="analysis",
            summary="Compute the volume of a part in mm³.",
            required_inputs=[], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        v = _safe_volume(part)
        return OperationResult(op_name="volume", ok=True, new_part=part,
            undo_data={}, effect=f"volume = {v:.0f} mm³ ({v/1000:.2f} cm³)",
            metrics={"volume_mm3": v, "volume_cm3": v/1000})
