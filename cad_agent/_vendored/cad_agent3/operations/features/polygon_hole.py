"""polygon_hole.py — drill a regular polygon hole (hex, square, etc.)."""
from __future__ import annotations
import math
from ..operation_base import Operation, OperationDecl, OperationResult, _bbox_dict
from ..catalog import register

@register
class PolygonHoleOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="polygon_hole", category="feature",
            summary="Cut a regular polygon hole (hex, square, etc.) through a part.",
            required_inputs=["sides", "across_flats_mm", "position"],
            optional_inputs=["depth_mm", "through", "rotation_deg"])
    @classmethod
    def apply(cls, part, inputs):
        try:
            from build123d import RegularPolygon, extrude, Pos, Rot
        except ImportError as e:
            return OperationResult(op_name="polygon_hole", ok=False, error=str(e))
        n = int(inputs["sides"])
        af = float(inputs["across_flats_mm"])
        # Convert "across flats" to circumradius
        r = af / (2 * math.cos(math.pi / n))
        x, y, z = inputs["position"]
        depth = inputs.get("depth_mm")
        through = inputs.get("through", depth is None)
        bb = _bbox_dict(part)
        h = max(bb.get("size_z", 100), 100) * 1.5 + 20 if through else float(depth) + 1
        rot = float(inputs.get("rotation_deg", 0))
        sk = RegularPolygon(r, n)
        cutter = extrude(sk, h)
        cutter = Pos(x, y, z) * Rot(0, 0, rot) * cutter
        try: new_part = part - cutter
        except Exception as e:
            return OperationResult(op_name="polygon_hole", ok=False, error=str(e))
        return OperationResult(op_name="polygon_hole", ok=True, new_part=new_part,
            undo_data={"previous_part": part},
            effect=f"{n}-sided hole, AF={af}mm at ({x:.1f},{y:.1f},{z:.1f})")
