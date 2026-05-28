"""shape_fix.py — heal small geometry defects using OCCT ShapeFix.

ShapeFix corrects: bad face orientations, gaps in shells, malformed wires.
Most useful after import of foreign STEP / STL files or after operations
that produce slightly invalid geometry.
"""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class ShapeFixOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="shape_fix", category="repair",
            summary="Heal geometry defects (gaps, wrong face orientations, bad wires).",
            required_inputs=[], optional_inputs=["tolerance_mm"])
    @classmethod
    def apply(cls, part, inputs):
        try:
            from OCP.ShapeFix import ShapeFix_Shape
            shape = part.wrapped if hasattr(part, "wrapped") else part
            fixer = ShapeFix_Shape(shape)
            tol = float(inputs.get("tolerance_mm", 0.01))
            fixer.SetPrecision(tol); fixer.SetMaxTolerance(tol * 100)
            ok = fixer.Perform()
            fixed_shape = fixer.Shape()
            from build123d import Solid, Compound
            new_part = Compound(fixed_shape) if fixed_shape.ShapeType() == 1 \
                       else Solid(fixed_shape)
        except Exception as e:
            return OperationResult(op_name="shape_fix", ok=False, error=str(e))
        return OperationResult(op_name="shape_fix", ok=bool(ok), new_part=new_part,
            undo_data={"previous_part": part},
            effect=("ShapeFix applied" if ok else "ShapeFix made no changes"),
            metrics={"applied": bool(ok)})
