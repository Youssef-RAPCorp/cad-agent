"""moment_of_inertia.py — compute principal moments of inertia."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult
from ..catalog import register

@register
class MomentOfInertiaOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="moment_of_inertia", category="analysis",
            summary="Principal moments of inertia about the centroid.",
            required_inputs=[], optional_inputs=[])
    @classmethod
    def apply(cls, part, inputs):
        try:
            from OCP.BRepGProp import BRepGProp as brepgprop
            from OCP.GProp import GProp_GProps
            shape = part.wrapped if hasattr(part, "wrapped") else part
            props = GProp_GProps()
            brepgprop.VolumeProperties_s(shape, props)
            m = props.MatrixOfInertia()
            ixx, iyy, izz = m.Value(1,1), m.Value(2,2), m.Value(3,3)
        except Exception as e:
            return OperationResult(op_name="moment_of_inertia", ok=False, error=str(e))
        return OperationResult(op_name="moment_of_inertia", ok=True, new_part=part,
            undo_data={}, effect=f"Ixx={ixx:.1f}, Iyy={iyy:.1f}, Izz={izz:.1f}",
            metrics={"ixx": ixx, "iyy": iyy, "izz": izz})
