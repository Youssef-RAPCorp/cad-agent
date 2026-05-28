"""mass.py — estimate part mass given a material density."""
from __future__ import annotations
from ..operation_base import Operation, OperationDecl, OperationResult, _safe_volume
from ..catalog import register

DENSITIES = {
    "PLA": 1.25, "ABS": 1.04, "PETG": 1.27, "TPU": 1.21, "Nylon": 1.13,
    "ASA": 1.07, "PC": 1.20,
    "Aluminum": 2.70, "Steel": 7.85, "Stainless": 8.00, "Brass": 8.50,
    "Copper": 8.96, "Cardboard": 0.7,
}

@register
class MassOp(Operation):
    @classmethod
    def declare(cls):
        return OperationDecl(name="mass", category="analysis",
            summary="Estimate mass given a material name or density.",
            required_inputs=["material"], optional_inputs=["density_g_per_cc"])
    @classmethod
    def apply(cls, part, inputs):
        mat = str(inputs["material"])
        rho = inputs.get("density_g_per_cc")
        if rho is None:
            rho = DENSITIES.get(mat)
            if rho is None:
                return OperationResult(op_name="mass", ok=False,
                    error=f"unknown material {mat!r}; supply density_g_per_cc")
        rho = float(rho)
        v_mm3 = _safe_volume(part)
        m_g = v_mm3 / 1000.0 * rho
        return OperationResult(op_name="mass", ok=True, new_part=part,
            undo_data={}, effect=f"mass ≈ {m_g:.1f}g ({mat}, ρ={rho:.2f} g/cc)",
            metrics={"mass_g": m_g, "density_g_per_cc": rho, "material": mat})
