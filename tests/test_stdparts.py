"""Tests for the deterministic precision-part helpers injected into the
codegen sandbox (accurate involute gears)."""
import pytest

build123d = pytest.importorskip("build123d")

from build123d import Pos, Rot

from cad_agent._vendored.cad_agent3.shape_generator import _execute_code
from cad_agent._vendored.cad_agent3.stdparts import involute_gear


def test_gear_proportions_exact():
    g = involute_gear(module=1.0, teeth=24, thickness=4.0, bore=6.0)
    bb = g.bounding_box()
    assert bb.max.X - bb.min.X == pytest.approx(26.0, abs=1e-3)  # m*(z+2)
    assert bb.max.Z - bb.min.Z == pytest.approx(4.0, abs=1e-6)
    assert g.volume > 0


def test_gear_pairs_mesh_without_interference():
    """At the exact center distance m*(z1+z2)/2, correctly phased pairs
    must not interpenetrate — this is what 'accurate cogs' means."""
    m, t = 0.8, 2.0
    for z1, z2 in ((36, 12), (48, 8), (24, 24)):
        g1 = Rot(0, 0, -90) * involute_gear(module=m, teeth=z1, thickness=t)
        phase = -90 + (180.0 / z2 if z2 % 2 == 0 else 0.0)
        g2 = (Pos(m * (z1 + z2) / 2, 0, 0) * Rot(0, 0, phase)
              * involute_gear(module=m, teeth=z2, thickness=t))
        inter = g1 & g2
        try:
            v = inter.volume
        except Exception:
            v = 0.0
        assert v == pytest.approx(0.0, abs=1e-6), f"z{z1}/z{z2} interferes"


def test_helper_available_in_codegen_sandbox():
    code = """
from build123d import *
g = involute_gear(module=1.0, teeth=12, thickness=3.0, bore=2.0)
part = g + Pos(12.0, 0, 0) * Rot(0, 0, 15) * g
"""
    part, err = _execute_code(code)
    assert part is not None, err
    assert part.volume > 0


def test_gear_rejects_degenerate_params():
    with pytest.raises(ValueError):
        involute_gear(module=1.0, teeth=3)
