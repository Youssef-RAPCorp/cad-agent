"""Tests for the multi-part orchestration pipeline (cad_agent.assembly).

Planner and per-part codegen are mocked, so everything runs offline.
These exercise: plan validation, deterministic gear primitives, LLM-part
envelope enforcement, instance placement, interference verification,
count checks, the revision loop, and export.
"""
import json
import types

import pytest

build123d = pytest.importorskip("build123d")
pytest.importorskip("pydantic")

from build123d import Box, Pos

import cad_agent.assembly as assembly
import cad_agent._vendored.cad_agent3 as backend
from cad_agent.assembly import AssemblyPlan, AssemblyResult, assemble

# Two meshed gears (m=1, z=24: center distance 24, half-pitch phase 7.5deg)
# plus one LLM part placed clear below them.
GOOD_PLAN = json.dumps({
    "name": "test gearbox",
    "parts": [
        {"id": "gear_z24", "primitive": {"kind": "involute_gear",
                                         "module": 1.0, "teeth": 24,
                                         "thickness": 4.0, "bore": 4.0}},
        {"id": "base_plate",
         "description": "a rectangular base plate 60x30x5mm",
         "envelope": [60, 30, 5]},
    ],
    "instances": [
        {"part": "gear_z24", "at": [0, 0, 0]},
        {"part": "gear_z24", "at": [24, 0, 0], "rotate": [0, 0, 7.5]},
        {"part": "base_plate", "at": [12, 0, -6]},
    ],
    "checks": [{"kind": "count", "pattern": "gear_*", "min": 2, "max": 4}],
})

# Same two gears rammed into each other — must fail interference.
OVERLAP_PLAN = json.dumps({
    "name": "clashing gearbox",
    "parts": [
        {"id": "gear_z24", "primitive": {"kind": "involute_gear",
                                         "module": 1.0, "teeth": 24,
                                         "thickness": 4.0}},
    ],
    "instances": [
        {"part": "gear_z24", "at": [0, 0, 0]},
        {"part": "gear_z24", "at": [10, 0, 0]},
    ],
})

# Fails its own count check (needs >= 3 gears, has 1).
UNDERCOUNT_PLAN = json.dumps({
    "name": "undercount",
    "parts": [
        {"id": "gear_z12", "primitive": {"kind": "involute_gear",
                                         "module": 1.0, "teeth": 12,
                                         "thickness": 3.0}},
    ],
    "instances": [{"part": "gear_z12", "at": [0, 0, 0]}],
    "checks": [{"kind": "count", "pattern": "gear_*", "min": 3}],
})


@pytest.fixture
def mock_planner(monkeypatch):
    def install(*responses):
        calls = {"n": 0, "prompts": []}

        def fake(prompt):
            i = min(calls["n"], len(responses) - 1)
            calls["n"] += 1
            calls["prompts"].append(prompt)
            return responses[i], None

        monkeypatch.setattr(assembly, "_call_planner", fake)
        return calls
    return install


@pytest.fixture
def mock_codegen(monkeypatch):
    """generate_shape returns a compliant 60x30x5 plate (base at z=0)."""
    def fake(desc, extra_constraints="", **kw):
        return types.SimpleNamespace(
            part=Pos(0, 0, 2.5) * Box(60, 30, 5), error=None, code="...")
    monkeypatch.setattr(backend, "generate_shape", fake)


def test_assemble_clean_first_round(tmp_path, mock_planner, mock_codegen):
    calls = mock_planner(GOOD_PLAN)
    result = assemble("a small test gearbox", output_dir=tmp_path)
    assert result.success, result.error
    assert calls["n"] == 1
    assert result.step_path.stat().st_size > 0
    assert result.stl_path.stat().st_size > 0
    assert json.loads(result.plan_path.read_text())["name"] == "test gearbox"
    assert (result.parts_dir / "gear_z24.stl").stat().st_size > 0
    assert (result.parts_dir / "base_plate.stl").stat().st_size > 0
    assert "2 unique parts, 3 instances" in result.summary()
    assert "2 x gear_z24" in result.summary()


def test_bad_layout_fails_preflight_before_generation(tmp_path, mock_planner,
                                                       monkeypatch):
    """Wrongly spaced gears are rejected by the soft pre-flight — no
    part generation happens for the bad plan."""
    gen_calls = {"n": 0}
    def counting(desc, extra_constraints="", **kw):
        gen_calls["n"] += 1
        return types.SimpleNamespace(
            part=Pos(0, 0, 2.5) * Box(60, 30, 5), error=None, code="...")
    monkeypatch.setattr(backend, "generate_shape", counting)
    calls = mock_planner(OVERLAP_PLAN, GOOD_PLAN)
    result = assemble("gearbox", output_dir=tmp_path)
    assert result.success, result.error
    assert calls["n"] == 2
    assert "pre-flight" in calls["prompts"][1]
    assert "meshing needs exactly 24.0mm" in calls["prompts"][1]
    # the bad round-1 plan cost zero codegen calls
    assert gen_calls["n"] == 1          # only GOOD_PLAN's base_plate


def test_unphased_gears_caught_by_precise_check(tmp_path, mock_planner,
                                                mock_codegen):
    """Correct center distance but no tooth phasing passes pre-flight
    (the layout is right) and is caught by the boolean interference
    check after generation."""
    plan = json.loads(GOOD_PLAN)
    plan["instances"][1]["rotate"] = [0, 0, 0]   # drop the 7.5deg phase
    calls = mock_planner(json.dumps(plan), GOOD_PLAN)
    result = assemble("gearbox", output_dir=tmp_path)
    assert result.success, result.error
    assert calls["n"] == 2
    assert "interference" in calls["prompts"][1]


def test_count_check_triggers_revision(tmp_path, mock_planner, mock_codegen):
    calls = mock_planner(UNDERCOUNT_PLAN, GOOD_PLAN)
    result = assemble("gearbox with at least 3 gears", output_dir=tmp_path)
    assert result.success
    assert calls["n"] == 2
    assert "count check failed" in calls["prompts"][1]


def test_invalid_plan_json_revised(tmp_path, mock_planner, mock_codegen):
    calls = mock_planner("not json at all", GOOD_PLAN)
    result = assemble("gearbox", output_dir=tmp_path)
    assert result.success
    assert calls["n"] == 2
    assert "plan validation failed" in calls["prompts"][1]


def test_envelope_is_orientation_agnostic(tmp_path, mock_planner, monkeypatch):
    """A plate modeled flat (110x140x3) must pass a standing envelope
    (160x3x160) — instances rotate parts, so only sorted dimensions
    matter. This was a real failure mode: same plate, wasted revisions."""
    plan = json.loads(GOOD_PLAN)
    plan["parts"][1]["envelope"] = [160, 3, 160]
    def flat_plate(desc, extra_constraints="", **kw):
        return types.SimpleNamespace(
            part=Pos(0, 0, 1.5) * Box(110, 140, 3), error=None, code="...")
    monkeypatch.setattr(backend, "generate_shape", flat_plate)
    # keep the plate's (tall) envelope proxy clear of the gears
    plan["instances"][2]["at"] = [250, 0, 0]
    calls = mock_planner(json.dumps(plan))
    result = assemble("gearbox", output_dir=tmp_path)
    assert result.success, result.error
    assert calls["n"] == 1


def test_envelope_has_absolute_floor(tmp_path, mock_planner, monkeypatch):
    """A 4mm-thick clock hand must pass a 2mm-thick budget: tiny envelope
    dimensions get a 2mm absolute allowance, not just 15%."""
    plan = json.loads(GOOD_PLAN)
    plan["parts"][1]["envelope"] = [20, 100, 2]
    plan["instances"][2]["at"] = [12, 0, -10]
    def hand(desc, extra_constraints="", **kw):
        return types.SimpleNamespace(
            part=Pos(0, 0, 2) * Box(18, 100, 4), error=None, code="...")
    monkeypatch.setattr(backend, "generate_shape", hand)
    calls = mock_planner(json.dumps(plan))
    result = assemble("clock hand", output_dir=tmp_path)
    assert result.success, result.error
    assert calls["n"] == 1


def test_envelope_violation_feeds_back(tmp_path, mock_planner, monkeypatch):
    """An LLM part that busts its bounding-box budget is rejected and the
    failure reaches the planner."""
    def oversized(desc, extra_constraints="", **kw):
        return types.SimpleNamespace(
            part=Pos(0, 0, 50) * Box(200, 200, 100), error=None, code="...")
    monkeypatch.setattr(backend, "generate_shape", oversized)
    calls = mock_planner(GOOD_PLAN)
    result = assemble("gearbox", output_dir=tmp_path, max_revisions=2)
    assert not result.success
    assert calls["n"] == 2
    assert "exceeds its envelope in any orientation" in calls["prompts"][1]


def test_exhausts_revisions(tmp_path, mock_planner, mock_codegen):
    calls = mock_planner(OVERLAP_PLAN)
    result = assemble("gearbox", output_dir=tmp_path, max_revisions=2)
    assert not result.success
    assert calls["n"] == 2
    assert "no clean assembly after 2" in result.error
    assert not result  # __bool__


def test_carve_gives_contents_clearance(tmp_path, mock_planner, monkeypatch):
    """A solid container marked carve:true gets exact pockets subtracted
    for its contents — interference cannot fail on interior fit."""
    plan = {
        "name": "carved box",
        "parts": [
            {"id": "case", "description": "a solid block 60x60x60mm",
             "envelope": [60, 60, 60], "carve": True},
            {"id": "gear_z12", "primitive": {"kind": "involute_gear",
                                             "module": 1.0, "teeth": 12,
                                             "thickness": 4.0}},
        ],
        "instances": [
            {"part": "case", "at": [0, 0, 0]},
            # gear buried in the middle of the solid block
            {"part": "gear_z12", "at": [0, 0, 28]},
        ],
    }
    def solid_block(desc, extra_constraints="", **kw):
        return types.SimpleNamespace(
            part=Pos(0, 0, 30) * Box(60, 60, 60), error=None, code="...")
    monkeypatch.setattr(backend, "generate_shape", solid_block)
    calls = mock_planner(json.dumps(plan))
    result = assemble("gear in a box", output_dir=tmp_path)
    assert result.success, result.error
    assert calls["n"] == 1          # no revision needed — carving handled it
    # carved case lost the gear's volume
    assert result.volume_mm3 < 60 * 60 * 60 + 1


def test_revision_feedback_includes_measured_sizes(tmp_path, mock_planner,
                                                   mock_codegen):
    plan = json.loads(GOOD_PLAN)
    plan["instances"][1]["rotate"] = [0, 0, 0]   # unphased: passes pre-flight,
    calls = mock_planner(json.dumps(plan), GOOD_PLAN)  # fails precise check
    result = assemble("gearbox", output_dir=tmp_path)
    assert result.success
    assert "MEASURED PART SIZES" in calls["prompts"][1]


def test_plan_schema_rejects_bad_references():
    with pytest.raises(Exception):
        AssemblyPlan.model_validate_json(json.dumps({
            "name": "x",
            "parts": [{"id": "a", "primitive": {"kind": "involute_gear",
                                                "module": 1, "teeth": 12,
                                                "thickness": 2}}],
            "instances": [{"part": "missing"}],
        }))
    with pytest.raises(Exception):  # LLM part without envelope
        AssemblyPlan.model_validate_json(json.dumps({
            "name": "x",
            "parts": [{"id": "a", "description": "a plate"}],
            "instances": [{"part": "a"}],
        }))
