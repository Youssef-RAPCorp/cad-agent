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


def test_exhausts_layout_revisions_fast(tmp_path, mock_planner, mock_codegen):
    """A never-clean layout burns only cheap planner iterations — the
    expensive build phase is never entered."""
    calls = mock_planner(OVERLAP_PLAN)
    result = assemble("gearbox", output_dir=tmp_path,
                      max_layout_revisions=3)
    assert not result.success
    assert calls["n"] == 3
    assert "no pre-flight-clean layout within 3" in result.error
    assert not result  # __bool__


def test_mounted_parts_exempt_from_preflight(tmp_path, mock_planner,
                                             monkeypatch):
    """A gear mounted on its arbor overlaps the arbor's envelope by
    definition — 'mounts' exempts the pair in pre-flight, and the
    precise check passes because the arbor fits through the bore."""
    plan = {
        "name": "gear on arbor",
        "parts": [
            {"id": "arbor", "description": "a steel rod 3.6mm dia x 40mm",
             "envelope": [4, 4, 40]},
            {"id": "gear_z24", "primitive": {"kind": "involute_gear",
                                             "module": 1.0, "teeth": 24,
                                             "thickness": 4.0, "bore": 4.0}},
        ],
        "instances": [
            {"part": "arbor", "at": [0, 0, -18]},
            {"part": "gear_z24", "at": [0, 0, 0], "mounts": "arbor"},
        ],
    }
    from build123d import Cylinder
    def rod(desc, extra_constraints="", **kw):
        return types.SimpleNamespace(
            part=Pos(0, 0, 20) * Cylinder(1.8, 40), error=None, code="...")
    monkeypatch.setattr(backend, "generate_shape", rod)
    calls = mock_planner(json.dumps(plan))
    result = assemble("gear on arbor", output_dir=tmp_path)
    assert result.success, result.error
    assert calls["n"] == 1

    # Without declared mounts the slender arbor is AUTO-mounted (a gear
    # overlapping a shaft-like part is riding it by definition).
    plan["instances"][1].pop("mounts")
    calls2 = mock_planner(json.dumps(plan))
    result2 = assemble("gear on arbor", output_dir=tmp_path)
    assert result2.success, result2.error
    assert calls2["n"] == 1

    # A NON-slender undeclared overlap is still rejected at pre-flight.
    plan["parts"][0]["envelope"] = [40, 40, 40]
    plan["parts"][0]["description"] = "a block 40x40x40mm"
    calls3 = mock_planner(json.dumps(plan))
    result3 = assemble("gear on block", output_dir=tmp_path,
                       max_layout_revisions=2)
    assert not result3.success
    assert "mounts" in calls3["prompts"][1]


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


def test_structure_carves_into_structure(tmp_path, mock_planner, monkeypatch):
    """Two carve-marked structural parts overlapping (seat board inside
    the case): the smaller carves a pocket into the larger instead of
    failing verification."""
    plan = {
        "name": "board in case",
        "parts": [
            {"id": "case", "description": "solid case block 100x100x100",
             "envelope": [100, 100, 100], "carve": True},
            {"id": "board", "description": "board 60x20x10",
             "envelope": [60, 20, 10], "carve": True},
        ],
        "instances": [
            {"part": "case", "at": [0, 0, 0]},
            {"part": "board", "at": [0, 0, 45]},   # buried mid-case
        ],
    }
    def by_desc(desc, extra_constraints="", **kw):
        if "case" in desc:
            solid = Pos(0, 0, 50) * Box(100, 100, 100)
        else:
            solid = Pos(0, 0, 5) * Box(60, 20, 10)
        return types.SimpleNamespace(part=solid, error=None, code="...")
    monkeypatch.setattr(backend, "generate_shape", by_desc)
    calls = mock_planner(json.dumps(plan))
    result = assemble("board in case", output_dir=tmp_path)
    assert result.success, result.error
    assert calls["n"] == 1
    # case lost the board's volume
    assert result.volume_mm3 == pytest.approx(100**3, rel=0.01)


def test_mounted_interference_message_hints_bore(tmp_path, mock_planner,
                                                 monkeypatch):
    """A gear whose bore is too small for its arbor fails precisely, and
    the feedback names the fix."""
    plan = {
        "name": "tight bore",
        "parts": [
            {"id": "arbor", "description": "rod 3.6mm dia x 40mm",
             "envelope": [4, 4, 40]},
            {"id": "gear_z24", "primitive": {"kind": "involute_gear",
                                             "module": 1.0, "teeth": 24,
                                             "thickness": 4.0, "bore": 1.0}},
        ],
        "instances": [
            {"part": "arbor", "at": [0, 0, -18]},
            {"part": "gear_z24", "at": [0, 0, 0], "mounts": "arbor"},
        ],
    }
    from build123d import Cylinder
    def rod(desc, extra_constraints="", **kw):
        return types.SimpleNamespace(
            part=Pos(0, 0, 20) * Cylinder(1.8, 40), error=None, code="...")
    monkeypatch.setattr(backend, "generate_shape", rod)
    calls = mock_planner(json.dumps(plan))
    result = assemble("tight bore", output_dir=tmp_path, max_revisions=1)
    assert not result.success
    assert "mounted pair" in result.error
    assert "bore" in result.error


def test_gear_train_links_compute_positions_and_phase(tmp_path,
                                                       mock_planner,
                                                       mock_codegen):
    """A 3-gear train + stacked pinion declared purely by topology
    (mesh_with / stack_on, no hand-computed coordinates) builds clean in
    one round — positions AND tooth phasing come from the engine."""
    plan = {
        "name": "linked train",
        "parts": [
            {"id": "gear_z40", "primitive": {"kind": "involute_gear",
                                             "module": 1.5, "teeth": 40,
                                             "thickness": 4.0}},
            {"id": "gear_z12", "primitive": {"kind": "involute_gear",
                                             "module": 1.5, "teeth": 12,
                                             "thickness": 4.0}},
        ],
        "instances": [
            {"part": "gear_z40", "at": [0, 0, 0]},
            {"part": "gear_z12", "mesh_with": "gear_z40#1",
             "mesh_angle_deg": 30},
            {"part": "gear_z40", "stack_on": "gear_z12#1",
             "axial_offset": 4},
            {"part": "gear_z12", "mesh_with": "gear_z40#2",
             "mesh_angle_deg": 275},
        ],
    }
    calls = mock_planner(json.dumps(plan))
    result = assemble("linked gear train", output_dir=tmp_path)
    assert result.success, result.error
    assert calls["n"] == 1
    # engine computed the meshing position: z40/z12 center distance is 39
    resolved = result.plan.instances[1]
    import math
    d = math.dist(resolved.at, result.plan.instances[0].at)
    assert d == pytest.approx(39.0, abs=1e-6)


def test_link_errors_feed_back_fast(tmp_path, mock_planner, mock_codegen):
    plan = {
        "name": "bad link",
        "parts": [
            {"id": "gear_z12", "primitive": {"kind": "involute_gear",
                                             "module": 1.0, "teeth": 12,
                                             "thickness": 3.0}},
        ],
        "instances": [
            {"part": "gear_z12", "mesh_with": "gear_z40#7"},
        ],
    }
    calls = mock_planner(json.dumps(plan), GOOD_PLAN)
    result = assemble("bad link", output_dir=tmp_path)
    assert result.success
    assert calls["n"] == 2
    assert "link target" in calls["prompts"][1]


def test_diagonal_gears_no_false_positive(tmp_path, mock_planner,
                                           mock_codegen):
    """Grid diagonals: two z30 m1.5 gears 63.6mm apart (tip radii sum
    48) do NOT touch — box proxies used to flag them and spiral the
    planner. The exact cylinder test must pass them."""
    plan = {
        "name": "diagonal gears",
        "parts": [
            {"id": "gear_z30", "primitive": {"kind": "involute_gear",
                                             "module": 1.5, "teeth": 30,
                                             "thickness": 4.0}},
        ],
        "instances": [
            {"part": "gear_z30", "at": [0, 0, 0]},
            {"part": "gear_z30", "at": [45, 0, 0], "rotate": [0, 0, 6],
             "mounts": None},
            {"part": "gear_z30", "at": [45.0, 45.0, 0]},
        ],
    }
    # 0-1 meshed at 45mm; 0-2 diagonal at 63.6mm (no contact); 1-2 at
    # 45mm vertical — mesh-phase it too via the engine formula.
    from cad_agent._vendored.cad_agent3.stdparts import mesh_phase
    plan["instances"][1]["rotate"] = [0, 0, mesh_phase(30, 30, 0, 0)]
    plan["instances"][2]["rotate"] = [
        0, 0, mesh_phase(30, 30, 90, plan["instances"][1]["rotate"][2])]
    calls = mock_planner(json.dumps(plan))
    result = assemble("diagonal gears", output_dir=tmp_path)
    assert result.success, result.error
    assert calls["n"] == 1


def test_instance_ids_tolerated(tmp_path, mock_planner, mock_codegen):
    plan = json.loads(GOOD_PLAN)
    for i, inst in enumerate(plan["instances"]):
        inst["id"] = f"inst_{i}"
    calls = mock_planner(json.dumps(plan))
    result = assemble("gearbox", output_dir=tmp_path)
    assert result.success, result.error
    assert calls["n"] == 1


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
