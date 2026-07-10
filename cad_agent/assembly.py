"""cad_agent.assembly — multi-part orchestration for complex models.

Single-script generation tops out quickly: one LLM call cannot reliably
produce a 40-part mechanism with accurate geometry, and the only check
it gets is "code ran, volume > 0". This pipeline splits the problem:

    plan -> generate parts -> assemble -> verify -> revise

1. PLAN: the LLM decomposes the spec into unique parts and placed
   instances (validated Pydantic JSON). Precision parts (gears) are
   declared as parametric primitives — built by exact math, never by
   LLM geometry.
2. GENERATE: primitives come from stdparts; every other part gets its
   own focused codegen call with a bounding-box budget, and is rejected
   if it busts its envelope. Parts are cached by content, so plan
   revisions only regenerate what changed.
3. ASSEMBLE: instances are placed (translate + rotate) by trusted code.
4. VERIFY: pairwise interference via boolean intersection (AABB
   prefiltered), plus plan-declared checks (e.g. 30 <= count(gear_*)
   <= 50).
5. REVISE: violations are fed back to the planner with numbers, up to
   max_revisions rounds.

Usage:
    from cad_agent.assembly import assemble
    result = assemble("A grandfather clock with 30-50 accurate cogs...")
    print(result.summary())

Or:  cad-agent --assembly "<spec>"
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "cad_agent.assembly requires pydantic. Install it with:\n\n"
        "    pip install -e \".[drawings]\"    # from the cad-agent repo root\n"
        "    # or: pip install pydantic\n"
    ) from exc


# ---------------------------------------------------------------------------
# Plan schema — the LLM I/O contract for assemblies
# ---------------------------------------------------------------------------

class GearPrimitive(BaseModel):
    """A mathematically exact involute spur gear (built by stdparts, not
    by LLM geometry). Axis +Z, extruded z=0..thickness, centered in XY.
    Tip diameter = module*(teeth+2)."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["involute_gear"] = "involute_gear"
    module: float = Field(gt=0)
    teeth: int = Field(ge=4)
    thickness: float = Field(gt=0)
    bore: float = Field(0.0, ge=0)


class PartSpec(BaseModel):
    """One unique part. Either a parametric `primitive` (preferred for
    precision parts) or a natural-language `description` for LLM
    generation. LLM parts are modeled centered on the origin in X/Y
    with their base at Z=0, and must fit inside `envelope`."""
    model_config = ConfigDict(extra="forbid")
    id: str
    description: str = ""
    primitive: Optional[GearPrimitive] = None
    envelope: Optional[Tuple[float, float, float]] = None  # max (X, Y, Z) mm
    # Container parts: carve exact clearance pockets for any contents
    # that would otherwise collide with this part (contents win).
    carve: bool = False

    @model_validator(mode="after")
    def _needs_source(self):
        if self.primitive is None and not self.description.strip():
            raise ValueError(f"part {self.id!r} needs a primitive or a description")
        if self.primitive is None and self.envelope is None:
            raise ValueError(f"LLM part {self.id!r} needs an envelope")
        return self


class Instance(BaseModel):
    """A placement of a part: rotate (XYZ Euler, degrees, applied first)
    then translate to `at` (mm, assembly frame). `mounts` names the part
    this instance is fixed on/around/through (a gear on its arbor, a
    hand on its arbor, a sleeve around a shaft): mounted pairs are
    exempt from the layout pre-flight — their true fit is still
    verified precisely after generation."""
    model_config = ConfigDict(extra="forbid")
    part: str
    at: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotate: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mounts: Optional[str] = None


class CountCheck(BaseModel):
    """Assembly-level requirement: number of instances whose part id
    matches `pattern` (fnmatch) must lie within [min, max]."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["count"] = "count"
    pattern: str
    min: int = 0
    max: int = 10 ** 9


class AssemblyPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    parts: List[PartSpec]
    instances: List[Instance]
    checks: List[CountCheck] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self):
        ids = [p.id for p in self.parts]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate part ids")
        known = set(ids)
        for inst in self.instances:
            if inst.part not in known:
                raise ValueError(f"instance references unknown part {inst.part!r}")
        if not self.instances:
            raise ValueError("plan has no instances")
        return self


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class AssemblyResult:
    """Outcome of an assemble() call."""
    spec: str
    success: bool
    name: str = ""
    plan: Optional[AssemblyPlan] = None
    step_path: Optional[Path] = None
    stl_path: Optional[Path] = None
    plan_path: Optional[Path] = None
    parts_dir: Optional[Path] = None
    compound: Optional[object] = None       # live build123d Compound
    volume_mm3: Optional[float] = None
    report: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return self.success

    def summary(self) -> str:
        if not self.success:
            return f"FAILED: {self.error or 'unknown error'}"
        bom: Dict[str, int] = {}
        for inst in self.plan.instances:
            bom[inst.part] = bom.get(inst.part, 0) + 1
        lines = [f"OK: assembly '{self.name}' — {len(self.plan.parts)} unique "
                 f"parts, {len(self.plan.instances)} instances"]
        if self.volume_mm3:
            lines.append(f"  volume: {self.volume_mm3:.0f} mm³")
        for pid, n in sorted(bom.items()):
            lines.append(f"    {n:3d} x {pid}")
        if self.step_path:
            lines.append(f"  STEP:  {self.step_path}")
        if self.stl_path:
            lines.append(f"  STL:   {self.stl_path}")
        if self.plan_path:
            lines.append(f"  plan:  {self.plan_path}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------

_PLAN_PROMPT = """You are a senior mechanical design engineer. Decompose the \
product described below into an ASSEMBLY PLAN: unique parts plus placed \
instances, as one JSON object valid against the schema.

Rules:
1. PRECISION PARTS: gears/cogs MUST be declared as primitives —
   {{"kind": "involute_gear", "module": m, "teeth": z, "thickness": t,
   "bore": d}} — never as described geometry. Meshing gears share the same
   module; their center distance MUST equal module*(teeth1+teeth2)/2 —
   compute instance positions from that rule. Phase meshing pairs by
   rotating one gear (rotate[2] = 180/teeth degrees is the usual
   correction). Gear primitives are modeled axis +Z, z=0..thickness,
   centered in XY; use `rotate` to orient them.
2. LLM PARTS: everything else gets a `description` — self-contained,
   fully dimensioned in mm, written like a spec for a machinist ("a
   rectangular seat board 200x180x20mm with two 10mm notches...") — and
   an `envelope` [X, Y, Z] it must fit inside. Parts are modeled centered
   on the origin in X/Y with their base at Z=0.
3. INSTANCES: place every physical occurrence (a part used 4 times = 4
   instances). `rotate` is XYZ Euler degrees applied before translating
   to `at`. Parts must NOT interpenetrate — leave real clearances;
   support every part on another (nothing floats).
4. CHECKS: encode countable requirements from the spec as checks, e.g.
   {{"kind": "count", "pattern": "gear_*", "min": 30, "max": 50}} — name
   parts so patterns work (gear_z36, gear_z12, ...).
5. Prefer FEWER, LARGER LLM parts (a case as one part, not 12 panels);
   use many instances of few gear primitives for trains.
6. STRUCTURE vs MECHANISM: set "carve": true on EVERY structural part
   — case sections, hoods, boxes, plates, boards, frames, anything
   other parts sit inside, rest on, or pass through. The pipeline
   carves exact clearance pockets into carve parts for whatever
   touches them (an arbor through a carve movement plate gets its
   pivot hole drilled automatically). NEVER set carve on mechanism
   parts: gears, arbors, hands, pendulum, weights, cables — those must
   stay intact, and mechanism-vs-mechanism collisions still fail
   verification. Describe containers as hollow shells with wall
   thickness and openings anyway; carving is the safety net, not the
   design.
7. LAYOUT PRE-FLIGHT: before any geometry is generated, every
   non-carve instance is checked as a proxy box (gears: their exact tip
   cylinder; LLM parts: their envelope). Two proxies may not overlap
   unless they are a correctly meshed gear pair (center distance =
   module*(z1+z2)/2), coaxial stacked wheels, or MOUNTED on each other:
   any instance fixed on/around/through another part (a gear or hand on
   its arbor, a sleeve around a shaft) MUST set "mounts": "<part_id>" of
   that part. Mounted fit is still verified precisely after generation,
   so size bores to their shafts. Compute all positions numerically and
   size envelopes honestly.
8. Output ONLY the JSON object — no markdown fences, no commentary.

JSON schema:
{schema}

Product to design:
{spec}
{feedback}
Return the assembly plan JSON now."""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _rot_matrix(rx: float, ry: float, rz: float):
    """Rotation matrix for XYZ Euler degrees (Rz @ Ry @ Rx)."""
    import math as m
    ax, ay, az = (m.radians(rx), m.radians(ry), m.radians(rz))
    cx, sx = m.cos(ax), m.sin(ax)
    cy, sy = m.cos(ay), m.sin(ay)
    cz, sz = m.cos(az), m.sin(az)
    return ((cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
            (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
            (-sy, cy * sx, cy * cx))


def _xform(pt, R, at):
    return (R[0][0] * pt[0] + R[0][1] * pt[1] + R[0][2] * pt[2] + at[0],
            R[1][0] * pt[0] + R[1][1] * pt[1] + R[1][2] * pt[2] + at[1],
            R[2][0] * pt[0] + R[2][1] * pt[1] + R[2][2] * pt[2] + at[2])


def _preflight(plan: AssemblyPlan) -> List[str]:
    """Soft layout check BEFORE any geometry is generated.

    Every non-carve instance is represented by a cheap proxy volume —
    a gear's exact tip cylinder (analytic from module/teeth) or an LLM
    part's declared envelope box — transformed by its placement. Two
    proxies overlapping by more than 2mm is a layout violation, except
    gear pairs that satisfy the meshing center-distance rule and
    coaxial stacked wheels (their true fit is verified precisely after
    generation). Count checks run here too. This costs microseconds,
    so bad layouts are rejected for the price of a planner call instead
    of a full generation round.
    """
    violations: List[str] = []
    for chk in plan.checks:
        n = sum(1 for inst in plan.instances
                if fnmatch.fnmatch(inst.part, chk.pattern))
        if not (chk.min <= n <= chk.max):
            violations.append(
                f"count check failed: {n} instances match "
                f"'{chk.pattern}' (need {chk.min}..{chk.max})")

    parts_by_id = {pt.id: pt for pt in plan.parts}
    proxies = []
    counters: Dict[str, int] = {}
    for inst in plan.instances:
        part = parts_by_id[inst.part]
        counters[inst.part] = counters.get(inst.part, 0) + 1
        label = f"{inst.part}#{counters[inst.part]}"
        if part.carve:
            continue                    # structure gets carved later
        g = part.primitive
        if g is not None:
            r = g.module * (g.teeth + 2) / 2.0
            box = (-r, -r, 0.0, r, r, g.thickness)
            center_local = (0.0, 0.0, g.thickness / 2.0)
        else:
            ex, ey, ez = part.envelope
            box = (-ex / 2, -ey / 2, 0.0, ex / 2, ey / 2, ez)
            center_local = (0.0, 0.0, ez / 2.0)
        R = _rot_matrix(*inst.rotate)
        pts = [_xform((x, y, z), R, inst.at)
               for x in (box[0], box[3])
               for y in (box[1], box[4])
               for z in (box[2], box[5])]
        aabb = (min(p[0] for p in pts), min(p[1] for p in pts),
                min(p[2] for p in pts), max(p[0] for p in pts),
                max(p[1] for p in pts), max(p[2] for p in pts))
        proxies.append((label, g, aabb, _xform(center_local, R, inst.at),
                        inst))

    contact = 2.0                       # ignore adjacency/stacking contact
    for i in range(len(proxies)):
        for j in range(i + 1, len(proxies)):
            la, ga, a, ca, ia = proxies[i]
            lb, gb, b, cb, ib = proxies[j]
            # Mounted pairs (gear on arbor, hand on arbor, sleeve around
            # shaft) legitimately overlap; precise verification after
            # generation judges their true fit (bores vs shafts).
            if (ia.mounts == ib.part or ib.mounts == ia.part
                    or (ia.mounts is not None and ia.mounts == ib.mounts)):
                continue
            if not (min(a[3], b[3]) - max(a[0], b[0]) > contact
                    and min(a[4], b[4]) - max(a[1], b[1]) > contact
                    and min(a[5], b[5]) - max(a[2], b[2]) > contact):
                continue
            if ga is not None and gb is not None:
                d = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2
                     + (ca[2] - cb[2]) ** 2) ** 0.5
                if d < 2.0:
                    continue            # coaxial stacked wheels
                if ga.module == gb.module:
                    need = ga.module * (ga.teeth + gb.teeth) / 2.0
                    if abs(d - need) <= 0.5:
                        continue        # correctly meshed pair
                    violations.append(
                        f"gears {la} and {lb} overlap but are {d:.1f}mm "
                        f"apart; meshing needs exactly {need:.1f}mm")
                    continue
                violations.append(
                    f"gears {la} and {lb} overlap with different modules "
                    f"({ga.module} vs {gb.module}) — they cannot mesh")
                continue
            violations.append(
                f"layout overlap: {la} at {tuple(round(v) for v in ia.at)} "
                f"and {lb} at {tuple(round(v) for v in ib.at)} "
                f"interpenetrate — move them apart, or set 'mounts' if "
                f"one is fixed on the other")
    return violations


def _call_planner(prompt: str):
    from ._vendored.cad_agent3 import gemini_codegen
    return gemini_codegen.call_gemini_for_code(prompt)


def _build_part(part: PartSpec, cache: Dict[str, object], verbose: bool):
    """Build one unique part solid (cached by content). Returns
    (solid, error)."""
    key = part.model_dump_json()
    if key in cache:
        return cache[key], None

    if part.primitive is not None:
        from ._vendored.cad_agent3.stdparts import involute_gear
        p = part.primitive
        solid = involute_gear(module=p.module, teeth=p.teeth,
                              thickness=p.thickness, bore=p.bore)
        cache[key] = solid
        return solid, None

    from ._vendored import cad_agent3 as backend
    ex, ey, ez = part.envelope
    constraints = (f"The part must fit inside a {ex:g} x {ey:g} x {ez:g} mm "
                   f"bounding box, centered on the origin in X and Y with "
                   f"its base at Z=0.")
    desc = part.description
    for attempt in range(2):
        if verbose:
            print(f"[cad_agent.assembly]   generating part '{part.id}' "
                  f"(attempt {attempt + 1})", file=sys.stderr)
        gen = backend.generate_shape(desc, extra_constraints=constraints)
        if gen.part is None:
            err = gen.error or "no geometry produced"
            continue
        bb = gen.part.bounding_box()
        size = (bb.max.X - bb.min.X, bb.max.Y - bb.min.Y, bb.max.Z - bb.min.Z)
        # Orientation-agnostic budget: instances rotate parts anyway, so
        # compare SORTED dimensions. Tolerance is 15% plus a 2mm
        # absolute floor — percentage-only tolerance made tiny budgets
        # (a 2mm-thick hand) fail on fractions of a millimeter.
        if all(s <= e * 1.15 + 2.0
               for s, e in zip(sorted(size), sorted(part.envelope))):
            cache[key] = gen.part
            return gen.part, None
        err = (f"part exceeds its envelope in any orientation: measured "
               f"{size[0]:.0f}x{size[1]:.0f}x{size[2]:.0f} vs budget "
               f"{ex:g}x{ey:g}x{ez:g}")
        constraints += (f"\nYOUR PREVIOUS ATTEMPT WAS TOO BIG "
                        f"({size[0]:.0f}x{size[1]:.0f}x{size[2]:.0f}mm). "
                        f"Scale the geometry to fit the budget.")
    return None, f"part '{part.id}': {err}"


def _place(solid, inst: Instance):
    from build123d import Pos, Rot
    rx, ry, rz = inst.rotate
    x, y, z = inst.at
    return Pos(x, y, z) * Rot(rx, ry, rz) * solid


def _verify_assembly(plan: AssemblyPlan, placed, max_pairs: int = 400,
                     tol_mm3: float = 0.5, verbose: bool = False):
    """Return a list of violation strings (empty = clean)."""
    violations: List[str] = []

    # 1. Plan-declared count checks.
    for chk in plan.checks:
        n = sum(1 for inst in plan.instances
                if fnmatch.fnmatch(inst.part, chk.pattern))
        if not (chk.min <= n <= chk.max):
            violations.append(
                f"count check failed: {n} instances match "
                f"'{chk.pattern}' (need {chk.min}..{chk.max})")

    # 2. Pairwise interference, AABB-prefiltered. Meshing gears at the
    # correct center distance intersect by ~0 and pass.
    boxes = []
    for label, solid in placed:
        bb = solid.bounding_box()
        boxes.append((label, solid,
                      (bb.min.X, bb.min.Y, bb.min.Z,
                       bb.max.X, bb.max.Y, bb.max.Z)))
    margin = 0.2  # ignore mere surface contact
    pairs = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][2], boxes[j][2]
            if (min(a[3], b[3]) - max(a[0], b[0]) > margin
                    and min(a[4], b[4]) - max(a[1], b[1]) > margin
                    and min(a[5], b[5]) - max(a[2], b[2]) > margin):
                pairs.append((i, j))
    if len(pairs) > max_pairs:
        violations.append(
            f"too many overlapping part pairs to verify "
            f"({len(pairs)} > {max_pairs}) — spread the layout out")
        return violations
    if verbose and pairs:
        print(f"[cad_agent.assembly] checking {len(pairs)} "
              f"AABB-overlapping pairs for interference", file=sys.stderr)
    for i, j in pairs:
        la, sa, _ = boxes[i]
        lb, sb, _ = boxes[j]
        try:
            inter = sa & sb
            v = float(inter.volume) if inter is not None else 0.0
        except Exception:
            v = 0.0
        if v > tol_mm3:
            violations.append(
                f"interference: {la} and {lb} overlap by {v:.1f} mm³")
    return violations


def assemble(
    spec: str,
    *,
    name: Optional[str] = None,
    output_dir: Union[str, Path] = "./cad_output",
    max_revisions: int = 3,
    max_layout_revisions: int = 8,
    write_parts: bool = True,
    verbose: bool = False,
) -> AssemblyResult:
    """Generate a multi-part assembly from a natural-language spec.

    Two nested loops with very different costs:

    - LAYOUT loop (fast, up to max_layout_revisions iterations per build
      round): plan -> validate -> proxy pre-flight. Each iteration is
      one planner LLM call plus microseconds of geometry-free checking.
    - BUILD loop (expensive, up to max_revisions rounds): generate parts
      in parallel, carve container clearances, verify precisely with
      boolean intersections. A build failure sends feedback back into
      the layout loop.

    Needs an LLM API key, like CADAgent.generate(). Returns
    AssemblyResult; result.success is False if no clean assembly emerged
    (last error in result.error, per-round details in result.report).
    """
    if (os.environ.get("CAD_AGENT_BACKEND", "").lower() == "anthropic"):
        os.environ.setdefault("LLM_BACKEND", "anthropic")

    result = AssemblyResult(spec=spec, success=False)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    schema_json = json.dumps(AssemblyPlan.model_json_schema(),
                             separators=(",", ":"))
    feedback = ""
    last_error = "no attempts made"
    cache: Dict[str, object] = {}

    def _say(msg):
        if verbose:
            print(f"[cad_agent.assembly] {msg}", file=sys.stderr)

    def _revise(round_tag: str, problem: str, raw: str) -> str:
        _say(f"  revising ({round_tag}): {problem[:200]}")
        result.report.append(f"{round_tag}: {problem[:500]}")
        return (f"\nYOUR PREVIOUS PLAN FAILED:\n{problem[:2500]}\n\n"
                f"Previous JSON (truncated):\n{raw[:2500]}\n\n"
                f"Fix these problems and output the corrected plan.\n")

    def _plan_layout(build_round: int):
        """Fast inner loop: planner call + proxy pre-flight, no geometry."""
        nonlocal feedback, last_error
        import time
        t0 = time.time()
        for it in range(1, max_layout_revisions + 1):
            _say(f"layout iteration {it}/{max_layout_revisions} "
                 f"(build round {build_round})")
            raw, err = _call_planner(_PLAN_PROMPT.format(
                schema=schema_json, spec=spec, feedback=feedback))
            if raw is None:
                last_error = f"LLM call failed: {err}"
                return None
            tag = f"build {build_round} / layout {it}"
            try:
                plan = AssemblyPlan.model_validate_json(raw)
            except ValidationError as e:
                last_error = f"plan validation failed: {e}"
                feedback = _revise(tag, last_error, raw)
                continue
            viols = _preflight(plan)
            if not viols:
                _say(f"layout clean after {it} iteration(s), "
                     f"{time.time() - t0:.0f}s")
                return plan, raw
            last_error = ("layout pre-flight failed (proxy boxes, no "
                          "geometry generated): " + "; ".join(viols[:8]))
            feedback = _revise(tag, last_error, raw)
        return None

    for round_no in range(1, max_revisions + 1):
        planned = _plan_layout(round_no)
        if planned is None:
            result.error = (f"no pre-flight-clean layout within "
                            f"{max_layout_revisions} layout iterations "
                            f"(build round {round_no}); last: {last_error}")
            return result
        plan, raw = planned
        _say(f"plan '{plan.name}': {len(plan.parts)} parts, "
             f"{len(plan.instances)} instances")
        tag = f"build {round_no}"

        # --- generate unique parts (parallel: independent LLM calls) --
        import time
        from concurrent.futures import ThreadPoolExecutor
        workers = max(1, int(os.environ.get("CAD_AGENT_PARALLEL", "20")))
        llm_parts = [pt for pt in plan.parts if pt.primitive is None]
        _say(f"generating {len(plan.parts)} unique parts "
             f"({len(llm_parts)} via LLM, {workers} parallel workers)")
        t0 = time.time()
        solids: Dict[str, object] = {}
        part_errors: List[str] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [(part, pool.submit(_build_part, part, cache, verbose))
                       for part in plan.parts]
            for part, fut in futures:
                solid, perr = fut.result()
                if solid is None:
                    part_errors.append(perr)
                else:
                    solids[part.id] = solid
        _say(f"part generation took {time.time() - t0:.0f}s")

        # Measured sizes — included in any revision feedback so the
        # planner places REAL geometry instead of guessed sizes.
        sizes = []
        for pid, solid in solids.items():
            try:
                bb = solid.bounding_box()
                sizes.append(f"{pid}: {bb.max.X - bb.min.X:.0f} x "
                             f"{bb.max.Y - bb.min.Y:.0f} x "
                             f"{bb.max.Z - bb.min.Z:.0f} mm")
            except Exception:
                pass
        measured = ("MEASURED PART SIZES (local frames, use these for "
                    "placement):\n  " + "\n  ".join(sizes)) if sizes else ""

        if part_errors:
            last_error = ("some parts could not be generated: "
                          + "; ".join(part_errors[:4])
                          + ". Simplify or re-envelope those parts.")
            feedback = _revise(tag, last_error, raw)
            continue

        # --- place instances ------------------------------------------
        carve_ids = {pt.id for pt in plan.parts if pt.carve}
        placed = []
        counters: Dict[str, int] = {}
        for inst in plan.instances:
            counters[inst.part] = counters.get(inst.part, 0) + 1
            label = f"{inst.part}#{counters[inst.part]}"
            placed.append([label, _place(solids[inst.part], inst),
                           inst.part in carve_ids])

        # --- carve container clearances --------------------------------
        if carve_ids:
            n_carved = 0
            for entry in placed:
                if not entry[2]:
                    continue
                for other in placed:
                    if other is entry or other[2]:
                        continue
                    try:
                        entry[1] = entry[1] - other[1]
                        n_carved += 1
                    except Exception:
                        pass
            _say(f"carved clearance for {n_carved} content instances "
                 f"out of {len(carve_ids)} container part(s)")
        placed = [(label, solid) for label, solid, _ in placed]

        # --- verify precisely -------------------------------------------
        violations = _verify_assembly(plan, placed, verbose=verbose)
        if violations:
            last_error = ("assembly verification failed: "
                          + "; ".join(violations[:6])
                          + ("\n" + measured if measured else ""))
            feedback = _revise(tag, last_error, raw)
            continue

        # --- export ----------------------------------------------------
        from build123d import Compound, export_step, export_stl
        compound = Compound([s for _, s in placed])
        aname = name or re.sub(r"[^a-z0-9]+", "_", plan.name.lower()).strip("_")[:40] \
            or "assembly"
        step_path = outdir / f"{aname}.step"
        stl_path = outdir / f"{aname}.stl"
        export_step(compound, str(step_path))
        export_stl(compound, str(stl_path))
        plan_path = outdir / f"{aname}_plan.json"
        plan_path.write_text(plan.model_dump_json(indent=2))
        parts_dir = None
        if write_parts:
            parts_dir = outdir / f"{aname}_parts"
            parts_dir.mkdir(exist_ok=True)
            for pid, solid in solids.items():
                export_stl(solid, str(parts_dir / f"{pid}.stl"))

        result.success = True
        result.name = aname
        result.plan = plan
        result.compound = compound
        result.step_path = step_path
        result.stl_path = stl_path
        result.plan_path = plan_path
        result.parts_dir = parts_dir
        try:
            result.volume_mm3 = float(compound.volume)
        except Exception:
            pass
        _say(f"assembly verified clean: {len(placed)} instances")
        return result

    result.error = (f"no clean assembly after {max_revisions} build "
                    f"rounds; last: {last_error}")
    return result
