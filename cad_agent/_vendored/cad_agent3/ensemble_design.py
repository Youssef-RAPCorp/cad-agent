"""ensemble_design.py — multi-agent design ensemble for hard constraint problems.

Runs N specialist agents in parallel, each given the same problem with a
different role-prompt. Their outputs are aggregated by a synthesis agent
that produces a single unified spec, which is then handed to the
build123d codegen.

USE WITH CAUTION. For most engineering parts this is overkill and produces
worse output than a single careful pass — see notes in the design_chat
documentation. It's intended for problems where the constraint set has
real internal tensions (e.g. fold-flat structures, deployable mechanisms,
parts where optics + mechanics + materials all interact).

API:
    from cad_agent3.ensemble_design import run_ensemble, run_single_pass

    spec, transcript = run_ensemble(
        problem="Foldable cardboard VR headset that fits in a pocket",
        constraints=["..."],
        roles=DEFAULT_ROLES,
    )

Each role is a dict with name + system prompt fragment. The synthesis
agent always runs last with full visibility of all role outputs.
"""

from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .gemini_codegen import call_gemini_for_code


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

DEFAULT_ROLES = [
    {
        "name": "optics",
        "title": "Optics Engineer",
        "focus": (
            "lens type and diameter, focal length, lens-to-eye distance, "
            "lens-to-screen distance, IPD (inter-pupillary distance), "
            "field of view, image quality at edges. Cite specific numbers "
            "wherever possible (e.g. 'biconvex acrylic, f=45mm, IPD=63mm')."
        ),
    },
    {
        "name": "structural",
        "title": "Structural / Foldability Specialist",
        "focus": (
            "fold topology, hinge mechanics, panel count, panel sizes, "
            "deployment kinematics, structural rigidity when deployed, "
            "fatigue life of hinges, what fold pattern (origami/accordion/"
            "Miura/clamshell/telescoping) actually achieves the size goal."
        ),
    },
    {
        "name": "materials",
        "title": "Materials Engineer",
        "focus": (
            "cardboard grade and thickness, fluting direction relative to "
            "fold lines, score-line depth as fraction of thickness, glue/"
            "tape points vs purely mechanical assembly, durability, water "
            "resistance, cost per unit."
        ),
    },
    {
        "name": "manufacturing",
        "title": "Manufacturing / DFM Engineer",
        "focus": (
            "single-sheet cut pattern (or count of pieces), nesting "
            "efficiency, laser cutter or die-cutter compatibility, "
            "internal cuts that would drop pieces, score-line tooling, "
            "material yield, assembly time per unit."
        ),
    },
    {
        "name": "ux",
        "title": "User Experience Designer",
        "focus": (
            "deployment steps (how many actions to go from pocket to "
            "viewable), phone insertion mechanism, light blocking quality, "
            "comfort against the face, ease of removing for shared use, "
            "what could go wrong in a user's hands."
        ),
    },
]


@dataclass
class AgentResponse:
    name: str
    title: str
    output: str
    duration_s: float
    error: Optional[str] = None


@dataclass
class EnsembleResult:
    problem: str
    constraints: List[str]
    role_responses: List[AgentResponse] = field(default_factory=list)
    synthesis: str = ""
    final_spec: str = ""
    transcript_md: str = ""


# ---------------------------------------------------------------------------
# Single role agent
# ---------------------------------------------------------------------------

def _build_role_prompt(problem: str, constraints: List[str], role: Dict) -> str:
    constraint_block = "\n".join(f"- {c}" for c in constraints)
    return f"""You are a {role['title']}, one of five specialists collaborating on the design problem below.

Your specific focus area: {role['focus']}

You will respond ONLY about your focus area. Other specialists are covering
the other aspects. Be concrete with numbers, materials, and dimensions.
If you see a constraint that's impossible from your specialty's perspective,
say so explicitly — that's more useful than a compromise.

# Design problem
{problem}

# Hard constraints
{constraint_block}

# Your task
Write a focused engineering analysis from YOUR specialty's viewpoint:

1. **Critical parameters** — the 2-5 numbers/choices in your focus area
   that this design MUST get right. Give exact values.

2. **Risks and tradeoffs** — what are the failure modes from your
   perspective? What would compromise design quality?

3. **Recommendations** — your concrete recommendations as bullet points.
   Be specific (e.g. "use 1.5mm E-flute corrugated cardboard with the
   flutes running perpendicular to the primary fold lines").

4. **Disagreement flags** — if you anticipate other specialists pulling
   in a direction that conflicts with yours, name it. Example: "An
   optics-first design will want fixed lens spacing, but I need..."

Keep your response under 400 words. Be technical and concrete, not florid.
"""


def _run_role(problem: str, constraints: List[str], role: Dict) -> AgentResponse:
    """Run one specialist agent."""
    prompt = _build_role_prompt(problem, constraints, role)
    t0 = time.time()
    output, err = call_gemini_for_code(prompt)
    dt = time.time() - t0
    return AgentResponse(
        name=role["name"], title=role["title"],
        output=output or "", duration_s=dt, error=err)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def _build_synthesis_prompt(problem: str, constraints: List[str],
                              responses: List[AgentResponse]) -> str:
    constraint_block = "\n".join(f"- {c}" for c in constraints)
    role_blocks = []
    for r in responses:
        if r.error or not r.output:
            role_blocks.append(f"## {r.title} — (FAILED: {r.error})")
            continue
        role_blocks.append(f"## {r.title}\n\n{r.output}")
    role_text = "\n\n".join(role_blocks)

    return f"""You are the lead designer integrating analyses from five specialists.

# Design problem
{problem}

# Hard constraints
{constraint_block}

# Specialist analyses
{role_text}

# Your task
Produce a SINGLE unified design specification that resolves any conflicts
between the specialists. Where they disagree, pick a side and say WHY —
do not compromise to the middle. Compromises produce mediocre designs;
commitment produces good ones.

Output a clear engineering spec with:

1. **Top-level approach** (1-2 sentences naming the chosen design strategy)
2. **Resolved parameters** — every number a CAD model would need:
   - Materials and thicknesses
   - Panel/component count and sizes (mm)
   - Fold topology (if applicable)
   - Lens/optic specs
   - Hinge/joint geometry
   - Assembly sequence
3. **Conflicts resolved** — list any specialist conflicts and your ruling
4. **Known weaknesses** — be honest about what this design sacrifices

Then output a CONCRETE BUILD PLAN as a numbered list of geometric features
the CAD model should construct. Each item should be specific enough that
a build123d coder could implement it without further interpretation.

Output as plain text, no markdown headers, no code fences. Be concise
but unambiguous.
"""


def _run_synthesis(problem: str, constraints: List[str],
                     responses: List[AgentResponse]) -> tuple:
    prompt = _build_synthesis_prompt(problem, constraints, responses)
    t0 = time.time()
    output, err = call_gemini_for_code(prompt)
    dt = time.time() - t0
    return output or "", err, dt


# ---------------------------------------------------------------------------
# Single-pass control (for comparison)
# ---------------------------------------------------------------------------

SINGLE_PASS_PROMPT = """You are a senior product designer working on the
problem below. You have full responsibility — there is no team — so you
must reason about ALL aspects (optics, mechanics, materials, manufacturing,
user experience) yourself.

# Design problem
{problem}

# Hard constraints
{constraint_block}

# Your task
Think carefully about the design. Where constraints conflict, pick the
most important one and commit to it. Then produce:

1. **Top-level approach** (1-2 sentences)
2. **Resolved parameters** — every number a CAD model would need
3. **Tradeoffs you accepted** — what you sacrificed and why
4. **Concrete build plan** as a numbered list of geometric features

Output as plain text. Be technical, specific, and honest about limitations.
"""


def run_single_pass(problem: str, constraints: List[str]) -> tuple:
    """Single-agent design pass for comparison with the ensemble."""
    constraint_block = "\n".join(f"- {c}" for c in constraints)
    prompt = SINGLE_PASS_PROMPT.format(
        problem=problem, constraint_block=constraint_block)
    t0 = time.time()
    output, err = call_gemini_for_code(prompt)
    dt = time.time() - t0
    return output or "", err, dt


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def run_ensemble(problem: str,
                  constraints: List[str],
                  roles: Optional[List[Dict]] = None,
                  verbose: bool = True) -> EnsembleResult:
    """Run the full ensemble: 5 specialists in sequence + 1 synthesis."""
    roles = roles or DEFAULT_ROLES
    result = EnsembleResult(problem=problem, constraints=constraints)

    if verbose:
        print(f"=== Multi-agent ensemble: {len(roles)} specialists ===")
    for role in roles:
        if verbose:
            print(f"  [{role['name']}] running...", flush=True)
        ar = _run_role(problem, constraints, role)
        result.role_responses.append(ar)
        if verbose:
            mark = "✓" if not ar.error else "✗"
            print(f"  [{role['name']}] {mark} ({ar.duration_s:.1f}s)"
                   + (f" — {ar.error}" if ar.error else ""), flush=True)

    if verbose:
        print(f"  [synthesis] running...", flush=True)
    syn, syn_err, syn_dt = _run_synthesis(
        problem, constraints, result.role_responses)
    result.synthesis = syn
    if verbose:
        mark = "✓" if not syn_err else "✗"
        print(f"  [synthesis] {mark} ({syn_dt:.1f}s)"
               + (f" — {syn_err}" if syn_err else ""), flush=True)

    # Build a markdown transcript for inspection
    md = [f"# Ensemble design transcript\n",
          f"**Problem:** {problem}\n",
          f"**Constraints:**"]
    for c in constraints:
        md.append(f"- {c}")
    md.append("\n---\n")
    for r in result.role_responses:
        md.append(f"## {r.title} ({r.duration_s:.1f}s)\n")
        if r.error:
            md.append(f"_FAILED: {r.error}_\n")
        else:
            md.append(r.output)
        md.append("\n---\n")
    md.append(f"## Synthesis ({syn_dt:.1f}s)\n")
    md.append(syn or f"_FAILED: {syn_err}_")
    result.transcript_md = "\n".join(md)
    result.final_spec = syn

    return result
