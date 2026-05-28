"""constraint_analyst.py — extract hard constraints + flag tensions.

Takes a free-form problem statement, returns a structured list of
constraints with their priorities and a list of identified TENSIONS
(where two constraints fight each other).

The tension detection is what catches problems before they reach
codegen. E.g. "must fold to phone size" + "must hold rigid lenses"
is a tension that demands a topology decision (separate lens carrier
vs. flexible lens mount).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import json
import time

from ..gemini_codegen import call_gemini_for_code


@dataclass
class Constraint:
    name: str           # short identifier
    description: str    # human-readable
    priority: str = "must"   # "must" | "should" | "nice"
    target_value: Optional[str] = None   # specific numeric or string target
    measurable: bool = True              # can we test this after?


@dataclass
class Tension:
    constraint_a: str
    constraint_b: str
    description: str
    resolution_options: List[str] = field(default_factory=list)


@dataclass
class ConstraintAnalysis:
    problem: str
    constraints: List[Constraint] = field(default_factory=list)
    tensions: List[Tension] = field(default_factory=list)
    raw_response: str = ""
    error: Optional[str] = None
    duration_s: float = 0.0

    def has_constraints(self) -> bool:
        return len(self.constraints) > 0

    def to_prompt_text(self) -> str:
        """Render as compact text for downstream LLM prompts."""
        lines = ["# Hard constraints"]
        for c in self.constraints:
            tgt = f" (target: {c.target_value})" if c.target_value else ""
            lines.append(f"- [{c.priority.upper()}] {c.description}{tgt}")
        if self.tensions:
            lines.append("\n# Constraint tensions (need design choice)")
            for t in self.tensions:
                lines.append(f"- {t.constraint_a} vs {t.constraint_b}: "
                              f"{t.description}")
                if t.resolution_options:
                    for opt in t.resolution_options:
                        lines.append(f"    option: {opt}")
        return "\n".join(lines)


PROMPT_TEMPLATE = """You are a senior product engineer analyzing a design request. Your job is to extract HARD ENGINEERING CONSTRAINTS and identify TENSIONS between them.

# Design request
{problem}

{research_text}

# Your task
Output a JSON object with two keys:

1. "constraints" — array of constraint objects with fields:
   - "name": short snake_case identifier
   - "description": one-line human-readable description
   - "priority": "must" / "should" / "nice"
   - "target_value": specific number or short string (or null)
   - "measurable": true if we can test it after the design is built

2. "tensions" — array of tension objects with fields:
   - "constraint_a": name of one constraint
   - "constraint_b": name of another constraint
   - "description": why these two fight each other
   - "resolution_options": array of 2-3 short strings naming ways to resolve

Be SPECIFIC. Numbers and dimensions where possible.

Output ONLY the JSON object, no markdown fences, no preamble.

Example for "design a watertight enclosure that opens easily":
{{
  "constraints": [
    {{"name": "watertight", "description": "must seal against water ingress at 1m depth", "priority": "must", "target_value": "IP67 or better", "measurable": true}},
    {{"name": "tool_free_open", "description": "user opens it without tools", "priority": "must", "target_value": "<5 seconds", "measurable": true}}
  ],
  "tensions": [
    {{"constraint_a": "watertight", "constraint_b": "tool_free_open", "description": "permanent seals (gaskets glued in place) achieve IP67 easily but require tools to open; tool-free seals (snap-fit lids) leak unless precisely engineered", "resolution_options": ["O-ring with quarter-turn cam latch", "magnetic seal with redundant secondary catch", "screw-on lid with finger-thread"]}}
  ]
}}
"""


def analyze(problem: str,
             research_result=None,
             verbose: bool = False) -> ConstraintAnalysis:
    """Extract structured constraints from a problem statement."""
    t0 = time.time()
    research_text = ""
    if research_result is not None and research_result.hits:
        research_text = ("\n# Prior research findings\n"
                         + research_result.to_prompt_text(max_hits=4)
                         + "\n")

    prompt = PROMPT_TEMPLATE.format(
        problem=problem, research_text=research_text)

    if verbose:
        print(f"[constraint_analyst] running (research={'yes' if research_text else 'no'})",
               flush=True)

    response, err = call_gemini_for_code(prompt)
    dt = time.time() - t0

    if err or not response:
        return ConstraintAnalysis(
            problem=problem, error=err or "empty response",
            raw_response=response or "", duration_s=dt)

    parsed = _parse_json_loose(response)
    if not isinstance(parsed, dict):
        return ConstraintAnalysis(
            problem=problem,
            error=f"could not parse JSON: response began with {response[:80]!r}",
            raw_response=response, duration_s=dt)

    constraints = []
    for c in parsed.get("constraints", []) or []:
        if not isinstance(c, dict): continue
        constraints.append(Constraint(
            name=str(c.get("name", "")),
            description=str(c.get("description", "")),
            priority=str(c.get("priority", "must")),
            target_value=(str(c["target_value"])
                          if c.get("target_value") is not None else None),
            measurable=bool(c.get("measurable", True)),
        ))

    tensions = []
    for t in parsed.get("tensions", []) or []:
        if not isinstance(t, dict): continue
        tensions.append(Tension(
            constraint_a=str(t.get("constraint_a", "")),
            constraint_b=str(t.get("constraint_b", "")),
            description=str(t.get("description", "")),
            resolution_options=[str(x) for x in
                                  (t.get("resolution_options") or [])],
        ))

    if verbose:
        print(f"[constraint_analyst] ✓ {len(constraints)} constraints, "
               f"{len(tensions)} tensions ({dt:.1f}s)", flush=True)
    return ConstraintAnalysis(
        problem=problem, constraints=constraints, tensions=tensions,
        raw_response=response, duration_s=dt)


def _parse_json_loose(text: str):
    """Try to parse JSON, tolerating markdown fences and surrounding text."""
    text = text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("```"):
                lines = lines[:i]; break
        text = "\n".join(lines).strip()
    # Find first { and last }
    start = text.find("{"); end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end+1])
    except json.JSONDecodeError:
        return None
