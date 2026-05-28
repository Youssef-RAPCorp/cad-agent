"""decomposer.py — turn TopologyChoice + constraints into a concrete geometric spec.

The decomposer takes the chosen topology and the constraint analysis,
and produces a structured GeometricSpec — a list of features with
concrete dimensions that codegen can implement directly.

This is the bridge between "we chose pattern X" and actual CAD code.
The spec is structured so it can be checked against constraints by
the spec_critic before any code is generated.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import json
import time

from ..gemini_codegen import call_gemini_for_code


@dataclass
class GeometricFeature:
    """A single geometric feature with its dimensions."""
    name: str                  # e.g. "back_panel", "lens_hole_left"
    kind: str                  # "panel", "hole", "slot", "fillet", "score_line", etc.
    dimensions: dict = field(default_factory=dict)  # numeric specs
    position: dict = field(default_factory=dict)    # x,y,z + rotation
    parent: Optional[str] = None       # parent feature for hinges
    notes: str = ""


@dataclass
class GeometricSpec:
    """The complete geometric specification, ready for codegen."""
    summary: str = ""
    overall_dimensions: dict = field(default_factory=dict)
    materials: dict = field(default_factory=dict)
    features: List[GeometricFeature] = field(default_factory=list)
    assembly_notes: List[str] = field(default_factory=list)
    raw_response: str = ""
    error: Optional[str] = None
    duration_s: float = 0.0

    def is_valid(self) -> bool:
        return (not self.error) and len(self.features) > 0

    def to_prompt_text(self) -> str:
        lines = [f"# Geometric specification"]
        if self.summary: lines.append(self.summary)
        if self.overall_dimensions:
            dims = ", ".join(f"{k}={v}" for k, v in self.overall_dimensions.items())
            lines.append(f"\nOverall: {dims}")
        if self.materials:
            mats = "; ".join(f"{k}: {v}" for k, v in self.materials.items())
            lines.append(f"Materials: {mats}")
        lines.append(f"\nFeatures ({len(self.features)}):")
        for f in self.features:
            dims = ", ".join(f"{k}={v}" for k, v in f.dimensions.items())
            pos = ", ".join(f"{k}={v}" for k, v in f.position.items())
            line = f"- [{f.kind}] {f.name}"
            if dims: line += f"  dims={{{dims}}}"
            if pos: line += f"  pos={{{pos}}}"
            if f.notes: line += f"  ({f.notes})"
            lines.append(line)
        if self.assembly_notes:
            lines.append("\nAssembly notes:")
            for n in self.assembly_notes:
                lines.append(f"- {n}")
        return "\n".join(lines)


PROMPT_TEMPLATE = """You are a senior CAD engineer translating a chosen design topology into a concrete GEOMETRIC SPECIFICATION.

# Problem
{problem}

# Chosen topology
{topology_text}

# Constraints (must satisfy all)
{constraints_text}

# Research findings
{research_text}

# Your task
Produce a structured geometric spec listing every feature the CAD model must include, with EXACT DIMENSIONS in millimeters. The spec will be implemented directly in build123d code, so be unambiguous.

Output a JSON object:
{{
  "summary": "1-2 sentence description of the design",
  "overall_dimensions": {{"width_mm": 130, "depth_mm": 50, "height_mm": 75, "folded_thickness_mm": 8}},
  "materials": {{"primary": "1.5mm E-flute corrugated cardboard"}},
  "features": [
    {{
      "name": "snake_case_id",
      "kind": "panel" | "hole" | "slot" | "score_line" | "fillet" | "chamfer" | "boss" | "rib" | "flap" | "tab",
      "dimensions": {{"width_mm": ..., "height_mm": ..., "thickness_mm": ..., "diameter_mm": ..., ...}},
      "position": {{"x_mm": ..., "y_mm": ..., "z_mm": ..., "rotation_deg": ...}},
      "parent": "name_of_parent_feature_if_hinged_to_one_or_null",
      "notes": "any clarifying detail"
    }}
  ],
  "assembly_notes": ["short string describing assembly step or critical relationship"]
}}

Rules:
- Every dimension is a number with units in the key name (e.g. "width_mm": 130).
- Every feature has a name, kind, dimensions (at minimum), and position.
- For features hinged to other features, set parent to the other feature's name.
- Be EXHAUSTIVE — every panel, hole, slot, score line, tab. Nothing implicit.
- If the topology requires a specific known parameter (e.g. IPD=63mm for VR), include it explicitly.

Output ONLY the JSON object. No fences, no preamble.
"""


def decompose(problem: str,
               topology,
               constraints,
               research_result,
               verbose: bool = False,
               previous_issues=None) -> GeometricSpec:
    """Generate a concrete geometric spec from the topology choice.

    If previous_issues is supplied (a list of SpecIssue from a prior
    spec_critic rejection), they're added to the prompt so the decomposer
    knows what to fix in this revision.
    """
    t0 = time.time()
    topology_text = (topology.to_prompt_text() if topology
                     else "(no topology selected)")
    constraints_text = (constraints.to_prompt_text() if constraints
                        else "(no constraints)")
    research_text = (research_result.to_prompt_text(max_hits=4)
                     if research_result else "(no research)")

    issues_text = ""
    if previous_issues:
        lines = ["# CRITIC FEEDBACK from previous revision (FIX THESE):"]
        for i in previous_issues:
            lines.append(f"- [{i.severity}] {i.constraint}"
                          + (f" / {i.feature}" if i.feature else "")
                          + f": {i.description}")
            if i.suggested_fix:
                lines.append(f"    suggested fix: {i.suggested_fix}")
        issues_text = "\n" + "\n".join(lines) + "\n"

    prompt = PROMPT_TEMPLATE.format(
        problem=problem, topology_text=topology_text,
        constraints_text=constraints_text,
        research_text=research_text + issues_text)

    if verbose:
        print(f"[decomposer] running...", flush=True)
    response, err = call_gemini_for_code(prompt)
    dt = time.time() - t0

    if err or not response:
        return GeometricSpec(
            error=err or "empty response",
            raw_response=response or "", duration_s=dt)

    parsed = _parse_json_loose(response)
    if not isinstance(parsed, dict):
        return GeometricSpec(
            error=f"could not parse JSON: {response[:80]!r}",
            raw_response=response, duration_s=dt)

    features = []
    for f in parsed.get("features", []) or []:
        if not isinstance(f, dict): continue
        features.append(GeometricFeature(
            name=str(f.get("name", "")),
            kind=str(f.get("kind", "")),
            dimensions=dict(f.get("dimensions") or {}),
            position=dict(f.get("position") or {}),
            parent=(str(f["parent"]) if f.get("parent") else None),
            notes=str(f.get("notes", "")),
        ))

    spec = GeometricSpec(
        summary=str(parsed.get("summary", "")),
        overall_dimensions=dict(parsed.get("overall_dimensions") or {}),
        materials=dict(parsed.get("materials") or {}),
        features=features,
        assembly_notes=[str(n) for n in (parsed.get("assembly_notes") or [])],
        raw_response=response, duration_s=dt,
    )
    if verbose:
        mark = "✓" if spec.is_valid() else "✗"
        print(f"[decomposer] {mark} {len(spec.features)} features ({dt:.1f}s)",
               flush=True)
    return spec


def _parse_json_loose(text: str):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"): lines = lines[1:]
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("```"):
                lines = lines[:i]; break
        text = "\n".join(lines).strip()
    start = text.find("{"); end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end+1])
    except json.JSONDecodeError:
        return None
