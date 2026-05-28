"""spec_critic.py — adversarial review of a GeometricSpec against constraints.

This is the LAST gate before codegen. The critic reads the spec and the
constraints, and either:
  - APPROVES the spec (returns Approved with no issues)
  - REJECTS with a list of specific issues that must be fixed

Issues are concrete and actionable: "feature X has dimension Y but
constraint Z requires <Y/2", not vague "this might not work".
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import json
import time

from ..gemini_codegen import call_gemini_for_code


@dataclass
class SpecIssue:
    severity: str           # "blocker" | "warning" | "advisory"
    constraint: str         # which constraint name is violated
    feature: Optional[str] = None    # which feature is the problem (if applicable)
    description: str = ""   # what's wrong
    suggested_fix: str = "" # how to fix it


@dataclass
class CriticReport:
    approved: bool
    issues: List[SpecIssue] = field(default_factory=list)
    overall_assessment: str = ""
    raw_response: str = ""
    error: Optional[str] = None
    duration_s: float = 0.0

    def has_blockers(self) -> bool:
        return any(i.severity == "blocker" for i in self.issues)

    def to_prompt_text(self) -> str:
        verdict = "APPROVED" if self.approved else "REJECTED"
        lines = [f"# Critic verdict: {verdict}"]
        if self.overall_assessment:
            lines.append(self.overall_assessment)
        if self.issues:
            lines.append(f"\nIssues ({len(self.issues)}):")
            for i in self.issues:
                lines.append(f"- [{i.severity}] {i.constraint}"
                              + (f" / {i.feature}" if i.feature else "")
                              + f": {i.description}")
                if i.suggested_fix:
                    lines.append(f"    fix: {i.suggested_fix}")
        return "\n".join(lines)


PROMPT_TEMPLATE = """You are an adversarial reviewer. Your job is to find PROBLEMS in a proposed design spec BEFORE any code is written. Be skeptical and concrete.

# Problem
{problem}

# Constraints (every one must be satisfied)
{constraints_text}

# Proposed geometric spec
{spec_text}

# Your task
Check the spec against EVERY constraint. For each constraint, ask:
1. Does the spec actually achieve this constraint? Compute it from the dimensions if possible.
2. Are there features that contradict each other?
3. Are there missing features the constraint requires that aren't in the spec?
4. Does any dimension violate physical / manufacturing reality?

Output a JSON object:
{{
  "approved": true | false,
  "overall_assessment": "1-3 sentence summary of whether the spec is sound",
  "issues": [
    {{
      "severity": "blocker" | "warning" | "advisory",
      "constraint": "name of the violated constraint",
      "feature": "name of the offending feature, or null",
      "description": "specific problem with concrete numbers if possible",
      "suggested_fix": "concrete fix"
    }}
  ]
}}

Rules:
- approved must be FALSE if there is even one blocker.
- Be specific. "Folded thickness 12mm exceeds constraint of 10mm" not "may be too thick".
- If the spec is sound, output approved=true with empty issues. Don't invent problems.

Output ONLY the JSON object. No fences, no preamble.
"""


def review(problem: str,
           spec,
           constraints,
           verbose: bool = False) -> CriticReport:
    """Adversarially review a spec against constraints."""
    t0 = time.time()
    spec_text = spec.to_prompt_text() if spec else "(no spec)"
    constraints_text = (constraints.to_prompt_text() if constraints
                        else "(no constraints)")

    prompt = PROMPT_TEMPLATE.format(
        problem=problem, constraints_text=constraints_text,
        spec_text=spec_text)

    if verbose:
        print(f"[spec_critic] running...", flush=True)
    response, err = call_gemini_for_code(prompt)
    dt = time.time() - t0

    if err or not response:
        return CriticReport(
            approved=False, error=err or "empty response",
            raw_response=response or "", duration_s=dt)

    parsed = _parse_json_loose(response)
    if not isinstance(parsed, dict):
        return CriticReport(
            approved=False,
            error=f"could not parse JSON: {response[:80]!r}",
            raw_response=response, duration_s=dt)

    issues = []
    for i in parsed.get("issues", []) or []:
        if not isinstance(i, dict): continue
        issues.append(SpecIssue(
            severity=str(i.get("severity", "warning")),
            constraint=str(i.get("constraint", "")),
            feature=(str(i["feature"]) if i.get("feature") else None),
            description=str(i.get("description", "")),
            suggested_fix=str(i.get("suggested_fix", "")),
        ))

    report = CriticReport(
        approved=bool(parsed.get("approved", False)),
        overall_assessment=str(parsed.get("overall_assessment", "")),
        issues=issues, raw_response=response, duration_s=dt,
    )
    # Force-disapprove if there are any blockers, even if LLM said approved
    if report.has_blockers():
        report.approved = False

    if verbose:
        mark = "✓ APPROVED" if report.approved else f"✗ REJECTED ({len(issues)} issues)"
        print(f"[spec_critic] {mark} ({dt:.1f}s)", flush=True)
    return report


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
