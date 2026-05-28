"""topology_chooser.py — pick a validated approach from research/KB.

Given research findings and constraint analysis, select a specific
topology / architectural pattern to commit to. NO compromise — pick
ONE pattern and justify it.

The output is a TopologyChoice that the decomposer turns into a spec.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import json
import time

from ..gemini_codegen import call_gemini_for_code


@dataclass
class TopologyChoice:
    chosen_pattern_id: str       # the pattern we're committing to
    chosen_pattern_aliases: List[str] = field(default_factory=list)
    rationale: str = ""           # why this and not the alternatives
    rejected_alternatives: List[dict] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    confidence: float = 0.5
    raw_response: str = ""
    error: Optional[str] = None
    duration_s: float = 0.0

    def is_valid(self) -> bool:
        return (not self.error) and bool(self.chosen_pattern_id)

    def to_prompt_text(self) -> str:
        lines = [f"# Chosen topology: {self.chosen_pattern_id}"]
        if self.chosen_pattern_aliases:
            lines.append(f"(also known as: {', '.join(self.chosen_pattern_aliases)})")
        if self.rationale:
            lines.append(f"\nRationale: {self.rationale}")
        if self.citations:
            lines.append(f"\nValidated by: {', '.join(self.citations)}")
        return "\n".join(lines)


PROMPT_TEMPLATE = """You are a senior design engineer making an architectural commitment for a project.

# Problem
{problem}

# Constraints and tensions (from earlier analysis)
{constraints_text}

# Research findings (real products / patterns / literature)
{research_text}

# Your task
COMMIT to ONE topology / architectural pattern that resolves the constraint tensions.

Important rules:
- DO NOT compromise toward the middle. Pick a side on every tension and explain why.
- If the research found a real commercial product or validated pattern that fits, use that exactly. Do NOT invent a novel approach when a proven one exists.
- Cite the validation source (URL, product name, or pattern ID).
- Reject and name the alternatives you considered.

Output a JSON object with these keys:
{{
  "chosen_pattern_id": "snake_case_identifier",
  "chosen_pattern_aliases": ["alt name 1", "alt name 2"],
  "rationale": "1-3 sentences explaining why THIS pattern over alternatives, with reference to specific constraints",
  "rejected_alternatives": [
    {{"id": "alt_pattern", "why_rejected": "specific reason"}}
  ],
  "citations": ["product name or URL", "..."],
  "confidence": 0.0-1.0
}}

Output ONLY the JSON object. No fences, no preamble.
"""


def choose(problem: str,
           constraints,
           research_result,
           verbose: bool = False) -> TopologyChoice:
    """Pick a topology given constraint analysis and research results."""
    t0 = time.time()
    constraints_text = (constraints.to_prompt_text()
                        if constraints else "(no constraint analysis available)")
    research_text = (research_result.to_prompt_text(max_hits=5)
                     if research_result else "(no research available)")

    prompt = PROMPT_TEMPLATE.format(
        problem=problem, constraints_text=constraints_text,
        research_text=research_text)

    if verbose:
        print(f"[topology_chooser] running...", flush=True)
    response, err = call_gemini_for_code(prompt)
    dt = time.time() - t0

    if err or not response:
        return TopologyChoice(
            chosen_pattern_id="", error=err or "empty response",
            raw_response=response or "", duration_s=dt)

    parsed = _parse_json_loose(response)
    if not isinstance(parsed, dict):
        return TopologyChoice(
            chosen_pattern_id="",
            error=f"could not parse JSON: {response[:80]!r}",
            raw_response=response, duration_s=dt)

    out = TopologyChoice(
        chosen_pattern_id=str(parsed.get("chosen_pattern_id", "")),
        chosen_pattern_aliases=[str(x) for x in
                                  (parsed.get("chosen_pattern_aliases") or [])],
        rationale=str(parsed.get("rationale", "")),
        rejected_alternatives=[
            d for d in (parsed.get("rejected_alternatives") or [])
            if isinstance(d, dict)],
        citations=[str(x) for x in (parsed.get("citations") or [])],
        confidence=float(parsed.get("confidence", 0.5) or 0.5),
        raw_response=response, duration_s=dt,
    )
    if verbose:
        mark = "✓" if out.is_valid() else "✗"
        print(f"[topology_chooser] {mark} chose {out.chosen_pattern_id!r} "
               f"(conf={out.confidence:.2f}, {dt:.1f}s)", flush=True)
    return out


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
