"""kb_loader.py — load and index the YAML pattern library.

The knowledge base is a directory of YAML files. Each YAML file
contains a `patterns` list, where each pattern has at minimum:
  id: unique identifier
  summary: human-readable description
And optionally:
  aliases, use_cases, key_parameters, geometry_constraints,
  real_world_validation, failure_modes, when_to_use, not_suitable_for.

This module loads them all and provides search by id/alias/keyword.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Dict, Optional

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PATTERNS_DIR = os.path.join(_THIS_DIR, "patterns")


@dataclass
class Pattern:
    id: str
    summary: str
    source_file: str
    aliases: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    key_parameters: dict = field(default_factory=dict)
    geometry_constraints: dict = field(default_factory=dict)
    real_world_validation: list = field(default_factory=list)
    failure_modes: list = field(default_factory=list)
    when_to_use: str = ""
    not_suitable_for: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict, source_file: str) -> "Pattern":
        # Pull known fields; everything else goes in `extra`.
        known = {
            "id", "summary", "aliases", "use_cases", "key_parameters",
            "geometry_constraints", "real_world_validation",
            "failure_modes", "when_to_use", "not_suitable_for",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            id=data.get("id", ""),
            summary=data.get("summary", "").strip(),
            source_file=source_file,
            aliases=list(data.get("aliases") or []),
            use_cases=list(data.get("use_cases") or []),
            key_parameters=dict(data.get("key_parameters") or {}),
            geometry_constraints=dict(data.get("geometry_constraints") or {}),
            real_world_validation=list(data.get("real_world_validation") or []),
            failure_modes=list(data.get("failure_modes") or []),
            when_to_use=(data.get("when_to_use") or "").strip(),
            not_suitable_for=list(data.get("not_suitable_for") or []),
            extra=extra,
        )

    def matches_keyword(self, kw: str) -> bool:
        kw_low = kw.lower()
        if kw_low == self.id.lower():
            return True
        if any(kw_low == a.lower() for a in self.aliases):
            return True
        # Fuzzy match in summary, use_cases, when_to_use
        haystacks = [self.summary, self.when_to_use, " ".join(self.use_cases)]
        for h in haystacks:
            if kw_low in h.lower():
                return True
        return False

    def to_prompt_block(self) -> str:
        """Render as compact text suitable for an LLM prompt."""
        lines = [f"## Pattern: {self.id}"]
        if self.aliases:
            lines.append(f"Aliases: {', '.join(self.aliases)}")
        lines.append(f"Summary: {self.summary}")
        if self.use_cases:
            lines.append(f"Use cases: {', '.join(self.use_cases[:3])}")
        if self.when_to_use:
            lines.append(f"When to use: {self.when_to_use}")
        if self.key_parameters:
            kp = ", ".join(f"{k}={v}" for k, v in
                            list(self.key_parameters.items())[:5])
            lines.append(f"Key parameters: {kp}")
        if self.geometry_constraints:
            gc = "; ".join(f"{k}: {v}" for k, v in
                            list(self.geometry_constraints.items())[:5])
            lines.append(f"Geometry constraints: {gc}")
        if self.real_world_validation:
            v = self.real_world_validation[0]
            v_str = (v.get("product") or v.get("reference") or str(v))
            lines.append(f"Validated by: {v_str}")
        if self.failure_modes:
            modes = [str(m) if not isinstance(m, str) else m
                     for m in self.failure_modes[:2]]
            lines.append(f"Failure modes: {'; '.join(modes)}")
        return "\n".join(lines)


@lru_cache(maxsize=1)
def _load_all_patterns() -> Dict[str, Pattern]:
    """Read every YAML in patterns/ and index by id."""
    if not _HAS_YAML:
        raise RuntimeError("PyYAML required. pip install pyyaml")
    if not os.path.isdir(PATTERNS_DIR):
        return {}
    out: Dict[str, Pattern] = {}
    for fname in sorted(os.listdir(PATTERNS_DIR)):
        if not fname.endswith((".yaml", ".yml")):
            continue
        path = os.path.join(PATTERNS_DIR, fname)
        try:
            with open(path) as f:
                doc = _yaml.safe_load(f) or {}
        except Exception:
            continue
        for p_data in doc.get("patterns", []) or []:
            if not isinstance(p_data, dict) or "id" not in p_data:
                continue
            p = Pattern.from_dict(p_data, fname)
            if p.id in out:
                # Duplicate id — keep first, skip
                continue
            out[p.id] = p
    return out


def list_all_patterns() -> List[Pattern]:
    return list(_load_all_patterns().values())


def get_pattern(pattern_id: str) -> Optional[Pattern]:
    return _load_all_patterns().get(pattern_id)


def search(query: str, limit: int = 5) -> List[Pattern]:
    """Return up to `limit` patterns matching the query, ranked by relevance.

    Ranking: exact id match > alias match > use-case match > summary match.
    """
    q = (query or "").strip().lower()
    if not q:
        return []
    all_patterns = _load_all_patterns()
    scored: List[tuple] = []
    # Tokenize query into words for keyword-style matching
    words = [w for w in re.findall(r"[a-z0-9_]+", q) if len(w) >= 2]

    for p in all_patterns.values():
        score = 0
        # Exact id or alias = strong signal
        if p.id.lower() == q:
            score += 100
        if any(a.lower() == q for a in p.aliases):
            score += 80
        # Word-level: each query word that appears in id/aliases/summary
        text = (p.id + " " + " ".join(p.aliases) + " " + p.summary
                 + " " + p.when_to_use + " " + " ".join(p.use_cases)).lower()
        for w in words:
            if w in p.id.lower() or any(w in a.lower() for a in p.aliases):
                score += 15
            elif w in p.summary.lower():
                score += 5
            elif w in text:
                score += 2
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda t: -t[0])
    return [p for _, p in scored[:limit]]


def summarize_kb() -> str:
    """One-line summary suitable for an LLM prompt header."""
    patterns = list_all_patterns()
    if not patterns:
        return "(knowledge base empty)"
    by_file: Dict[str, List[str]] = {}
    for p in patterns:
        by_file.setdefault(p.source_file, []).append(p.id)
    lines = [f"Knowledge base: {len(patterns)} patterns across "
             f"{len(by_file)} categories"]
    for f, ids in sorted(by_file.items()):
        cat = f.replace(".yaml", "").replace("_", " ")
        lines.append(f"  {cat}: {', '.join(ids[:3])}"
                      + (f" (+{len(ids)-3} more)" if len(ids) > 3 else ""))
    return "\n".join(lines)
