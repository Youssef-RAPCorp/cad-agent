"""knowledge_base.py — research backend that queries the static YAML pattern library.

This is the always-available fallback. Returns ResearchHits derived from
the pattern library when web search is unavailable or yields nothing.
"""
from __future__ import annotations
import time
from typing import Optional

from .types import ResearchHit, ResearchResult


def search(query: str, max_hits: int = 5) -> ResearchResult:
    """Run a query against the local YAML pattern KB."""
    t0 = time.time()
    try:
        from ...knowledge.kb_loader import search as kb_search, summarize_kb
    except Exception as e:
        return ResearchResult(
            query=query, backend_used="kb",
            error=f"KB import failed: {type(e).__name__}: {e}",
            duration_s=time.time() - t0)

    try:
        patterns = kb_search(query, limit=max_hits)
    except Exception as e:
        return ResearchResult(
            query=query, backend_used="kb",
            error=f"KB search failed: {type(e).__name__}: {e}",
            duration_s=time.time() - t0)

    hits = []
    for i, p in enumerate(patterns):
        # Higher relevance for higher-ranked results (simple linear decay)
        rel = max(0.2, 1.0 - i * 0.15)
        hits.append(ResearchHit(
            title=f"Pattern: {p.id}" + (f" ({', '.join(p.aliases)})"
                                          if p.aliases else ""),
            summary=p.to_prompt_block(),
            source=f"kb:{p.source_file}#{p.id}",
            relevance=rel,
            extra={"pattern_id": p.id,
                    "use_cases": p.use_cases,
                    "geometry_constraints": p.geometry_constraints},
        ))
    return ResearchResult(
        query=query, backend_used="kb",
        hits=hits, duration_s=time.time() - t0)


def is_available() -> bool:
    """The KB backend is always available if YAML is installed and the
    knowledge directory has any patterns."""
    try:
        from ...knowledge.kb_loader import list_all_patterns
        return len(list_all_patterns()) > 0
    except Exception:
        return False
