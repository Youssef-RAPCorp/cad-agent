"""hybrid.py — research backend that tries web first, falls back to KB.

Selects the best available web backend automatically based on env:
  - LLM_BACKEND=anthropic         -> anthropic_websearch
  - GEMINI_API_KEY set            -> gemini_websearch
  - ANTHROPIC_API_KEY set         -> anthropic_websearch (fallback)
  - neither                        -> kb only

If web search succeeds with >=1 hit, uses those. If web fails or returns
nothing, falls back to KB. Optionally MERGES web + KB hits (web first).
"""
from __future__ import annotations
import os
from typing import List

from .types import ResearchResult, ResearchHit
from . import knowledge_base


def _pick_web_backend():
    """Return the (search_fn, is_available_fn, name) tuple for the
    preferred web backend based on environment, or (None, None, None)
    if none are available."""
    explicit = os.environ.get("LLM_BACKEND", "").lower()

    # Honor explicit override first
    if explicit == "anthropic":
        from . import anthropic_websearch as a
        if a.is_available():
            return (a.search, a.is_available, "anthropic")
    if explicit == "gemini":
        from . import gemini_websearch as g
        if g.is_available():
            return (g.search, g.is_available, "gemini")

    # Otherwise try Gemini (production default), then Anthropic
    try:
        from . import gemini_websearch as g
        if g.is_available():
            return (g.search, g.is_available, "gemini")
    except Exception:
        pass
    try:
        from . import anthropic_websearch as a
        if a.is_available():
            return (a.search, a.is_available, "anthropic")
    except Exception:
        pass
    return (None, None, None)


def search(query: str, max_hits: int = 5,
           merge_kb: bool = True) -> ResearchResult:
    """Hybrid research: web first, KB fallback or merge."""
    web_search, _, web_name = _pick_web_backend()

    if web_search is not None:
        web_result = web_search(query, max_hits=max_hits)
        if not web_result.error and web_result.hits:
            # Web succeeded; optionally augment with KB
            if merge_kb:
                kb_result = knowledge_base.search(query, max_hits=3)
                merged = list(web_result.hits) + list(kb_result.hits)
                # Dedupe by source
                seen = set(); unique = []
                for h in merged:
                    if h.source in seen: continue
                    seen.add(h.source); unique.append(h)
                return ResearchResult(
                    query=query,
                    backend_used=f"hybrid:web:{web_name}+kb",
                    hits=unique[:max_hits + 2],
                    duration_s=web_result.duration_s
                                + kb_result.duration_s,
                )
            return web_result
        # Web failed or empty — fall back to KB
        kb_result = knowledge_base.search(query, max_hits=max_hits)
        kb_result.backend_used = (
            f"kb (web:{web_name} returned: {web_result.error or 'no hits'})")
        return kb_result

    # No web backend available
    kb_result = knowledge_base.search(query, max_hits=max_hits)
    kb_result.backend_used = "kb (no web backend available)"
    return kb_result
