"""researcher.py — unified research API.

The Researcher is the layer that the constraint analyst, topology
chooser, and decomposer call. It hides which backend is being used.

Default backend selection: hybrid (web with KB fallback). Can be
forced via Researcher(backend="kb"|"web"|"hybrid").
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

from .research_backends.types import ResearchResult, ResearchHit
from .research_backends import hybrid, knowledge_base


@dataclass
class Researcher:
    backend: str = "hybrid"     # "hybrid" | "kb" | "web"
    max_hits: int = 5
    verbose: bool = False

    def research(self, query: str) -> ResearchResult:
        """Execute a research query through the selected backend."""
        if self.verbose:
            print(f"[researcher] backend={self.backend}  query={query!r}",
                   flush=True)
        if self.backend == "kb":
            result = knowledge_base.search(query, max_hits=self.max_hits)
        elif self.backend == "web":
            # web-only: try gemini, fall back to anthropic, no kb
            from .research_backends.hybrid import _pick_web_backend
            web_fn, _, web_name = _pick_web_backend()
            if web_fn is None:
                result = ResearchResult(
                    query=query, backend_used="web (none available)",
                    error="No web backend available — no API keys set")
            else:
                result = web_fn(query, max_hits=self.max_hits)
        else:
            result = hybrid.search(query, max_hits=self.max_hits)
        if self.verbose:
            mark = "✓" if not result.error else "✗"
            print(f"[researcher] {mark} {result.backend_used} "
                   f"({result.duration_s:.1f}s, {len(result.hits)} hits)"
                   + (f" — {result.error}" if result.error else ""),
                   flush=True)
        return result

    def research_multi(self, queries: List[str]) -> List[ResearchResult]:
        """Run multiple research queries (for compound problems)."""
        return [self.research(q) for q in queries]
