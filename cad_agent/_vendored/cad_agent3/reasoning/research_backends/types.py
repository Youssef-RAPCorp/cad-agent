"""Common types for research backends."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ResearchHit:
    """A single piece of research evidence."""
    title: str
    summary: str
    source: str           # URL, pattern_id, or "user-provided"
    relevance: float = 0.5  # 0-1, backend's own confidence
    extra: dict = field(default_factory=dict)


@dataclass
class ResearchResult:
    """Result of a research call. Multiple hits, ranked by relevance."""
    query: str
    backend_used: str          # "web", "kb", "hybrid:web+kb"
    hits: List[ResearchHit] = field(default_factory=list)
    error: Optional[str] = None
    duration_s: float = 0.0

    def to_prompt_text(self, max_hits: int = 5) -> str:
        """Format hits as compact text for LLM consumption."""
        if not self.hits:
            return f"Research found nothing relevant for: {self.query}"
        lines = [f"Research results for: {self.query}",
                 f"(via {self.backend_used}, {len(self.hits)} hits)"]
        for h in self.hits[:max_hits]:
            lines.append("")
            lines.append(f"[{h.source}]")
            lines.append(h.title)
            if h.summary:
                lines.append(h.summary[:600])
        return "\n".join(lines)
