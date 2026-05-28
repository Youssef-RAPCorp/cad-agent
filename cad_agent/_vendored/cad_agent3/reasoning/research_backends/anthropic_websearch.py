"""anthropic_websearch.py — research backend using Anthropic's web_search tool.

Uses the Anthropic API's server-side web_search_20250305 tool. Allows
real web research from any environment with ANTHROPIC_API_KEY set,
including this development container.

This backend is what runs when LLM_BACKEND=anthropic is set; the
Gemini production deployment should use gemini_websearch.py instead.

Environment:
    ANTHROPIC_API_KEY     required
    ANTHROPIC_MODEL       optional, defaults to claude-haiku-4-5
"""
from __future__ import annotations
import os
import time
from typing import Optional, List

from .types import ResearchHit, ResearchResult


def search(query: str, max_hits: int = 5) -> ResearchResult:
    """Use Claude with the web_search tool to research a query.

    The model performs the search, reads the results, and synthesizes
    a response. We parse that response into ResearchHits.
    """
    t0 = time.time()
    try:
        import anthropic
    except ImportError:
        return ResearchResult(
            query=query, backend_used="web:anthropic",
            error="anthropic SDK not installed (pip install anthropic)",
            duration_s=time.time() - t0)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ResearchResult(
            query=query, backend_used="web:anthropic",
            error="ANTHROPIC_API_KEY not set",
            duration_s=time.time() - t0)

    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
    client = anthropic.Anthropic(api_key=api_key)

    # We frame the prompt as a research task so the model uses web_search
    # purposefully and reports findings in a parseable structure.
    prompt = f"""You are a CAD engineering researcher. Research the following query and report findings as STRUCTURED text.

Query: {query}

Use web_search to find at most {max_hits} authoritative results. Focus on:
  - Real commercial products that solve this problem (with names + URLs)
  - Engineering specifications (dimensions, materials, parameters)
  - Validated design patterns or standards
  - Failure modes documented in the field

Format each finding EXACTLY like this (one block per finding, no numbering):

---
TITLE: <one-line title>
SOURCE: <URL or product name>
SUMMARY: <2-4 sentences with the engineering substance>
---

Do not include preamble, follow-up text, or commentary outside the blocks.
"""

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
        )
    except Exception as e:
        return ResearchResult(
            query=query, backend_used="web:anthropic",
            error=f"Anthropic API call failed: {type(e).__name__}: {e}",
            duration_s=time.time() - t0)

    # Extract the text blocks from the response
    text = ""
    for block in (resp.content or []):
        if hasattr(block, "text") and block.text:
            text += block.text + "\n"

    if not text.strip():
        return ResearchResult(
            query=query, backend_used="web:anthropic",
            error="Empty response from Claude (no text blocks)",
            duration_s=time.time() - t0)

    hits = _parse_findings(text)
    return ResearchResult(
        query=query, backend_used="web:anthropic",
        hits=hits[:max_hits], duration_s=time.time() - t0)


def _parse_findings(text: str) -> List[ResearchHit]:
    """Parse the structured TITLE/SOURCE/SUMMARY blocks."""
    hits = []
    # Split on delimiter lines
    blocks = text.split("---")
    for block in blocks:
        block = block.strip()
        if not block or "TITLE:" not in block:
            continue
        title, source, summary = "", "", ""
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith("TITLE:"):
                title = stripped[6:].strip()
            elif stripped.startswith("SOURCE:"):
                source = stripped[7:].strip()
            elif stripped.startswith("SUMMARY:"):
                summary = stripped[8:].strip()
            elif summary and stripped and not stripped.startswith(
                ("TITLE:", "SOURCE:")):
                # Continuation of summary
                summary += " " + stripped
        if title or source or summary:
            hits.append(ResearchHit(
                title=title or "(no title)",
                summary=summary,
                source=source or "(no source)",
                relevance=0.8,   # web hits trusted higher than KB
                extra={},
            ))
    return hits


def is_available() -> bool:
    """Available iff anthropic SDK and API key are present."""
    try:
        import anthropic  # noqa
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
