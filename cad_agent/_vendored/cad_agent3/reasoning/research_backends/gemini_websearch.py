"""gemini_websearch.py — research backend using Gemini's grounding tool.

Uses the google-genai SDK's google_search grounding feature. This is
what runs in the production Gemini deployment.

Environment:
    GEMINI_API_KEY (or GOOGLE_API_KEY)    required
    GEMINI_RESEARCH_MODEL                 optional, defaults to gemini-flash-latest
"""
from __future__ import annotations
import os
import time
from typing import List

from .types import ResearchHit, ResearchResult


def search(query: str, max_hits: int = 5) -> ResearchResult:
    """Use Gemini with grounding/web_search to research a query."""
    t0 = time.time()
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        return ResearchResult(
            query=query, backend_used="web:gemini",
            error="google-genai not installed (pip install google-genai)",
            duration_s=time.time() - t0)

    api_key = (os.environ.get("GEMINI_API_KEY")
               or os.environ.get("GOOGLE_API_KEY"))
    if not api_key:
        return ResearchResult(
            query=query, backend_used="web:gemini",
            error="GEMINI_API_KEY (or GOOGLE_API_KEY) not set",
            duration_s=time.time() - t0)

    model = os.environ.get("GEMINI_RESEARCH_MODEL", "gemini-flash-latest")

    prompt = f"""You are a CAD engineering researcher. Research the following query and report findings as STRUCTURED text.

Query: {query}

Use the search tool to find at most {max_hits} authoritative results. Focus on:
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

    client = genai.Client(api_key=api_key)
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                tools=[genai_types.Tool(
                    google_search=genai_types.GoogleSearch())],
            ),
        )
    except Exception as e:
        return ResearchResult(
            query=query, backend_used="web:gemini",
            error=f"Gemini API call failed: {type(e).__name__}: {e}",
            duration_s=time.time() - t0)

    text = getattr(resp, "text", None) or ""
    if not text:
        try:
            chunks = []
            for cand in (resp.candidates or []):
                content = getattr(cand, "content", None)
                if content is None: continue
                for part in (getattr(content, "parts", None) or []):
                    t = getattr(part, "text", None)
                    if t: chunks.append(t)
            text = "\n".join(chunks)
        except Exception:
            text = ""
    if not text.strip():
        return ResearchResult(
            query=query, backend_used="web:gemini",
            error="Empty response from Gemini",
            duration_s=time.time() - t0)

    hits = _parse_findings(text)
    return ResearchResult(
        query=query, backend_used="web:gemini",
        hits=hits[:max_hits], duration_s=time.time() - t0)


def _parse_findings(text: str) -> List[ResearchHit]:
    """Parse the structured TITLE/SOURCE/SUMMARY blocks."""
    hits = []
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
                summary += " " + stripped
        if title or source or summary:
            hits.append(ResearchHit(
                title=title or "(no title)",
                summary=summary,
                source=source or "(no source)",
                relevance=0.8,
                extra={},
            ))
    return hits


def is_available() -> bool:
    try:
        from google import genai  # noqa
    except ImportError:
        return False
    return bool(os.environ.get("GEMINI_API_KEY")
                 or os.environ.get("GOOGLE_API_KEY"))
