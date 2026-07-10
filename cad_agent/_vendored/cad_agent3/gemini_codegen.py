"""gemini_codegen.py — shared text-to-code helper using Gemini Flash.

Single source of truth for all LLM-driven build123d code generation:
  - shape_generator.py (text → Part)
  - image_to_shape.py (after vision pass)
  - llm_fitter.py (reconstruction repair tier)
  - builder.py add_feature (modify existing part)

Backend: Google Gemini via the google-genai SDK.
Default model: gemini-3.5-flash. Override with GEMINI_CODEGEN_MODEL.

Why one shared helper: keeps the SDK call shape, error handling,
markdown-fence stripping, and retry semantics identical everywhere.
Swap the backend in ONE place if needed.

Environment:
  GEMINI_API_KEY (or GOOGLE_API_KEY)   required for production
  GEMINI_CODEGEN_MODEL                 optional; default 'gemini-3.5-flash'
  LLM_BACKEND                          optional; 'gemini' (default) or 'anthropic'
                                        — 'anthropic' route exists for testing
                                        in environments where the Gemini host is
                                        unreachable; uses ANTHROPIC_API_KEY and
                                        claude-haiku-4-5 by default.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple


def call_gemini_for_code(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Send `prompt` to the configured codegen LLM, return (code, error).

    Default backend is Gemini Flash. If LLM_BACKEND=anthropic is set,
    routes to Claude instead (testing utility).
    """
    backend = os.environ.get("LLM_BACKEND", "gemini").lower()
    if backend == "anthropic":
        return _call_anthropic(prompt)
    return _call_gemini(prompt)


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("```"):
                lines = lines[:i]
                break
        text = "\n".join(lines)
    return text.strip()


def _call_gemini(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        from google import genai
    except ImportError:
        return None, "google-genai not installed (pip install google-genai)"

    api_key = (os.environ.get("GEMINI_API_KEY")
               or os.environ.get("GOOGLE_API_KEY"))
    if not api_key:
        return None, "GEMINI_API_KEY (or GOOGLE_API_KEY) not set"

    model = os.environ.get("GEMINI_CODEGEN_MODEL", "gemini-3.5-flash")

    client = genai.Client(api_key=api_key)
    try:
        resp = client.models.generate_content(model=model, contents=prompt)
    except Exception as e:
        return None, f"Gemini API call failed: {type(e).__name__}: {e}"

    text = getattr(resp, "text", None) or ""
    if not text:
        try:
            chunks = []
            for cand in (resp.candidates or []):
                content = getattr(cand, "content", None)
                if content is None:
                    continue
                for part in (getattr(content, "parts", None) or []):
                    t = getattr(part, "text", None)
                    if t:
                        chunks.append(t)
            text = "\n".join(chunks)
        except Exception:
            text = ""
    if not text:
        return None, "Empty response from Gemini"
    return _strip_fences(text), None


def _call_anthropic(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Test-only backend: Claude via Anthropic API.
    Used when LLM_BACKEND=anthropic. Production path stays Gemini.
    """
    try:
        import anthropic
    except ImportError:
        return None, "anthropic not installed (pip install anthropic)"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None, "ANTHROPIC_API_KEY not set"

    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
    client = anthropic.Anthropic(api_key=api_key)
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        return None, f"Anthropic API call failed: {type(e).__name__}: {e}"

    if not resp.content:
        return None, "Empty response from Claude"
    text = ""
    for block in resp.content:
        if hasattr(block, "text"):
            text += block.text
    if not text:
        return None, "Empty response (no text blocks)"
    return _strip_fences(text), None

