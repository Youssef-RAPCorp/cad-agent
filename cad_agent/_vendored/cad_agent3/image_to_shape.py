"""image_to_shape.py — extract shape features from an image, convert to 3D.

This is a SIDE MODULE. It is not wired into the main reconstruction
pipeline in engine.py — it's meant for offline use when you have a
photograph, sketch, or drawing of a part and want a starting-point
build123d model.

Pipeline:
  1. Send the image to Gemini (gemini-flash-latest by default) with a
     structured prompt asking for the primitive breakdown: what shape
     is this, what are the approximate dimensions, are there holes /
     fillets / features, etc.
  2. Gemini returns a JSON blob describing the shape.
  3. Feed that description to the shape_generator module, which calls
     Gemini Flash to produce executable build123d code.
  4. Return the resulting Part.

This is deliberately a two-model pipeline: Gemini is good at VISION
(reading the image), Gemini Flash is good at CODE (writing build123d). Neither
is optimized for the other's job.

Caveats:
  - The extracted dimensions are estimates. Without a reference scale
    in the image, absolute sizes are guesses. Pass `scale_hint_mm` if
    you know one dimension (e.g. "the hole is 5mm diameter").
  - Complex free-form shapes will not come back well. This works best
    for recognizable mechanical parts: brackets, plates, blocks with
    holes, simple fixtures.

Environment:
  GEMINI_API_KEY        required (or GOOGLE_API_KEY)
  GEMINI_VISION_MODEL   optional; defaults to 'gemini-flash-latest'
  GEMINI_API_KEY        required (used for both passes)

Usage:
    from cad_agent3.image_to_shape import image_to_shape

    result = image_to_shape(
        image_path="/tmp/bracket_photo.jpg",
        scale_hint_mm="The longest dimension is approximately 80mm",
    )
    if result.part is not None:
        result.save_step("/tmp/bracket.step")
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Optional, List

from .shape_generator import (
    GenerationResult, generate_shape,
)


# ---------------------------------------------------------------------------
# Vision prompt — asks for structured feature extraction.
# ---------------------------------------------------------------------------

VISION_PROMPT = """You are analyzing an image of a mechanical part or \
geometric shape to extract its CAD-relevant features.

Respond with a SINGLE JSON object (no markdown, no prose, no explanation) \
with the following fields:

{{
  "overall_description": "one-sentence description of the part",
  "primitive_decomposition": [
    {{
      "role": "base" | "feature" | "subtractive",
      "primitive": "box" | "cylinder" | "prism" | "extruded_profile" | "other",
      "dimensions_mm": {{"length": <num>, "width": <num>, "height": <num>,
                         "radius": <num>, "diameter": <num>}},
      "position_note": "where in the part this piece is (e.g. 'centered', \
'top-left corner', 'through the middle')",
      "notes": "anything else relevant"
    }}
  ],
  "notable_features": [
    "fillets on all top edges, approx 2mm radius",
    "through-hole diameter approx 5mm, centered",
    "..."
  ],
  "estimated_bbox_mm": {{"length": <num>, "width": <num>, "height": <num>}},
  "confidence": "high" | "medium" | "low",
  "ambiguities": ["things you're unsure about, e.g. 'hidden back features'"]
}}

Use millimeters throughout. If absolute scale is unclear, USE THE SCALE \
HINT below if provided; otherwise state your assumption in `ambiguities`.

{scale_hint}

Only output the JSON. No backticks, no prefix, no suffix."""


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class VisionExtraction:
    """Structured output from the vision pass."""
    raw_json: Optional[dict]
    raw_text: str                  # exact text returned by the model
    overall_description: str = ""
    confidence: str = "unknown"
    ambiguities: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ImageToShapeResult:
    """End-to-end result: vision extraction + generated part."""
    vision: VisionExtraction
    generation: Optional[GenerationResult] = None

    @property
    def part(self):
        return self.generation.part if self.generation else None

    def save_step(self, path: str) -> bool:
        return bool(self.generation and self.generation.save_step(path))

    def save_stl(self, path: str) -> bool:
        return bool(self.generation and self.generation.save_stl(path))

    def save_code(self, path: str) -> bool:
        return bool(self.generation and self.generation.save_code(path))


# ---------------------------------------------------------------------------
# Vision call (Gemini)
# ---------------------------------------------------------------------------

def _call_vision(image_path: str, scale_hint: str = "") -> VisionExtraction:
    """Call Gemini with the image and return a VisionExtraction."""
    try:
        from google import genai
    except ImportError:
        return VisionExtraction(
            raw_json=None, raw_text="",
            error="google-genai not installed (pip install google-genai)",
        )

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return VisionExtraction(
            raw_json=None, raw_text="",
            error="GEMINI_API_KEY (or GOOGLE_API_KEY) not set",
        )

    model = os.environ.get("GEMINI_VISION_MODEL", "gemini-flash-latest")

    if not os.path.isfile(image_path):
        return VisionExtraction(
            raw_json=None, raw_text="",
            error=f"image file not found: {image_path}",
        )

    # Load image bytes + infer mime type.
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    ext = image_path.lower().rsplit(".", 1)[-1]
    mime_map = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "webp": "image/webp",
        "gif": "image/gif", "bmp": "image/bmp",
    }
    mime = mime_map.get(ext, "image/jpeg")

    hint_line = (
        f"SCALE HINT: {scale_hint}" if scale_hint
        else "SCALE HINT: (none provided — state any assumptions in `ambiguities`)"
    )
    prompt = VISION_PROMPT.format(scale_hint=hint_line)

    client = genai.Client(api_key=api_key)
    try:
        from google.genai import types as genai_types
        resp = client.models.generate_content(
            model=model,
            contents=[
                genai_types.Part.from_bytes(data=image_bytes, mime_type=mime),
                prompt,
            ],
        )
    except Exception as e:
        return VisionExtraction(
            raw_json=None, raw_text="",
            error=f"Gemini API call failed: {e}",
        )

    text = getattr(resp, "text", None) or ""
    if not text:
        try:
            text = "\n".join(
                p.text for c in (resp.candidates or [])
                for p in (c.content.parts or []) if getattr(p, "text", None)
            )
        except Exception:
            text = ""
    text = text.strip()

    # Strip markdown fences if present.
    cleaned = text
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("```"):
                lines = lines[:i]
                break
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except Exception as e:
        return VisionExtraction(
            raw_json=None, raw_text=text,
            error=f"could not parse JSON from vision response: {e}",
        )

    return VisionExtraction(
        raw_json=parsed,
        raw_text=text,
        overall_description=str(parsed.get("overall_description", "")),
        confidence=str(parsed.get("confidence", "unknown")),
        ambiguities=list(parsed.get("ambiguities", []) or []),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def image_to_shape(
    image_path: str,
    scale_hint_mm: str = "",
    max_iterations: int = 3,
) -> ImageToShapeResult:
    """Convert an image of a part into a build123d Part.

    Args:
        image_path: path to a JPEG/PNG/WebP image of the part.
        scale_hint_mm: optional natural-language scale reference,
            e.g. "the diameter of the central hole is 5mm". Without it
            the absolute size will be a guess.
        max_iterations: code-generation retry budget.

    Returns:
        ImageToShapeResult. If `.part` is None, check `.vision.error`
        or `.generation.error` for what went wrong.
    """
    extraction = _call_vision(image_path, scale_hint=scale_hint_mm)
    if extraction.error or extraction.raw_json is None:
        return ImageToShapeResult(vision=extraction, generation=None)

    # Build the natural-language description for the shape generator
    # from the structured JSON. We don't hand-parse each primitive —
    # we feed the whole JSON to Gemini so it has full context.
    features_json = json.dumps(extraction.raw_json, indent=2)
    description = (
        f"{extraction.overall_description}\n\n"
        f"FEATURE BREAKDOWN (from vision analysis):\n{features_json}"
    )
    if scale_hint_mm:
        description += f"\n\nSCALE HINT: {scale_hint_mm}"

    gen = generate_shape(description=description, max_iterations=max_iterations)
    return ImageToShapeResult(vision=extraction, generation=gen)
