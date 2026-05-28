"""Result objects returned by the CAD agent.

These are plain dataclasses — no behavior. Inspect fields, save files,
or pipe into another tool. The `part` field holds the live build123d
object (None if execution was disabled or failed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional


@dataclass
class CADResult:
    """Outcome of a single CAD generation request.

    Fields:
      spec:           original natural-language input
      success:        True if a valid build123d Part was produced
      script:         the generated build123d Python source
      part:           live build123d Part object (None if not executed)
      volume_mm3:     volume of the part, if it was executed
      step_path:      path to the exported STEP file (None if not written)
      stl_path:       path to the exported STL file
      script_path:    path to the saved .py source
      output_dir:     directory containing all artifacts
      reasoning_log:  per-revision notes from the critic-feedback loop
      error:          error string if success is False
      metadata:       backend, model, token counts, latency, etc.
    """

    spec: str
    success: bool
    script: str = ""
    part: Optional[Any] = None
    volume_mm3: Optional[float] = None
    step_path: Optional[Path] = None
    stl_path: Optional[Path] = None
    script_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    reasoning_log: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.success

    def summary(self) -> str:
        """A human-readable one-paragraph summary."""
        if not self.success:
            return f"FAILED: {self.error or 'unknown error'}"
        lines = [f"OK: '{self.spec[:60]}{'...' if len(self.spec) > 60 else ''}'"]
        if self.volume_mm3 is not None:
            lines.append(f"  volume: {self.volume_mm3/1e9:.4f} m³ ({self.volume_mm3:.0f} mm³)")
        if self.step_path:
            lines.append(f"  STEP:   {self.step_path}")
        if self.stl_path:
            lines.append(f"  STL:    {self.stl_path}")
        if self.script_path:
            lines.append(f"  script: {self.script_path}")
        if self.reasoning_log:
            lines.append(f"  revisions: {len(self.reasoning_log)}")
        return "\n".join(lines)
