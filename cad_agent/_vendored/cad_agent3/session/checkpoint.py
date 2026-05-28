"""checkpoint.py — save and restore session state.

A checkpoint captures the entire session at a given log sequence:
  - The current part state (the artifact registry's latest 'part' artifact)
  - The full log up to that sequence
  - The full artifact registry up to that sequence

Restoring a checkpoint:
  - Truncates the log back to the checkpoint
  - Replays operations that were popped (or simply restores the saved part)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class Checkpoint:
    label: str
    sequence: int                 # log sequence at the time of the checkpoint
    saved_part: Any               # the current part at this point
    saved_artifact_count: int = 0
    timestamp: float = 0.0

    def __repr__(self):
        return (f"<Checkpoint '{self.label}' @ seq={self.sequence}>")


class CheckpointManager:
    """Manages a stack of named checkpoints."""

    def __init__(self):
        self._checkpoints: List[Checkpoint] = []

    def save(self, label: str, sequence: int, current_part: Any,
              artifact_count: int = 0) -> Checkpoint:
        import time
        cp = Checkpoint(
            label=label, sequence=sequence, saved_part=current_part,
            saved_artifact_count=artifact_count, timestamp=time.time())
        self._checkpoints.append(cp); return cp

    def get(self, label: str) -> Optional[Checkpoint]:
        for cp in self._checkpoints:
            if cp.label == label: return cp
        return None

    def latest(self) -> Optional[Checkpoint]:
        return self._checkpoints[-1] if self._checkpoints else None

    def all(self) -> List[Checkpoint]:
        return list(self._checkpoints)

    def remove(self, label: str) -> bool:
        for i, cp in enumerate(self._checkpoints):
            if cp.label == label:
                del self._checkpoints[i]; return True
        return False

    def __len__(self): return len(self._checkpoints)
