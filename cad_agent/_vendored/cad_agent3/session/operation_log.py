"""operation_log.py — append-only log of operations with rollback support.

Each entry records:
  - sequence: monotonically increasing
  - op_name: which operation ran
  - inputs: the inputs dict (sans large objects)
  - result: OperationResult (sans the large new_part)
  - undo_data: opaque dict the operation can use to undo
  - artifact_id: id of the artifact produced (if any)
  - timestamp

Rollback: pop entries until reaching a target sequence, calling
op.undo(undo_data) on each.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class LogEntry:
    sequence: int
    op_name: str
    inputs_summary: Dict[str, Any]    # truncated for printability
    effect: str                        # human-readable
    ok: bool
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifact_id: Optional[int] = None
    undo_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class OperationLog:
    """Append-only log of all operations applied in a session."""

    def __init__(self):
        self._entries: List[LogEntry] = []
        self._next_seq = 1

    def append(self, op_name: str, inputs: dict, result, artifact_id=None) -> LogEntry:
        """Append a record of an applied operation."""
        seq = self._next_seq; self._next_seq += 1
        entry = LogEntry(
            sequence=seq, op_name=op_name,
            inputs_summary=_summarize_inputs(inputs),
            effect=result.effect, ok=result.ok,
            error=result.error, metrics=dict(result.metrics or {}),
            artifact_id=artifact_id,
            undo_data=dict(result.undo_data or {}),
        )
        self._entries.append(entry); return entry

    def entries(self) -> List[LogEntry]:
        return list(self._entries)

    def latest(self) -> Optional[LogEntry]:
        return self._entries[-1] if self._entries else None

    def at_sequence(self, seq: int) -> Optional[LogEntry]:
        for e in self._entries:
            if e.sequence == seq: return e
        return None

    def pop(self) -> Optional[LogEntry]:
        """Remove and return the most recent entry."""
        return self._entries.pop() if self._entries else None

    def truncate_to(self, seq: int):
        """Drop all entries with sequence > seq."""
        self._entries = [e for e in self._entries if e.sequence <= seq]
        if self._entries:
            self._next_seq = self._entries[-1].sequence + 1
        else:
            self._next_seq = 1

    def __len__(self): return len(self._entries)

    def summary(self) -> str:
        if not self._entries:
            return "OperationLog: empty"
        ok = sum(1 for e in self._entries if e.ok)
        bad = len(self._entries) - ok
        return f"OperationLog: {len(self._entries)} entries ({ok} ok, {bad} failed)"

    def to_text(self, max_entries: int = 20) -> str:
        """Render as a compact log."""
        lines = [self.summary()]
        for e in self._entries[-max_entries:]:
            mark = "✓" if e.ok else "✗"
            ln = f"  #{e.sequence} {mark} {e.op_name}: {e.effect}"
            if e.error: ln += f"  ERR: {e.error}"
            lines.append(ln)
        return "\n".join(lines)


def _summarize_inputs(inputs: dict) -> dict:
    """Drop large geometry objects from input dict for printable logging."""
    out = {}
    for k, v in (inputs or {}).items():
        if hasattr(v, "wrapped") or hasattr(v, "edges"):
            out[k] = f"<{type(v).__name__}>"
        elif isinstance(v, (list, tuple)) and len(v) > 5:
            out[k] = f"<{type(v).__name__} of {len(v)}>"
        elif isinstance(v, dict) and len(v) > 5:
            out[k] = f"<dict of {len(v)}>"
        else:
            try:
                out[k] = v if len(repr(v)) <= 100 else f"<{type(v).__name__}>"
            except Exception:
                out[k] = f"<{type(v).__name__}>"
    return out
