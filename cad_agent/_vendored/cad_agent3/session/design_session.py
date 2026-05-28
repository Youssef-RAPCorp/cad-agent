"""design_session.py — top-level session that coordinates everything.

A DesignSession holds:
  - the current part (the active build state)
  - the artifact registry (everything created in this session)
  - the operation log (every applied op)
  - the dependency graph
  - the checkpoint manager
  - a reference to the operation catalog

The session is what user code interacts with for fine-grained operations.
The Builder (existing) wraps it for higher-level natural-language flows.

API:
    sess = DesignSession()
    sess.start_with(part)              # initialize with an existing part
    sess.apply("hole", diameter_mm=5, position=(0,0,10))  # run a registered op
    sess.checkpoint("after_holes")
    sess.apply("fillet", radius_mm=1)
    sess.rollback_to("after_holes")     # undo back to checkpoint
    sess.summary()
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..operations import catalog
from ..operations.operation_base import OperationResult
from .artifact_registry import ArtifactRegistry, Artifact
from .operation_log import OperationLog, LogEntry
from .dependency_graph import DependencyGraph
from .checkpoint import CheckpointManager, Checkpoint


@dataclass
class SessionStatus:
    has_part: bool
    op_count: int
    artifact_count: int
    checkpoint_count: int
    last_op: Optional[str] = None
    last_error: Optional[str] = None

    def to_text(self) -> str:
        lines = [f"Session status:"]
        lines.append(f"  current part: {'yes' if self.has_part else 'none'}")
        lines.append(f"  operations applied: {self.op_count}")
        lines.append(f"  artifacts: {self.artifact_count}")
        lines.append(f"  checkpoints: {self.checkpoint_count}")
        if self.last_op:
            lines.append(f"  last op: {self.last_op}")
        if self.last_error:
            lines.append(f"  last error: {self.last_error}")
        return "\n".join(lines)


class DesignSession:
    """Top-level coordinator for a CAD design session."""

    def __init__(self, verbose: bool = False):
        self.part: Any = None
        self.registry = ArtifactRegistry()
        self.log = OperationLog()
        self.graph = DependencyGraph()
        self.checkpoints = CheckpointManager()
        self.verbose = verbose
        self._last_artifact_id: Optional[int] = None

    # ---- starting state ----

    def start_with(self, part: Any, name: str = "initial_part") -> Artifact:
        """Initialize the session with an existing part."""
        self.part = part
        art = self.registry.add(kind="part", data=part, name=name)
        self._last_artifact_id = art.id
        if self.verbose:
            print(f"[session] start_with: registered {art!r}", flush=True)
        return art

    # ---- applying operations ----

    def apply(self, op_name: str, **inputs) -> OperationResult:
        """Look up and run a registered operation. Updates session state."""
        op_class = catalog.get(op_name)
        if op_class is None:
            err = f"unknown operation '{op_name}'"
            if self.verbose: print(f"[session] {err}", flush=True)
            return OperationResult(op_name=op_name, ok=False, error=err)

        # Validate
        check = op_class.validate(self.part, inputs)
        if not check.ok:
            err = f"validation failed: {'; '.join(check.issues)}"
            if self.verbose: print(f"[session] {err}", flush=True)
            return OperationResult(op_name=op_name, ok=False, error=err)

        # Apply
        if self.verbose:
            print(f"[session] applying {op_name}({list(inputs.keys())})...",
                   flush=True)
        result = op_class.apply(self.part, inputs)

        if result.ok and result.new_part is not None:
            self.part = result.new_part
            # Register as artifact
            parent_ids = ([self._last_artifact_id]
                          if self._last_artifact_id else [])
            art = self.registry.add(
                kind="part", data=result.new_part,
                created_by_op=op_name, parents=parent_ids,
                **(result.metrics or {}))
            for p in parent_ids:
                self.graph.add_edge(p, art.id)
            self._last_artifact_id = art.id
            entry = self.log.append(op_name, inputs, result, artifact_id=art.id)
        else:
            entry = self.log.append(op_name, inputs, result)

        if self.verbose:
            mark = "✓" if result.ok else "✗"
            print(f"[session] {mark} {result.effect or result.error}",
                   flush=True)
        return result

    # ---- checkpoint / rollback ----

    def checkpoint(self, label: str) -> Checkpoint:
        """Save a named checkpoint of the current state."""
        seq = self.log.latest().sequence if len(self.log) else 0
        cp = self.checkpoints.save(label, seq, self.part,
                                     artifact_count=len(self.registry))
        if self.verbose:
            print(f"[session] checkpoint '{label}' @ seq={seq}", flush=True)
        return cp

    def rollback_to(self, label: str) -> bool:
        """Restore the session to a named checkpoint.

        Truncates the log and resets the current part to the saved value.
        Returns True if the checkpoint was found.
        """
        cp = self.checkpoints.get(label)
        if cp is None:
            if self.verbose:
                print(f"[session] no checkpoint '{label}'", flush=True)
            return False
        self.part = cp.saved_part
        self.log.truncate_to(cp.sequence)
        if self.verbose:
            print(f"[session] rolled back to '{label}' (seq={cp.sequence})",
                   flush=True)
        return True

    def undo_last(self) -> bool:
        """Undo the most recent operation by reverting to its previous part."""
        entry = self.log.latest()
        if entry is None or not entry.ok:
            return False
        prev = entry.undo_data.get("previous_part")
        if prev is None:
            return False
        self.part = prev
        self.log.pop()
        if self.verbose:
            print(f"[session] undid #{entry.sequence} {entry.op_name}",
                   flush=True)
        return True

    # ---- inspection ----

    def status(self) -> SessionStatus:
        last = self.log.latest()
        return SessionStatus(
            has_part=self.part is not None,
            op_count=len(self.log),
            artifact_count=len(self.registry),
            checkpoint_count=len(self.checkpoints),
            last_op=last.op_name if last else None,
            last_error=last.error if last and not last.ok else None,
        )

    def summary(self) -> str:
        parts = [self.status().to_text(),
                 self.registry.summary(),
                 self.log.summary(),
                 self.graph.summary()]
        cps = self.checkpoints.all()
        if cps:
            parts.append("checkpoints: " +
                          ", ".join(f"{c.label}@{c.sequence}" for c in cps))
        return "\n".join(parts)
