"""artifact_registry.py — track every artifact created in a session.

Each artifact has:
  - id: monotonically increasing integer
  - name: optional human-readable label
  - kind: "part" | "sketch" | "spec" | "research" | "selection" | "render"
  - state: "draft" | "validated" | "emitted" | "archived"
  - data: the actual object (build123d Part, Sketch, dict, etc.)
  - created_by_op: which operation produced it (or None for imports)
  - parents: list of artifact ids this depends on
  - metadata: free-form dict
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Artifact:
    id: int
    kind: str
    state: str = "draft"
    name: str = ""
    data: Any = None
    created_by_op: Optional[str] = None
    parents: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        nm = f" '{self.name}'" if self.name else ""
        return (f"<Artifact #{self.id}{nm} kind={self.kind} state={self.state}>")


class ArtifactRegistry:
    """In-memory registry of artifacts in a single session."""

    def __init__(self):
        self._next_id = 1
        self._by_id: Dict[int, Artifact] = {}
        self._by_name: Dict[str, int] = {}

    def add(self, kind: str, data: Any, name: str = "",
             created_by_op: Optional[str] = None,
             parents: Optional[List[int]] = None,
             **metadata) -> Artifact:
        """Create and register a new artifact. Returns the artifact."""
        aid = self._next_id; self._next_id += 1
        art = Artifact(id=aid, kind=kind, name=name, data=data,
                        created_by_op=created_by_op,
                        parents=list(parents or []), metadata=metadata)
        self._by_id[aid] = art
        if name:
            self._by_name[name] = aid
        return art

    def get(self, key) -> Optional[Artifact]:
        if isinstance(key, int):
            return self._by_id.get(key)
        if isinstance(key, str):
            aid = self._by_name.get(key)
            return self._by_id.get(aid) if aid is not None else None
        return None

    def by_kind(self, kind: str) -> List[Artifact]:
        return [a for a in self._by_id.values() if a.kind == kind]

    def latest(self, kind: Optional[str] = None) -> Optional[Artifact]:
        """Return the most recently added artifact (optionally filtered by kind)."""
        for aid in sorted(self._by_id, reverse=True):
            a = self._by_id[aid]
            if kind is None or a.kind == kind:
                return a
        return None

    def all(self) -> List[Artifact]:
        return [self._by_id[i] for i in sorted(self._by_id)]

    def update_state(self, key, new_state: str) -> bool:
        a = self.get(key)
        if a is None: return False
        a.state = new_state; return True

    def __len__(self): return len(self._by_id)

    def summary(self) -> str:
        if not self._by_id:
            return "ArtifactRegistry: empty"
        by_kind: Dict[str, int] = {}
        for a in self._by_id.values():
            by_kind[a.kind] = by_kind.get(a.kind, 0) + 1
        bits = ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items()))
        return f"ArtifactRegistry: {len(self._by_id)} artifacts ({bits})"
