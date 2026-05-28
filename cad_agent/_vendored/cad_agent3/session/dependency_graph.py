"""dependency_graph.py — track which artifacts depend on which.

When an artifact's parents change, all descendants need re-running.
Used by checkpoint/replay and by future incremental-rebuild logic.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class DependencyGraph:
    edges: Dict[int, Set[int]] = field(default_factory=dict)   # parent -> set of children

    def add_node(self, node_id: int):
        if node_id not in self.edges:
            self.edges[node_id] = set()

    def add_edge(self, parent: int, child: int):
        self.add_node(parent); self.add_node(child)
        self.edges[parent].add(child)

    def descendants(self, node_id: int) -> Set[int]:
        """All transitive descendants of node_id."""
        seen: Set[int] = set()
        stack = list(self.edges.get(node_id, set()))
        while stack:
            n = stack.pop()
            if n in seen: continue
            seen.add(n); stack.extend(self.edges.get(n, set()))
        return seen

    def root_ancestors(self, node_id: int) -> Set[int]:
        """Walk up reverse edges to find nodes that nothing depends on."""
        # Build reverse adjacency
        rev: Dict[int, Set[int]] = {}
        for p, kids in self.edges.items():
            for k in kids:
                rev.setdefault(k, set()).add(p)
        seen: Set[int] = set()
        stack = [node_id]
        while stack:
            n = stack.pop()
            if n in seen: continue
            seen.add(n); stack.extend(rev.get(n, set()))
        return {n for n in seen if not rev.get(n)}

    def __len__(self): return len(self.edges)

    def summary(self) -> str:
        n_nodes = len(self.edges)
        n_edges = sum(len(v) for v in self.edges.values())
        return f"DependencyGraph: {n_nodes} nodes, {n_edges} edges"
