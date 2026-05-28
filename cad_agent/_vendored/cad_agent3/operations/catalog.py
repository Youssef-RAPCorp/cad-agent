"""catalog.py — operation registry.

Operations register themselves at import time. The catalog provides:
  - register(op_class)
  - get(name) -> op_class
  - list_by_category(category) -> list of op_classes
  - all_decls() -> list of OperationDecl

Auto-discovery: importing cad_agent3.operations causes all sub-packages
(features, selectors, analysis, repair, transforms, booleans) to import
their modules, each of which calls register() at module top level.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Type

from .operation_base import Operation, OperationDecl


_REGISTRY: Dict[str, Type[Operation]] = {}


def register(op_class: Type[Operation]) -> Type[Operation]:
    """Decorator (or callable) to register an operation."""
    if not issubclass(op_class, Operation):
        raise TypeError(f"{op_class!r} is not an Operation subclass")
    decl = op_class.declare()
    if decl.name in _REGISTRY:
        # Allow re-registration for hot-reload during development
        pass
    _REGISTRY[decl.name] = op_class
    return op_class


def get(name: str) -> Optional[Type[Operation]]:
    return _REGISTRY.get(name)


def all_names() -> List[str]:
    return sorted(_REGISTRY.keys())


def all_decls() -> List[OperationDecl]:
    return [op.declare() for op in _REGISTRY.values()]


def list_by_category(category: str) -> List[Type[Operation]]:
    return [op for op in _REGISTRY.values()
            if op.declare().category == category]


def summarize() -> str:
    """Return a categorized listing of all registered operations."""
    by_cat: Dict[str, List[OperationDecl]] = {}
    for op in _REGISTRY.values():
        d = op.declare()
        by_cat.setdefault(d.category, []).append(d)
    lines = [f"Operation catalog: {len(_REGISTRY)} operations across "
             f"{len(by_cat)} categories"]
    for cat in sorted(by_cat):
        decls = by_cat[cat]
        lines.append(f"  {cat} ({len(decls)}):")
        for d in sorted(decls, key=lambda x: x.name):
            lines.append(f"    - {d.name}: {d.summary}")
    return "\n".join(lines)
