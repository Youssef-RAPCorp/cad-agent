"""operation_base.py — base class for every CAD operation.

Every operation in cad_agent3.operations.* inherits Operation. The
contract is:

  - declare() returns metadata (name, inputs, outputs)
  - validate(inputs) checks preconditions; returns OperationCheck
  - apply(part, inputs) performs the operation; returns OperationResult
                       with new_part, undo_data, and effect description
  - undo(part, undo_data) reverses the operation

Operations are STATELESS — they don't hold a part reference. The
session passes the current part in. This makes operations composable,
testable in isolation, and easy to log/replay.

OperationResult.new_part is the part AFTER applying. If the operation
fails, OperationResult.error is set and new_part is None.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OperationDecl:
    """Describes an operation. Used by the catalog and orchestrator."""
    name: str                          # snake_case unique id
    category: str                      # "feature" | "selector" | etc.
    summary: str                       # one-line human description
    required_inputs: List[str] = field(default_factory=list)
    optional_inputs: List[str] = field(default_factory=list)
    output_type: str = "Part"          # what apply returns


@dataclass
class OperationCheck:
    """Result of validate(). If ok==False, apply will refuse."""
    ok: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class OperationResult:
    """Result of apply(). new_part is None on failure."""
    op_name: str
    ok: bool
    new_part: Any = None              # the modified part (build123d Part/Compound)
    undo_data: Dict[str, Any] = field(default_factory=dict)
    effect: str = ""                  # human-readable change description
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class Operation(ABC):
    """Base class for all CAD operations."""

    @classmethod
    @abstractmethod
    def declare(cls) -> OperationDecl:
        """Return operation metadata."""
        raise NotImplementedError

    @classmethod
    def validate(cls, part: Any, inputs: Dict[str, Any]) -> OperationCheck:
        """Check preconditions. Default: every required_input is present."""
        decl = cls.declare()
        issues = []
        for req in decl.required_inputs:
            if req not in inputs:
                issues.append(f"missing required input: {req}")
        return OperationCheck(ok=not issues, issues=issues)

    @classmethod
    @abstractmethod
    def apply(cls, part: Any, inputs: Dict[str, Any]) -> OperationResult:
        """Execute. Returns OperationResult with new_part and undo_data."""
        raise NotImplementedError

    @classmethod
    def undo(cls, part: Any, undo_data: Dict[str, Any]) -> Any:
        """Reverse the operation. Default: return undo_data['previous_part']."""
        return undo_data.get("previous_part")


# ---------------------------------------------------------------------------
# Helpers operations use a lot
# ---------------------------------------------------------------------------

def _bbox_dict(part: Any) -> Dict[str, float]:
    """Bounding box as a dict, or empty dict if part has no .bounding_box()."""
    try:
        bb = part.bounding_box()
        return {
            "min_x": float(bb.min.X), "max_x": float(bb.max.X),
            "min_y": float(bb.min.Y), "max_y": float(bb.max.Y),
            "min_z": float(bb.min.Z), "max_z": float(bb.max.Z),
            "size_x": float(bb.max.X - bb.min.X),
            "size_y": float(bb.max.Y - bb.min.Y),
            "size_z": float(bb.max.Z - bb.min.Z),
        }
    except Exception:
        return {}


def _safe_volume(part: Any) -> float:
    """Volume in mm^3, or 0.0 if not measurable."""
    try:
        return float(part.volume)
    except Exception:
        return 0.0
