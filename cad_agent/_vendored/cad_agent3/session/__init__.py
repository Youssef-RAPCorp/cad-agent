"""cad_agent3.session — coordinated session management.

Public API:
    DesignSession           -- top-level coordinator
    Artifact, ArtifactRegistry
    LogEntry, OperationLog
    DependencyGraph
    Checkpoint, CheckpointManager
"""
from .design_session import DesignSession, SessionStatus
from .artifact_registry import ArtifactRegistry, Artifact
from .operation_log import OperationLog, LogEntry
from .dependency_graph import DependencyGraph
from .checkpoint import CheckpointManager, Checkpoint

__all__ = [
    "DesignSession", "SessionStatus",
    "ArtifactRegistry", "Artifact",
    "OperationLog", "LogEntry",
    "DependencyGraph",
    "CheckpointManager", "Checkpoint",
]
