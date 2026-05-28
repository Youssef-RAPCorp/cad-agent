"""cad_agent3 -- solid-aware CAD reconstruction + LLM-driven design."""

# Reconstruction stack (existing)
from .scanner import scan_source, ScanResult, CongruenceGroup, GridPattern
from .emitter import emit_recipe, EmissionResult
from .verifier import compute_intersection, safe_volume
from .engine import FitEngine, Tier, FitDiag, default_tiers

# LLM codegen primitives (existing)
from .shape_generator import generate_shape, GenerationResult
from .image_to_shape import image_to_shape, ImageToShapeResult, VisionExtraction
from . import build123d_reference

# Design stack (new)
from .reference import (lookup, list_available, summarize_for_prompt,
                          resolve_keywords)
from .builder import Builder, HistoryEntry
from .builder_v2 import BuilderV2
from .orchestrator import Orchestrator, Plan, PlanStep, plan_from_request
from .validator import validate, ValidationReport, ValidationIssue
from .render_preview import render_preview, PreviewResult

# Reasoning layer (Phase 1)
from .reasoning.researcher import Researcher
from .reasoning.reasoning_session import ReasoningSession, ReasoningResult
from .reasoning.constraint_analyst import (
    analyze as analyze_constraints, ConstraintAnalysis, Constraint, Tension)
from .reasoning.topology_chooser import choose as choose_topology, TopologyChoice
from .reasoning.decomposer import (
    decompose as decompose_spec, GeometricSpec, GeometricFeature)
from .reasoning.spec_critic import review as critic_review, CriticReport, SpecIssue
from .reasoning.research_backends.types import ResearchHit, ResearchResult
from .knowledge.kb_loader import (
    list_all_patterns, get_pattern, search as kb_search, summarize_kb, Pattern)

# Operations + session (Phase 2)
from . import operations
from .operations import catalog as op_catalog
from .operations.operation_base import (
    Operation, OperationDecl, OperationResult, OperationCheck)
from .session import (
    DesignSession, SessionStatus,
    ArtifactRegistry, Artifact,
    OperationLog, LogEntry,
    DependencyGraph,
    CheckpointManager, Checkpoint,
)

__all__ = [
    # Reconstruction
    "scan_source", "ScanResult", "CongruenceGroup", "GridPattern",
    "emit_recipe", "EmissionResult",
    "compute_intersection", "safe_volume",
    "FitEngine", "Tier", "FitDiag", "default_tiers",
    # LLM codegen
    "generate_shape", "GenerationResult",
    "image_to_shape", "ImageToShapeResult", "VisionExtraction",
    "build123d_reference",
    # Design
    "lookup", "list_available", "summarize_for_prompt", "resolve_keywords",
    "Builder", "HistoryEntry", "BuilderV2",
    "Orchestrator", "Plan", "PlanStep", "plan_from_request",
    "validate", "ValidationReport", "ValidationIssue",
    "render_preview", "PreviewResult",
    # Reasoning layer
    "Researcher", "ReasoningSession", "ReasoningResult",
    "analyze_constraints", "ConstraintAnalysis", "Constraint", "Tension",
    "choose_topology", "TopologyChoice",
    "decompose_spec", "GeometricSpec", "GeometricFeature",
    "critic_review", "CriticReport", "SpecIssue",
    "ResearchHit", "ResearchResult",
    "list_all_patterns", "get_pattern", "kb_search", "summarize_kb", "Pattern",
    # Operations and session (Phase 2)
    "operations", "op_catalog",
    "Operation", "OperationDecl", "OperationResult", "OperationCheck",
    "DesignSession", "SessionStatus",
    "ArtifactRegistry", "Artifact",
    "OperationLog", "LogEntry",
    "DependencyGraph",
    "CheckpointManager", "Checkpoint",
]
