"""reasoning_session.py — coordinate the full reasoning pipeline.

A ReasoningSession runs the steps in order and holds the state:
  1. research(problem)        -> ResearchResult
  2. analyze_constraints      -> ConstraintAnalysis
  3. choose_topology          -> TopologyChoice
  4. decompose                -> GeometricSpec
  5. critique                 -> CriticReport
  6. (loop: revise spec if rejected, up to N times)

The output is a final approved GeometricSpec, ready for codegen.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import time

from .researcher import Researcher
from .constraint_analyst import analyze as analyze_constraints, ConstraintAnalysis
from .topology_chooser import choose as choose_topology, TopologyChoice
from .decomposer import decompose as decompose_spec, GeometricSpec
from .spec_critic import review as critic_review, CriticReport
from .research_backends.types import ResearchResult


@dataclass
class ReasoningResult:
    problem: str
    research: Optional[ResearchResult] = None
    constraints: Optional[ConstraintAnalysis] = None
    topology: Optional[TopologyChoice] = None
    spec: Optional[GeometricSpec] = None
    critic: Optional[CriticReport] = None
    revision_count: int = 0
    final_approved: bool = False
    total_duration_s: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_summary(self) -> str:
        lines = [f"# Reasoning result for: {self.problem}",
                 f"Total time: {self.total_duration_s:.1f}s"]
        if self.research:
            lines.append(f"Research: {len(self.research.hits)} hits "
                          f"via {self.research.backend_used}")
        if self.constraints:
            lines.append(f"Constraints: {len(self.constraints.constraints)} extracted, "
                          f"{len(self.constraints.tensions)} tensions")
        if self.topology:
            lines.append(f"Topology: {self.topology.chosen_pattern_id} "
                          f"(confidence {self.topology.confidence:.2f})")
        if self.spec:
            lines.append(f"Spec: {len(self.spec.features)} features")
        if self.critic:
            verdict = "APPROVED" if self.critic.approved else "REJECTED"
            lines.append(f"Critic: {verdict} ({len(self.critic.issues)} issues)")
        if self.revision_count > 0:
            lines.append(f"Revisions: {self.revision_count}")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for e in self.errors[:3]:
                lines.append(f"  - {e}")
        return "\n".join(lines)


@dataclass
class ReasoningSession:
    """A reusable reasoning pipeline."""
    researcher: Optional[Researcher] = None
    max_revisions: int = 2
    verbose: bool = False

    def __post_init__(self):
        if self.researcher is None:
            self.researcher = Researcher(backend="hybrid",
                                           verbose=self.verbose)

    def run(self, problem: str) -> ReasoningResult:
        """Run the complete reasoning pipeline for one problem."""
        t0 = time.time()
        result = ReasoningResult(problem=problem)

        if self.verbose:
            print(f"\n=== Reasoning session ===\n  problem: {problem}",
                   flush=True)

        # Step 1: research
        result.research = self.researcher.research(problem)
        if result.research.error:
            result.errors.append(f"research: {result.research.error}")

        # Step 2: extract constraints
        result.constraints = analyze_constraints(
            problem, result.research, verbose=self.verbose)
        if result.constraints.error:
            result.errors.append(f"constraints: {result.constraints.error}")

        # Step 3: pick topology
        result.topology = choose_topology(
            problem, result.constraints, result.research,
            verbose=self.verbose)
        if result.topology.error:
            result.errors.append(f"topology: {result.topology.error}")

        # Step 4-5: decompose + critique loop
        previous_critic_issues = None
        for revision in range(self.max_revisions + 1):
            result.revision_count = revision

            # Decompose (with critic feedback if this is a revision)
            spec = decompose_spec(
                problem, result.topology, result.constraints,
                result.research, verbose=self.verbose,
                previous_issues=previous_critic_issues)
            result.spec = spec
            if spec.error:
                result.errors.append(f"decompose#{revision}: {spec.error}")
                break

            # Critique
            critic = critic_review(
                problem, spec, result.constraints, verbose=self.verbose)
            result.critic = critic
            if critic.error:
                result.errors.append(f"critic#{revision}: {critic.error}")
                break

            if critic.approved:
                result.final_approved = True
                break

            # Feed the issues into the next revision so the decomposer
            # knows what to fix.
            previous_critic_issues = critic.issues
            if self.verbose:
                print(f"[reasoning_session] critic rejected; "
                       f"feeding {len(critic.issues)} issues to revision "
                       f"#{revision + 1}", flush=True)
            if revision >= self.max_revisions:
                break

        result.total_duration_s = time.time() - t0
        if self.verbose:
            print(f"\n=== session done ({result.total_duration_s:.1f}s, "
                   f"approved={result.final_approved}) ===\n", flush=True)
        return result
