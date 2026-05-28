"""orchestrator.py — translate user intent into Builder operations.

The orchestrator sits between the chat layer and the Builder. Its job:

1. Parse natural-language requests into a STRUCTURED PLAN (a list of
   operations + parameters).
2. Resolve named standards (NEMA17, M3, 2020 extrusion) into actual
   numeric dimensions via reference.py.
3. Inject the resolved dimensions into the prompt that goes to the
   Builder's text-to-Part operations, so the LLM has accurate numbers
   to work with rather than hallucinating.
4. Execute the plan against a Builder instance.

The orchestrator does NOT generate build123d code itself. It calls the
Builder, which calls shape_generator, which calls Codex.

The orchestrator is also where "configuration files" (YAML specs) get
consulted. A user request that mentions "NEMA17" gets the full NEMA17
spec dict appended to the LLM prompt before codegen runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Any

from .builder import Builder
from .reference import resolve_keywords, summarize_for_prompt, lookup


# ---------------------------------------------------------------------------
# Plan structure
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    operation: str           # builder method name
    args: dict = field(default_factory=dict)
    rationale: str = ""      # one-line explanation for the chat


@dataclass
class Plan:
    steps: List[PlanStep] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    resolved_specs: List[tuple] = field(default_factory=list)
    # ^ list of (category, name, spec_dict) from resolve_keywords

    def summary(self) -> str:
        lines = ["Plan:"]
        for i, s in enumerate(self.steps, 1):
            lines.append(f"  {i}. {s.operation}({_fmt_args(s.args)})")
            if s.rationale:
                lines.append(f"      → {s.rationale}")
        if self.resolved_specs:
            lines.append("Resolved standards:")
            for cat, name, _ in self.resolved_specs:
                lines.append(f"  - {cat}/{name}")
        if self.notes:
            lines.append("Notes:")
            for n in self.notes:
                lines.append(f"  - {n}")
        return "\n".join(lines)


def _fmt_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 50:
            parts.append(f"{k}={v[:47]!r}...")
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Spec packaging — turn resolved YAML data into prompt text
# ---------------------------------------------------------------------------

def _format_spec_for_prompt(category: str, name: str, spec: dict) -> str:
    """Convert a YAML spec dict into compact text for an LLM prompt."""
    out = [f"# {category.upper()} STANDARD: {name}"]
    if spec.get("description"):
        out.append(f"# {spec['description']}")
    # Include all numeric/scalar fields. Skip nested dicts (e.g. fastener
    # tables) and metadata we don't need.
    skip_keys = {"name", "category", "description", "_designation", "_table"}
    for k, v in spec.items():
        if k in skip_keys:
            continue
        if isinstance(v, (int, float)):
            out.append(f"{k}: {v}")
        elif isinstance(v, str):
            out.append(f"{k}: {v}")
        elif isinstance(v, list):
            if all(isinstance(x, (int, float)) for x in v):
                out.append(f"{k}: {v}")
        elif isinstance(v, dict):
            # one level of nesting (e.g. mounting_holes)
            out.append(f"{k}:")
            for kk, vv in v.items():
                out.append(f"  {kk}: {vv}")
    return "\n".join(out)


def _bundle_resolved_specs(resolved: list) -> str:
    if not resolved:
        return ""
    blocks = []
    for cat, name, spec in resolved:
        blocks.append(_format_spec_for_prompt(cat, name, spec))
    return ("\n\nRESOLVED ENGINEERING STANDARDS (use these dimensions):\n"
            + "\n\n".join(blocks)
            + "\n\nUse these EXACT numbers in your design. Do not "
              "approximate or guess; the values above come from "
              "engineering reference data.")


# ---------------------------------------------------------------------------
# Plan generation — for now, simple keyword-driven planner
# ---------------------------------------------------------------------------

def plan_from_request(request: str,
                       has_existing_part: bool = False) -> Plan:
    """Turn a free-form user request into an executable Plan.

    Default policy:
    - If no current part: emit a single `start_from_text` step using the
      request as description, with resolved-spec context appended.
    - If a part exists and the request reads like a modification
      ("add a hole", "fillet the edges", etc.): emit the matching
      modification step.
    - Otherwise treat it as starting fresh again.

    This is intentionally simple. The chat layer may override by
    asking the LLM to produce a plan in JSON, but a deterministic
    fallback is essential.
    """
    plan = Plan()
    req_lower = request.lower()

    # Always run keyword resolution
    resolved = resolve_keywords(request)
    plan.resolved_specs = resolved

    spec_text = _bundle_resolved_specs(resolved)

    # Detect modification-style requests
    fillet_keywords = ("fillet", "round the edges", "round edges")
    chamfer_keywords = ("chamfer", "bevel")
    add_keywords = ("add a", "add an", "add some")
    sub_keywords = ("subtract", "cut a", "drill", "make a hole",
                     "add a hole", "remove")
    render_keywords = ("show me", "render", "preview")
    emit_keywords = ("export", "save", "write")
    validate_keywords = ("check", "validate", "does it fit")

    if has_existing_part and any(k in req_lower for k in fillet_keywords):
        # try to extract a radius
        import re
        m = re.search(r"(\d+(?:\.\d+)?)\s*mm", req_lower)
        radius = float(m.group(1)) if m else 1.0
        # try to extract a selector
        sel = "all"
        for s in ("vertical", "top", "bottom", "horizontal"):
            if s in req_lower:
                sel = s
                break
        plan.steps.append(PlanStep(
            "fillet_edges",
            {"selector": sel, "radius": radius},
            f"fillet {sel} edges with r={radius}mm"))
        return plan

    if has_existing_part and any(k in req_lower for k in chamfer_keywords):
        import re
        m = re.search(r"(\d+(?:\.\d+)?)\s*mm", req_lower)
        length = float(m.group(1)) if m else 1.0
        sel = "all"
        for s in ("vertical", "top", "bottom"):
            if s in req_lower:
                sel = s
                break
        plan.steps.append(PlanStep(
            "chamfer_edges",
            {"selector": sel, "length": length},
            f"chamfer {sel} edges with len={length}mm"))
        return plan

    if has_existing_part and any(k in req_lower for k in sub_keywords):
        plan.steps.append(PlanStep(
            "subtract_feature",
            {"description": (request + spec_text).strip()},
            "subtract feature from existing part"))
        return plan

    if has_existing_part and any(k in req_lower for k in add_keywords):
        plan.steps.append(PlanStep(
            "add_feature",
            {"description": (request + spec_text).strip()},
            "add feature to existing part"))
        return plan

    if has_existing_part and any(k in req_lower for k in render_keywords):
        plan.steps.append(PlanStep(
            "render", {"out_path": "preview.png"},
            "render preview of current part"))
        return plan

    if has_existing_part and any(k in req_lower for k in emit_keywords):
        plan.steps.append(PlanStep(
            "emit", {"out_path": "design.py"},
            "emit standalone build123d script"))
        return plan

    if has_existing_part and any(k in req_lower for k in validate_keywords):
        plan.steps.append(PlanStep(
            "validate", {},
            "run validator on current part"))
        return plan

    # Default: start fresh from text
    full_description = request + spec_text
    plan.steps.append(PlanStep(
        "start_from_text",
        {"description": full_description},
        "generate a new part from the request"))
    return plan


# ---------------------------------------------------------------------------
# Plan execution
# ---------------------------------------------------------------------------

class Orchestrator:
    """Owns a Builder and executes Plans against it."""

    def __init__(self, builder: Optional[Builder] = None,
                 verbose: bool = True):
        self.builder = builder or Builder(verbose=verbose)
        self.verbose = verbose
        self.last_plan: Optional[Plan] = None

    def execute(self, plan: Plan) -> List[Any]:
        """Run all steps of a plan. Returns a list of step-result objects."""
        self.last_plan = plan
        results = []
        for step in plan.steps:
            result = self._run_step(step)
            results.append(result)
            # Stop early if a step raised
            if isinstance(result, dict) and result.get("error"):
                break
        return results

    def _run_step(self, step: PlanStep):
        method = getattr(self.builder, step.operation, None)
        if method is None or not callable(method):
            return {"error": f"Builder has no operation '{step.operation}'"}
        try:
            ret = method(**step.args)
            return {"ok": True, "result": ret, "step": step}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "step": step}

    def handle(self, request: str) -> dict:
        """One-shot: parse request, plan, execute, summarize.

        Returns a dict suitable for chat presentation.
        """
        has_part = self.builder.part is not None
        plan = plan_from_request(request, has_existing_part=has_part)
        if self.verbose:
            print(plan.summary(), flush=True)
        results = self.execute(plan)
        return {
            "request": request,
            "plan": plan,
            "results": results,
            "current": self.builder.summary(),
        }

    def handle_with_reasoning(self, request: str,
                                max_revisions: int = 2) -> dict:
        """Two-phase: run reasoning pipeline first, then feed approved
        spec into the builder.

        The reasoning phase researches the problem, extracts constraints,
        picks a validated topology, decomposes it into a concrete spec,
        and adversarially critiques it before any code is generated.

        If the critic approves, the spec is rendered as a structured
        prompt and passed to the existing handle() flow. If the critic
        rejects after max_revisions, returns a result dict with the
        critic's issues so the user can adjust the request.
        """
        from .reasoning.reasoning_session import ReasoningSession

        session = ReasoningSession(max_revisions=max_revisions,
                                     verbose=self.verbose)
        reasoning = session.run(request)

        if self.verbose:
            print("\n" + reasoning.to_summary(), flush=True)

        if not reasoning.final_approved:
            return {
                "request": request,
                "reasoning": reasoning,
                "results": [],
                "current": self.builder.summary(),
                "approved": False,
                "reason": ("reasoning pipeline did not produce an approved "
                           "spec; see reasoning.critic.issues for details"),
            }

        # Build an enriched request that gives the codegen full context
        enriched = _build_enriched_request(request, reasoning)
        has_part = self.builder.part is not None
        plan = plan_from_request(enriched, has_existing_part=has_part)
        if self.verbose:
            print(plan.summary(), flush=True)
        results = self.execute(plan)
        return {
            "request": request,
            "enriched_request": enriched,
            "reasoning": reasoning,
            "plan": plan,
            "results": results,
            "current": self.builder.summary(),
            "approved": True,
        }


def _build_enriched_request(original_request: str, reasoning) -> str:
    """Embed the reasoning result into a request the existing planner
    and code generator can use as additional context."""
    parts = [f"Design request: {original_request}", ""]
    if reasoning.topology and reasoning.topology.is_valid():
        parts.append(reasoning.topology.to_prompt_text())
        parts.append("")
    if reasoning.constraints and reasoning.constraints.has_constraints():
        parts.append(reasoning.constraints.to_prompt_text())
        parts.append("")
    if reasoning.spec and reasoning.spec.is_valid():
        parts.append(reasoning.spec.to_prompt_text())
    return "\n".join(parts)
