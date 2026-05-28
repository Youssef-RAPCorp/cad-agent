# cad_agent3 — Phase 1: Reasoning Layer

This update adds a research-and-reason-before-coding layer on top of the
existing CAD agent. When called via `Orchestrator.handle_with_reasoning(request)`
instead of `handle(request)`, every design request now runs through:

1. **Research** — looks for prior art (web + knowledge base)
2. **Constraint analysis** — extracts hard constraints, identifies tensions
3. **Topology selection** — commits to a validated pattern (no compromise)
4. **Decomposition** — produces a concrete geometric spec
5. **Spec critique** — adversarial review against the constraints
6. **(loop)** — if rejected, revise up to N times before falling through

Only when the critic *approves* the spec does it reach the existing
codegen stack. Approved specs include explicit dimensions, citations, and
constraint-tracking that make codegen far less likely to drift.

## What's new

```
cad_agent3/
├── reasoning/
│   ├── researcher.py             # Unified research API
│   ├── research_backends/
│   │   ├── types.py              # ResearchHit, ResearchResult
│   │   ├── knowledge_base.py     # YAML pattern lookup (always available)
│   │   ├── anthropic_websearch.py # Real web search via Claude tool use
│   │   ├── gemini_websearch.py   # Real web search via Gemini grounding
│   │   └── hybrid.py             # web first, KB fallback
│   ├── constraint_analyst.py     # Problem -> constraints + tensions
│   ├── topology_chooser.py       # Constraints + research -> chosen pattern
│   ├── decomposer.py             # Pattern -> structured GeometricSpec
│   ├── spec_critic.py            # Adversarial review pre-codegen
│   └── reasoning_session.py      # Coordinates all of the above
└── knowledge/
    ├── kb_loader.py              # YAML indexer with search
    └── patterns/                 # 27 YAML files, 82 patterns total
        ├── foldable_cardboard.yaml
        ├── printer_kinematics.yaml
        ├── motor_mounting.yaml
        ├── extrusion_joints.yaml
        ├── enclosure_topologies.yaml
        ├── fastener_patterns.yaml
        ├── lens_optics.yaml
        ├── hinge_mechanisms.yaml
        ├── bearing_mounts.yaml
        ├── linear_motion.yaml
        ├── belt_drives.yaml
        ├── leadscrew_drives.yaml
        ├── end_stops_homing.yaml
        ├── heated_bed.yaml
        ├── hot_ends.yaml
        ├── extruders.yaml
        ├── gantries.yaml
        ├── pcb_mounting.yaml
        ├── psu_mounting.yaml
        ├── wire_management.yaml
        ├── fan_mounts.yaml
        ├── vr_phone_holder.yaml
        ├── phone_clamping.yaml
        ├── face_padding.yaml
        ├── phone_screen_optics.yaml
        ├── light_sealing.yaml
        └── strap_attachment.yaml
```

## Backend selection

Web research can run on either runtime, picked automatically:

- **`LLM_BACKEND=anthropic` + `ANTHROPIC_API_KEY`** — uses Claude with the
  `web_search_20250305` tool. Useful in development (e.g. this container).
- **`GEMINI_API_KEY` (or `GOOGLE_API_KEY`)** — uses Gemini with the
  `GoogleSearch` grounding tool. The production default.
- **Neither** — falls back to KB-only.

In all modes the KB serves as a fallback when the web call fails or
returns empty.

## Usage

```python
from cad_agent3 import Orchestrator

orch = Orchestrator(verbose=True)
result = orch.handle_with_reasoning(
    "design a foldable cardboard VR headset that fits in a pocket"
)
if result["approved"]:
    # spec was approved; codegen has run
    print(result["current"])
else:
    # critic rejected — see result["reasoning"].critic.issues
    for issue in result["reasoning"].critic.issues:
        print(f"  [{issue.severity}] {issue.constraint}: {issue.description}")
```

You can also run the reasoning step in isolation:

```python
from cad_agent3 import ReasoningSession

session = ReasoningSession(verbose=True, max_revisions=2)
reasoning = session.run("design a CoreXY 3D printer with 300mm cube envelope")
print(reasoning.to_summary())
print(reasoning.spec.to_prompt_text())
```

## What's coming in Phase 2

- `operations/` — fine-grained CAD feature operations (~30-50 modules:
  hole, pocket, boss, rib, fillet, chamfer, etc., each with explicit
  preconditions and undo data)
- `session/` — persistent design sessions with rollback and dependency
  tracking
- Builder refactor to be a thin orchestrator over operations/session
- Spec-revision feedback loop (currently the decomposer doesn't see the
  critic's issues; it should)

## Honest limits

The reasoning layer makes the agent commit to validated patterns rather
than inventing topologies. It does not make the agent omniscient. It
helps most on hard problems with multiple competing constraints (the
VR headset failure that motivated this work). For simple parts (a
bracket, a basic enclosure), it adds latency without much gain.
