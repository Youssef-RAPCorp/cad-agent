# cad_agent3 — Phase 2: Operations + Session Layer

Phase 2 builds on the reasoning layer from Phase 1, adding:

- **53 fine-grained operations** across 6 categories, each in its own
  module with explicit pre/post conditions and undo data
- **Session management** with checkpoint, rollback, undo
- **Artifact registry** + **operation log** + **dependency graph**
- **BuilderV2** — thin convenience layer over the session/ops API
- **Critic-feedback revision loop** — when the spec critic rejects,
  the decomposer now sees the issues and tries to fix them

## What's where

```
cad_agent3/operations/
├── operation_base.py          # Operation, OperationResult, OperationCheck
├── catalog.py                 # @register decorator + lookup
├── features/                  # 19 feature ops
│   ├── hole, counterbore, countersink, pocket, slot,
│   ├── boss, rib, fillet, chamfer, shell,
│   ├── extrude, revolve, loft, sweep,
│   ├── thread, draft, polygon_hole, tab_and_slot, score_line
├── selectors/                 # 10 selector ops
│   ├── faces_by_normal, edges_by_axis, edges_by_length,
│   ├── faces_by_area, vertices_by_position,
│   ├── top_face, bottom_face, planar_faces,
│   ├── circular_holes, faces_at_z
├── analysis/                  # 9 analysis ops
│   ├── volume, mass, bbox, centroid, face_count,
│   ├── interference, surface_area, manifold_check, moment_of_inertia
├── repair/                    # 5 OCCT shape-healing wrappers
│   ├── shape_fix, sew_shells, fix_wireorder, simplify, remove_small_features
├── transforms/                # 6 transform ops
│   ├── translate, rotate, scale, mirror, pattern_linear, pattern_circular
└── booleans/                  # 4 boolean ops
    └── union, cut, intersect, split

cad_agent3/session/
├── design_session.py          # DesignSession — top-level coordinator
├── artifact_registry.py       # Tracks every artifact created
├── operation_log.py           # Append-only log with rollback
├── dependency_graph.py        # Tracks artifact dependencies
└── checkpoint.py              # Named save/restore points
```

## Operation contract

Every operation inherits `Operation` and provides:

- `declare()` returns `OperationDecl` (name, category, summary, required/optional inputs)
- `validate(part, inputs)` checks preconditions
- `apply(part, inputs)` performs the op, returns `OperationResult` with
  `new_part`, `undo_data`, `effect`, `metrics`
- `undo(part, undo_data)` reverses (default: returns `undo_data["previous_part"]`)

Operations register themselves at import time via the `@register` decorator,
so the catalog is automatically populated when the package is imported.

## Usage

```python
import cad_agent3 as c

# High-level builder API (recommended)
b = c.BuilderV2(verbose=True)
b.start_from_box(60, 40, 10)
b.checkpoint("blank")

# 4 corner mounting holes
for x, y in [(-25, -15), (25, -15), (-25, 15), (25, 15)]:
    b.drill_hole(diameter=3.4, position=(x, y, 5))

b.counterbore(thru_diameter=3.4, cb_diameter=6, cb_depth=3, position=(0, 0, 5))
b.cut_slot(length=20, width=4, position=(0, 0, 5))
b.fillet_edges(radius=1.5, axis="Z")

# Inspection
b.get_volume()
b.estimate_mass(material="PLA")
b.check_manifold()

# Export
b.export_step("part.step")

# Rollback
b.rollback("blank")  # back to the original blank box
```

```python
# Lower-level direct session use
sess = c.DesignSession()
sess.start_with(some_part)
result = sess.apply("hole", diameter_mm=3.4, position=(0, 0, 5))
# result is an OperationResult with .ok, .new_part, .effect, .metrics, .undo_data

# The whole catalog is discoverable
print(c.op_catalog.summarize())
```

## Phase 1 + Phase 2 together

When the reasoning layer (Phase 1) approves a spec, that spec can now
be implemented through the operations catalog (Phase 2) instead of as
free-form code generation. This lets the codegen layer call structured
operations whose behavior is documented and tested, instead of producing
ad-hoc build123d code that may or may not work.

A spec like:
```json
{"features": [
  {"name": "h1", "kind": "hole",
   "dimensions": {"diameter_mm": 3.4},
   "position": {"x_mm": 0, "y_mm": 0, "z_mm": 10}}
]}
```
can be applied directly:
```python
b.session.apply("hole", diameter_mm=3.4, position=(0, 0, 10))
```

This is the foundation for spec-driven codegen that doesn't drift from
the spec.

## Critic feedback revision loop

When the spec critic in Phase 1 rejects a spec, the issues are now fed
back into the decomposer for the next revision. Previously this was a
gap (the decomposer didn't see the critic's feedback).

```python
session = c.ReasoningSession(max_revisions=3, verbose=True)
result = session.run(problem)
# If revision 0's spec is rejected, revision 1's prompt now includes
# "CRITIC FEEDBACK from previous revision (FIX THESE): ..."
```

## Total scope after Phase 2

- 104 Python modules
- 37 YAML files (27 patterns + 10 configs)
- 53 registered operations
- 82 indexed knowledge patterns
- Web search via Anthropic OR Gemini, with KB fallback
- 5-stage reasoning pipeline (research → constraints → topology → spec → critique)
- Session management with checkpoints, rollback, undo, dependency tracking

## Honest limits

- Operations don't yet have a unified "preview before apply" mode
  (you can `validate()` separately, but rendering isn't built in).
- Some OCCT-shape-healing operations (`fix_wireorder`, `simplify`,
  `remove_small_features`) are wrapped but not exhaustively tested
  on weird input geometry — they may pass-through unchanged on shapes
  that don't need them.
- The `draft` operation is a placeholder; build123d 0.10 doesn't ship
  with native draft and a real implementation needs face-by-face
  offset which isn't done.
- The dependency graph tracks artifacts but isn't yet used for
  incremental rebuild (when you change parameter X, recompute only
  the descendants). That's a Phase 3 concern.
