# cad-agent

LLM-powered CAD model generation from natural-language specs.

`cad-agent` turns a sentence like

> "A 6m × 6m SIP residential pod with a south-facing 914mm door, one east window, and a 2% sloped roof draining north."

into an executable [build123d](https://github.com/gumyr/build123d) Python script, runs it in a sandbox, and exports STEP + STL files.

Under the hood it uses an LLM (Gemini Flash by default; Claude available) to generate the build123d code, runs a critic-feedback loop to catch geometric tensions before code is emitted, and resolves named standards (NEMA17, M3, 2020 T-slot extrusion) from a bundled spec library so the model gets accurate numbers instead of hallucinated ones.

---

## Install

`cad-agent` is not published to PyPI yet — install it from the GitHub repo.

### From GitHub (recommended)

```bash
git clone git@github.com:Youssef-RAPCorp/cad-agent.git
cd cad-agent

# Create and activate a virtual environment
python3 -m venv cad_venv
source cad_venv/bin/activate

# Install in editable mode with the extras you need
pip install -e ".[gemini]"             # Gemini backend (default)
pip install -e ".[anthropic]"          # Claude backend
pip install -e ".[drawings]"           # 2D engineering drawings (no LLM needed)
pip install -e ".[gemini,drawings]"    # typical setup
pip install -e ".[all]"                # everything
```

Editable mode (`-e`) means `git pull` picks up updates without reinstalling.

Or install straight from GitHub without cloning:

```bash
pip install "cad-agent[gemini] @ git+https://github.com/Youssef-RAPCorp/cad-agent.git"
```

### Set your API key

Model generation calls an LLM, so it needs a key (the 2D drawing engine works without one):

```bash
export GEMINI_API_KEY=...          # default backend
# or
export ANTHROPIC_API_KEY=...
export CAD_AGENT_BACKEND=anthropic
```

### Run it

From the command line:

```bash
cad-agent "A 20mm cube with a 5mm through-hole" --name cube -o ./cad_output
```

This writes `cube.step`, `cube.stl`, and the generated `cube.py` script into `./cad_output`. Run `cad-agent --help` for all flags.

Or from Python — see the quick start below.

> **Note:** don't run files inside the `cad_agent/` package directly (`python cad_agent/agent.py` fails with a relative-import error). Use the `cad-agent` CLI, `python -m cad_agent`, or `import cad_agent` from your own script.

### Verify the install

```bash
pip install pytest
pytest tests/          # all tests run offline — no API key required
```

---

## Quick start

```python
from cad_agent import CADAgent

agent = CADAgent()
result = agent.generate("A hex nut with M6 thread hole and 5mm thickness")
print(result.summary())
# OK: 'A hex nut with M6 thread hole and 5mm thickness'
#   volume: 0.0005 m³ (469 mm³)
#   STEP:   cad_output/part_4f8e2c19.step
#   STL:    cad_output/part_4f8e2c19.stl
#   script: cad_output/part_4f8e2c19.py
```

The returned `CADResult` holds the live build123d `Part` object too:

```python
result.part.volume       # → 469.3 mm³
result.part.bounding_box # → BoundBox(...)
```

---

## Configuration

For non-default behavior, build a `CADAgentConfig`:

```python
from cad_agent import CADAgent, CADAgentConfig
from pathlib import Path

cfg = CADAgentConfig(
    backend="gemini",
    api_key="...",                 # or omit and rely on env
    model="gemini-flash-latest",
    output_dir=Path("/tmp/cad"),
    max_revisions=5,               # critic-feedback rounds
    use_reasoning_loop=True,       # adds the planning + critique stage
    inject_reference_specs=True,   # auto-resolve "NEMA17" → real dims
    write_step=True,
    write_stl=True,
    write_script=True,
    verbose=False,
)
agent = CADAgent(cfg)
```

Every field has a sensible default; only set what you need to override.

---

## Specs that work well

Mention dimensions in mm explicitly. Be specific about features (holes, fillets, patterns) and any standards involved:

- `"A 50×30×10mm bracket with two M3 mounting holes at (10, 10) and (40, 10), and a 5mm corner fillet"`
- `"A NEMA17 motor mount plate, 4mm thick, with the standard 31mm bolt circle and a 22mm clearance hole in the center"`
- `"A V-shaped jig for holding 2020 T-slot extrusion at 90°, 100mm long, with two M5 clamping screws"`

Vague specs work too but quality drops:

- `"a small box with holes"` → you'll get *something*, but expect surprises.

---

## Programmatic batch generation

```python
from cad_agent import CADAgent

agent = CADAgent()

specs = [
    ("nut",     "M6 hex nut, 5mm thick"),
    ("washer",  "M6 flat washer, 1.6mm thick, 12mm outer diameter"),
    ("bolt",    "M6×20 socket head cap screw with hex drive"),
]
for name, spec in specs:
    result = agent.generate(spec, name=name)
    if result:
        print(f"✓ {name}: {result.volume_mm3:.0f} mm³")
    else:
        print(f"✗ {name}: {result.error}")
```

---

## Multi-part assemblies

Single-script generation cannot reliably produce a 40-part mechanism. `--assembly` runs a plan → generate → assemble → verify → revise pipeline instead:

```bash
cad-agent --assembly "A grandfather clock movement with 30-50 accurate cogs, \
a movement box, seat board, pendulum and two drive weights" --name clockwork -v
```

The LLM decomposes the spec into unique parts and placed instances (a validated JSON plan). **Gears are declared as parametric primitives and built by exact involute math — never LLM geometry.** Every other part gets its own focused codegen call with a bounding-box budget it must fit. Trusted code places the instances, then the assembly is verified: pairwise interference via boolean intersection, plus plan-declared checks (`30 ≤ count(gear_*) ≤ 50`). Violations go back to the planner with numbers, for up to `--max-revisions` rounds. Output: merged STEP/STL, per-part STLs, and the plan JSON (a bill of materials).

From Python: `from cad_agent.assembly import assemble; result = assemble("...")`.

---

## 2D engineering drawings

`cad_agent.drawings` turns models into production-style drawing sheets — ASME/ISO-style DXF with title blocks, revision blocks, dimensions, leaders, and **collision-aware annotation placement** (every label and dim is placed by a ring search against the ink footprint of everything else on the sheet).

```bash
pip install -e ".[drawings]"       # from the repo root
```

No API key needed — the drawing engine is pure geometry, and also works standalone on any existing STL/OBJ/PLY/GLB file.

One call goes from a generated model to a third-angle multi-view sheet (FRONT, TOP, RIGHT, ISO) with an auto-fit ISO 5455 scale. Views are true **hidden-line-removed** projections (depth-buffer occlusion testing): visible edges solid, occluded edges dashed per ASME Y14.2 (`hidden=False` omits them), and the isometric renders as an opaque object:

```python
from cad_agent import CADAgent
from cad_agent.drawings import draw_multiview

result = CADAgent().generate("A 50×30×10mm bracket with two M3 mounting holes")
sheet = draw_multiview(result)          # or draw_multiview("part.stl")
print(sheet.summary())
# OK: drawing 'part_4f8e2c19' on A2 at 2:1
#   DXF:     cad_output/part_4f8e2c19_sheet.dxf
#   preview: cad_output/part_4f8e2c19_sheet.png
```

**Smart drafting** hands the whole sheet design to the LLM, grounded in the *real* model: it receives the measured specs (overall sizes, per-view extents) plus rendered images of the hidden-line-classified views, and composes the `DrawingSpec` itself — which views to show, where to place them, what to dimension, feature callouts, notes, title block — embedding true projections of the 3D shape via `Mesh3DView` entities. Validation, build, and collision errors are fed back for revision:

```python
from cad_agent.drawings import draft_drawing

sheet = draft_drawing("cad_output/clock.stl",
                      notes="material is oak; callout the pendulum")
print(sheet.summary())
```

**LLM-assisted drafting from text** goes the other way — describe a part and the LLM emits the full dimensioned `DrawingSpec` (geometry, dims, hole callouts, title block), Pydantic-validated, built, and collision-checked, with errors fed back for revision (needs an API key, like `generate()`):

```python
from cad_agent.drawings import generate_drawing

sheet = generate_drawing(
    "A flange plate, OD 120mm, with 8 M6 clearance holes on a 95mm "
    "bolt circle and a 40mm center bore"
)
print(sheet.summary())
```

All of it is available from the terminal:

```bash
cad-agent-draw cad_output/part.stl                        # smart LLM draft (default with an API key)
cad-agent-draw cad_output/part.step --notes "6061-T6"     # STEP works too; guide the drafter
cad-agent-draw cad_output/part.stl --basic                # offline multi-view template (no LLM)
cad-agent-draw "spacer plate 80x40mm with two M3 holes"   # LLM-drafted from a description
cad-agent "M6 hex nut, 5mm thick" --draw                  # generate 3D model + sheet in one go
```

Model files use the smart LLM draft when an API key is in the environment and fall back to the offline template otherwise (`--smart` / `--basic` force either).

For full manual control, build a `DrawingSpec` declaratively — every entity and annotation is a validated Pydantic model:

```python
from cad_agent.drawings import (
    DrawingBuilder, DrawingSpec, TitleBlock, Units,
    Circle, Line, LinearDim, DiameterDim, Ref, Snap,
    render_preview, validate,
)

spec = DrawingSpec(
    sheet="A3",
    units=Units.MILLIMETERS,
    workflow="mech",
    title_block=TitleBlock(title="WIDGET BRACKET", drawing_no="RAP-0001", rev="A"),
    entities=[
        Circle(id="H1", center=(20, 20), radius=4.0),
        Line(id="L1", start=(0, 0), end=(100, 0)),
    ],
    annotations=[
        DiameterDim(id="D1", target=Ref(entity_id="H1", snap=Snap.CENTER)),
        LinearDim(id="DH", p1=Ref(entity_id="L1", snap=Snap.START),
                  p2=Ref(entity_id="L1", snap=Snap.END), side="below"),
    ],
)

builder = DrawingBuilder(spec)
doc = builder.build()
builder.save("widget.dxf")
render_preview(doc, "widget_sheet.png", layout="paperspace")
```

Sheets cover ISO A0–A4 (landscape and portrait — `A3P`, `A2P`, …) and ASME ANSI_A–E; dimstyles follow ASME Y14.5 conventions (mech / arch / struct, mm and inch). Multi-view sheets auto-pick the sheet and orientation that the drawing fills best, and carry overall dimensions (true model mm) by default. A post-build `validate()` pass flags any residual annotation overlaps. End-to-end examples live in `tests/test_drawings_*.py`.

---

## Viewing outputs

`cad_agent.viewer` opens any generated model in an interactive browser viewer (three.js orbit controls). It writes a single self-contained HTML file next to the model — no server, works offline. Requires `trimesh` (included in the `drawings` extra).

```bash
# open an existing output
cad-agent-view cad_output/part.stl

# or generate and view in one go
cad-agent "M6 hex nut, 5mm thick" --view
```

From Python:

```python
from cad_agent.viewer import view

view(result)                    # a CADResult from agent.generate()
view("cad_output/part.step")    # STL, OBJ, PLY, OFF, GLB/GLTF, or STEP
```

STEP files are tessellated through build123d; mesh formats load directly. Use `--no-open` (or `open_browser=False`) to just write the HTML.

---

## Lower-level APIs

`CADAgent.generate(...)` is one-shot. For multi-step workflows with rollback and explicit operation control, use `cad_agent.advanced`:

```python
import cad_agent.advanced as advanced

# Reasoning-only — analyze constraints, propose topology, decompose
# into geometric features, without generating any code yet.
session = advanced.ReasoningSession()
plan = session.run("A bracket that holds a NEMA17 motor at 30° tilt", max_iterations=3)
print(plan.final_spec)

# Full design session with checkpointing
sess = advanced.DesignSession(name="bracket")
sess.apply("create_box", dims=(50, 30, 10))
cp = sess.checkpoint()
sess.apply("hole", diameter=5, location=(10, 10, 5))
if not sess.validate():
    sess.rollback(cp)
```

The operation catalog (53 registered ops across 6 categories: features, selectors, analysis, repair, transforms, booleans) is at `advanced.op_catalog`.

The standard parts library (bearings, motors, T-slot extrusions, common dev boards) is at `agent.list_known_parts()`.

---

## CLI

```bash
cad-agent "A 20mm cube with a 5mm through-hole" --output ./out
```

Same options as the Python config, exposed as flags. Run `cad-agent --help` for the full list.

---

## How it works

1. **Spec analysis** — the input is sent to a constraint-analyst LLM call that identifies dimensional constraints, tensions, and ambiguities.
2. **Topology choice** — a topology-chooser proposes the right primitives + operations (box+hole vs. cylinder+slot, etc.).
3. **Decomposition** — the spec is broken into ordered geometric features, each with parameters resolved against the standards library.
4. **Critic loop** — a spec-critic reviews the decomposition and asks for revisions until it converges or hits `max_revisions`.
5. **Code emission** — the final decomposed spec drives an LLM call that produces build123d code in algebra mode.
6. **Validation** — the code is executed in a sandbox; the resulting Part is checked for nonzero volume and exported.

Steps 1–4 can be skipped with `use_reasoning_loop=False` for one-shot generation (faster, lower quality).

---

## License

MIT.
