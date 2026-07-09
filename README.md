# cad-agent

LLM-powered CAD model generation from natural-language specs.

`cad-agent` turns a sentence like

> "A 6m × 6m SIP residential pod with a south-facing 914mm door, one east window, and a 2% sloped roof draining north."

into an executable [build123d](https://github.com/gumyr/build123d) Python script, runs it in a sandbox, and exports STEP + STL files.

Under the hood it uses an LLM (Gemini Flash by default; Claude available) to generate the build123d code, runs a critic-feedback loop to catch geometric tensions before code is emitted, and resolves named standards (NEMA17, M3, 2020 T-slot extrusion) from a bundled spec library so the model gets accurate numbers instead of hallucinated ones.

---

## Install

```bash
pip install cad-agent[gemini]      # for Gemini backend
pip install cad-agent[anthropic]   # for Claude backend
pip install cad-agent[all]         # both + extras
```

Set your API key:

```bash
export GEMINI_API_KEY=...          # default backend
# or
export ANTHROPIC_API_KEY=...
export CAD_AGENT_BACKEND=anthropic
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

## 2D engineering drawings

`cad_agent.drawings` turns models into production-style drawing sheets — ASME/ISO-style DXF with title blocks, revision blocks, dimensions, leaders, and **collision-aware annotation placement** (every label and dim is placed by a ring search against the ink footprint of everything else on the sheet).

```bash
pip install cad-agent[drawings]
```

One call goes from a generated model to a third-angle multi-view sheet (FRONT, TOP, RIGHT, ISO) with an auto-fit ISO 5455 scale:

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

For dimensioned sheets, build a `DrawingSpec` declaratively — every entity and annotation is a validated Pydantic model:

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

Sheets cover ISO A0–A4 and ASME ANSI_A–E; dimstyles follow ASME Y14.5 conventions (mech / arch / struct, mm and inch). A post-build `validate()` pass flags any residual annotation overlaps. End-to-end examples live in `tests/test_drawings_*.py`.

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
