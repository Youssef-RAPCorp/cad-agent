"""build123d_reference.py — curated build123d API reference for LLM prompts.

This module provides three constants that are inserted into the system
prompt for any LLM call that produces build123d code (shape_generator,
llm_fitter repair tier, and the design chat orchestrator).

The reference is OPINIONATED — it picks Algebra mode as the canonical
style and a focused subset of operations. The LLM will produce better
code if asked to stay inside this subset than if given the full 447-
export API surface.

Sources:
- build123d 0.10.0 source code & docs (key_concepts_algebra.rst,
  cheat_sheet.rst, tips.rst, operations.rst, objects.rst).
- Algebra-mode examples in build123d's `examples/` directory.

Distilled in our own words; no copyrighted prose was reproduced.
"""

# ---------------------------------------------------------------------------
# REFERENCE CARD — inserted at the top of every codegen system prompt.
# ---------------------------------------------------------------------------

REFERENCE_CARD = """\
# build123d Algebra-Mode Reference (canonical style — use this)

Always start with: `from build123d import *`

## Core principle: Object Arithmetic
Shapes combine with `+` (union), `-` (cut), `&` (intersect):
    part = Box(10, 10, 5) - Cylinder(2, 5)
    plate = Rectangle(50, 30) - Circle(3)
    panel = pcb + standoffs - mounting_holes

## Core principle: Placement Arithmetic
Use `*` to place a shape on a plane and/or at a location.
Place at location:    Pos(x, y, z) * Box(10, 10, 5)
Place + rotate:       Location((x, y, z), (rx, ry, rz)) * Box(...)
                       Pos(x, y, z) * Rot(0, 0, 45) * Box(...)
Place on plane:       Plane.XZ * Box(10, 10, 5)
Combined:             Plane.XZ * Pos(0, 0, 5) * Rot(0, 0, 45) * Box(...)

`Plane.XY` is the default and can be omitted.
`*` binds tighter than `+`/`-`/`&` — no parentheses needed normally.

## 3D Primitives (Part)
Box(length, width, height)                          # rectangular block
Cylinder(radius, height)                             # cylinder, axis = Z
Cone(bottom_radius, top_radius, height)              # truncated cone
Sphere(radius)
Torus(major_radius, minor_radius)
Wedge(xsize, ysize, zsize, xmin, zmin, xmax, zmax)  # prismatic wedge
Hole(radius, depth=None)                             # cuts material; use with `-`
CounterBoreHole(radius, counter_bore_radius, counter_bore_depth, depth=None)
CounterSinkHole(radius, counter_sink_radius, depth=None, counter_sink_angle=82)

## 2D Sketch Objects
Rectangle(width, height)
RectangleRounded(width, height, radius)
Circle(radius)
Ellipse(x_radius, y_radius)
RegularPolygon(radius, side_count)
Polygon(*points)                                # explicit vertex list
SlotOverall(width, height)                       # rounded-end slot
Trapezoid(width, height, left_side_angle, right_side_angle=None)
Text(txt, font_size, font="Arial", font_style=FontStyle.REGULAR)

## 1D Curves (BuildLine objects, used by Polyline/sweep paths)
Line((x1, y1), (x2, y2))
Polyline(*points, close=False)
RadiusArc((x1, y1), (x2, y2), radius)        # signed: + bulges left of chord
TangentArc(*points, tangent=(tx, ty))
Spline(*points, tangents=None)
Bezier(*points)                                # control polygon
Helix(pitch, height, radius)
JernArc(start, tangent, radius, arc_size)

## Conversions: 2D → 3D
extrude(sketch, amount, dir=None, taper=0)         # straight extrude
revolve(sketch, axis=Axis.Z, revolution_arc=360)   # rotate around axis
loft(sections, ruled=False)                         # blend between sketches
sweep(section, path, multisection=False)            # sweep section along path
section(part, plane=Plane.XZ)                       # 2D section through 3D part

## Modifying Operations
fillet(edges, radius)                  # round edges; ALWAYS apply LATE
chamfer(edges, length, length2=None)   # bevel edges; ALWAYS apply LATE
mirror(shape, about=Plane.XZ)
offset(shape, amount)                   # grow/shrink shape
scale(shape, by)
split(shape, bisect_by=Plane.XY, keep=Keep.TOP)

## Patterning (multiple copies in one expression)
GridLocations(x_spacing, y_spacing, x_count, y_count)
PolarLocations(radius, count, start_angle=0)
HexLocations(apothem, x_count, y_count)
Locations(loc1, loc2, ...)

# Example: 4 corner holes
hole_locs = GridLocations(40, 30, 2, 2)
plate = Rectangle(50, 40) - [loc * Circle(2.5) for loc in hole_locs]

## Selectors (find specific edges/faces by criteria)
part.edges()                 # all edges of a Part
part.faces() | Plane.XY      # faces parallel to XY plane (filter_by)
part.edges() | GeomType.LINE # only straight edges
part.edges() > Axis.Z        # sort along Z axis
part.faces() << Axis.Z       # bottom-most face along Z
part.faces() >> Axis.Z       # top-most face along Z

# Edge sub-position: @ (point), % (tangent), ^ (Location)
edge @ 0.5    # midpoint of edge as Vector
edge ^ 0.5    # Location at midpoint with edge tangent

## Locations & Planes
Plane.XY, Plane.XZ, Plane.YZ                   # standard planes
Plane(origin=(x,y,z), x_dir=..., z_dir=...)    # custom plane
Location((x, y, z), (rx_deg, ry_deg, rz_deg))  # combined translate+rotate
Pos(x, y, z), Rot(rx, ry, rz)                  # convenience builders
Axis.X, Axis.Y, Axis.Z                          # standard axes

## Assembly
parts = Compound([part_a, part_b, part_c])
# Or with names for export:
parts = Compound(label="my_assembly", children=[part_a, part_b])

## Import / Export
from build123d import import_step, import_brep, import_stl
shape = import_step("part.step")
export_step(part, "out.step")
export_stl(part, "out.stl")
"""


# ---------------------------------------------------------------------------
# WORKED PATTERNS — short, complete examples covering common cases.
# Each one runs as-is and demonstrates a canonical pattern.
# ---------------------------------------------------------------------------

WORKED_PATTERNS = """\
# Worked patterns (each is a complete, runnable script)

## Pattern 1: Plate with mounting holes (parametric)
from build123d import *
plate_w, plate_h, thick = 60, 40, 3
hole_dia, hole_inset = 3.4, 5
plate = Rectangle(plate_w, plate_h)
for x in (-plate_w/2 + hole_inset, plate_w/2 - hole_inset):
    for y in (-plate_h/2 + hole_inset, plate_h/2 - hole_inset):
        plate -= Pos(x, y) * Circle(hole_dia / 2)
part = extrude(plate, thick)

## Pattern 2: Bracket — extruded L profile with reinforcing rib
from build123d import *
leg_a, leg_b, thick, width = 30, 25, 3, 20
profile = Polygon((0, 0), (leg_a, 0), (leg_a, thick),
                   (thick, thick), (thick, leg_b), (0, leg_b))
bracket = extrude(profile, width)
# add a triangular gusset at one end
with BuildSketch(Plane.XZ) as gusset:
    Polygon((thick, thick), (leg_a-2, thick), (thick, leg_b-2))
part = bracket + extrude(gusset.sketch, 3)

## Pattern 3: Tube/ring (cylinder with concentric hole)
from build123d import *
od, id, h = 10, 6, 20
part = Cylinder(od/2, h) - Cylinder(id/2, h)

## Pattern 4: NEMA17 mount face (4 corner holes around shaft)
from build123d import *
face_size, bolt_circle, bolt_dia, shaft_dia, plate_thick = 42, 31, 3.4, 22, 5
plate = Rectangle(face_size, face_size) - Circle(shaft_dia / 2)
# 4 mounting bolts at +/- bolt_circle/2 in X,Y (i.e., on a 31mm bolt circle)
half = bolt_circle / 2 / (2 ** 0.5) * (2 ** 0.5)  # = bolt_circle / 2 — diagonal grid
for x, y in [(-half, -half), (half, -half), (half, half), (-half, half)]:
    plate -= Pos(x, y) * Circle(bolt_dia / 2)
part = extrude(plate, plate_thick)

## Pattern 5: Threaded part outline using revolve
from build123d import *
profile = Polygon((0, 0), (5, 0), (5, 10), (3, 10), (3, 5), (0, 5))
part = revolve(profile, axis=Axis.Y)

## Pattern 6: Loft between two cross-sections (frustum)
from build123d import *
bottom = Pos(0, 0, 0) * Circle(10)
top = Pos(0, 0, 30) * Circle(5)
part = loft([bottom, top])

## Pattern 7: Sweep a section along a curved path (handle, hose, wire)
from build123d import *
path = Spline((-15, 0, 0), (0, 0, 10), (15, 0, 0),
               tangents=((0, 0, 1), (0, 0, -1)))
section = Pos(-15, 0, 0) * Circle(2)
part = sweep(section, path=path)

## Pattern 8: Hole pattern using GridLocations
from build123d import *
plate = Rectangle(80, 60)
plate -= [loc * Circle(2) for loc in GridLocations(20, 15, 3, 3)]
part = extrude(plate, 4)

## Pattern 9: Fillet only specific edges (use selectors!)
from build123d import *
part = Box(50, 30, 10)
# Fillet only the four vertical edges (parallel to Z)
vert_edges = part.edges() | Axis.Z
part = fillet(vert_edges, radius=3)

## Pattern 10: Assembly of two named parts
from build123d import *
base = Box(50, 30, 5)
peg = Pos(0, 0, 5) * Cylinder(3, 10)
asm = Compound(label="dowel_pin", children=[
    base.label_set("base") if hasattr(base, "label_set") else base,
    peg,
])
"""


# ---------------------------------------------------------------------------
# ANTI-PATTERNS — explicit "DON'T do X, DO do Y" rules.
# ---------------------------------------------------------------------------

ANTI_PATTERNS = """\
# Anti-patterns — common mistakes to AVOID

DON'T mix Algebra mode and Builder mode.
   BAD:   with BuildPart() as bp:
              add(some_algebra_part + Box(10, 10, 5))
   GOOD:  part = some_algebra_part + Box(10, 10, 5)

DON'T translate parts with .translate() — use Location * part.
   BAD:   bracket.translate((10, 0, 0))
   GOOD:  Pos(10, 0, 0) * bracket

DON'T forget to assign a final `part` variable. The shape_generator
sandbox looks for it.
   BAD:   result = Box(10, 10, 5)              # 'part' is missing
   GOOD:  part = Box(10, 10, 5)

DON'T fillet/chamfer early. Apply them LAST.
   BAD:   base = fillet(Box(50, 30, 10).edges(), 2)  # then add holes
          base -= Pos(0, 0) * Cylinder(3, 10)
   GOOD:  base = Box(50, 30, 10)
          base -= Pos(0, 0) * Cylinder(3, 10)
          base = fillet(base.edges() | Axis.Z, 2)

DON'T close polylines manually — pass close=True or use Polygon.
   BAD:   Polyline((0,0), (10,0), (10,10), (0,10), (0,0))   # extra point
   GOOD:  Polyline((0,0), (10,0), (10,10), (0,10), close=True)
   GOOD:  Polygon((0,0), (10,0), (10,10), (0,10))

DON'T pass collinear or duplicate points to make_face — it will throw
TopoDS::Face TypeMismatch when ShapeFix can't close the wire.
Inspect your point list first.

DON'T import non-build123d modules in code intended for the sandbox
(shape_generator strips imports of os/sys/file IO/network).

DON'T use `Vector.__add__` on Compounds — Compounds compose with `+`,
Vectors compose with `+`, but mixing them silently does the wrong thing.

DON'T forget units. Build123d works in ANY unit but is conventionally MM.
Always include units in comments / docstrings.

DON'T use `Box(...).Solid()` — that's the cadquery API; in build123d a
Box() is already usable as a Part.
"""


# ---------------------------------------------------------------------------
# Compact philosophy — distilled from build123d's tips.rst
# ---------------------------------------------------------------------------

PHILOSOPHY = """\
# build123d design philosophy (distilled)

1. 2D before 3D. Build the 2D profile (sketch), THEN extrude/revolve/sweep.
   3D operations are slower and fail more often.

2. Parameterize. Use named variables for critical dimensions; derive other
   dimensions from them. Future-you will thank present-you.

3. Delay fillets and chamfers. Apply them as one of the LAST steps. Doing
   them early creates non-planar faces that cripple later operations.

4. Use shallow copies for repeated parts (fasteners, bearings). Don't
   construct each instance from scratch.

5. When something fails, try a different approach. If sweep fails, try
   loft. If revolve fails, try extrude+rotate+combine. CAD is finicky.

6. Choose a convenient origin. Pick one that exploits your part's symmetry
   so the math is simple and bugs reveal themselves.

7. Plan for assembly. Holes need clearance, mating surfaces need tolerance.
   M3 clearance hole = 3.4mm dia (NOT 3.0mm).
"""


# ---------------------------------------------------------------------------
# Bundled prompt fragment — all four constants concatenated, ready to drop
# into a system prompt verbatim.
# ---------------------------------------------------------------------------

FULL_PROMPT_FRAGMENT = "\n\n".join([
    REFERENCE_CARD,
    WORKED_PATTERNS,
    ANTI_PATTERNS,
    PHILOSOPHY,
])
