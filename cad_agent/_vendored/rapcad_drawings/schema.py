"""
Pydantic schema: the LLM I/O contract.

An LLM produces a `DrawingSpec` (JSON-serializable). The Builder consumes
it and emits a DXF + optional preview. Geometric references inside
annotations use entity ids — coordinates are not duplicated, so labels
stay attached to features.

Key invariants enforced by validators:
  - Every `entity_id` is unique within a drawing.
  - Every annotation `target` references a known entity id.
  - Polyline must have >=2 points; circles/arcs must have radius>0; arcs
    must have a non-zero sweep.
  - Units are consistent: dimstyle suffix matches drawing units unless
    explicitly overridden.
"""
from __future__ import annotations

import math
from typing import List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .geometry import GeomRef, Snap
from .standards import SHEETS, Units


Point2D = Tuple[float, float]


# ---------------------------------------------------------------------------
# Annotation reference (validated form of GeomRef)
# ---------------------------------------------------------------------------

class Ref(BaseModel):
    """Reference to a snap point on an entity. Mirrors geometry.GeomRef."""
    model_config = ConfigDict(extra="forbid")

    entity_id: str
    snap: Snap = Snap.MID
    index: int = 0
    t: float = Field(0.5, ge=0.0, le=1.0)

    def to_geomref(self) -> GeomRef:
        return GeomRef(entity_id=self.entity_id, snap=self.snap,
                       index=self.index, t=self.t)


# ---------------------------------------------------------------------------
# Geometry entities
# ---------------------------------------------------------------------------

class _EntityBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    layer: str = "VISIBLE"


class Line(_EntityBase):
    kind: Literal["line"] = "line"
    start: Point2D
    end:   Point2D

    @model_validator(mode="after")
    def _nonzero(self):
        if self.start == self.end:
            raise ValueError(f"Line {self.id}: start and end are identical")
        return self


class Polyline(_EntityBase):
    kind: Literal["polyline"] = "polyline"
    points: List[Point2D]
    closed: bool = False

    @field_validator("points")
    @classmethod
    def _at_least_two(cls, v):
        if len(v) < 2:
            raise ValueError("polyline needs at least 2 points")
        return v


class Rectangle(_EntityBase):
    """Axis-aligned rectangle (a closed polyline under the hood)."""
    kind: Literal["rectangle"] = "rectangle"
    corner: Point2D                  # lower-left
    width:  float = Field(gt=0)
    height: float = Field(gt=0)


class Circle(_EntityBase):
    kind: Literal["circle"] = "circle"
    center: Point2D
    radius: float = Field(gt=0)


class Arc(_EntityBase):
    kind: Literal["arc"] = "arc"
    center: Point2D
    radius: float = Field(gt=0)
    start_angle: float                # degrees CCW from +x
    end_angle:   float

    @model_validator(mode="after")
    def _nonzero_sweep(self):
        sweep = (self.end_angle - self.start_angle) % 360.0
        if sweep == 0.0:
            raise ValueError(f"Arc {self.id}: zero sweep")
        return self


class Ellipse(_EntityBase):
    kind: Literal["ellipse"] = "ellipse"
    center: Point2D
    major_axis: Point2D              # vector from center to end of major
    ratio: float = Field(gt=0, le=1) # minor/major


class Hatch(_EntityBase):
    """Hatch fill bounded by an entity id (must be a closed polyline,
    circle, ellipse, or arc)."""
    kind: Literal["hatch"] = "hatch"
    boundary_id: str
    pattern: str = "ANSI31"          # default cross-hatch
    pattern_scale: float = 1.0
    pattern_angle: float = 0.0


class Mesh3DView(_EntityBase):
    """A projected 2D view of a 3D mesh file (STL/OBJ/PLY/OFF/GLB).

    When the builder encounters this entity, it loads the mesh, computes
    silhouette + sharp-feature edges for the named view, and emits the
    resulting line segments into modelspace at the specified origin and
    scale. Each emitted line is registered in the spatial index so
    subsequent annotations can avoid colliding with the projected view.

    Multiple Mesh3DView entities can reference the same `path` with
    different `view` values to produce a multi-view drawing (FRONT, TOP,
    RIGHT, ISO) of one part. Multiple entities with different paths can
    place several parts on one sheet.
    """
    kind: Literal["mesh3d_view"] = "mesh3d_view"
    path: str
    view: Literal["front", "top", "bottom", "right", "left", "back", "iso"] = "front"
    origin: Tuple[float, float] = (0.0, 0.0)         # placement origin
    scale:  float = 1.0
    angle_threshold_deg: float = 30.0                # feature-edge cutoff
    show_hidden: bool = False                        # occluded edges, dashed
    hidden_layer: str = "HIDDEN"                     # layer for hidden edges
    label:  Optional[str] = None                     # text below the view
    label_height: Optional[float] = None             # else default_text_height * 1.4
    label_offset: float = 6.0                        # mm gap below the view


# Union for entity discriminator
Entity = Union[Line, Polyline, Rectangle, Circle, Arc, Ellipse, Hatch,
               Mesh3DView]


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

class _AnnBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    layer: Optional[str] = None      # default chosen by kind


class TextLabel(_AnnBase):
    kind: Literal["text"] = "text"
    text: str
    target: Ref                      # what the label is pointing at
    height: Optional[float] = None   # default from drawing
    # Optional manual offset/rotation. If omitted, placement engine decides.
    offset: Optional[Point2D] = None
    rotation: Optional[float] = None
    # Force a leader even if there's room without one
    force_leader: bool = False


class LinearDim(_AnnBase):
    kind: Literal["linear_dim"] = "linear_dim"
    p1: Ref                          # measurement point 1
    p2: Ref                          # measurement point 2
    angle: Optional[float] = None    # dim-line angle in deg; default = along p1->p2
    side: Literal["auto", "left", "right", "above", "below"] = "auto"
    base_offset: Optional[float] = None
    text_override: Optional[str] = None
    dimstyle: Optional[str] = None


class RadialDim(_AnnBase):
    kind: Literal["radial_dim"] = "radial_dim"
    target: Ref                      # must reference a circle/arc center or quadrant
    angle_deg: float = 45.0          # where the leader exits the circle
    dimstyle: Optional[str] = None


class DiameterDim(_AnnBase):
    kind: Literal["diameter_dim"] = "diameter_dim"
    target: Ref                      # reference center of a circle
    angle_deg: float = 45.0
    dimstyle: Optional[str] = None


class AngularDim(_AnnBase):
    kind: Literal["angular_dim"] = "angular_dim"
    line1_id: str                    # two line entities defining the angle
    line2_id: str
    radius: float = 20.0             # arc radius for the dim
    dimstyle: Optional[str] = None


Annotation = Union[TextLabel, LinearDim, RadialDim, DiameterDim, AngularDim]


# ---------------------------------------------------------------------------
# Title block + revision
# ---------------------------------------------------------------------------

class TitleBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title:        str = ""
    subtitle:     str = ""
    drawing_no:   str = ""
    sheet:        str = "1/1"
    rev:          str = "A"
    date:         str = ""
    drawn_by:     str = ""
    checked_by:   str = ""
    approved_by:  str = ""
    scale:        str = "1:1"
    units_label:  str = ""           # filled from Drawing.units if blank
    org:          str = "RAPCorp"
    project:      str = ""
    tolerance:    str = "ASME Y14.5"
    material:     str = ""
    finish:       str = ""
    notes:        List[str] = Field(default_factory=list)


class RevisionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rev:         str
    description: str
    date:        str
    by:          str = ""


# ---------------------------------------------------------------------------
# Drawing spec (top level)
# ---------------------------------------------------------------------------

class DrawingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sheet:       Literal[
        "A4", "A3", "A2", "A1", "A0",
        "A4P", "A3P", "A2P", "A1P", "A0P",   # portrait
        "ANSI_A", "ANSI_B", "ANSI_C", "ANSI_D", "ANSI_E"
    ] = "A3"
    units:       Units = Units.MILLIMETERS
    workflow:    Literal["mech", "arch", "struct"] = "mech"
    dxf_version: Literal["R2010", "R2013", "R2018"] = "R2018"

    title_block:  TitleBlock = Field(default_factory=TitleBlock)
    revisions:    List[RevisionEntry] = Field(default_factory=list)

    entities:     List[Entity] = Field(default_factory=list)
    annotations:  List[Annotation] = Field(default_factory=list)

    # Default text height for labels (drawing units). If None, the dimstyle
    # text height is reused.
    default_text_height: Optional[float] = None
    # Minimum clearance (halo) between annotations and geometry, drawing units.
    # If None, derived as 0.5 * text height.
    annotation_halo: Optional[float] = None

    # ---- cross-entity validation ----
    @model_validator(mode="after")
    def _validate_refs(self):
        ids = set()
        for e in self.entities:
            if e.id in ids:
                raise ValueError(f"duplicate entity id: {e.id}")
            ids.add(e.id)
        for a in self.annotations:
            if a.id in ids:
                raise ValueError(f"annotation id collides with entity id: {a.id}")
            ids.add(a.id)

        # All annotation references point to known entities.
        def check_ref(r: Ref, where: str):
            if r.entity_id not in ids:
                raise ValueError(f"{where}: unknown entity_id {r.entity_id!r}")

        for a in self.annotations:
            if isinstance(a, TextLabel):
                check_ref(a.target, f"text {a.id}")
            elif isinstance(a, LinearDim):
                check_ref(a.p1, f"linear_dim {a.id} p1")
                check_ref(a.p2, f"linear_dim {a.id} p2")
            elif isinstance(a, (RadialDim, DiameterDim)):
                check_ref(a.target, f"{a.kind} {a.id}")
            elif isinstance(a, AngularDim):
                if a.line1_id not in ids:
                    raise ValueError(f"angular_dim {a.id}: unknown line1 {a.line1_id}")
                if a.line2_id not in ids:
                    raise ValueError(f"angular_dim {a.id}: unknown line2 {a.line2_id}")

        # Hatch boundary references
        for e in self.entities:
            if isinstance(e, Hatch) and e.boundary_id not in ids:
                raise ValueError(f"hatch {e.id}: unknown boundary {e.boundary_id}")

        return self
