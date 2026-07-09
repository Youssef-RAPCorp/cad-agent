"""
DrawingBuilder: takes a DrawingSpec and produces a DXF document.

Pipeline:
  1. Create ezdxf document, register layers + dimstyles + linetypes
  2. Draw entities into modelspace, registering each in the SpatialIndex
  3. Run placement engine for each annotation, querying the index for
     collisions, then write the annotation
  4. Draw border + title block in paperspace
  5. Validate (delegated to validator.py) and save
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import ezdxf
from ezdxf.document import Drawing
from ezdxf.enums import MTextEntityAlignment, TextEntityAlignment

from .geometry import (AABB, GeomEntity, Point, aabb_of, perp_left,
                       resolve_point, sub, unit)
from .placement import (CHAR_W_RATIO, linear_dim_aabb, place_label,
                        place_linear_dim, text_aabb)
from .schema import (AngularDim, Annotation, Arc, Circle, DiameterDim,
                     DrawingSpec, Ellipse, Entity, Hatch, LinearDim, Line,
                     Mesh3DView, Polyline, RadialDim, Rectangle, Ref,
                     TextLabel)
from .spatial_index import SpatialIndex
from .standards import (RAPCAD_TEXT_STYLE, SHEETS, dim_text_height_for_sheet,
                        dimstyle_for, register_dimstyles, register_layers,
                        snap_text_height)
from .title_block import draw_border_and_titleblock


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class BuildReport:
    """Diagnostics from a build pass — useful for the LLM to self-correct."""
    def __init__(self):
        self.warnings: List[str] = []
        self.leaders_added: int = 0
        self.placements_resolved: int = 0
        self.unresolved_collisions: List[str] = []

    def __repr__(self):
        return (f"<BuildReport placements={self.placements_resolved} "
                f"leaders={self.leaders_added} "
                f"warnings={len(self.warnings)} "
                f"unresolved={len(self.unresolved_collisions)}>")


class DrawingBuilder:
    def __init__(self, spec: DrawingSpec):
        self.spec = spec
        self.doc: Drawing = ezdxf.new(dxfversion=spec.dxf_version, setup=True)
        self.doc.units = spec.units.dxf_insunits
        self.doc.header["$INSUNITS"] = spec.units.dxf_insunits
        self.msp = self.doc.modelspace()
        self.index = SpatialIndex()
        self.report = BuildReport()

        # Text height resolution order:
        #   1. spec.default_text_height (if explicitly set)
        #   2. sheet-aware default (DIM_TEXT_HEIGHT_FOR_SHEET)
        # then snapped to the √2 series so different specs read uniformly.
        if spec.default_text_height is not None:
            raw_h = spec.default_text_height
        else:
            raw_h = dim_text_height_for_sheet(spec.sheet)
        self._text_height = snap_text_height(raw_h)
        self._halo = (spec.annotation_halo
                      if spec.annotation_halo is not None
                      else 0.5 * self._text_height)

        register_layers(self.doc)
        register_dimstyles(self.doc, spec.units, text_height=self._text_height)

        self._default_dimstyle = dimstyle_for(spec.workflow, spec.units)

    # -- public --

    def build(self) -> Drawing:
        for ent in self.spec.entities:
            self._draw_entity(ent)
        for ann in self.spec.annotations:
            self._draw_annotation(ann)
        self._setup_paperspace()
        return self.doc

    def save(self, path: str) -> None:
        # ezdxf audit catches structural issues before write
        auditor = self.doc.audit()
        if auditor.has_errors:
            self.report.warnings.append(
                f"audit reported {len(auditor.errors)} unfixable issue(s)"
            )
        self.doc.saveas(path)

    # =====================================================================
    # Entity drawing
    # =====================================================================

    def _draw_entity(self, ent: Entity) -> None:
        if isinstance(ent, Line):
            self._draw_line(ent)
        elif isinstance(ent, Polyline):
            self._draw_polyline(ent)
        elif isinstance(ent, Rectangle):
            self._draw_rectangle(ent)
        elif isinstance(ent, Circle):
            self._draw_circle(ent)
        elif isinstance(ent, Arc):
            self._draw_arc(ent)
        elif isinstance(ent, Ellipse):
            self._draw_ellipse(ent)
        elif isinstance(ent, Hatch):
            self._draw_hatch(ent)
        elif isinstance(ent, Mesh3DView):
            self._draw_mesh3d_view(ent)
        else:
            self.report.warnings.append(f"unknown entity kind {type(ent).__name__}")

    def _draw_line(self, ent: Line) -> None:
        self.msp.add_line(ent.start, ent.end, dxfattribs={"layer": ent.layer})
        ge = GeomEntity(entity_id=ent.id, kind="line",
                        points=[ent.start, ent.end])
        ge.aabb = aabb_of(ge)
        self.index.add(ge)

    def _draw_polyline(self, ent: Polyline) -> None:
        self.msp.add_lwpolyline(ent.points, close=ent.closed,
                                dxfattribs={"layer": ent.layer})
        ge = GeomEntity(entity_id=ent.id, kind="polyline",
                        points=list(ent.points), closed=ent.closed)
        ge.aabb = aabb_of(ge)
        self.index.add(ge)

    def _draw_rectangle(self, ent: Rectangle) -> None:
        x, y = ent.corner
        pts = [(x, y), (x + ent.width, y),
               (x + ent.width, y + ent.height), (x, y + ent.height)]
        self.msp.add_lwpolyline(pts, close=True,
                                dxfattribs={"layer": ent.layer})
        ge = GeomEntity(entity_id=ent.id, kind="polyline",
                        points=pts, closed=True)
        ge.aabb = aabb_of(ge)
        self.index.add(ge)

    def _draw_circle(self, ent: Circle) -> None:
        self.msp.add_circle(ent.center, ent.radius,
                            dxfattribs={"layer": ent.layer})
        ge = GeomEntity(entity_id=ent.id, kind="circle",
                        center=ent.center, radius=ent.radius)
        ge.aabb = aabb_of(ge)
        self.index.add(ge)

    def _draw_arc(self, ent: Arc) -> None:
        self.msp.add_arc(ent.center, ent.radius,
                         start_angle=ent.start_angle,
                         end_angle=ent.end_angle,
                         dxfattribs={"layer": ent.layer})
        ge = GeomEntity(entity_id=ent.id, kind="arc",
                        center=ent.center, radius=ent.radius,
                        start_angle=ent.start_angle,
                        end_angle=ent.end_angle)
        ge.aabb = aabb_of(ge)
        self.index.add(ge)

    def _draw_ellipse(self, ent: Ellipse) -> None:
        self.msp.add_ellipse(ent.center, major_axis=ent.major_axis,
                             ratio=ent.ratio,
                             dxfattribs={"layer": ent.layer})
        # AABB approximation: bounding box of axes endpoints
        cx, cy = ent.center
        ax, ay = ent.major_axis
        major_len = math.hypot(ax, ay)
        minor_len = major_len * ent.ratio
        pts = [(cx+ax, cy+ay), (cx-ax, cy-ay),
               (cx-ay*ent.ratio, cy+ax*ent.ratio),
               (cx+ay*ent.ratio, cy-ax*ent.ratio)]
        ge = GeomEntity(entity_id=ent.id, kind="polyline", points=pts,
                        closed=True)
        ge.aabb = AABB.from_points(pts)
        # Tag actual kind so we don't accidentally treat it as polyline ref
        ge.kind = "ellipse"
        self.index.add(ge)

    def _draw_hatch(self, ent: Hatch) -> None:
        boundary = self.index.get(ent.boundary_id)
        hatch = self.msp.add_hatch(
            dxfattribs={"layer": ent.layer},
        )
        hatch.set_pattern_fill(ent.pattern,
                               scale=ent.pattern_scale,
                               angle=ent.pattern_angle)
        if boundary.kind == "polyline":
            hatch.paths.add_polyline_path(boundary.points,
                                          is_closed=boundary.closed)
        elif boundary.kind == "circle":
            from ezdxf.math import Vec3
            hatch.paths.add_edge_path().add_arc(
                center=boundary.center, radius=boundary.radius,
                start_angle=0, end_angle=360)
        elif boundary.kind == "arc":
            hatch.paths.add_edge_path().add_arc(
                center=boundary.center, radius=boundary.radius,
                start_angle=boundary.start_angle,
                end_angle=boundary.end_angle)
        else:
            self.report.warnings.append(
                f"hatch {ent.id}: cannot bound on {boundary.kind}")
            return
        # Hatches don't participate in collision detection as obstacles —
        # they're fill, not lines.
        ge = GeomEntity(entity_id=ent.id, kind="hatch",
                        points=boundary.points,
                        center=boundary.center, radius=boundary.radius)
        ge.aabb = boundary.aabb
        ge.obstacle = False
        self.index.add(ge)

    def _draw_mesh3d_view(self, ent: Mesh3DView) -> None:
        """Load a 3D mesh, project to 2D, and emit one Line per edge.

        Each emitted line is registered in the spatial index so dim /
        text placement can avoid the projected geometry. The view's
        bounding box is also recorded as a parent entity for collision
        culling.
        """
        try:
            from .model3d import project_file
        except ImportError as e:
            self.report.warnings.append(
                f"mesh3d_view {ent.id}: {e}")
            return

        try:
            view = project_file(ent.path,
                                view=ent.view,
                                angle_threshold_deg=ent.angle_threshold_deg)
        except Exception as e:
            self.report.warnings.append(
                f"mesh3d_view {ent.id}: failed to load {ent.path}: {e}")
            return

        if not view.edges_2d:
            self.report.warnings.append(
                f"mesh3d_view {ent.id}: no edges produced for view '{ent.view}'")
            return

        # Translate the view so its bounding-box center sits at the
        # requested origin, then scale.
        vcx, vcy = view.center
        ox, oy = ent.origin
        sc = ent.scale
        layer = ent.layer

        all_pts: list = []
        for i, ((x0, y0), (x1, y1)) in enumerate(view.edges_2d):
            # Translate to origin, scale around new origin
            p0 = (ox + (x0 - vcx) * sc, oy + (y0 - vcy) * sc)
            p1 = (ox + (x1 - vcx) * sc, oy + (y1 - vcy) * sc)
            self.msp.add_line(p0, p1, dxfattribs={"layer": layer})
            sub_id = f"{ent.id}__e{i}"
            ge = GeomEntity(entity_id=sub_id, kind="line", points=[p0, p1])
            ge.aabb = AABB.from_points([p0, p1])
            self.index.add(ge)
            all_pts.extend([p0, p1])

        # Register a parent entity covering the whole view's bounds.
        # This lets annotations target the view as a whole (e.g. a label
        # below it can reference ent.id as a target).
        parent_aabb = AABB.from_points(all_pts) if all_pts else AABB()
        parent = GeomEntity(entity_id=ent.id, kind="polyline",
                            points=[(parent_aabb.xmin, parent_aabb.ymin),
                                    (parent_aabb.xmax, parent_aabb.ymin),
                                    (parent_aabb.xmax, parent_aabb.ymax),
                                    (parent_aabb.xmin, parent_aabb.ymax)],
                            closed=True)
        parent.aabb = parent_aabb
        # Mark the parent as a non-obstacle wrapper so it doesn't double-
        # collide with its own child edges.
        parent.obstacle = False
        self.index.add(parent)

        # Optional label below the view
        if ent.label:
            text_h = (ent.label_height
                      if ent.label_height is not None
                      else self._text_height * 1.4)
            ly = parent_aabb.ymin - ent.label_offset - text_h
            lx = (parent_aabb.xmin + parent_aabb.xmax) / 2
            label_text = ent.label
            label_w = len(label_text) * text_h * CHAR_W_RATIO
            label_aabb = AABB(lx - label_w / 2, ly,
                              lx + label_w / 2, ly + text_h)
            t = self.msp.add_text(
                label_text,
                dxfattribs={"layer": "TEXT", "height": text_h,
                            "style": RAPCAD_TEXT_STYLE},
            )
            t.set_placement((lx, ly + text_h / 2),
                            align=TextEntityAlignment.MIDDLE_CENTER)
            label_id = f"{ent.id}__label"
            ge_label = GeomEntity(entity_id=label_id, kind="text",
                                  points=[(label_aabb.xmin, label_aabb.ymin),
                                          (label_aabb.xmax, label_aabb.ymin),
                                          (label_aabb.xmax, label_aabb.ymax),
                                          (label_aabb.xmin, label_aabb.ymax)],
                                  closed=True)
            ge_label.aabb = label_aabb
            ge_label.host_ids = [ent.id]
            self.index.add(ge_label)

    # =====================================================================
    # Annotation drawing with placement
    # =====================================================================

    def _draw_annotation(self, ann: Annotation) -> None:
        if isinstance(ann, TextLabel):
            self._draw_text_label(ann)
        elif isinstance(ann, LinearDim):
            self._draw_linear_dim(ann)
        elif isinstance(ann, RadialDim):
            self._draw_radial_dim(ann)
        elif isinstance(ann, DiameterDim):
            self._draw_diameter_dim(ann)
        elif isinstance(ann, AngularDim):
            self._draw_angular_dim(ann)
        else:
            self.report.warnings.append(f"unknown annotation {type(ann).__name__}")

    # ---- text label ----

    def _draw_text_label(self, ann: TextLabel) -> None:
        host = self.index.get(ann.target.entity_id)
        anchor = resolve_point(host, ann.target.to_geomref())
        height = ann.height if ann.height is not None else self._text_height
        layer = ann.layer or "TEXT"

        # If the LLM supplied an offset, honor it but still warn on collision.
        if ann.offset is not None:
            insert = (anchor[0] + ann.offset[0], anchor[1] + ann.offset[1])
            ab = text_aabb(insert, ann.text, height,
                           rotation_deg=ann.rotation or 0.0)
            collisions = self.index.collisions(ab, clearance=self._halo,
                                               exclude_ids=[host.entity_id])
            if collisions:
                self.report.warnings.append(
                    f"text {ann.id}: manual offset collides with "
                    f"{[c.entity_id for c in collisions]}")
            self._emit_text(ann, insert, height, layer, ann.rotation or 0.0)
            if ann.force_leader:
                self._emit_leader(anchor, insert, ann.id)
            return

        # Auto-place
        # Prefer the outward direction from the host's AABB center to anchor —
        # this is the natural side to put a label.
        host_center = host.aabb.center
        outward = (anchor[0] - host_center[0], anchor[1] - host_center[1])
        if outward == (0.0, 0.0):
            outward = (1.0, 0.0)

        result = place_label(
            self.index, anchor, ann.text, height,
            preferred_dir=outward,
            base_gap=1.5 * height,
            clearance=self._halo,
            exclude_ids=[host.entity_id],
        )
        self._emit_text(ann, result.position, height, layer, result.rotation_deg)
        # Register the placed text in the index so subsequent annotations
        # know to avoid it.
        text_ent = GeomEntity(
            entity_id=ann.id, kind="text",
            points=[(result.aabb.xmin, result.aabb.ymin),
                    (result.aabb.xmax, result.aabb.ymin),
                    (result.aabb.xmax, result.aabb.ymax),
                    (result.aabb.xmin, result.aabb.ymax)],
            closed=True,
        )
        text_ent.aabb = result.aabb
        text_ent.obstacle = True
        text_ent.host_ids = [host.entity_id]
        self.index.add(text_ent)

        if result.leader_required or ann.force_leader:
            self._emit_leader(anchor, result.position, ann.id)
            self.report.leaders_added += 1
            if result.leader_required:
                self.report.warnings.append(
                    f"text {ann.id}: no clear placement, leader added")
                self.report.unresolved_collisions.append(ann.id)
        self.report.placements_resolved += 1

    def _emit_text(self, ann: TextLabel, insert: Point, height: float,
                   layer: str, rotation_deg: float):
        t = self.msp.add_text(
            ann.text,
            dxfattribs={"layer": layer, "height": height,
                        "rotation": rotation_deg,
                        "style": RAPCAD_TEXT_STYLE},
        )
        t.set_placement(insert, align=TextEntityAlignment.BOTTOM_LEFT)

    def _emit_leader(self, start: Point, end_aabb_point: Point, ann_id: str):
        """Draw a leader from start (on geometry) to the nearest edge of
        the text's bounding region. We approximate the destination as a
        small offset from end_aabb_point toward start."""
        dx = start[0] - end_aabb_point[0]
        dy = start[1] - end_aabb_point[1]
        n = math.hypot(dx, dy)
        if n < 1e-9:
            return
        # Leave a small gap so the leader doesn't overstrike the text
        gap = 1.0
        ex = end_aabb_point[0] + (dx / n) * gap
        ey = end_aabb_point[1] + (dy / n) * gap
        self.msp.add_leader(
            [(ex, ey), (start[0], start[1])],
            dxfattribs={"layer": "LEADERS"},
        )

    # ---- linear dim ----

    def _draw_linear_dim(self, ann: LinearDim) -> None:
        p1_host = self.index.get(ann.p1.entity_id)
        p2_host = self.index.get(ann.p2.entity_id)
        p1 = resolve_point(p1_host, ann.p1.to_geomref())
        p2 = resolve_point(p2_host, ann.p2.to_geomref())
        dimstyle = ann.dimstyle or self._default_dimstyle
        base_offset = (ann.base_offset
                       if ann.base_offset is not None
                       else 4.0 * self._text_height)

        base, angle = place_linear_dim(
            self.index, p1, p2, side=ann.side,
            base_offset=base_offset,
            text_height=self._text_height,
            exclude_ids=[p1_host.entity_id, p2_host.entity_id],
        )
        # If the user provided an explicit angle (e.g. for an aligned dim
        # along a non-axis-aligned feature), honour it. Otherwise let
        # ezdxf place along the feature direction.
        dim_angle = ann.angle if ann.angle is not None else angle

        text_arg = ann.text_override if ann.text_override else "<>"
        dim = self.msp.add_linear_dim(
            base=base, p1=p1, p2=p2,
            angle=dim_angle,
            text=text_arg,
            dimstyle=dimstyle,
            dxfattribs={"layer": ann.layer or "DIMENSIONS"},
        )
        dim.render()
        # Register dimension's tight AABB so later annotations avoid the
        # actual rendered footprint, not a bloated padded box.
        ab = linear_dim_aabb(p1, p2, base, dim_angle,
                             text_height=self._text_height)
        ge = GeomEntity(entity_id=ann.id, kind="dimension",
                        points=[p1, p2, base])
        ge.aabb = ab
        ge.obstacle = True
        ge.host_ids = list({p1_host.entity_id, p2_host.entity_id})
        self.index.add(ge)
        self.report.placements_resolved += 1

    # ---- radial / diameter dim ----

    def _draw_radial_dim(self, ann: RadialDim) -> None:
        host = self.index.get(ann.target.entity_id)
        if host.kind not in ("circle", "arc"):
            self.report.warnings.append(
                f"radial_dim {ann.id}: target {host.entity_id} is not a circle/arc")
            return
        cx, cy = host.center
        r = host.radius
        a = math.radians(ann.angle_deg)
        mpoint = (cx + r * math.cos(a), cy + r * math.sin(a))
        dimstyle = ann.dimstyle or self._default_dimstyle
        dim = self.msp.add_radius_dim_2p(
            center=(cx, cy), mpoint=mpoint,
            dimstyle=dimstyle,
            dxfattribs={"layer": ann.layer or "DIMENSIONS"},
        )
        dim.render()
        # AABB: small region around the radial leader endpoint + text
        text_w = max(1, len(f"R{r:.1f}")) * self._text_height * CHAR_W_RATIO
        # Leader extends outward another ~text_height past mpoint
        out_x = cx + (r + self._text_height * 1.5) * math.cos(a)
        out_y = cy + (r + self._text_height * 1.5) * math.sin(a)
        ab = AABB.from_points([
            (cx, cy), mpoint,
            (out_x - text_w / 2, out_y - self._text_height),
            (out_x + text_w / 2, out_y + self._text_height),
        ])
        self._register_dim_aabb(ann.id, ab, host_ids=[host.entity_id])

    def _draw_diameter_dim(self, ann: DiameterDim) -> None:
        host = self.index.get(ann.target.entity_id)
        if host.kind != "circle":
            self.report.warnings.append(
                f"diameter_dim {ann.id}: target must be a circle")
            return
        cx, cy = host.center
        r = host.radius
        a = math.radians(ann.angle_deg)
        p1 = (cx + r * math.cos(a), cy + r * math.sin(a))
        p2 = (cx - r * math.cos(a), cy - r * math.sin(a))
        dimstyle = ann.dimstyle or self._default_dimstyle
        dim = self.msp.add_diameter_dim_2p(
            p1=p1, p2=p2,
            dimstyle=dimstyle,
            dxfattribs={"layer": ann.layer or "DIMENSIONS"},
        )
        dim.render()
        # AABB covers the diameter chord + a small text band on one side
        text_w = max(1, len(f"\u00d8{2*r:.1f}")) * self._text_height * CHAR_W_RATIO
        out_x = cx + (r + self._text_height * 1.5) * math.cos(a)
        out_y = cy + (r + self._text_height * 1.5) * math.sin(a)
        ab = AABB.from_points([
            p1, p2,
            (out_x - text_w / 2, out_y - self._text_height),
            (out_x + text_w / 2, out_y + self._text_height),
        ])
        self._register_dim_aabb(ann.id, ab, host_ids=[host.entity_id])

    def _draw_angular_dim(self, ann: AngularDim) -> None:
        l1 = self.index.get(ann.line1_id)
        l2 = self.index.get(ann.line2_id)
        if l1.kind != "line" or l2.kind != "line":
            self.report.warnings.append(
                f"angular_dim {ann.id}: needs two line entities")
            return
        dimstyle = ann.dimstyle or self._default_dimstyle
        dim = self.msp.add_angular_dim_2l(
            base=None,
            line1=(l1.points[0], l1.points[1]),
            line2=(l2.points[0], l2.points[1]),
            dimstyle=dimstyle,
            dxfattribs={"layer": ann.layer or "DIMENSIONS"},
        )
        dim.render()
        # Approximate AABB: union of the two lines
        pts = l1.points + l2.points
        self._register_dim_aabb(ann.id, AABB.from_points(pts).expanded(ann.radius), host_ids=[l1.entity_id, l2.entity_id])

    def _register_dim_aabb(self, ann_id: str, aabb: AABB,
                           host_ids: Optional[List[str]] = None):
        ge = GeomEntity(entity_id=ann_id, kind="dimension")
        ge.aabb = aabb
        ge.obstacle = True
        ge.host_ids = list(host_ids or [])
        self.index.add(ge)
        self.report.placements_resolved += 1

    # =====================================================================
    # Paperspace
    # =====================================================================

    def _setup_paperspace(self):
        spec = self.spec
        sheet = SHEETS[spec.sheet]
        psp = self.doc.paperspace()

        # Set paper size on layout
        psp.page_setup(
            size=(sheet.width_mm, sheet.height_mm),
            margins=(sheet.border_top, sheet.border_right,
                     sheet.border_bottom, sheet.border_left),
            units="mm",
        )

        # Fill units label if not set
        tb = spec.title_block.model_copy()
        if not tb.units_label:
            tb.units_label = spec.units.value.upper()

        draw_border_and_titleblock(psp, sheet, tb, spec.revisions)

        # Build a paperspace viewport showing all modelspace geometry.
        # Layout (mm from sheet origin):
        #   - Title block occupies bottom 60 mm of the right portion
        #   - Notes / revision block sit above the title block
        #   - Drawing area: from border_left+5 to border_right-5 horizontally,
        #     from (border_bottom + title_block + safety) to (top - 5) vertically
        bounds = self.index.overall_bounds()
        if not bounds.is_empty:
            from .title_block import TB_HEIGHT, REV_BLOCK_W
            vp_left   = sheet.border_left + 5
            vp_bottom = sheet.border_bottom + TB_HEIGHT + 10   # clearance above TB + notes band
            vp_right  = sheet.width_mm - sheet.border_right - 5
            vp_top    = sheet.height_mm - sheet.border_top - 10
            vp_w = max(40.0, vp_right - vp_left)
            vp_h = max(40.0, vp_top - vp_bottom)
            vp_cx = (vp_left + vp_right) / 2
            vp_cy = (vp_bottom + vp_top) / 2

            # Scale so the model fits with a 12% margin on the limiting axis.
            margin = 1.12
            model_w = max(1e-6, bounds.width  * margin)
            model_h = max(1e-6, bounds.height * margin)
            scale = min(vp_w / model_w, vp_h / model_h)
            # ezdxf's viewport view_height is the HEIGHT of the modelspace
            # area shown, in modelspace units. With scale = paper/model,
            # view_h must be vp_h/scale so the displayed range fills the
            # viewport. (The previous formula used model_h/scale*1.05 which
            # shows only model_h units even when the viewport is much
            # taller, cropping the drawing.)
            view_h = vp_h / scale

            vp = psp.add_viewport(
                center=(vp_cx, vp_cy),
                size=(vp_w, vp_h),
                view_center_point=bounds.center,
                view_height=view_h,
                dxfattribs={"layer": "VIEWPORT"},
            )
            vp.dxf.status = 1   # active


def build_dxf(spec: DrawingSpec, output_path: str) -> BuildReport:
    """One-shot helper: build and save."""
    b = DrawingBuilder(spec)
    b.build()
    b.save(output_path)
    return b.report
