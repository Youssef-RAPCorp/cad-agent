"""
rapcad_drawings — produce expert-grade CAD drawings from structured LLM input.

Public API
==========

    from rapcad_drawings import DrawingSpec, DrawingBuilder, build_dxf
    from rapcad_drawings import render_preview, validate
    from rapcad_drawings.schema import (
        Line, Polyline, Rectangle, Circle, Arc, Ellipse, Hatch,
        TextLabel, LinearDim, RadialDim, DiameterDim, AngularDim,
        Ref, TitleBlock, RevisionEntry,
    )
    from rapcad_drawings.geometry import Snap
    from rapcad_drawings.standards import Units

Workflow:
    spec = DrawingSpec(... )         # validated by Pydantic
    builder = DrawingBuilder(spec)
    doc = builder.build()             # in-memory ezdxf Document
    builder.save("output.dxf")        # write DXF
    render_preview(doc, "preview.png", layout="paperspace")
"""

from .builder import BuildReport, DrawingBuilder, build_dxf
from .geometry import Snap
from .renderer import render_preview
from .schema import (AngularDim, Annotation, Arc, Circle, DiameterDim,
                     DrawingSpec, Ellipse, Entity, Hatch, LinearDim, Line,
                     Mesh3DView, Polyline, Rectangle, Ref, RadialDim,
                     RevisionEntry, TextLabel, TitleBlock)
from .standards import Units
from .validator import Finding, validate

__all__ = [
    # Top level
    "DrawingSpec", "DrawingBuilder", "BuildReport", "build_dxf",
    "render_preview", "validate", "Finding",
    # Geometry kinds
    "Line", "Polyline", "Rectangle", "Circle", "Arc", "Ellipse", "Hatch",
    "Mesh3DView", "Entity",
    # Annotations
    "TextLabel", "LinearDim", "RadialDim", "DiameterDim", "AngularDim",
    "Annotation",
    # Misc
    "Ref", "Snap", "Units",
    "TitleBlock", "RevisionEntry",
]

__version__ = "0.1.0"
