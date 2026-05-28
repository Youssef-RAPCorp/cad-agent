"""
Verifier for cad_agent3.

Compound-safe boolean volumes. Same logic as cad_agent2's verifier:
OCCT's native intersection returns empty on multi-solid Compound operands,
so we iterate into solids and sum piecewise.
"""
from __future__ import annotations

from typing import Optional


def _get_solids(shape) -> list:
    if shape is None:
        return []
    cname = type(shape).__name__
    if cname == "Solid":
        return [shape]
    if hasattr(shape, "solids"):
        try:
            return list(shape.solids())
        except Exception:
            return [shape]
    return [shape]


def compute_intersection(a, b) -> float:
    solids_a = _get_solids(a)
    solids_b = _get_solids(b)
    total = 0.0
    for sa in solids_a:
        for sb in solids_b:
            try:
                inter = sa & sb
            except Exception:
                inter = None
            v = 0.0
            if inter is not None:
                v = getattr(inter, "volume", None)
                if v is None and hasattr(inter, "__iter__"):
                    try:
                        v = sum(getattr(p, "volume", 0.0) for p in inter)
                    except Exception:
                        v = 0.0
                if v is None:
                    v = 0.0
            # If standard boolean failed, retry with fuzzy tolerance.
            # Only attempt when bboxes overlap (otherwise result truly is 0).
            if (not v or v == 0.0):
                bboxes_overlap = True
                try:
                    ba = sa.bounding_box(); bb = sb.bounding_box()
                    if (ba.max.X < bb.min.X or ba.min.X > bb.max.X or
                        ba.max.Y < bb.min.Y or ba.min.Y > bb.max.Y or
                        ba.max.Z < bb.min.Z or ba.min.Z > bb.max.Z):
                        bboxes_overlap = False
                except Exception:
                    pass
                if bboxes_overlap:
                    try:
                        from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
                        from OCP.GProp import GProp_GProps
                        from OCP.BRepGProp import BRepGProp
                        from OCP.TopTools import TopTools_ListOfShape
                        args1 = TopTools_ListOfShape(); args1.Append(sa.wrapped)
                        args2 = TopTools_ListOfShape(); args2.Append(sb.wrapped)
                        op = BRepAlgoAPI_Common()
                        op.SetArguments(args1)
                        op.SetTools(args2)
                        op.SetFuzzyValue(1e-5)
                        op.Build()
                        res = op.Shape()
                        props = GProp_GProps()
                        BRepGProp.VolumeProperties_s(res, props)
                        v = props.Mass()
                    except Exception:
                        v = 0.0
                # Last-resort: Monte-Carlo point sampling in bbox overlap.
                # Use this when OCCT booleans give zero but shapes overlap —
                # happens with layered reconstructions of cone/cylinder solids.
                if (not v or v == 0.0) and bboxes_overlap:
                    try:
                        import random
                        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
                        from OCP.gp import gp_Pnt
                        from OCP.TopAbs import TopAbs_IN
                        ba = sa.bounding_box(); bb = sb.bounding_box()
                        ox0, ox1 = max(ba.min.X, bb.min.X), min(ba.max.X, bb.max.X)
                        oy0, oy1 = max(ba.min.Y, bb.min.Y), min(ba.max.Y, bb.max.Y)
                        oz0, oz1 = max(ba.min.Z, bb.min.Z), min(ba.max.Z, bb.max.Z)
                        if ox1 > ox0 and oy1 > oy0 and oz1 > oz0:
                            ca = BRepClass3d_SolidClassifier(sa.wrapped)
                            cb = BRepClass3d_SolidClassifier(sb.wrapped)
                            N = 4000
                            rng = random.Random(42)
                            both_in = 0
                            for _ in range(N):
                                x = rng.uniform(ox0, ox1)
                                y = rng.uniform(oy0, oy1)
                                z = rng.uniform(oz0, oz1)
                                p = gp_Pnt(x, y, z)
                                ca.Perform(p, 1e-8)
                                if ca.State() != TopAbs_IN: continue
                                cb.Perform(p, 1e-8)
                                if cb.State() == TopAbs_IN:
                                    both_in += 1
                            overlap_vol = (ox1-ox0)*(oy1-oy0)*(oz1-oz0)
                            v = overlap_vol * both_in / N
                    except Exception:
                        v = 0.0
            if v and v > 0:
                total += float(v)
    return total


def safe_volume(shape) -> float:
    if shape is None:
        return 0.0
    v = getattr(shape, "volume", None)
    if isinstance(v, (int, float)):
        return float(v)
    if hasattr(shape, "__iter__"):
        try:
            return float(sum(getattr(s, "volume", 0.0) for s in shape))
        except Exception:
            return 0.0
    return 0.0
