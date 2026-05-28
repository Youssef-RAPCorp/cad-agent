"""FitEngine — central dispatcher for solid reconstruction.

The engine owns the tier list of fitters and decides which to call and in
what order per solid. Tier order:

  1. Parametric primitives (Box, BoxFillets, Cylinder, Extrude) — fastest,
     exact for shapes that ARE those primitives.
  2. Halfspace hull — for planar-convex solids.
  3. Axis-stack (flat extrude) — Z/Y/X-stackable solids like housings.
  4. Axis-stack (loft) — tapered stackable solids.
  5. Face-extrude subtract — generic planar-boundary solids.
  6. LLM repair (optional) — last resort when earlier tiers leave
     sym-diff > target_sym_pct. Feeds the best partial result to an LLM
     and asks for a corrected recipe.

Control flow is METRIC-DRIVEN, not LLM-driven. The engine checks
completeness and accuracy after each tier and short-circuits when the
quality threshold is met. Tiers only run as needed.

Parallelism: an engine instance can process multiple solids concurrently
via `fit_part(solids, workers=N)`. Each solid runs in its own process.
"""

from __future__ import annotations

import os
import sys
import time
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from .fitter import (
    FitResult, _verify_fit, _FORCE_PRIMITIVES, set_force_primitives,
    try_fit_box, try_fit_box_with_fillets, try_fit_cylinder,
    try_fit_extrude, try_fit_halfspace_hull, try_fit_axis_stack,
    try_fit_face_extrude,
)


# ---------------------------------------------------------------------------
# Tier definition
# ---------------------------------------------------------------------------

@dataclass
class Tier:
    """One fitter tier in the engine's pipeline."""
    name: str
    fn: Callable[..., FitResult]
    kwargs: dict = field(default_factory=dict)
    # If best_q is already ≥ this value, skip this tier.
    skip_if_best_q_gte: float = 0.995
    # Soft wall-clock budget (seconds). If a tier exceeds this it still
    # completes; this is advisory, used for diagnostics.
    budget_s: float = 60.0


def default_tiers(enable_llm: bool = False) -> List[Tier]:
    """Default fitter pipeline."""
    tiers = [
        Tier("box", try_fit_box, skip_if_best_q_gte=0.999),
        Tier("box_fillets", try_fit_box_with_fillets, skip_if_best_q_gte=0.999),
        Tier("cylinder", try_fit_cylinder, skip_if_best_q_gte=0.999),
        Tier("extrude", try_fit_extrude, skip_if_best_q_gte=0.999),
        Tier("halfspace_hull", try_fit_halfspace_hull, skip_if_best_q_gte=0.998),
        Tier("axis_stack_extrude", try_fit_axis_stack,
             kwargs={"use_loft": False}, skip_if_best_q_gte=0.995,
             budget_s=60.0),
        Tier("axis_stack_loft", try_fit_axis_stack,
             kwargs={"use_loft": True}, skip_if_best_q_gte=0.995,
             budget_s=60.0),
        Tier("face_extrude", try_fit_face_extrude, skip_if_best_q_gte=0.99),
    ]
    if enable_llm:
        from .llm_fitter import try_fit_llm
        tiers.append(Tier("llm_repair", try_fit_llm,
                          kwargs={"target_sym_pct": 1.0},
                          skip_if_best_q_gte=0.99,
                          budget_s=120.0))
    return tiers


# ---------------------------------------------------------------------------
# Per-solid fit
# ---------------------------------------------------------------------------

@dataclass
class FitDiag:
    """Diagnostic record for one solid fit."""
    solid_idx: int
    source_vol: float
    tier_runs: List[Tuple[str, float, float, float]] = field(default_factory=list)
    # each tier_run: (tier_name, time_s, completeness, accuracy)
    best_tier: Optional[str] = None
    best_comp: float = 0.0
    best_acc: float = 0.0
    total_time: float = 0.0

    @property
    def sym_pct(self) -> float:
        """Approximate per-solid sym-diff percent."""
        return (1 - self.best_comp) * 100 + (1 - self.best_acc) * 100

    def summary(self) -> str:
        lines = [f"solid#{self.solid_idx} vol={self.source_vol:.3f}"
                 f" best={self.best_tier} "
                 f"comp={self.best_comp*100:.2f}% "
                 f"acc={self.best_acc*100:.2f}% "
                 f"sym~{self.sym_pct:.3f}% "
                 f"({self.total_time:.1f}s total)"]
        for name, dt, c, a in self.tier_runs:
            mark = " " if (c, a) != (self.best_comp, self.best_acc) else "*"
            lines.append(f"  {mark} {name:22s} {dt:5.1f}s  c={c*100:6.2f}%  a={a*100:6.2f}%")
        return "\n".join(lines)


class FitEngine:
    """Owns a tier list, dispatches fits, optionally runs in parallel."""

    def __init__(
        self,
        tiers: Optional[List[Tier]] = None,
        tol: float = 0.01,
        target_sym_pct: float = 1.0,
        verbose: bool = False,
        enable_llm: bool = False,
    ):
        self.tiers = tiers if tiers is not None else default_tiers(enable_llm=enable_llm)
        self.tol = tol
        self.target_sym_pct = target_sym_pct
        self.verbose = verbose
        self.enable_llm = enable_llm

    # -------------------------------------------------------------------
    # Single-solid path
    # -------------------------------------------------------------------
    def fit_solid(self, solid, solid_idx: int = 0) -> Tuple[FitResult, FitDiag]:
        """Run tier chain on a single solid, return best FitResult + diag."""
        from .verifier import safe_volume
        src_vol = safe_volume(solid)
        diag = FitDiag(solid_idx=solid_idx, source_vol=src_vol)
        best = FitResult(None, 0.0, 0.0, "none", "no fit")
        best_q = -1.0
        total_t0 = time.time()

        target_q_equiv = 1.0 - (self.target_sym_pct / 200.0)

        for tier in self.tiers:
            if best_q >= tier.skip_if_best_q_gte:
                if self.verbose:
                    print(f"    [{tier.name}] skipped (best_q={best_q:.4f} "
                          f"≥ {tier.skip_if_best_q_gte})", flush=True)
                continue

            t0 = time.time()
            try:
                if tier.name == "llm_repair":
                    # LLM tier takes (solid, tol, ...) but needs the best
                    # partial result for repair context. Pass via kwargs.
                    r = tier.fn(solid, self.tol,
                                best_result=best, **tier.kwargs)
                else:
                    r = tier.fn(solid, self.tol, **tier.kwargs)
            except TypeError:
                # Tier doesn't accept extra kwargs; fall back.
                try:
                    r = tier.fn(solid, self.tol)
                except Exception as e:
                    r = FitResult(None, 0.0, 0.0, tier.name, f"error: {e}")
            except Exception as e:
                r = FitResult(None, 0.0, 0.0, tier.name, f"error: {e}")
            dt = time.time() - t0

            comp = r.completeness if r and r.code_body else 0.0
            acc = r.accuracy if r and r.code_body else 0.0
            diag.tier_runs.append((tier.name, dt, comp, acc))

            q = (comp + acc) / 2.0
            if self.verbose:
                status = "ok" if r and r.code_body else "fail"
                print(f"    [{tier.name}] {dt:.1f}s  {status}  "
                      f"c={comp*100:.2f}%  a={acc*100:.2f}%", flush=True)

            if q > best_q:
                best = r
                best_q = q
                diag.best_tier = tier.name
                diag.best_comp = comp
                diag.best_acc = acc

            # Short-circuit if target met.
            if best_q >= target_q_equiv:
                if self.verbose:
                    print(f"    target met (sym~{(1-best_q)*200:.2f}%); "
                          f"stopping", flush=True)
                break

        diag.total_time = time.time() - total_t0
        return best, diag

    # -------------------------------------------------------------------
    # Multi-solid path (with optional parallelism)
    # -------------------------------------------------------------------
    def fit_part(
        self,
        solids: list,
        workers: int = 1,
    ) -> List[Tuple[FitResult, FitDiag]]:
        """Fit all solids in a part. If workers>1, run in parallel via
        multiprocessing.Pool. Returns list of (FitResult, FitDiag) in
        original solid order."""
        if workers <= 1 or len(solids) <= 1:
            return [self.fit_solid(s, i) for i, s in enumerate(solids)]

        # Parallel: pool with workers processes.
        # Each worker needs to re-import build123d (cold start ~30s in this
        # container). Only worth it for parts with many solids.
        workers = min(workers, len(solids))
        if self.verbose:
            print(f"  parallel: {len(solids)} solids across {workers} workers", flush=True)

        # Serialize solids as STEP strings for cross-process transport.
        # Easier & more reliable than pickling build123d objects.
        payloads = []
        for i, s in enumerate(solids):
            try:
                payloads.append((i, _solid_to_step_bytes(s),
                                 self.tol, self.target_sym_pct,
                                 self.enable_llm))
            except Exception:
                payloads.append((i, None, self.tol, self.target_sym_pct,
                                 self.enable_llm))

        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            results = pool.map(_pool_worker, payloads)
        # results are (idx, FitResult_as_dict, FitDiag_as_dict)
        out: List[Optional[Tuple[FitResult, FitDiag]]] = [None] * len(solids)
        for idx, fr_dict, diag_dict in results:
            fr = _dict_to_fitresult(fr_dict)
            diag = _dict_to_diag(diag_dict)
            out[idx] = (fr, diag)
        return out  # type: ignore


# ---------------------------------------------------------------------------
# Parallel-worker helpers (module-level so they pickle)
# ---------------------------------------------------------------------------

def _solid_to_step_bytes(solid) -> bytes:
    """Export a single solid to STEP bytes for cross-process transport."""
    import tempfile
    from build123d import export_step
    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
        tmp = f.name
    try:
        export_step(solid, tmp)
        with open(tmp, "rb") as fr:
            return fr.read()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _step_bytes_to_solid(data: bytes):
    """Reconstruct a single solid from STEP bytes."""
    import tempfile
    from build123d import import_step
    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        solids = list(import_step(tmp).solids())
        return solids[0] if solids else None
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _fitresult_to_dict(fr: FitResult) -> dict:
    return {
        "code_body": fr.code_body,
        "completeness": fr.completeness,
        "accuracy": fr.accuracy,
        "kind": fr.kind,
        "details": fr.details,
    }


def _dict_to_fitresult(d: dict) -> FitResult:
    return FitResult(
        code_body=d.get("code_body"),
        completeness=d.get("completeness", 0.0),
        accuracy=d.get("accuracy", 0.0),
        kind=d.get("kind", "none"),
        details=d.get("details", ""),
    )


def _diag_to_dict(diag: FitDiag) -> dict:
    return {
        "solid_idx": diag.solid_idx,
        "source_vol": diag.source_vol,
        "tier_runs": diag.tier_runs,
        "best_tier": diag.best_tier,
        "best_comp": diag.best_comp,
        "best_acc": diag.best_acc,
        "total_time": diag.total_time,
    }


def _dict_to_diag(d: dict) -> FitDiag:
    diag = FitDiag(
        solid_idx=d.get("solid_idx", 0),
        source_vol=d.get("source_vol", 0.0),
        tier_runs=d.get("tier_runs", []),
        best_tier=d.get("best_tier"),
        best_comp=d.get("best_comp", 0.0),
        best_acc=d.get("best_acc", 0.0),
        total_time=d.get("total_time", 0.0),
    )
    return diag


def _pool_worker(args):
    """Subprocess entrypoint — fits one solid via a fresh FitEngine."""
    idx, step_bytes, tol, target_sym_pct, enable_llm = args
    if step_bytes is None:
        return (idx,
                _fitresult_to_dict(FitResult(None, 0.0, 0.0, "none",
                                             "serialization failed")),
                _diag_to_dict(FitDiag(solid_idx=idx, source_vol=0.0)))
    try:
        set_force_primitives(True)
        solid = _step_bytes_to_solid(step_bytes)
        if solid is None:
            return (idx,
                    _fitresult_to_dict(FitResult(None, 0.0, 0.0, "none",
                                                 "deserialization failed")),
                    _diag_to_dict(FitDiag(solid_idx=idx, source_vol=0.0)))
        engine = FitEngine(tol=tol, target_sym_pct=target_sym_pct,
                           enable_llm=enable_llm, verbose=False)
        fr, diag = engine.fit_solid(solid, solid_idx=idx)
        return (idx, _fitresult_to_dict(fr), _diag_to_dict(diag))
    except Exception as e:
        return (idx,
                _fitresult_to_dict(FitResult(None, 0.0, 0.0, "none",
                                             f"worker error: {e}")),
                _diag_to_dict(FitDiag(solid_idx=idx, source_vol=0.0)))
