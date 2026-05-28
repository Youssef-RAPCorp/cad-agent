"""
Emit a build123d reconstruction from a ScanResult.

The emitter strategy is:
  - For each congruence group with a detected grid: export ONE canonical
    solid to its own STEP file, then emit import_step + GridLocations.
  - For each ungrouped solid: export it to its own STEP file, emit
    import_step.
  - Fuse everything into a final Part.
  - Optionally re-export the final Part to a single STEP/STL for
    verification.

The emitted script references companion STEP files (one per group + one
per ungrouped solid) that must be kept alongside the script.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .scanner import ScanResult, CongruenceGroup, GridPattern


@dataclass
class EmissionResult:
    script_path: str
    component_step_paths: list[str]
    total_groups: int
    total_individual_solids: int
    # Breakdown: how many solids were emitted by each strategy
    n_box: int = 0
    n_box_fillets: int = 0
    n_cylinder: int = 0
    n_extrude: int = 0
    n_box_forced: int = 0   # bbox-Box approximation from --force-primitives
    n_brep_exact: int = 0   # exact BREP embedding (complex shapes, 0% sym-diff)
    n_halfspace_hull: int = 0   # halfspace hull+cavity decomposition
    n_llm: int = 0   # LLM-refined recipes
    n_voxel: int = 0   # voxel-based reconstruction
    n_face_extrude: int = 0   # face-extrude subtraction
    n_axis_stack: int = 0   # axis-stack 2D profile extrudes
    n_import_step: int = 0
    # List of (solid_index, reason) tuples for solids that fell back.
    # Useful for --verbose diagnosis of why primitives failed.
    fallback_reasons: list = None


def _fmt(x: float) -> str:
    return f"{x:.9g}"


def _prefit_worker(args):
    """Subprocess entry: load one solid from the source STEP by index,
    run fit_primitive, return (solid_index, FitResult-as-dict).

    Loads the full source STEP and picks solid at the given index.
    Each worker cold-imports build123d (~5-30s depending on machine).
    Returns a dict so it pickles cleanly across processes.
    """
    source_path, solid_idx, force_primitives, verbose_worker = args
    try:
        from .fitter import (fit_primitive, set_force_primitives,
                             set_verbose)
        from .scanner import _load_solids
        set_force_primitives(bool(force_primitives))
        # Worker stays quiet regardless: the parent handles output.
        set_verbose(False)
        solids = _load_solids(source_path)
        if solid_idx >= len(solids):
            return (solid_idx, {
                "code_body": None, "completeness": 0.0, "accuracy": 0.0,
                "kind": "none",
                "details": f"worker: solid index {solid_idx} out of range",
            })
        fit = fit_primitive(solids[solid_idx], tol=0.01)
        return (solid_idx, {
            "code_body": fit.code_body,
            "completeness": fit.completeness,
            "accuracy": fit.accuracy,
            "kind": fit.kind,
            "details": fit.details,
        })
    except Exception as e:
        return (solid_idx, {
            "code_body": None, "completeness": 0.0, "accuracy": 0.0,
            "kind": "none",
            "details": f"worker exception: {type(e).__name__}: {e}",
        })


def _prefit_in_parallel(source_path, n_solids, workers,
                        force_primitives, verbose, solids_list):
    """Run fit_primitive for all solids in parallel. Returns
    {id(solid): FitResult} keyed by the parent-process solid identity.
    """
    import multiprocessing as mp
    from .fitter import FitResult
    ctx = mp.get_context("spawn")
    tasks = [(source_path, i, force_primitives, verbose)
             for i in range(n_solids)]
    cache: dict[int, object] = {}
    with ctx.Pool(workers) as pool:
        for solid_idx, fr_dict in pool.imap_unordered(_prefit_worker, tasks):
            fit = FitResult(
                code_body=fr_dict.get("code_body"),
                completeness=fr_dict.get("completeness", 0.0),
                accuracy=fr_dict.get("accuracy", 0.0),
                kind=fr_dict.get("kind", "none"),
                details=fr_dict.get("details", ""),
            )
            if 0 <= solid_idx < len(solids_list):
                cache[id(solids_list[solid_idx])] = fit
            if verbose:
                if fit.code_body is not None:
                    print(f"    [worker] solid #{solid_idx}: {fit.kind} "
                          f"c={fit.completeness*100:.2f}% "
                          f"a={fit.accuracy*100:.2f}%", flush=True)
                else:
                    print(f"    [worker] solid #{solid_idx}: FAIL "
                          f"({fit.details[:50]})", flush=True)
    return cache


def _rename_part_variable(code: str, new_var: str) -> str:
    """Rename the fitter's _part variable to a unique name.

    The fitter emits code that ends with either:
      - `with BuildPart() as _part: ...` (Box, Cylinder)
      - `_part = extrude(_sk.sketch, amount=...)` (Extrude)

    We do a simple text replace. Only `_part` is used in fitter code,
    so word-boundary isn't strictly needed, but we handle it safely.
    """
    import re
    return re.sub(r"\b_part\b", new_var, code)


def _emit_whole_source(scan: ScanResult, out_dir: str, script_name: str,
                        export_step: Optional[str],
                        export_stl: Optional[str],
                        verbose: bool) -> EmissionResult:
    """Fallback emitter: reference the whole source STEP and extract
    solids by index. Preserves geometry exactly — no per-solid round-trip.

    Produces a script that:
      1. Imports the source STEP (copied next to the script).
      2. Gets the list of source solids.
      3. Emits one `add(source_solids[i])` per solid, WITHOUT any tiling.
         (Tiling would require copying the solid, and copies go through
         the same export-import that caused problems. Simpler to just
         reference all source solids in-place.)
    """
    import shutil
    src_basename = os.path.basename(scan.source_path)
    src_copy = os.path.join(out_dir, src_basename)
    if not os.path.exists(src_copy):
        shutil.copy(scan.source_path, src_copy)

    script_lines = [
        '"""Auto-generated build123d reconstruction (whole-source fallback)."""',
        "from build123d import (import_step, Compound, export_step, export_stl)",
        "",
        f"# Source: {scan.source_path}",
        f"# Source volume: {scan.total_volume:.6f} mm^3",
        "",
        f"_src = import_step(r'{src_basename}')",
        "_all_solids = list(_src.solids())",
        "result = Compound(children=_all_solids)",
        "",
    ]
    if export_step:
        script_lines.append(f'export_step(result, r"{export_step}")')
    if export_stl:
        script_lines.append(f'export_stl(result, r"{export_stl}")')

    script_path = os.path.join(out_dir, script_name)
    with open(script_path, "w") as f:
        f.write("\n".join(script_lines) + "\n")

    if verbose:
        print(f"[emit] WHOLE-SOURCE fallback mode")
        print(f"[emit] script: {script_path}")
        print(f"[emit] source copied to: {src_copy}")

    return EmissionResult(
        script_path=script_path,
        component_step_paths=[src_copy],
        total_groups=0,
        total_individual_solids=len(scan.solids_info),
    )


def emit_recipe(scan: ScanResult,
                out_dir: str,
                script_name: str = "recipe.py",
                export_step: Optional[str] = None,
                export_stl: Optional[str] = None,
                verbose: bool = True,
                whole_source_fallback: bool = False,
                force_primitives: Optional[bool] = None,
                workers: int = 1) -> EmissionResult:
    """Emit the reconstruction script and its companion STEP files.

    All companion files and the script are written to `out_dir`. The
    script uses relative paths so the whole directory is portable.

    `whole_source_fallback`: if True, skip per-solid export and emit a
    script that imports the whole source and references solids by index.
    This avoids OCCT's occasional export-import nondeterminism on complex
    shapes at the cost of requiring the source STEP alongside the recipe.

    `force_primitives`: if True, NEVER emit `import_step()` fallbacks.
    Solids whose fit returned None cause the whole emission to raise a
    RuntimeError with a list of failing solids. If None, reads the
    fitter module's _FORCE_PRIMITIVES global (set by --force-primitives).

    `workers`: number of parallel worker processes for per-solid fitting.
    With workers=1 (default) fitting is serial inline with emission.
    With workers>1, all solids are pre-fit in a process pool BEFORE the
    emission loop runs; the emission loop then reads cached results.
    Each worker cold-imports build123d, so parallelism only pays off
    when there are many solids and each solid's fit is slow.
    """
    from build123d import import_step as b_import_step
    from build123d import export_step as b_export_step
    from build123d import Location, Vector

    # Honor the fitter module's global unless caller overrode.
    if force_primitives is None:
        try:
            from . import fitter as _fm
            force_primitives = bool(getattr(_fm, "_FORCE_PRIMITIVES", False))
        except Exception:
            force_primitives = False

    os.makedirs(out_dir, exist_ok=True)

    # Wrap fit_primitive with per-solid verbose progress when requested.
    # Without this wrapper, users see nothing between "scan done" and
    # the summary line — even on parts with 15+ solids that take
    # minutes to fit. The wrapper prints one line per solid showing
    # which solid is being attempted, which tier won, and the time.
    import time as _time
    _fit_counter = {"n": 0}
    # Cache: id(solid) -> FitResult, populated by the optional
    # parallel pre-fit pass. Lookup is by identity since build123d
    # Solids don't have stable hashes; the same instance is passed
    # to both pre-fit and emission loops.
    _prefit_cache: dict[int, object] = {}

    def _fit_with_progress(solid, label, tol=0.01):
        _fit_counter["n"] += 1
        n = _fit_counter["n"]
        cached = _prefit_cache.get(id(solid))
        if verbose:
            try:
                v = solid.volume
                nf = len(list(solid.faces()))
            except Exception:
                v, nf = 0.0, 0
            tag = " [cached]" if cached is not None else ""
            # Full line (with newline) so Windows terminals flush
            # immediately. The prior end=" " meant users saw nothing
            # for 30+ seconds while the fit ran. Better to get two
            # lines per solid than zero.
            print(f"  [fit {n}]{tag} {label} vol={v:.3f} faces={nf} "
                  f"-> fitting...", flush=True)
        t0 = _time.time()
        if cached is not None:
            fit = cached
        else:
            fit = fit_primitive(solid, tol=tol)
        dt = _time.time() - t0
        if verbose:
            if fit.code_body is not None:
                print(f"  [fit {n}] done: {fit.kind} "
                      f"c={fit.completeness*100:.2f}% "
                      f"a={fit.accuracy*100:.2f}% ({dt:.1f}s)", flush=True)
            else:
                print(f"  [fit {n}] FAIL ({dt:.1f}s): "
                      f"{fit.details[:60]}", flush=True)
        return fit

    if whole_source_fallback:
        return _emit_whole_source(scan, out_dir, script_name,
                                   export_step, export_stl, verbose)

    # Reload source to get solids we can export individually.
    # This can take a few seconds on large STEP files; note it.
    if verbose:
        print(f"  [emit] reloading source to enumerate solids...",
              flush=True)
    from .scanner import _load_solids
    solids = _load_solids(scan.source_path)
    if verbose:
        print(f"  [emit] {len(solids)} solids loaded; "
              f"starting per-solid fits...", flush=True)

    # Parallel pre-fit pass: fit all original solids in a process pool
    # before the emission loop starts. Emission then reads from cache.
    if workers > 1 and len(solids) > 1:
        if verbose:
            print(f"  [parallel] pre-fitting {len(solids)} solids "
                  f"across {workers} workers...", flush=True)
        t_par = _time.time()
        _prefit_cache.update(
            _prefit_in_parallel(scan.source_path, len(solids),
                                min(workers, len(solids)),
                                force_primitives, verbose, solids)
        )
        if verbose:
            print(f"  [parallel] pre-fit done in "
                  f"{_time.time()-t_par:.1f}s "
                  f"({len(_prefit_cache)} results cached)", flush=True)

    component_paths: list[str] = []
    emitted_blocks: list[str] = []
    covered: set[int] = set()

    # Each group produces either:
    #   - ADD ONE canonical solid + GridLocations (if grid detected)
    #   - ADD every member individually (if no grid)
    #   - ADD the single member (if group has 1 member)

    from .fitter import fit_primitive

    # Per-kind counters. For grid groups we count per-member (so a 7-pin
    # grid fit as extrude counts as 7 extrudes, since 7 solids are
    # effectively rebuilt from the emitted primitive).
    n_box = 0
    n_box_fillets = 0
    n_cylinder = 0
    n_extrude = 0
    n_box_forced = 0
    n_brep_exact = 0
    n_halfspace_hull = 0
    n_llm = 0
    n_voxel = 0
    n_face_extrude = 0
    n_axis_stack = 0
    n_import_step = 0
    fallback_reasons = []  # list of (solid_idx, reason)

    group_idx = 0
    for group in scan.groups:
        group_idx += 1
        if group.grid is not None:
            # Grid group: try to fit canonical to a primitive first.
            # If that works, emit the primitive code + GridLocations
            # (no STEP file needed). Otherwise fall back to the old
            # export-canonical-STEP approach.
            canon_idx = group.grid.start_index
            canon_solid = solids[canon_idx]
            bb = canon_solid.bounding_box()
            center = ((bb.min.X + bb.max.X) / 2.0,
                      (bb.min.Y + bb.max.Y) / 2.0,
                      (bb.min.Z + bb.max.Z) / 2.0)

            # Try primitive fit on a canonical-at-origin version of the
            # solid so the emitted code produces a shape centered at the
            # origin that we can then translate via Locations.
            canon_at_origin = canon_solid.moved(
                Location(Vector(-center[0], -center[1], -center[2])))
            fit = _fit_with_progress(
                canon_at_origin,
                f"group{group_idx:02d} canonical",
                tol=0.01,
            )

            axis = group.grid.axis
            pitch = group.grid.pitch
            count = group.grid.count
            # Anchor position: such that grid position k=0 == original center
            anchor = list(group.grid.start_center)
            anchor[axis] += pitch * (count - 1) / 2.0

            if fit.code_body is not None:
                # Emit fitter's code to build _canon_{group_idx} from
                # primitives, then tile with GridLocations.
                # The fitter emits "with BuildPart() as _part: ..." or
                # a similar block that ends with `_part`. We rename
                # _part -> _canon_{group_idx}.
                canon_code = _rename_part_variable(
                    fit.code_body, f"_canon_{group_idx}")
                header = (f"# Group {group_idx}: {count} copies along "
                          f"{'XYZ'[axis]} at pitch {pitch:.4f} mm "
                          f"(canonical = {fit.kind})\n")

                # Tile
                if axis == 0:
                    tile_code = (
                        f"with BuildPart() as _tile_{group_idx}:\n"
                        f"    with Locations(({_fmt(anchor[0])}, "
                        f"{_fmt(anchor[1])}, {_fmt(anchor[2])})):\n"
                        f"        with GridLocations({_fmt(pitch)}, 1, "
                        f"{count}, 1):\n"
                        f"            add(_canon_{group_idx}.part if "
                        f"hasattr(_canon_{group_idx}, 'part') else "
                        f"_canon_{group_idx})\n"
                        f"_parts.append(_tile_{group_idx}.part)\n"
                    )
                elif axis == 1:
                    tile_code = (
                        f"with BuildPart() as _tile_{group_idx}:\n"
                        f"    with Locations(({_fmt(anchor[0])}, "
                        f"{_fmt(anchor[1])}, {_fmt(anchor[2])})):\n"
                        f"        with GridLocations(1, {_fmt(pitch)}, "
                        f"1, {count}):\n"
                        f"            add(_canon_{group_idx}.part if "
                        f"hasattr(_canon_{group_idx}, 'part') else "
                        f"_canon_{group_idx})\n"
                        f"_parts.append(_tile_{group_idx}.part)\n"
                    )
                else:
                    locs = [(anchor[0], anchor[1],
                             group.grid.start_center[2] + k * pitch)
                            for k in range(count)]
                    loc_code = ", ".join(
                        f"({_fmt(l[0])}, {_fmt(l[1])}, {_fmt(l[2])})"
                        for l in locs
                    )
                    tile_code = (
                        f"with BuildPart() as _tile_{group_idx}:\n"
                        f"    with Locations({loc_code}):\n"
                        f"        add(_canon_{group_idx}.part if "
                        f"hasattr(_canon_{group_idx}, 'part') else "
                        f"_canon_{group_idx})\n"
                        f"_parts.append(_tile_{group_idx}.part)\n"
                    )

                block = header + canon_code + tile_code
                emitted_blocks.append(block)
                covered.update(group.member_indices)
                # Count: all `count` members are rebuilt from this primitive
                if fit.kind == "box":
                    n_box += count
                elif fit.kind == "box_fillets":
                    n_box_fillets += count
                elif fit.kind == "cylinder":
                    n_cylinder += count
                elif fit.kind == "extrude":
                    n_extrude += count
                elif fit.kind == "box_forced":
                    n_box_forced += count
                elif fit.kind == "brep_exact":
                    n_brep_exact += count
                elif fit.kind == "halfspace_hull":
                    n_halfspace_hull += count
                elif fit.kind == "llm":
                    n_llm += count
                elif fit.kind == "voxel":
                    n_voxel += count
                elif fit.kind == "face_extrude":
                    n_face_extrude += count
                elif fit.kind == "axis_stack":
                    n_axis_stack += count
                continue

            # Fitter failed: record why, then either skip (force_primitives)
            # or export canonical STEP and emit import_step fallback.
            fallback_reasons.append(
                (canon_idx, f"group canonical: {fit.details}"))
            if force_primitives:
                # Don't emit import_step fallback. Mark this group as
                # unrecoverable; we'll raise at the end.
                covered.update(group.member_indices)
                continue
            canon_name = f"group{group_idx:02d}_canonical.step"
            canon_path = os.path.join(out_dir, canon_name)
            b_export_step(canon_at_origin, canon_path)
            component_paths.append(canon_path)

            if axis == 0:
                block = (
                    f"# Group {group_idx}: {count} copies along X at "
                    f"pitch {pitch:.4f} mm (canonical = import_step)\n"
                    f"_canon_{group_idx} = import_step(r'{canon_name}')\n"
                    f"with BuildPart() as _tile_{group_idx}:\n"
                    f"    with Locations(({_fmt(anchor[0])}, "
                    f"{_fmt(anchor[1])}, {_fmt(anchor[2])})):\n"
                    f"        with GridLocations({_fmt(pitch)}, 1, "
                    f"{count}, 1):\n"
                    f"            add(_canon_{group_idx})\n"
                    f"_parts.append(_tile_{group_idx}.part)\n"
                )
            elif axis == 1:
                block = (
                    f"# Group {group_idx}: {count} copies along Y at "
                    f"pitch {pitch:.4f} mm (canonical = import_step)\n"
                    f"_canon_{group_idx} = import_step(r'{canon_name}')\n"
                    f"with BuildPart() as _tile_{group_idx}:\n"
                    f"    with Locations(({_fmt(anchor[0])}, "
                    f"{_fmt(anchor[1])}, {_fmt(anchor[2])})):\n"
                    f"        with GridLocations(1, {_fmt(pitch)}, "
                    f"1, {count}):\n"
                    f"            add(_canon_{group_idx})\n"
                    f"_parts.append(_tile_{group_idx}.part)\n"
                )
            else:
                locs = [(anchor[0], anchor[1],
                         group.grid.start_center[2] + k * pitch)
                        for k in range(count)]
                loc_code = ", ".join(
                    f"({_fmt(l[0])}, {_fmt(l[1])}, {_fmt(l[2])})"
                    for l in locs
                )
                block = (
                    f"# Group {group_idx}: {count} copies along Z at "
                    f"pitch {pitch:.4f} mm (canonical = import_step)\n"
                    f"_canon_{group_idx} = import_step(r'{canon_name}')\n"
                    f"with BuildPart() as _tile_{group_idx}:\n"
                    f"    with Locations({loc_code}):\n"
                    f"        add(_canon_{group_idx})\n"
                    f"_parts.append(_tile_{group_idx}.part)\n"
                )
            emitted_blocks.append(block)
            covered.update(group.member_indices)
            n_import_step += count
        else:
            # Non-grid group: try primitive fit per solid, else export
            for member_idx in group.member_indices:
                if member_idx in covered:
                    continue
                solid = solids[member_idx]
                fit = _fit_with_progress(
                    solid, f"group{group_idx:02d} member#{member_idx}",
                    tol=0.01,
                )
                if fit.code_body is not None:
                    # Rename _part to a unique variable
                    var = f"_solid_{member_idx:03d}"
                    code = _rename_part_variable(fit.code_body, var)
                    block = (
                        f"# Individual solid {member_idx} "
                        f"(primitive: {fit.kind})\n"
                        f"{code}"
                        f"_parts.append({var}.part if hasattr({var}, "
                        f"'part') else {var})\n"
                    )
                    emitted_blocks.append(block)
                    covered.add(member_idx)
                    if fit.kind == "box":
                        n_box += 1
                    elif fit.kind == "box_fillets":
                        n_box_fillets += 1
                    elif fit.kind == "cylinder":
                        n_cylinder += 1
                    elif fit.kind == "extrude":
                        n_extrude += 1
                    elif fit.kind == "box_forced":
                        n_box_forced += 1
                    elif fit.kind == "brep_exact":
                        n_brep_exact += 1
                    elif fit.kind == "halfspace_hull":
                        n_halfspace_hull += 1
                    elif fit.kind == "llm":
                        n_llm += 1
                    elif fit.kind == "voxel":
                        n_voxel += 1
                    elif fit.kind == "face_extrude":
                        n_face_extrude += 1
                    elif fit.kind == "axis_stack":
                        n_axis_stack += 1
                    continue

                fallback_reasons.append((member_idx, fit.details))
                if force_primitives:
                    covered.add(member_idx)
                    continue
                comp_name = f"solid_{member_idx:03d}.step"
                comp_path = os.path.join(out_dir, comp_name)
                b_export_step(solid, comp_path)
                component_paths.append(comp_path)
                block = (
                    f"# Individual solid {member_idx} (import_step)\n"
                    f"_parts.append(import_step(r'{comp_name}'))\n"
                )
                emitted_blocks.append(block)
                covered.add(member_idx)
                n_import_step += 1

    # Any solid not already covered
    for i in range(len(solids)):
        if i in covered:
            continue
        solid = solids[i]
        fit = _fit_with_progress(solid, f"uncategorized solid #{i}",
                                  tol=0.01)
        if fit.code_body is not None:
            var = f"_solid_{i:03d}"
            code = _rename_part_variable(fit.code_body, var)
            block = (
                f"# Uncategorized solid {i} (primitive: {fit.kind})\n"
                f"{code}"
                f"_parts.append({var}.part if hasattr({var}, 'part') "
                f"else {var})\n"
            )
            emitted_blocks.append(block)
            covered.add(i)
            if fit.kind == "box":
                n_box += 1
            elif fit.kind == "box_fillets":
                n_box_fillets += 1
            elif fit.kind == "cylinder":
                n_cylinder += 1
            elif fit.kind == "extrude":
                n_extrude += 1
            elif fit.kind == "box_forced":
                n_box_forced += 1
            elif fit.kind == "brep_exact":
                n_brep_exact += 1
            elif fit.kind == "halfspace_hull":
                n_halfspace_hull += 1
            elif fit.kind == "llm":
                n_llm += 1
            elif fit.kind == "voxel":
                n_voxel += 1
            elif fit.kind == "face_extrude":
                n_face_extrude += 1
            elif fit.kind == "axis_stack":
                n_axis_stack += 1
            continue
        fallback_reasons.append((i, fit.details))
        if force_primitives:
            covered.add(i)
            continue
        comp_name = f"solid_{i:03d}.step"
        comp_path = os.path.join(out_dir, comp_name)
        b_export_step(solid, comp_path)
        component_paths.append(comp_path)
        block = (
            f"# Uncategorized solid {i} (import_step)\n"
            f"_parts.append(import_step(r'{comp_name}'))\n"
        )
        emitted_blocks.append(block)
        covered.add(i)
        n_import_step += 1

    # If force_primitives and any fits failed, raise with a report.
    if force_primitives and fallback_reasons:
        reasons_str = "\n".join(
            f"  solid #{idx}: {reason}" for idx, reason in fallback_reasons
        )
        raise RuntimeError(
            f"force_primitives: {len(fallback_reasons)} solid(s) could "
            f"not be reconstructed from parametric primitives:\n"
            f"{reasons_str}\n"
            f"No recipe emitted. Consider: running without --strict to "
            f"allow import_step fallback, or enabling the LLM tier "
            f"(OPENAI_API_KEY) to repair failing solids."
        )

    # Choose header text + imports based on whether import_step was used.
    if force_primitives or n_import_step == 0:
        header_doc = (
            '"""Auto-generated build123d reconstruction (solid-aware).\n\n'
            'Fully parametric: every solid is built from build123d\n'
            'primitives (Box, Cylinder, extrude, loft, plane splits, etc.).\n'
            'No import_step fallbacks, no companion STEP files.\n'
            '"""'
        )
        imports_line = "    export_step, export_stl,"
    else:
        header_doc = (
            '"""Auto-generated build123d reconstruction (solid-aware).\n\n'
            'Primitive-first: each solid is built from build123d primitives\n'
            '(Box, Cylinder, extrude of polyline sketch) where possible.\n'
            'Complex solids fall back to import_step from companion files.\n'
            '"""'
        )
        imports_line = "    import_step, export_step, export_stl,"

    # Build the script
    script_lines = [
        header_doc,
        "from build123d import (",
        "    BuildPart, BuildSketch, BuildLine, Polyline,",
        "    Line, RadiusArc, Locations, GridLocations, Plane, Mode, Keep,",
        "    Box, Cylinder, Circle, extrude, loft, make_face, add, fillet,",
        "    Edge, Wire, Face, Vector,",
        imports_line,
        ")",
        "",
        f"# Source: {scan.source_path}",
        f"# Source volume: {scan.total_volume:.6f} mm^3",
        f"# {len(scan.groups)} congruence groups, "
        f"{sum(1 for g in scan.groups if g.grid is not None)} grid groups",
        "",
        "_parts = []",
        "",
    ]
    script_lines.extend(emitted_blocks)
    script_lines.extend([
        "",
        "# Collect all solids (as a Compound, not fused -- preserves",
        "# overlapping-solid volumes as in the original source).",
        "from build123d import Compound",
        "_all_solids = []",
        "for _p in _parts:",
        "    try:",
        "        _all_solids.extend(list(_p.solids()))",
        "    except Exception:",
        "        _all_solids.append(_p)",
        "result = Compound(children=_all_solids)",
        "",
    ])

    if export_step:
        script_lines.append(f'export_step(result, r"{export_step}")')
    if export_stl:
        script_lines.append(f'export_stl(result, r"{export_stl}")')

    script_path = os.path.join(out_dir, script_name)
    with open(script_path, "w") as f:
        f.write("\n".join(script_lines) + "\n")

    n_grid = sum(1 for g in scan.groups if g.grid is not None)
    n_indiv = len(solids) - sum(len(g.member_indices)
                                 for g in scan.groups if g.grid is not None)

    if verbose:
        print(f"[emit] script: {script_path}")
        print(f"[emit] {len(component_paths)} component STEP files")
        print(f"[emit] {n_grid} grid groups, {n_indiv} individual solids")
        total_fit = n_box + n_box_fillets + n_cylinder + n_extrude
        print(f"[emit] primitives: {total_fit} "
              f"(box={n_box}, box_fillets={n_box_fillets}, "
              f"cylinder={n_cylinder}, extrude={n_extrude})")
        if n_brep_exact:
            print(f"[emit] exact BREP reconstructions: {n_brep_exact}")
        if n_halfspace_hull:
            print(f"[emit] halfspace hull+cavity reconstructions: "
                  f"{n_halfspace_hull}")
        if n_llm:
            print(f"[emit] LLM-refined reconstructions: {n_llm}")
        if n_voxel:
            print(f"[emit] voxel reconstructions: {n_voxel}")
        if n_face_extrude:
            print(f"[emit] face-extrude reconstructions: {n_face_extrude}")
        if n_axis_stack:
            print(f"[emit] axis-stack reconstructions: {n_axis_stack}")
        if n_box_forced:
            print(f"[emit] forced bbox-Box approximations: "
                  f"{n_box_forced}")
        print(f"[emit] import_step fallbacks: {n_import_step}")
        if fallback_reasons:
            print(f"[emit] fallback details:")
            for idx, reason in fallback_reasons:
                print(f"  solid {idx}: {reason}")

    return EmissionResult(
        script_path=script_path,
        component_step_paths=component_paths,
        total_groups=n_grid,
        total_individual_solids=n_indiv,
        n_box=n_box,
        n_box_fillets=n_box_fillets,
        n_cylinder=n_cylinder,
        n_extrude=n_extrude,
        n_box_forced=n_box_forced,
        n_brep_exact=n_brep_exact,
        n_halfspace_hull=n_halfspace_hull,
        n_llm=n_llm,
        n_voxel=n_voxel,
        n_face_extrude=n_face_extrude,
        n_axis_stack=n_axis_stack,
        n_import_step=n_import_step,
        fallback_reasons=fallback_reasons,
    )
