"""fcstd_to_assembly.py — standalone FCStd -> build123d assembly converter.

Reads a FreeCAD .FCStd file and writes a Python script that, when run,
reproduces the entire assembly as a build123d Compound containing all
constituent solids in their original world-coordinate positions.

This does NOT go through cad_agent3's fitter chain. It's a direct
geometry re-hydration:

  1. Unzip the .FCStd (zip archive)
  2. Extract each PartShape*.brp (OpenCascade native BREP files)
  3. Read each with BRepTools, walk topology to collect every solid
  4. Re-serialize each solid to a companion .brep file in the output dir
  5. Emit a Python script that loads those .brep files and unions them
     into one Compound

The emitted script reproduces the assembly EXACTLY (byte-perfect
geometry). Trade-off: the script references companion .brep files; it
is not fully-parametric "pure algebra". Use cad_agent3 proper if you
want parametric primitives.

Usage (CLI):
    python fcstd_to_assembly.py path/to/file.FCStd --out out_dir

Usage (programmatic):
    from cad_agent3.fcstd_to_assembly import convert
    result = convert("file.FCStd", "out_dir")
    print(result.summary())
    # then:  cd out_dir && python assembly.py  -> produces assembly.step
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, List


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ConversionResult:
    source_path: str
    out_dir: str
    script_path: str
    brep_dir: str
    n_partshapes: int        # PartShape*.brp files in the FCStd
    n_solids_extracted: int  # total solids after topology walk
    total_volume_mm3: float
    bbox_size_mm: tuple      # (X, Y, Z) extent of the assembly
    load_time_s: float
    write_time_s: float

    def summary(self) -> str:
        x, y, z = self.bbox_size_mm
        return (
            f"FCStd -> assembly conversion\n"
            f"  source:             {self.source_path}\n"
            f"  output dir:         {self.out_dir}\n"
            f"  assembly script:    {self.script_path}\n"
            f"  companion BREPs:    {self.brep_dir}\n"
            f"  input PartShapes:   {self.n_partshapes}\n"
            f"  extracted solids:   {self.n_solids_extracted}\n"
            f"  total volume:       {self.total_volume_mm3:.3f} mm^3\n"
            f"  assembly bbox:      {x:.1f} x {y:.1f} x {z:.1f} mm\n"
            f"  load time:          {self.load_time_s:.1f}s\n"
            f"  write time:         {self.write_time_s:.1f}s"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sort_brp_names(names: List[str]) -> List[str]:
    """Sort PartShape.brp, PartShape1.brp, ... in numeric order."""
    def key(name):
        stem = name.rsplit(".", 1)[0]
        digits = ""
        for c in reversed(stem):
            if c.isdigit():
                digits = c + digits
            else:
                break
        return (int(digits) if digits else -1, name)
    return sorted(names, key=key)


def _parse_labels(document_xml_bytes: bytes) -> dict:
    """Parse Document.xml and return {shape_file: label} mapping.
    Returns empty dict on any parse failure — labels are nice-to-have.
    """
    try:
        root = ET.fromstring(document_xml_bytes)
    except Exception:
        return {}
    mapping = {}
    objects_elem = root.find('ObjectData')
    if objects_elem is None:
        return {}
    for obj in objects_elem.findall('Object'):
        shape_file = None
        label = None
        for prop in obj.findall('.//Property'):
            pname = prop.attrib.get('name', '')
            if pname == 'Shape':
                part = prop.find('.//Part')
                if part is not None:
                    shape_file = part.attrib.get('file')
            elif pname == 'Label':
                s = prop.find('.//String')
                if s is not None:
                    label = s.attrib.get('value')
        if shape_file:
            mapping[shape_file] = label or ""
    return mapping


def _safe_label(label: str, fallback: str) -> str:
    """Turn a FreeCAD label into a safe Python identifier fragment."""
    if not label:
        return fallback
    # Strip non-alnum, replace with _
    out = "".join(c if c.isalnum() else "_" for c in label)
    out = out.strip("_")
    if not out:
        return fallback
    if out[0].isdigit():
        out = "_" + out
    return out


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(
    source_path: str,
    out_dir: str,
    script_name: str = "assembly.py",
    brep_subdir: str = "shapes",
    verbose: bool = True,
) -> ConversionResult:
    """Convert a .FCStd file into a standalone build123d assembly script.

    Emitted script structure:

        # assembly.py
        from build123d import Compound, Solid, export_step, export_stl
        from OCP.BRepTools import BRepTools
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS_Shape
        from OCP.TopAbs import TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        import os

        HERE = os.path.dirname(os.path.abspath(__file__))

        def _load(relpath):
            shape = TopoDS_Shape()
            BRepTools.Read_s(shape, os.path.join(HERE, relpath), BRep_Builder())
            out = []
            exp = TopExp_Explorer(shape, TopAbs_SOLID)
            while exp.More():
                out.append(Solid(exp.Current()))
                exp.Next()
            return out

        solids = []
        solids.extend(_load("shapes/0000_SOLID.brep"))
        solids.extend(_load("shapes/0001_COMPOUND.brep"))
        ...

        assembly = Compound(solids)
        if __name__ == "__main__":
            export_step(assembly, os.path.join(HERE, "assembly.step"))
            export_stl(assembly,  os.path.join(HERE, "assembly.stl"))
    """
    from OCP.BRepTools import BRepTools
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Shape
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from build123d import Solid

    os.makedirs(out_dir, exist_ok=True)
    brep_dir_abs = os.path.join(out_dir, brep_subdir)
    os.makedirs(brep_dir_abs, exist_ok=True)

    # ---- Load phase ----
    t_load0 = time.time()
    if verbose:
        print(f"[1/3] Unpacking {source_path}...", flush=True)
    with zipfile.ZipFile(source_path, "r") as zf:
        # Pull Document.xml for labels (optional but nicer output)
        labels = {}
        try:
            with zf.open("Document.xml") as f:
                labels = _parse_labels(f.read())
        except KeyError:
            pass

        all_names = zf.namelist()
        brp_names = [n for n in all_names
                     if n.lower().endswith((".brp", ".brep"))]
        brp_names = _sort_brp_names(brp_names)
        if not brp_names:
            raise ValueError(f"No .brp shapes inside {source_path}")
        n_partshapes = len(brp_names)
        if verbose:
            print(f"      found {n_partshapes} PartShape*.brp files; "
                  f"{len(labels)} have labels", flush=True)

        # Extract every BRP to a temp dir, read each, collect solids.
        collected = []   # list of (serial_index, label, Solid)
        running_idx = 0
        total_volume = 0.0
        bbox_min = [float("inf")] * 3
        bbox_max = [float("-inf")] * 3

        with tempfile.TemporaryDirectory() as tmpdir:
            for brp_idx, name in enumerate(brp_names):
                zf.extract(name, tmpdir)
                src_file = os.path.join(tmpdir, name)
                shape = TopoDS_Shape()
                try:
                    ok = BRepTools.Read_s(shape, src_file, BRep_Builder())
                except Exception as e:
                    if verbose:
                        print(f"      [{brp_idx}] {name}: read err "
                              f"{type(e).__name__}; skipped", flush=True)
                    continue
                if not ok or shape.IsNull():
                    if verbose:
                        print(f"      [{brp_idx}] {name}: null; skipped",
                              flush=True)
                    continue

                group_label = labels.get(name, "") or f"Shape{brp_idx}"
                exp = TopExp_Explorer(shape, TopAbs_SOLID)
                n_in_shape = 0
                while exp.More():
                    try:
                        s = Solid(exp.Current())
                    except Exception:
                        exp.Next()
                        continue
                    try:
                        v = s.volume
                    except Exception:
                        v = 0.0
                    if v <= 1e-9:
                        exp.Next()
                        continue
                    # Update assembly bbox
                    try:
                        bb = s.bounding_box()
                        bbox_min[0] = min(bbox_min[0], bb.min.X)
                        bbox_min[1] = min(bbox_min[1], bb.min.Y)
                        bbox_min[2] = min(bbox_min[2], bb.min.Z)
                        bbox_max[0] = max(bbox_max[0], bb.max.X)
                        bbox_max[1] = max(bbox_max[1], bb.max.Y)
                        bbox_max[2] = max(bbox_max[2], bb.max.Z)
                    except Exception:
                        pass
                    total_volume += v
                    collected.append((running_idx, group_label, s))
                    running_idx += 1
                    n_in_shape += 1
                    exp.Next()

                if verbose and (brp_idx % 20 == 0 or n_in_shape > 3):
                    print(f"      [{brp_idx+1}/{n_partshapes}] {name} "
                          f"(label={group_label!r}): "
                          f"{n_in_shape} solid(s)", flush=True)

    load_time = time.time() - t_load0
    if verbose:
        print(f"      loaded {len(collected)} solids total "
              f"({load_time:.1f}s)", flush=True)

    if not collected:
        raise ValueError(f"No solids extracted from {source_path}")

    # ---- Serialize each solid as its own .brep in the output dir ----
    t_write0 = time.time()
    if verbose:
        print(f"[2/3] Writing {len(collected)} .brep files to "
              f"{brep_dir_abs}...", flush=True)

    brep_rel_paths = []
    for idx, label, solid in collected:
        safe = _safe_label(label, "Solid")
        # Zero-pad so lexical sort matches insertion order
        fname = f"{idx:04d}_{safe}.brep"
        fpath = os.path.join(brep_dir_abs, fname)
        try:
            BRepTools.Write_s(solid.wrapped, fpath)
        except Exception as e:
            if verbose:
                print(f"      skip {fname}: {type(e).__name__}: {e}",
                      flush=True)
            continue
        brep_rel_paths.append(os.path.join(brep_subdir, fname).replace("\\", "/"))

    # ---- Emit the assembly script ----
    if verbose:
        print(f"[3/3] Writing assembly script...", flush=True)

    script_abs = os.path.join(out_dir, script_name)
    src_basename = os.path.basename(source_path)
    bbox_size = (
        bbox_max[0] - bbox_min[0] if bbox_max[0] > bbox_min[0] else 0.0,
        bbox_max[1] - bbox_min[1] if bbox_max[1] > bbox_min[1] else 0.0,
        bbox_max[2] - bbox_min[2] if bbox_max[2] > bbox_min[2] else 0.0,
    )

    script_lines = [
        '"""Auto-generated build123d assembly.',
        '',
        f'Reconstructs {src_basename} from companion .brep files.',
        f'Each shape is loaded via OpenCascade\'s BRepTools and all solids',
        f'are wrapped into one build123d Compound, preserving the original',
        f'world-coordinate positions.',
        '',
        f'Source: {src_basename}',
        f'Solids: {len(collected)}',
        f'Total volume: {total_volume:.3f} mm^3',
        f'Bounding box: {bbox_size[0]:.1f} x {bbox_size[1]:.1f} x '
        f'{bbox_size[2]:.1f} mm',
        '"""',
        'import os',
        '',
        'from build123d import Compound, Solid, export_step, export_stl',
        'from OCP.BRepTools import BRepTools',
        'from OCP.BRep import BRep_Builder',
        'from OCP.TopoDS import TopoDS_Shape',
        'from OCP.TopAbs import TopAbs_SOLID',
        'from OCP.TopExp import TopExp_Explorer',
        '',
        'HERE = os.path.dirname(os.path.abspath(__file__))',
        '',
        '',
        'def _load_solids_from_brep(relpath):',
        '    """Read a .brep file and return every TopoDS_SOLID inside,',
        '    wrapped as build123d Solid objects.',
        '    """',
        '    shape = TopoDS_Shape()',
        '    ok = BRepTools.Read_s(',
        '        shape, os.path.join(HERE, relpath), BRep_Builder())',
        '    if not ok or shape.IsNull():',
        '        return []',
        '    out = []',
        '    exp = TopExp_Explorer(shape, TopAbs_SOLID)',
        '    while exp.More():',
        '        try:',
        '            out.append(Solid(exp.Current()))',
        '        except Exception:',
        '            pass',
        '        exp.Next()',
        '    return out',
        '',
        '',
        '# Load every constituent solid in assembly order.',
        'solids = []',
    ]
    for rel in brep_rel_paths:
        script_lines.append(f'solids.extend(_load_solids_from_brep({rel!r}))')

    script_lines += [
        '',
        '# Wrap everything into one Compound — this IS the assembly.',
        '# Positions are preserved from the original FreeCAD placements.',
        'assembly = Compound(solids)',
        '',
        'if __name__ == "__main__":',
        '    step_out = os.path.join(HERE, "assembly.step")',
        '    stl_out  = os.path.join(HERE, "assembly.stl")',
        '    export_step(assembly, step_out)',
        '    export_stl(assembly, stl_out)',
        '    total_vol = sum(s.volume for s in solids)',
        '    bb = assembly.bounding_box()',
        '    print(f"assembly: {len(solids)} solids, "',
        '          f"vol={total_vol:.3f} mm^3, "',
        '          f"bbox={bb.size.X:.1f}x{bb.size.Y:.1f}x{bb.size.Z:.1f} mm")',
        '    print(f"wrote: {step_out}")',
        '    print(f"wrote: {stl_out}")',
        '',
    ]
    with open(script_abs, "w", encoding="utf-8") as f:
        f.write("\n".join(script_lines))

    write_time = time.time() - t_write0

    return ConversionResult(
        source_path=source_path,
        out_dir=out_dir,
        script_path=script_abs,
        brep_dir=brep_dir_abs,
        n_partshapes=n_partshapes,
        n_solids_extracted=len(collected),
        total_volume_mm3=total_volume,
        bbox_size_mm=bbox_size,
        load_time_s=load_time,
        write_time_s=write_time,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Convert a FreeCAD .FCStd file to a standalone "
                    "build123d assembly script (no fitter, no approximation)."
    )
    p.add_argument("source", help="Path to the .FCStd file")
    p.add_argument("--out", default="fcstd_out",
                   help="Output directory (default: fcstd_out)")
    p.add_argument("--script-name", default="assembly.py",
                   help="Emitted script filename (default: assembly.py)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress output.")
    args = p.parse_args(argv)

    try:
        result = convert(
            args.source, args.out,
            script_name=args.script_name,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    print()
    print(result.summary())
    print()
    print(f"To build the assembly into a STEP + STL file, run:")
    print(f"  python {os.path.relpath(result.script_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
