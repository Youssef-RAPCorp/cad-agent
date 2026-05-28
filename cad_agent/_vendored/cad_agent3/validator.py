"""validator.py — sanity checks on generated parts.

Given a build123d Part and a structured spec describing what it should
be, validate the result. Returns a list of issues (or empty if all
checks pass).

Checks performed:
- Volume is positive and within an order of magnitude of an estimate
- Bounding box fits within a requested envelope (if given)
- Number of through-holes of an expected diameter matches expectation
- Topology is non-degenerate (one connected solid, not many shards)

This is NOT FEA. It catches gross errors — wrong scale, wrong hole
count, missing features — before the user looks at the rendered part.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


# ---------------------------------------------------------------------------
# Issues
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    severity: str   # "error" | "warning" | "info"
    code: str       # short tag
    message: str    # human-readable
    details: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    part_summary: str
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.issues)

    def summary(self) -> str:
        if not self.issues:
            return f"{self.part_summary}\n  ✓ all checks passed"
        lines = [self.part_summary]
        for issue in self.issues:
            mark = {"error": "✗", "warning": "!", "info": "·"}.get(
                issue.severity, "·")
            lines.append(f"  {mark} [{issue.code}] {issue.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_volume(part, expected_min: Optional[float],
                   expected_max: Optional[float]) -> List[ValidationIssue]:
    issues = []
    try:
        v = part.volume
    except Exception as e:
        issues.append(ValidationIssue(
            "error", "volume_unreadable",
            f"could not read part.volume: {type(e).__name__}: {e}"))
        return issues
    if v is None or v <= 0:
        issues.append(ValidationIssue(
            "error", "non_positive_volume",
            f"part has volume={v} (must be > 0)"))
        return issues
    if expected_min is not None and v < expected_min:
        issues.append(ValidationIssue(
            "warning", "volume_below_expected",
            f"volume {v:.2f} mm³ is below expected min {expected_min:.2f}",
            {"actual": v, "expected_min": expected_min}))
    if expected_max is not None and v > expected_max:
        issues.append(ValidationIssue(
            "warning", "volume_above_expected",
            f"volume {v:.2f} mm³ is above expected max {expected_max:.2f}",
            {"actual": v, "expected_max": expected_max}))
    return issues


def _check_envelope(part, envelope: Optional[tuple]) -> List[ValidationIssue]:
    """envelope: (length_mm, width_mm, height_mm) — max allowed extent."""
    if envelope is None:
        return []
    issues = []
    try:
        bb = part.bounding_box()
        sx = bb.max.X - bb.min.X
        sy = bb.max.Y - bb.min.Y
        sz = bb.max.Z - bb.min.Z
    except Exception as e:
        return [ValidationIssue(
            "error", "bbox_unreadable",
            f"could not read bounding box: {type(e).__name__}: {e}")]
    ex_len, ex_wid, ex_hgt = envelope
    actual = sorted([sx, sy, sz], reverse=True)
    expected = sorted([ex_len, ex_wid, ex_hgt], reverse=True)
    # Compare sorted to be orientation-agnostic (a 50×30×5 envelope
    # accepts a 50×30×5 part regardless of which axis maps to which).
    for a, e, axis_name in zip(actual, expected, ("longest", "second", "shortest")):
        if a > e + 0.01:   # 0.01mm tolerance
            issues.append(ValidationIssue(
                "warning", "envelope_exceeded",
                f"{axis_name} dim {a:.2f} exceeds envelope {e:.2f}",
                {"actual": a, "envelope": e, "axis": axis_name}))
    return issues


def _check_hole_count(part, expected_count: Optional[int],
                       expected_diameter: Optional[float],
                       diameter_tol: float = 0.2) -> List[ValidationIssue]:
    """Count cylindrical faces matching the expected diameter.

    A through-hole shows up as one or two cylindrical surfaces (one for
    each side wall, depending on whether the cylinder is split). We
    count UNIQUE radii — so 4 holes = 4 cylindrical faces of the same
    radius.
    """
    if expected_count is None or expected_diameter is None:
        return []
    issues = []
    try:
        from build123d import GeomType
    except ImportError:
        return [ValidationIssue(
            "info", "build123d_missing",
            "could not import build123d for hole-count check")]
    target_r = expected_diameter / 2
    cyl_radii = []
    try:
        for face in part.faces():
            if face.geom_type == GeomType.CYLINDER:
                # Approximate the radius from the face's bounding box width
                bb = face.bounding_box()
                # cylinder radius ~ half the smaller bbox dim in XY
                dx = bb.max.X - bb.min.X
                dy = bb.max.Y - bb.min.Y
                r_est = min(dx, dy) / 2
                cyl_radii.append(r_est)
    except Exception as e:
        return [ValidationIssue(
            "info", "face_iter_failed",
            f"could not iterate faces: {type(e).__name__}: {e}")]
    matching = [r for r in cyl_radii if abs(r - target_r) <= diameter_tol]
    if len(matching) != expected_count:
        # If no matches at all, this is more serious than mis-count
        sev = "error" if len(matching) == 0 else "warning"
        issues.append(ValidationIssue(
            sev, "hole_count_mismatch",
            f"expected {expected_count} holes of "
            f"~{expected_diameter:.2f}mm dia, found "
            f"{len(matching)} cylindrical faces matching",
            {"expected_count": expected_count,
             "found_count": len(matching),
             "all_cyl_radii": sorted(cyl_radii)}))
    return issues


def _check_solid_count(part) -> List[ValidationIssue]:
    """Warn if the part is multiple disconnected solids."""
    try:
        n = len(list(part.solids()))
    except Exception:
        return []
    if n == 0:
        return [ValidationIssue(
            "error", "no_solids",
            "part contains no solid geometry")]
    if n > 1:
        return [ValidationIssue(
            "warning", "multi_solid",
            f"part contains {n} disconnected solids; "
            f"may need to be unioned",
            {"solid_count": n})]
    return []


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def validate(
    part,
    expected_volume_min: Optional[float] = None,
    expected_volume_max: Optional[float] = None,
    envelope: Optional[tuple] = None,
    expected_hole_count: Optional[int] = None,
    expected_hole_diameter: Optional[float] = None,
) -> ValidationReport:
    """Run all sanity checks on a part. Returns a ValidationReport."""
    try:
        v = part.volume
        bb = part.bounding_box()
        sx = bb.max.X - bb.min.X
        sy = bb.max.Y - bb.min.Y
        sz = bb.max.Z - bb.min.Z
        n_faces = len(list(part.faces()))
        summary = (f"part: vol={v:.3f} mm³, bbox={sx:.1f}×{sy:.1f}×{sz:.1f} mm, "
                   f"{n_faces} faces")
    except Exception as e:
        summary = f"part: (unreadable: {type(e).__name__}: {e})"

    issues = []
    issues.extend(_check_volume(part, expected_volume_min, expected_volume_max))
    issues.extend(_check_envelope(part, envelope))
    issues.extend(_check_hole_count(
        part, expected_hole_count, expected_hole_diameter))
    issues.extend(_check_solid_count(part))
    return ValidationReport(part_summary=summary, issues=issues)
