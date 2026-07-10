"""Deterministic, mathematically accurate standard parts for generated code.

LLM-generated build123d scripts are good at composition but unreliable
at precision geometry (gear teeth in particular tend to come out as
trapezoids with wrong meshing distances). The helpers here are injected
into the codegen sandbox namespace so generated code can call them
directly — the LLM chooses parameters, the math guarantees the shape.
"""

from __future__ import annotations

import math


def involute_gear(module: float, teeth: int, thickness: float = 5.0,
                  bore: float = 0.0, pressure_angle: float = 20.0,
                  backlash: float = 0.1):
    """A mathematically accurate involute spur gear (ISO 53 proportions).

    Args:
        module: gear module m in mm (tooth size). Two meshing gears must
            share the same module, and their shaft center distance must
            be  m * (teeth1 + teeth2) / 2.
        teeth: number of teeth (z >= 4).
        thickness: face width; the gear is extruded from z=0 to z=thickness,
            axis +Z, centered on the origin in XY.
        bore: central hole diameter (0 = no bore).
        pressure_angle: degrees; 20 is the modern standard.
        backlash: circumferential backlash allowance as a fraction of the
            module (each gear's teeth are thinned by half of it), so a
            meshing pair at the exact center distance runs with clearance
            instead of interference. 0 gives the theoretical zero-backlash
            profile.

    Returns:
        A build123d Part. Tip diameter = m*(z+2), root diameter =
        m*(z-2.5), pitch diameter = m*z.
    """
    from build123d import Circle, Polygon, Rot, extrude

    m = float(module)
    z = int(teeth)
    if z < 4:
        raise ValueError("involute_gear needs teeth >= 4")
    alpha = math.radians(float(pressure_angle))

    rp = m * z / 2.0                 # pitch radius
    rb = rp * math.cos(alpha)        # base radius
    ra = rp + m                      # addendum (tip) radius
    rf = rp - 1.25 * m               # dedendum (root) radius
    if rf <= 0:
        raise ValueError("gear too small: root radius <= 0")

    def inv(a: float) -> float:      # involute function
        return math.tan(a) - a

    # Angular half-thickness of a tooth at radius r (standard involute
    # gear relation): psi(r) = pi/(2z) + inv(alpha) - inv(alpha_r),
    # where cos(alpha_r) = rb / r. Thinned by half the backlash
    # allowance (arc j/2 => angle j/(4r)).
    j = float(backlash) * m

    def half_thick(r: float) -> float:
        if r <= rb:
            base = math.pi / (2 * z) + inv(alpha)
        else:
            a_r = math.acos(rb / r)
            base = math.pi / (2 * z) + inv(alpha) - inv(a_r)
        return base - j / (4.0 * r)

    # One tooth as a polygon from the root circle to the tip: right
    # flank up, tip arc, left flank down. 16 samples per flank keeps
    # the profile within ~0.1 um of the true involute at clock scales.
    n_flank, n_tip = 16, 7
    r_start = max(rf, rb)
    pts = []
    for i in range(n_flank + 1):                       # right flank
        r = r_start + (ra - r_start) * i / n_flank
        a = -half_thick(r)
        pts.append((r * math.sin(a), r * math.cos(a)))
    tip_a = half_thick(ra)
    for i in range(1, n_tip + 1):                      # tip arc
        a = -tip_a + 2 * tip_a * i / (n_tip + 1)
        pts.append((ra * math.sin(a), ra * math.cos(a)))
    for i in range(n_flank + 1):                       # left flank
        r = ra - (ra - r_start) * i / n_flank
        a = half_thick(r)
        pts.append((r * math.sin(a), r * math.cos(a)))
    if r_start > rf:                                   # radial root stubs
        a0 = half_thick(r_start)
        pts.append((rf * math.sin(a0), rf * math.cos(a0)))
        pts.insert(0, (rf * math.sin(-a0), rf * math.cos(-a0)))

    tooth = Polygon(*pts)
    profile = Circle(rf)
    for i in range(z):
        profile += Rot(0, 0, i * 360.0 / z) * tooth
    if bore > 0:
        profile -= Circle(bore / 2.0)
    return extrude(profile, float(thickness))


# Everything the sandbox should expose to generated code.
SANDBOX_HELPERS = {
    "involute_gear": involute_gear,
}
