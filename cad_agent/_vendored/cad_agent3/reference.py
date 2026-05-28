"""reference.py — look up engineering reference data from YAML configs.

Reference data lives in cad_agent3/configs/<category>/<name>.yaml. This
module loads them lazily and exposes lookup helpers used by the
orchestrator and design chat to resolve standards (NEMA17, M3, 2020
extrusion, etc.) into actual numeric dimensions before code generation.

Users can add their own YAMLs without code changes:

    cad_agent3/configs/motors/my_custom_motor.yaml

Categories currently shipped: motors, extrusions, fasteners, bearings, boards.

Usage:
    from cad_agent3.reference import lookup, list_available, summarize

    spec = lookup("motor", "nema17")
    print(spec["bolt_spacing"])   # 31.0

    spec = lookup("fastener", "M3")          # nested-key shortcut
    print(spec["clearance_diameter"])        # 3.4

    list_available()                          # dict of {category: [names]}
    summarize_for_prompt()                    # text dump suitable for an LLM
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

# yaml is a small dep; install instructions documented in NOTES.md
try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(_THIS_DIR, "configs")


# ---------------------------------------------------------------------------
# Category aliases
# ---------------------------------------------------------------------------
# Map plural / singular / common names to the canonical directory name.
CATEGORY_ALIASES = {
    "motor": "motors", "motors": "motors", "stepper": "motors",
    "extrusion": "extrusions", "extrusions": "extrusions",
    "tslot": "extrusions", "t-slot": "extrusions", "rail": "extrusions",
    "fastener": "fasteners", "fasteners": "fasteners",
    "screw": "fasteners", "screws": "fasteners", "bolt": "fasteners",
    "bearing": "bearings", "bearings": "bearings",
    "board": "boards", "boards": "boards", "pcb": "boards",
}


def _canonical_category(name: str) -> str:
    return CATEGORY_ALIASES.get(name.lower(), name.lower())


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    if not _HAS_YAML:
        raise RuntimeError(
            "PyYAML not installed. Run: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return _yaml.safe_load(f) or {}


@lru_cache(maxsize=128)
def _load_category(category: str) -> dict:
    """Return {name: spec_dict} for one category. Cached per-process."""
    cat = _canonical_category(category)
    cat_dir = os.path.join(CONFIGS_DIR, cat)
    if not os.path.isdir(cat_dir):
        return {}
    out = {}
    for fname in sorted(os.listdir(cat_dir)):
        if not fname.endswith(".yaml") and not fname.endswith(".yml"):
            continue
        path = os.path.join(cat_dir, fname)
        try:
            data = _load_yaml(path)
        except Exception as e:
            # bad YAML — skip, don't crash the whole lookup
            continue
        # canonical key = filename stem
        stem = fname.rsplit(".", 1)[0].lower()
        # also accept the `name` field if present
        explicit_name = (data.get("name") or stem).lower()
        out[stem] = data
        if explicit_name != stem:
            out[explicit_name] = data
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup(category: str, name: str) -> Optional[dict]:
    """Get a spec dict by category + name.

    For the `fastener` category, supports the shortcut `lookup("fastener",
    "M3")` which returns the M3 sub-spec from `metric_screws.yaml`.

    Returns None if not found.
    """
    cat = _canonical_category(category)
    cat_data = _load_category(cat)
    name_lower = name.lower()

    # Direct hit (e.g. "nema17", "2020_tslot")
    if name_lower in cat_data:
        return cat_data[name_lower]

    # Fastener shortcut: lookup("fastener", "M3") → metric_screws[sizes][M3]
    if cat == "fasteners":
        for fastener_table in cat_data.values():
            sizes = fastener_table.get("sizes")
            if isinstance(sizes, dict):
                # match case-insensitively
                for k, v in sizes.items():
                    if k.lower() == name_lower:
                        return {**v, "_designation": k,
                                "_table": fastener_table.get("name", "")}
    return None


def list_available() -> dict:
    """Return {category: [names]} of every available reference."""
    out = {}
    if not os.path.isdir(CONFIGS_DIR):
        return out
    for cat in sorted(os.listdir(CONFIGS_DIR)):
        cat_dir = os.path.join(CONFIGS_DIR, cat)
        if not os.path.isdir(cat_dir):
            continue
        names = []
        for fname in sorted(os.listdir(cat_dir)):
            if fname.endswith((".yaml", ".yml")):
                names.append(fname.rsplit(".", 1)[0])
        out[cat] = names
    return out


def summarize_for_prompt() -> str:
    """Return a compact text summary of all available references, suitable
    for inclusion in an LLM system prompt.
    """
    avail = list_available()
    if not avail:
        return "(no reference data available)"
    lines = ["Available reference standards (use lookup-resolved dimensions):"]
    for cat, names in avail.items():
        if not names:
            continue
        lines.append(f"  {cat}: {', '.join(names)}")
    # For fasteners: enumerate the sizes inside metric_screws
    fastener_table = _load_category("fasteners").get("metric_screws")
    if fastener_table and "sizes" in fastener_table:
        sizes = list(fastener_table["sizes"].keys())
        lines.append(f"  fastener sizes: {', '.join(sizes)}")
    return "\n".join(lines)


def resolve_keywords(text: str) -> list:
    """Scan free-form text for references to known standards, return
    list of (category, name, spec) triples for any matches.

    Used by the orchestrator to auto-extract specs from a user request:

        text = "design a bracket for a NEMA17 motor on 2020 extrusion with M3 screws"
        -> [("motors", "nema17", {...}), ("extrusions", "2020_tslot", {...}),
            ("fasteners", "M3", {...})]
    """
    text_low = text.lower()
    hits = []

    # Scan motors and extrusions: match against name / aliases
    for cat in ("motors", "extrusions", "bearings", "boards"):
        for stem, spec in _load_category(cat).items():
            # Build search terms: the stem, the explicit name, any aliases
            terms = {stem, (spec.get("name") or "").lower()}
            terms.discard("")
            # Also: nema17 → nema 17 etc.; 2020_tslot → 2020
            for term in list(terms):
                if "nema" in term:
                    terms.add(term.replace("nema", "nema "))
                if "_tslot" in term:
                    # bare "2020", "3030", "4040" should match
                    terms.add(term.split("_")[0])
                if "_" in term:
                    terms.add(term.replace("_", " "))
                    terms.add(term.replace("_", "-"))
            if any(t in text_low for t in terms):
                hits.append((cat, stem, spec))

    # Scan fastener designators (M3, M4, etc.) — use word boundaries
    fastener_table = _load_category("fasteners").get("metric_screws", {})
    sizes = fastener_table.get("sizes", {})
    import re
    for designation, size_spec in sizes.items():
        # Match e.g. "M3" but not "M30" or "M3-something"
        if re.search(r"\b" + re.escape(designation) + r"\b", text, re.IGNORECASE):
            hits.append(("fasteners", designation,
                         {**size_spec, "_designation": designation}))

    # Deduplicate (cat, name) pairs
    seen = set()
    out = []
    for cat, name, spec in hits:
        key = (cat, name)
        if key in seen:
            continue
        seen.add(key)
        out.append((cat, name, spec))
    return out
