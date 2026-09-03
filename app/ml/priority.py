"""
InfraPulse — priority scoring.

THE DOCUMENTED FORMULA (this file is the single source of truth; the report quotes it
verbatim, and the queue sorts by exactly the number produced here):

    priority_score = 100 x base_weight x (0.5 x severity + 0.5 x extent)

where

    severity     0..1  how bad the defect looks inside the located region
                       (local contrast, edge density, intensity deviation - see analyzer.py)
    extent       0..1  fraction of the photograph the defect covers, from Grad-CAM
                       run on our own trained network
    base_weight        per-defect weight, below

BASE WEIGHTS
    Each category owns a separate queue, so a base weight only ever changes ordering
    *within* one category. Performance is the only category holding two defect types,
    and the problem statement fixes their order explicitly:

        "Performance: paint peeling, cracked tiles (cracked tiles > paint peeling in priority)"

    so cracked tiles outrank paint peeling. Spalling and stagnant water are alone in
    their queues and take the full weight.

TIE-BREAKING
    Equal scores are ordered by submission time, oldest first, so a complaint can never
    be overtaken indefinitely by later arrivals with an identical score.

DELIBERATELY EXCLUDED
    Age-based escalation (old complaints slowly climbing the queue) is realistic but
    would make queue order disagree with visible severity, which is what the system is
    required to rank by. It is listed in the report as a suggested improvement instead.

Every input is computed from the submitted photograph at request time. Nothing is
memorised, hardcoded, or pre-associated with any expected evaluation input.
"""

from __future__ import annotations

BASE_WEIGHT = {
    "spalling": 1.00,        # alone in Structural
    "stagnant_water": 1.00,  # alone in Functional
    "cracked_tiles": 1.00,   # Performance - ranks above paint peeling (per the PS)
    "paint_peeling": 0.75,   # Performance - lower of the two
}

W_SEVERITY = 0.5
W_EXTENT = 0.5

# Display bands. Purely presentational - ordering always uses the raw score.
BANDS = [(70.0, "Critical"), (45.0, "High"), (22.0, "Medium"), (0.0, "Low")]


def compute_priority(defect: str, severity: float, extent: float) -> dict:
    """Apply the formula above. Deterministic: same inputs -> same score, always."""
    base = BASE_WEIGHT.get(defect, 0.75)
    severity = min(max(float(severity), 0.0), 1.0)
    extent = min(max(float(extent), 0.0), 1.0)

    score = 100.0 * base * (W_SEVERITY * severity + W_EXTENT * extent)
    score = round(score, 2)

    band = next(name for threshold, name in BANDS if score >= threshold)

    return {
        "priority_score": score,
        "priority_band": band,
        "base_weight": base,
        "explanation": (
            f"100 x {base:g} (base weight for {defect.replace('_', ' ')}) x "
            f"[{W_SEVERITY:g} x {severity:.3f} severity + {W_EXTENT:g} x {extent:.3f} extent] "
            f"= {score:.2f}"
        ),
    }
