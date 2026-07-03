"""De Leva (1996) segmental anthropometry: whole-body centre of mass (COM) from keypoints.

Reference
---------
De Leva, P. (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters.
Journal of Biomechanics 29(9):1223-1230.

Scope and honesty
-----------------
This is a *simplified* 14-segment model driven by MediaPipe-style joint keypoints.
- Segment MASS fractions are the De Leva (1996) values (they sum to 1.0 per sex; see test).
- Limb COM longitudinal ratios (fraction from the proximal endpoint) are De Leva values.
- HEAD and TRUNK COM are placed at the segment midpoint (ratio 0.5) as a documented
  simplification, because vertex / suprasternale / C7 landmarks are not available from a
  pose estimator. The head (~7% mass) and the midpoint trunk assumption are the main
  approximations; they are listed in docs/LIMITATIONS.md.

All functions are pure and operate on plain numpy arrays so the maths is testable without
any video or MediaPipe dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Canonical keypoint names this model consumes (a subset of MediaPipe Pose landmarks
# plus three midpoints derived in `derive_midpoints`).
REQUIRED_KEYPOINTS: tuple[str, ...] = (
    "nose", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_foot_index", "right_foot_index",
)  # raw keypoints a caller supplies; mid_shoulder/mid_hip/mid_ear are derived from these.


@dataclass(frozen=True)
class Segment:
    """One rigid segment: proximal/distal endpoints, mass fraction, and COM ratio (from proximal)."""
    name: str
    proximal: str
    distal: str
    mass_m: float   # mass as fraction of total body mass, male (De Leva 1996)
    mass_f: float   # mass as fraction of total body mass, female (De Leva 1996)
    com_m: float    # COM position as fraction of segment length from proximal, male
    com_f: float    # ... female
    bilateral: bool # True for paired limb segments (counted once per side L and R)


# De Leva (1996), Table 4. Mass = % body mass / 100. COM ratio = % segment length / 100 from proximal.
# Limb + trunk COM ratios are the authoritative De Leva (1996) values (verified against the
# Visual3D/HAS-Motion tabulation). HEAD COM is kept at 0.5 of the mid_ear->nose segment, which is
# a pose-keypoint proxy NOT on De Leva's head axis (vertex-based), so the De Leva head ratio does
# not apply; head is ~7%% mass so the effect on whole-body COM is small.
_SEGMENTS: tuple[Segment, ...] = (
    Segment("head",      "mid_ear",       "nose",            0.0694, 0.0668, 0.500, 0.500, False),
    Segment("trunk",     "mid_shoulder",  "mid_hip",         0.4346, 0.4257, 0.5138, 0.4964, False),
    Segment("upper_arm", "{s}_shoulder",  "{s}_elbow",       0.0271, 0.0255, 0.5772, 0.5754, True),
    # forearm+hand combined (De Leva): hand mass folded in because pose has no knuckle landmark;
    # COM ratio kept at the forearm value as a documented proxy (hand is ~0.6%% mass, negligible for whole-body COM).
    Segment("forearm_hand", "{s}_elbow",  "{s}_wrist",       0.0223, 0.0194, 0.4574, 0.4559, True),
    Segment("thigh",     "{s}_hip",       "{s}_knee",        0.1416, 0.1478, 0.4095, 0.3612, True),
    Segment("shank",     "{s}_knee",      "{s}_ankle",       0.0433, 0.0481, 0.4395, 0.4352, True),
    Segment("foot",      "{s}_ankle",     "{s}_foot_index",  0.0137, 0.0129, 0.4415, 0.4014, True),
)


def total_mass_fraction(sex: str) -> float:
    """Sum of all segment mass fractions (should be ~1.0). Used by tests."""
    total = 0.0
    for seg in _SEGMENTS:
        m = seg.mass_m if sex == "male" else seg.mass_f
        total += m * (2 if seg.bilateral else 1)
    return total


def derive_midpoints(kp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add mid_shoulder, mid_hip, mid_ear from left/right keypoints. Returns a new dict."""
    out = dict(kp)
    out["mid_shoulder"] = 0.5 * (kp["left_shoulder"] + kp["right_shoulder"])
    out["mid_hip"] = 0.5 * (kp["left_hip"] + kp["right_hip"])
    out["mid_ear"] = 0.5 * (kp["left_ear"] + kp["right_ear"])
    return out


def _segment_com(kp: dict[str, np.ndarray], prox: str, dist: str, ratio: float) -> np.ndarray:
    p = np.asarray(kp[prox], dtype=float)
    d = np.asarray(kp[dist], dtype=float)
    return p + ratio * (d - p)


def compute_com(kp: dict[str, np.ndarray], sex: str = "female") -> np.ndarray:
    """Whole-body COM = mass-weighted sum of segmental COMs (De Leva 1996).

    Parameters
    ----------
    kp : dict mapping canonical keypoint name -> 3D position (numpy array). The COM maths is
         dimension-agnostic, but the balance pipeline (detect_support_foot, IA) requires 3D, so
         3D is expected. Must contain `REQUIRED_KEYPOINTS`, or already-derived midpoints.
    sex : "male" or "female" (selects the De Leva column).

    Returns
    -------
    numpy array, same dimensionality as the input keypoints.
    """
    if sex not in ("male", "female"):
        raise ValueError(f"sex must be 'male' or 'female', got {sex!r}")
    if not all(k in kp for k in ("mid_shoulder", "mid_hip", "mid_ear")):
        need = ("left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_hip", "right_hip")
        miss = [k for k in need if k not in kp]
        if miss:
            raise ValueError(f"compute_com cannot derive midpoints; missing keypoints: {miss}")
        kp = derive_midpoints(kp)
    seg_keys = ("nose", "mid_ear", "mid_shoulder", "mid_hip", "left_shoulder", "right_shoulder",
                "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle",
                "left_foot_index", "right_foot_index")
    miss = [k for k in seg_keys if k not in kp]
    if miss:
        raise ValueError(f"compute_com missing keypoints: {miss}")

    contributions: list[np.ndarray] = []
    total_mass = 0.0
    for seg in _SEGMENTS:
        m = seg.mass_m if sex == "male" else seg.mass_f
        ratio = seg.com_m if sex == "male" else seg.com_f
        sides = ("left", "right") if seg.bilateral else (None,)
        for s in sides:
            prox = seg.proximal.format(s=s) if s else seg.proximal
            dist = seg.distal.format(s=s) if s else seg.distal
            contributions.append(m * _segment_com(kp, prox, dist, ratio))
            total_mass += m
    weighted_sum = np.sum(np.stack(contributions), axis=0)
    return weighted_sum / total_mass
