"""Centre of pressure (COP) proxy from kinematics, and the COM-COP pairing per frame.

Why a proxy is defensible
-------------------------
Lu's IA/RCIA needs the COP, normally from a force plate. But Lin, Su & Lin (2019)
(Front Bioeng Biotechnol 7:290), who applied the *same* COM-COP inclination method to
pirouette, approximate the COP during single-leg phases as the midpoint between the
ankle-joint centre and the metatarsal marker (i.e. a kinematics-only proxy). This module
reuses that idea, approximated with the available pose keypoints (the ankle and the foot-index toe
keypoint in place of the ankle joint centre and metatarsal marker; see docs/LIMITATIONS.md), so a
markerless video pipeline can estimate IA/RCIA during single-leg ballet tasks (releve, passe,
pirouette) without a force plate.

Limitations are documented in docs/LIMITATIONS.md: the proxy is only valid while the foot
is the sole support; it ignores true pressure distribution; it is 2D/3D-pose-dependent.
"""
from __future__ import annotations

import numpy as np

from .anthropometry import compute_com


def cop_proxy_single_leg(kp: dict[str, np.ndarray], support: str) -> np.ndarray:
    """COP proxy for a single-leg support phase (Lin et al. 2019 method).

    support : "left" or "right" -- the supporting foot.
    Returns the midpoint of the ankle and the foot-index (metatarsal) of that foot.
    """
    if support not in ("left", "right"):
        raise ValueError("support must be 'left' or 'right'")
    ankle = np.asarray(kp[f"{support}_ankle"], dtype=float)
    toe = np.asarray(kp[f"{support}_foot_index"], dtype=float)
    return 0.5 * (ankle + toe)


def cop_proxy_double_leg(kp: dict[str, np.ndarray]) -> np.ndarray:
    """COP proxy for a double-leg phase: midpoint of the two single-foot proxies.

    A crude approximation (true double-support COP depends on the load split between feet,
    which kinematics cannot recover). Flagged in LIMITATIONS.md; prefer single-leg tasks.
    """
    left = cop_proxy_single_leg(kp, "left")
    right = cop_proxy_single_leg(kp, "right")
    return 0.5 * (left + right)


def detect_support_foot(kp: dict[str, np.ndarray], vert_axis: int = 2) -> str:
    """Heuristic: the lower ankle (smaller vertical coordinate) is the support foot.

    Assumes the vertical axis increases upward. For an image/pose frame where y grows
    downward, pass the appropriate axis and the caller should flip the sign upstream.
    """
    la_pt = np.asarray(kp["left_ankle"], dtype=float)
    if la_pt.shape[0] <= vert_axis:
        raise ValueError(f"keypoints are {la_pt.shape[0]}D but vert_axis={vert_axis}; need 3D for balance")
    la = la_pt[vert_axis]
    ra = np.asarray(kp["right_ankle"], dtype=float)[vert_axis]
    return "left" if la <= ra else "right"


def com_and_cop(kp: dict[str, np.ndarray], sex: str = "female",
                support: str | None = None, vert_axis: int = 2
                ) -> tuple[np.ndarray, np.ndarray]:
    """Return (COM, COP_proxy) for one frame.

    support : "left"/"right" to force a single-leg proxy; "double" for a double-leg proxy;
              None -> auto-detect the lower foot as support (single-leg).
    """
    # compute_com derives any missing midpoints itself, with a descriptive
    # ValueError when source keypoints are absent (occluded frames)
    com = compute_com(kp, sex=sex)
    if support == "double":
        cop = cop_proxy_double_leg(kp)
    else:
        s = support or detect_support_foot(kp, vert_axis=vert_axis)
        cop = cop_proxy_single_leg(kp, s)
    return com, cop
