"""COM-to-COP inclination angle (IA) and its rate of change (RCIA).

Faithful to the Lu-lab definition used across Kuo et al. 2022 (Sci Rep 12:2660),
Lee et al. 2021 (Sci Rep 11:3742) and Yu et al. 2023 (Sensors 23:9040), whose
deep-notes record the equations:

    t = Z x (P / |P|)               (Eq. 1)   P = COM-to-COP vector, Z = vertical unit
    sagittal IA = arcsin(t_Y)       (Eq. 2)
    frontal  IA = +/- arcsin(t_X)   (Eq. 3)

With Z = (0,0,1) the cross product gives t = (-P_y, P_x, 0), so
    sagittal IA = arcsin(P_x / |P|)   (anterior-posterior plane)
    frontal  IA = arcsin(-P_y / |P|)  (medio-lateral plane).

We expose both the exact cross-product form (`inclination_cross_product`, for the L2
reproduction on force-plate data) and an axis-labelled convenience form
(`inclination_angles`) that is equivalent and easier to use with arbitrary frames.

RCIA is the first time-derivative of IA. Lu's papers smooth + differentiate IA with the
GCVSPL package; here we use scipy's `UnivariateSpline` with a generalized-cross-validation
smoothing factor as the practical open-source analogue (documented in docs/METHOD.md).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline

_Z = np.array([0.0, 0.0, 1.0])


def inclination_cross_product(com: np.ndarray, cop: np.ndarray, deg: bool = True
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Exact Lu Eq.(1)-(3). Input frame must be X=anterior-posterior, Y=medio-lateral, Z=vertical.

    com, cop : (...,3) arrays. Returns (sagittal_IA, frontal_IA), broadcast over leading axes.
    WARNING: no axis-order validation is performed; wrong-ordered input silently gives wrong IA.
    Use `inclination_angles` with explicit ap/ml axes when the frame order is non-standard.
    """
    com = np.asarray(com, dtype=float)
    cop = np.asarray(cop, dtype=float)
    if com.shape[-1] != 3 or cop.shape[-1] != 3:
        raise ValueError("inclination_cross_product needs 3D arrays (last dim = 3: X=AP, Y=ML, Z=vert)")
    p = com - cop
    norm = np.linalg.norm(p, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, np.nan, norm)
    p_hat = p / norm
    t = np.cross(np.broadcast_to(_Z, p_hat.shape), p_hat)  # (-P_y, P_x, 0)
    sagittal = np.arcsin(np.clip(t[..., 1], -1.0, 1.0))    # arcsin(P_x)
    frontal = np.arcsin(np.clip(t[..., 0], -1.0, 1.0))     # arcsin(-P_y)
    if deg:
        sagittal, frontal = np.degrees(sagittal), np.degrees(frontal)
    return sagittal, frontal


def inclination_angles(com: np.ndarray, cop: np.ndarray, ap_axis: int = 0,
                       ml_axis: int = 1, vert_axis: int = 2, deg: bool = True
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Axis-labelled IA. Returns (sagittal_AP_IA, frontal_ML_IA).

    sagittal_IA = arcsin(P_ap / |P|); frontal_IA = arcsin(-P_ml / |P|) [Lu Eq.(3) sign],
    where P = COM - COP. Identical to `inclination_cross_product` when the frame is
    (ap=0, ml=1, vert=2). Use this for markerless video where you choose the axes.

    Note: `vert_axis` is accepted for call-site symmetry with `com_and_cop` but is NOT used
    here; the vertical component enters only implicitly through the 3D norm |P|.
    """
    if ap_axis == ml_axis:
        raise ValueError("ap_axis and ml_axis must differ")
    com = np.asarray(com, dtype=float)
    cop = np.asarray(cop, dtype=float)
    p = com - cop
    norm = np.linalg.norm(p, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, np.nan, norm)
    p_hat = (p / norm)
    # frontal uses arcsin(-P_ml) to MATCH Lu Eq.(3) (= inclination_cross_product); this makes
    # the pipeline output sign-consistent with the faithful reproduction in notebook 02.
    sagittal = np.arcsin(np.clip(p_hat[..., ap_axis], -1.0, 1.0))
    frontal = np.arcsin(np.clip(-p_hat[..., ml_axis], -1.0, 1.0))
    if deg:
        sagittal, frontal = np.degrees(sagittal), np.degrees(frontal)
    return sagittal, frontal


def rcia(ia: np.ndarray, time: np.ndarray, smoothing: float | None = None
         ) -> np.ndarray:
    """Rate of change of IA (dIA/dt) via a smoothing spline (open-source GCVSPL analogue).

    ia : 1D inclination-angle time series (deg or rad). time : 1D matching timestamps (s).
    smoothing : spline smoothing factor s. None -> a mild default proportional to n*var,
                which behaves like generalized cross-validation for typical mocap noise.
    Returns dIA/dt in (units of ia)/second, same length as ia.
    """
    ia = np.asarray(ia, dtype=float)
    time = np.asarray(time, dtype=float)
    if ia.ndim != 1 or time.ndim != 1 or ia.size != time.size:
        raise ValueError("ia and time must be 1D arrays of equal length")
    if ia.size < 4:
        raise ValueError("need at least 4 samples to fit a cubic smoothing spline")
    if np.isnan(ia).any():
        raise ValueError("ia contains NaN; filter or interpolate dropped frames before rcia "
                         "(analyze_frames does this automatically)")
    if not np.all(np.diff(time) > 0):
        raise ValueError("time must be strictly increasing (no duplicates / reordering); "
                         f"found {int((np.diff(time) <= 0).sum())} non-increasing step(s)")
    std = float(np.std(ia))
    if std < 1e-9:
        return np.zeros_like(ia)  # constant signal -> zero rate (avoids spline instability)
    if smoothing is None:
        # GCV-like default: assume measurement noise ~10%% of the signal SD.
        smoothing = ia.size * (0.1 * std) ** 2
    spline = UnivariateSpline(time, ia, k=3, s=smoothing)
    return spline.derivative()(time)


def decompose_ap_ml(sagittal_ia: np.ndarray, frontal_ia: np.ndarray) -> dict[str, float]:
    """Summary of AP vs ML dominance for a movement: amplitude and ending-phase variability.

    Returns the standard-deviation of each plane plus the AP/(AP+ML) dominance ratio, the
    quantities Plan B and Lin et al. 2019 use (e.g. ending-phase sway SD, AP-dominance).
    """
    ap = np.asarray(sagittal_ia, dtype=float)
    ml = np.asarray(frontal_ia, dtype=float)
    if np.all(np.isnan(ap)) and np.all(np.isnan(ml)):
        return {"ap_sd": float("nan"), "ml_sd": float("nan"), "ap_range": float("nan"),
                "ml_range": float("nan"), "ap_dominance": float("nan")}
    ap_sd = float(np.nanstd(ap))
    ml_sd = float(np.nanstd(ml))
    denom = ap_sd + ml_sd
    return {
        "ap_sd": ap_sd,
        "ml_sd": ml_sd,
        "ap_range": float(np.nanmax(ap) - np.nanmin(ap)),
        "ml_range": float(np.nanmax(ml) - np.nanmin(ml)),
        "ap_dominance": float(ap_sd / denom) if denom else float("nan"),
    }
