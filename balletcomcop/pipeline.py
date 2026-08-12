"""End-to-end: a per-frame keypoint table -> COM/COP/IA/RCIA time series."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .com_cop import com_and_cop
from .inclination import inclination_angles, rcia


def analyze_frames(kp_by_frame: dict[int, dict[str, np.ndarray]], fps: float,
                   sex: str = "female", support: str | None = None,
                   ap_axis: int = 0, ml_axis: int = 1, vert_axis: int = 2) -> pd.DataFrame:
    """Compute COM, COP proxy, sagittal/frontal IA and RCIA for each frame.

    kp_by_frame : {frame_index: {keypoint: [ap, ml, vert]}}
    fps         : frames per second (for the time axis and RCIA derivative).
    Returns a DataFrame indexed by frame with columns:
        time, com_ap/ml/vert, cop_ap/ml/vert, ia_sagittal, ia_frontal, rcia_sagittal, rcia_frontal.
    """
    if not kp_by_frame:
        raise ValueError("kp_by_frame is empty; no frames to analyse")
    frames = sorted(kp_by_frame)
    recs: list[dict] = []
    for f in frames:
        com, cop = com_and_cop(kp_by_frame[f], sex=sex, support=support, vert_axis=vert_axis)
        sag, fro = inclination_angles(com, cop, ap_axis=ap_axis, ml_axis=ml_axis,
                                      vert_axis=vert_axis)
        recs.append({"frame": f, "time": f / fps,
                     "com_ap": com[ap_axis], "com_ml": com[ml_axis], "com_vert": com[vert_axis],
                     "cop_ap": cop[ap_axis], "cop_ml": cop[ml_axis], "cop_vert": cop[vert_axis],
                     "ia_sagittal": float(sag), "ia_frontal": float(fro)})
    df = pd.DataFrame(recs).set_index("frame")
    df["rcia_sagittal"] = _safe_rcia(df["ia_sagittal"], df["time"])
    df["rcia_frontal"] = _safe_rcia(df["ia_frontal"], df["time"])
    return df


def _safe_rcia(ia: "pd.Series", time: "pd.Series") -> np.ndarray:
    """RCIA that tolerates NaN frames (e.g. a dropped/occluded pose): fit on the finite
    samples, leave NaN where IA was NaN. Returns NaN everywhere if < 4 finite samples."""
    ia = ia.to_numpy(dtype=float)
    time = time.to_numpy(dtype=float)
    out = np.full_like(ia, np.nan)
    finite = ~np.isnan(ia)
    if finite.sum() >= 4:
        out[finite] = rcia(ia[finite], time[finite])
    return out
