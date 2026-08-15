"""Generate a synthetic single-leg-balance (releve-like) keypoint sequence.

This is NOT real data. It exists only so the pipeline and the demo notebook run end-to-end
without a video or a force plate (useful for CI and for first-time readers). It injects a
small anterior-posterior COM sway plus tiny noise to mimic a balance task.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

BASE_POSE = {
    "nose": (0.10, 0.00, 1.62), "left_ear": (0.0, 0.08, 1.60), "right_ear": (0.0, -0.08, 1.60),
    "left_shoulder": (0.0, 0.18, 1.40), "right_shoulder": (0.0, -0.18, 1.40),
    "left_elbow": (0.0, 0.30, 1.20), "right_elbow": (0.0, -0.30, 1.20),
    "left_wrist": (0.0, 0.40, 1.30), "right_wrist": (0.0, -0.40, 1.30),
    "left_hip": (0.0, 0.10, 0.90), "right_hip": (0.0, -0.10, 0.90),
    "left_knee": (0.0, 0.10, 0.50), "right_knee": (0.05, -0.10, 0.55),
    "left_ankle": (0.0, 0.10, 0.10), "right_ankle": (0.05, -0.10, 0.30),
    "left_foot_index": (0.15, 0.10, 0.02), "right_foot_index": (0.18, -0.10, 0.22),
}


def generate(seconds: float = 4.0, fps: float = 60.0, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    t = np.arange(n) / fps
    # anterior-posterior COM sway: a slow drift + small oscillation + noise
    ap_sway = 0.02 * np.sin(2 * np.pi * 0.4 * t) + 0.01 * np.sin(2 * np.pi * 1.3 * t)
    ml_sway = 0.008 * np.sin(2 * np.pi * 0.7 * t)
    rows = []
    for i in range(n):
        jitter = rng.normal(0, 0.002, size=3)
        for name, (x, y, z) in BASE_POSE.items():
            # The support foot (right) stays planted = the COP proxy is ~fixed; the rest of
            # the body sways, so the COM moves RELATIVE to the COP and IA/RCIA actually vary.
            planted = name in ("right_ankle", "right_foot_index")
            dx = 0.0 if planted else ap_sway[i]
            dy = 0.0 if planted else ml_sway[i]
            rows.append({"frame": i, "keypoint": name,
                         "x": x + dx + jitter[0],
                         "y": y + dy + jitter[1],
                         "z": z + jitter[2],
                         "visibility": 1.0})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "sample_synthetic.csv")
    df = generate()
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows, {df['frame'].nunique()} frames")
