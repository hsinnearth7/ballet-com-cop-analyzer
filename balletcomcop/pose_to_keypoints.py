"""Video -> per-frame canonical keypoints via MediaPipe Pose.

MediaPipe is imported lazily inside the functions so the deterministic core (anthropometry,
inclination, com_cop) and its tests run without MediaPipe installed. Install it only when
you actually process a video:  pip install mediapipe opencv-python

MediaPipe Pose returns 33 landmarks in normalized image coords (x,y in [0,1], z relative
depth, plus visibility). For balance work prefer `world_landmarks` (metric, origin at hips)
when available. NOTE: image y grows DOWNWARD, so we flip it to make z grow upward before
computing COM/COP; world landmarks are already in a metric frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# MediaPipe Pose landmark index -> our canonical keypoint name.
MP_INDEX_TO_NAME: dict[int, str] = {
    0: "nose",
    7: "left_ear", 8: "right_ear",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    31: "left_foot_index", 32: "right_foot_index",
}


def landmark_to_xyz(lm) -> tuple[float, float, float]:
    """Map a MediaPipe landmark to our convention (pure, unit-testable).

    Returns (x = image-horizontal, y = depth, z = VERTICAL up-positive). MediaPipe y grows
    DOWNWARD in both world and image frames, so vertical-up = -lm.y. Keeping this as a separate
    function lets us test the vertical flip without a real video / MediaPipe install.
    """
    return (lm.x, lm.z, -lm.y)


def keypoints_from_video(path: str, use_world: bool = True, model_complexity: int = 2,
                         fallback_to_image: bool = True) -> pd.DataFrame:
    """Run MediaPipe Pose over a video and return a tidy DataFrame.

    Columns: frame, keypoint, x, y, z, visibility. One row per (frame, keypoint). The same
    (x, y, z) = (lm.x, lm.z, -lm.y) remapping (see `landmark_to_xyz`) is applied to BOTH world and
    image landmarks, giving z = vertical up-positive. Raises FileNotFoundError if the path cannot
    be opened. If `use_world` and a frame has no world landmarks, falls back to image landmarks
    when `fallback_to_image` (else that frame is skipped).
    """
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when video is processed
        raise ImportError(
            "Processing video needs mediapipe + opencv-python. "
            "Install with: pip install mediapipe opencv-python"
        ) from exc

    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=model_complexity,
                                  enable_segmentation=False)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        pose.close()
        raise FileNotFoundError(f"cv2 could not open video: {path!r}")
    rows: list[dict] = []
    dropped = 0
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lms = res.pose_world_landmarks if use_world else res.pose_landmarks
            if lms is None and use_world and fallback_to_image:
                lms = res.pose_landmarks
            if lms is None:
                dropped += 1
            if lms is not None:
                for idx, name in MP_INDEX_TO_NAME.items():
                    lm = lms.landmark[idx]
                    # MediaPipe y grows DOWNWARD in both world and image frames. Store a
                    # consistent convention so the rest of the code is unambiguous:
                    #   x = image-horizontal (~medio-lateral for a front view)
                    #   y = depth (~anterior-posterior; weakly observed from a single camera)
                    #   z = VERTICAL, up-positive  (= -lm.y)
                    x, y, z = landmark_to_xyz(lm)
                    rows.append({"frame": frame_idx, "keypoint": name,
                                 "x": x, "y": y, "z": z, "visibility": lm.visibility})
            frame_idx += 1
    finally:
        cap.release()
        pose.close()
    if dropped:
        import warnings
        warnings.warn(f"{dropped} frame(s) had no detectable pose and were skipped", stacklevel=2)
    return pd.DataFrame(rows)


def frame_to_kp_dict(df: pd.DataFrame, frame: int, axes: tuple[str, str, str] = ("x", "y", "z")
                     ) -> dict[str, np.ndarray]:
    """Extract one frame as {keypoint: array([ap, ml, vert])}.

    `axes` maps (anterior_posterior, medio_lateral, vertical) to DataFrame columns. The DEFAULT
    ("x","y","z") matches the bundled synthetic data and conftest (x=AP, y=ML, z=vertical-up).

    For `keypoints_from_video` output (x=image-horizontal, y=depth, z=vertical-up) choose axes by
    camera view: side view -> ("x","y","z"); front/back view -> ("y","x","z"). The vertical axis
    is always the z column (up-positive), so `detect_support_foot` works without extra flags.
    See docs/METHOD.md.
    """
    sub = df[df["frame"] == frame]
    out: dict[str, np.ndarray] = {}
    for _, r in sub.iterrows():
        out[r["keypoint"]] = np.array([r[axes[0]], r[axes[1]], r[axes[2]]], dtype=float)
    return out
