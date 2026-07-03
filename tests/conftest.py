import numpy as np
import pytest


def _sym_pose() -> dict[str, np.ndarray]:
    """A symmetric upright skeleton. Frame: x=anterior-posterior, y=medio-lateral, z=vertical (up)."""
    return {
        "nose":             np.array([0.10, 0.00, 1.62]),
        "left_ear":         np.array([0.00, 0.08, 1.60]),
        "right_ear":        np.array([0.00, -0.08, 1.60]),
        "left_shoulder":    np.array([0.00, 0.18, 1.40]),
        "right_shoulder":   np.array([0.00, -0.18, 1.40]),
        "left_elbow":       np.array([0.00, 0.20, 1.10]),
        "right_elbow":      np.array([0.00, -0.20, 1.10]),
        "left_wrist":       np.array([0.00, 0.20, 0.85]),
        "right_wrist":      np.array([0.00, -0.20, 0.85]),
        "left_hip":         np.array([0.00, 0.10, 0.90]),
        "right_hip":        np.array([0.00, -0.10, 0.90]),
        "left_knee":        np.array([0.00, 0.10, 0.50]),
        "right_knee":       np.array([0.00, -0.10, 0.50]),
        "left_ankle":       np.array([0.00, 0.10, 0.08]),
        "right_ankle":      np.array([0.00, -0.10, 0.08]),
        "left_foot_index":  np.array([0.15, 0.10, 0.00]),
        "right_foot_index": np.array([0.15, -0.10, 0.00]),
    }


@pytest.fixture
def sym_pose():
    return _sym_pose()
