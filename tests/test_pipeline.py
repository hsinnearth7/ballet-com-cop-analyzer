import numpy as np
import pandas as pd
import pytest

from balletcomcop.pipeline import analyze_frames
from balletcomcop.pose_to_keypoints import frame_to_kp_dict


def _moving_pose(base, ap_shift):
    kp = {k: v.copy() for k, v in base.items()}
    for k in kp:
        if k not in ("right_ankle", "right_foot_index"):  # plant the support foot
            kp[k] = kp[k] + np.array([ap_shift, 0.0, 0.0])
    return kp


def test_analyze_frames_runs_and_varies(sym_pose):
    kp_by_frame = {i: _moving_pose(sym_pose, 0.01 * np.sin(i / 5.0)) for i in range(30)}
    res = analyze_frames(kp_by_frame, fps=60.0, sex="female", support="right")
    assert len(res) == 30
    assert res["ia_sagittal"].std() > 0  # IA actually moves
    assert res["rcia_sagittal"].notna().all()


def test_analyze_frames_tolerates_a_nan_frame(sym_pose):
    # A collapsed frame (all keypoints identical) -> COM == COP -> NaN IA. Must NOT crash.
    kp_by_frame = {i: _moving_pose(sym_pose, 0.01 * i) for i in range(20)}
    collapsed = {k: np.array([0.0, 0.0, 0.0]) for k in sym_pose}
    kp_by_frame[10] = collapsed
    res = analyze_frames(kp_by_frame, fps=60.0, sex="female", support="right")
    assert np.isnan(res.loc[10, "ia_sagittal"])          # bad frame is NaN
    assert res["rcia_sagittal"].notna().sum() >= 15      # the rest still computed


def test_frame_to_kp_dict_default_axes_match_synthetic_convention():
    # Default ("x","y","z") must map (ap, ml, vert) = (x, y, z) for the repo's data convention.
    df = pd.DataFrame([
        {"frame": 0, "keypoint": "left_hip", "x": 0.1, "y": 0.2, "z": 0.9},
    ])
    kp = frame_to_kp_dict(df, 0)
    assert np.allclose(kp["left_hip"], [0.1, 0.2, 0.9])


def test_analyze_frames_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        analyze_frames({}, fps=60.0)


def test_safe_rcia_all_nan_when_too_few_finite(sym_pose):
    # 20 frames but 18 collapsed -> < 4 finite IA -> rcia must be all-NaN, not crash.
    kp_by_frame = {i: {k: np.zeros(3) for k in sym_pose} for i in range(20)}
    kp_by_frame[0] = _moving_pose(sym_pose, 0.0)
    kp_by_frame[1] = _moving_pose(sym_pose, 0.01)
    res = analyze_frames(kp_by_frame, fps=60.0, sex="female", support="right")
    assert res["rcia_sagittal"].isna().all()
