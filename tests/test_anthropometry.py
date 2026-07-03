import numpy as np
import pytest

from balletcomcop.anthropometry import _SEGMENTS, compute_com, derive_midpoints, total_mass_fraction


@pytest.mark.parametrize("sex", ["male", "female"])
def test_mass_fractions_sum_to_one(sex):
    assert total_mass_fraction(sex) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("sex", ["male", "female"])
def test_com_of_collapsed_skeleton_is_that_point(sex, sym_pose):
    # every keypoint at the same location -> COM must equal that location
    pt = np.array([0.3, -0.2, 1.0])
    kp = {k: pt.copy() for k in sym_pose}
    com = compute_com(kp, sex=sex)
    assert com == pytest.approx(pt, abs=1e-9)


def test_com_is_translation_equivariant(sym_pose):
    shift = np.array([1.0, 2.0, 3.0])
    base = compute_com(sym_pose, sex="female")
    shifted_kp = {k: v + shift for k, v in derive_midpoints(sym_pose).items()}
    shifted = compute_com(shifted_kp, sex="female")
    assert shifted == pytest.approx(base + shift, abs=1e-9)


def test_symmetric_pose_has_zero_mediolateral_com(sym_pose):
    com = compute_com(sym_pose, sex="female")
    assert com[1] == pytest.approx(0.0, abs=1e-9)  # ML axis


def test_com_inside_bounding_box(sym_pose):
    kp = derive_midpoints(sym_pose)
    pts = np.stack(list(kp.values()))
    com = compute_com(sym_pose, sex="male")
    assert np.all(com >= pts.min(axis=0) - 1e-9)
    assert np.all(com <= pts.max(axis=0) + 1e-9)


def test_invalid_sex_raises(sym_pose):
    with pytest.raises(ValueError):
        compute_com(sym_pose, sex="other")


def test_deleva_com_ratios_match_authoritative_table():
    # Guard the De Leva (1996) CoM ratios against regression (verified vs Visual3D/HAS-Motion).
    ratios = {s.name: (s.com_m, s.com_f) for s in _SEGMENTS}
    assert ratios["trunk"] == pytest.approx((0.5138, 0.4964), abs=1e-4)
    assert ratios["shank"] == pytest.approx((0.4395, 0.4352), abs=1e-4)
    assert ratios["thigh"] == pytest.approx((0.4095, 0.3612), abs=1e-4)
    assert ratios["upper_arm"] == pytest.approx((0.5772, 0.5754), abs=1e-4)


def test_compute_com_with_partial_midpoints_does_not_crash(sym_pose):
    # mid_shoulder/mid_hip present but mid_ear absent must NOT KeyError (it must re-derive).
    kp = dict(sym_pose)
    kp["mid_shoulder"] = 0.5 * (kp["left_shoulder"] + kp["right_shoulder"])
    kp["mid_hip"] = 0.5 * (kp["left_hip"] + kp["right_hip"])
    com = compute_com(kp, sex="female")
    assert com.shape == (3,)


def test_compute_com_missing_keypoint_raises(sym_pose):
    kp = dict(sym_pose)
    del kp["left_knee"]
    with pytest.raises(ValueError, match="missing keypoints"):
        compute_com(kp, sex="female")


def test_compute_com_cannot_derive_midpoints_raises(sym_pose):
    kp = {k: v for k, v in sym_pose.items() if k not in ("left_ear", "right_ear")}
    with pytest.raises(ValueError, match="cannot derive midpoints"):
        compute_com(kp, sex="female")
