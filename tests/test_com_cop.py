import numpy as np
import pytest

from balletcomcop.com_cop import (
    com_and_cop,
    cop_proxy_double_leg,
    cop_proxy_single_leg,
    detect_support_foot,
)


def test_single_leg_proxy_is_ankle_toe_midpoint(sym_pose):
    cop = cop_proxy_single_leg(sym_pose, "left")
    expected = 0.5 * (sym_pose["left_ankle"] + sym_pose["left_foot_index"])
    assert cop == pytest.approx(expected, abs=1e-12)


def test_double_leg_proxy_between_feet(sym_pose):
    cop = cop_proxy_double_leg(sym_pose)
    assert cop[1] == pytest.approx(0.0, abs=1e-12)  # symmetric -> ML midline


def test_detect_support_foot_picks_lower_ankle(sym_pose):
    kp = dict(sym_pose)
    kp["right_ankle"] = kp["right_ankle"].copy()
    kp["right_ankle"][2] = 0.30  # lift right foot -> left is support
    assert detect_support_foot(kp) == "left"


def test_com_and_cop_returns_two_3d_points(sym_pose):
    com, cop = com_and_cop(sym_pose, sex="female", support="left")
    assert com.shape == (3,)
    assert cop.shape == (3,)


def test_invalid_support_raises(sym_pose):
    with pytest.raises(ValueError):
        cop_proxy_single_leg(sym_pose, "both")


def test_detect_support_foot_rejects_2d(sym_pose):
    kp = {k: v[:2] for k, v in sym_pose.items()}  # drop to 2D
    with pytest.raises(ValueError, match="vert_axis"):
        detect_support_foot(kp)


def test_com_and_cop_double_leg_support(sym_pose):
    com, cop = com_and_cop(sym_pose, sex="female", support="double")
    expected = 0.5 * (cop_proxy_single_leg(sym_pose, "left") + cop_proxy_single_leg(sym_pose, "right"))
    assert np.allclose(cop, expected)


def test_com_and_cop_missing_keypoint_raises_value_error(sym_pose):
    # an occluded upper-body landmark must give compute_com's descriptive
    # ValueError, not a raw KeyError from midpoint derivation
    kp = {k: v for k, v in sym_pose.items() if k != "right_shoulder"}
    with pytest.raises(ValueError, match="right_shoulder"):
        com_and_cop(kp, sex="female", support="left")
