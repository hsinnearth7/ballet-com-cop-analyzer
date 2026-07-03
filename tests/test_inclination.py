import numpy as np
import pytest

from balletcomcop.inclination import (
    decompose_ap_ml,
    inclination_angles,
    inclination_cross_product,
    rcia,
)


def test_com_above_cop_gives_zero_ia():
    com = np.array([0.0, 0.0, 1.0])
    cop = np.array([0.0, 0.0, 0.0])
    sag, fro = inclination_cross_product(com, cop)
    assert sag == pytest.approx(0.0, abs=1e-9)
    assert fro == pytest.approx(0.0, abs=1e-9)


def test_known_ap_offset_matches_arcsin():
    com = np.array([0.1, 0.0, 1.0])  # 0.1 m anterior, 1.0 m up
    cop = np.array([0.0, 0.0, 0.0])
    p = com - cop
    expected = np.degrees(np.arcsin(0.1 / np.linalg.norm(p)))
    sag, fro = inclination_angles(com, cop, ap_axis=0, ml_axis=1, vert_axis=2)
    assert sag == pytest.approx(expected, abs=1e-9)
    assert fro == pytest.approx(0.0, abs=1e-9)


def test_cross_product_and_axis_form_agree():
    rng = np.random.default_rng(0)
    com = rng.normal(size=(20, 3))
    com[:, 2] += 3.0  # keep COM above COP so |P| nonzero and mostly vertical
    cop = rng.normal(size=(20, 3)) * 0.05
    a_sag, a_fro = inclination_cross_product(com, cop)
    b_sag, b_fro = inclination_angles(com, cop, ap_axis=0, ml_axis=1, vert_axis=2)
    # Both now use Lu's convention (frontal = arcsin(-P_ml)), so they agree exactly.
    assert np.allclose(a_sag, b_sag, atol=1e-9)
    assert np.allclose(a_fro, b_fro, atol=1e-9)


def test_frontal_sign_matches_lu_for_positive_ml_lean():
    # COM leaning toward +ML should give NEGATIVE frontal IA under Lu Eq.(3) (arcsin(-P_ml)).
    com = np.array([0.0, 0.10, 1.0])
    cop = np.array([0.0, 0.0, 0.0])
    _, fro = inclination_angles(com, cop, ap_axis=0, ml_axis=1, vert_axis=2)
    assert fro < 0
    _, fro_cp = inclination_cross_product(com, cop)
    assert fro == pytest.approx(fro_cp, abs=1e-12)


def test_rcia_of_linear_ramp_is_constant_slope():
    t = np.linspace(0.0, 2.0, 200)
    slope = 3.5
    ia = slope * t + 1.0
    d = rcia(ia, t)
    # interior points should recover the slope closely
    assert np.allclose(d[5:-5], slope, atol=1e-2)


def test_rcia_rejects_short_series():
    with pytest.raises(ValueError):
        rcia(np.array([1.0, 2.0]), np.array([0.0, 1.0]))


def test_decompose_ap_ml_dominance():
    t = np.linspace(0, 1, 100)
    ap = 5.0 * np.sin(2 * np.pi * t)   # large AP swing
    ml = 0.5 * np.sin(2 * np.pi * t)   # small ML swing
    summ = decompose_ap_ml(ap, ml)
    assert summ["ap_sd"] > summ["ml_sd"]
    assert summ["ap_dominance"] > 0.8


def test_ia_matches_independent_atan2_reference():
    # arctan2(P_i, sqrt(sum of the other two squared)) == arcsin(P_i/|P|): an independent
    # code path that must equal Lu's cross-product IA exactly (NOT the planar atan2(P_x,P_z),
    # which is a DIFFERENT definition that only agrees at small inclinations).
    rng = np.random.default_rng(3)
    com = rng.normal(size=(40, 3))
    com[:, 2] = np.abs(com[:, 2]) + 2.0
    cop = rng.normal(size=(40, 3)) * 0.05
    p = com - cop
    ref_sag = np.degrees(np.arctan2(p[:, 0], np.hypot(p[:, 1], p[:, 2])))
    ref_fro = np.degrees(np.arctan2(-p[:, 1], np.hypot(p[:, 0], p[:, 2])))
    sag, fro = inclination_cross_product(com, cop, deg=True)
    assert np.allclose(sag, ref_sag, atol=1e-9)
    assert np.allclose(fro, ref_fro, atol=1e-9)


def test_sagittal_and_frontal_couple_through_magnitude():
    # Lu's IA = arcsin(component/|P|): adding ML lean GROWS |P|, so the same AP offset yields a
    # SMALLER sagittal IA. The two planes are not independent projections. Documented in METHOD.md.
    cop = np.array([0.0, 0.0, 0.0])
    sag_no_ml, _ = inclination_cross_product(np.array([0.2, 0.0, 1.0]), cop)
    sag_with_ml, _ = inclination_cross_product(np.array([0.2, 0.4, 1.0]), cop)
    assert sag_with_ml < sag_no_ml


def test_rcia_rejects_non_increasing_time():
    ia = np.array([1.0, 2.0, 1.5, 3.0])
    bad_time = np.array([0.0, 0.1, 0.1, 0.2])  # duplicate timestamp
    with pytest.raises(ValueError, match="strictly increasing"):
        rcia(ia, bad_time)


def test_inclination_angles_rejects_equal_axes():
    with pytest.raises(ValueError, match="must differ"):
        inclination_angles(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]),
                           ap_axis=0, ml_axis=0)


def test_cross_product_rejects_non_3d():
    with pytest.raises(ValueError, match="3D"):
        inclination_cross_product(np.array([0.0, 1.0]), np.array([0.0, 0.0]))


def test_decompose_all_nan_returns_nan_dict():
    nan = np.full(5, np.nan)
    out = decompose_ap_ml(nan, nan)
    assert np.isnan(out["ap_sd"]) and np.isnan(out["ap_dominance"])


def test_rcia_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="1D arrays of equal length"):
        rcia(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 1.0, 2.0]))


def test_rcia_direct_nan_raises():
    ia = np.array([1.0, np.nan, 3.0, 4.0])
    t = np.array([0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="NaN"):
        rcia(ia, t)


def test_rcia_constant_signal_returns_zeros():
    t = np.linspace(0, 1, 10)
    ia = np.full(10, 5.0)
    assert np.allclose(rcia(ia, t), 0.0)
