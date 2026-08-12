"""Property-based tests (hypothesis): IA invariants over thousands of random inputs."""
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from balletcomcop.inclination import inclination_cross_product

_coord = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)


@given(_coord, _coord, _coord, _coord)
@settings(max_examples=400)
def test_ia_always_within_plus_minus_90_deg(px, py, vert, copz):
    com = np.array([px, py, abs(vert) + 1.0])  # keep COM above COP -> |P| > 0
    cop = np.array([0.0, 0.0, copz])
    sag, fro = inclination_cross_product(com, cop, deg=True)
    if not (np.isnan(sag) or np.isnan(fro)):
        assert -90.0 <= sag <= 90.0
        assert -90.0 <= fro <= 90.0


@given(_coord, _coord, _coord, st.floats(min_value=0.1, max_value=10.0, allow_nan=False))
@settings(max_examples=300)
def test_ia_invariant_to_scaling_the_com_cop_vector(px, py, pz, scale):
    cop = np.array([0.3, -0.2, 0.1])
    p = np.array([px, py, abs(pz) + 0.5])
    com1 = cop + p
    com2 = cop + scale * p          # same direction, scaled magnitude
    a = inclination_cross_product(com1, cop)
    b = inclination_cross_product(com2, cop)
    assert np.allclose(a, b, atol=1e-9)


@given(_coord, _coord, _coord, _coord, _coord, _coord)
@settings(max_examples=300)
def test_ia_invariant_to_common_translation(px, py, pz, tx, ty, tz):
    com = np.array([px, py, abs(pz) + 1.0])
    cop = np.array([0.0, 0.0, 0.0])
    shift = np.array([tx, ty, tz])
    a = inclination_cross_product(com, cop)
    b = inclination_cross_product(com + shift, cop + shift)  # P unchanged
    assert np.allclose(a, b, atol=1e-9)
