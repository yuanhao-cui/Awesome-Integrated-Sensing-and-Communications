"""Semantic tests for the validated ISAC system model."""

import numpy as np
import pytest

from ..src.system_model import ISACSystemModel, dbm_to_watt


def test_dbm_conversion_uses_power_units() -> None:
    assert dbm_to_watt(30.0) == pytest.approx(1.0)
    assert dbm_to_watt(0.0) == pytest.approx(1e-3)
    assert dbm_to_watt(-80.0) == pytest.approx(1e-11)


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_dbm_conversion_rejects_nonfinite_input(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        dbm_to_watt(value)


def test_dbm_conversion_has_explicit_binary64_domain() -> None:
    assert dbm_to_watt(3000.0) == pytest.approx(1.0e297, rel=2e-13)
    assert dbm_to_watt(-3000.0) == pytest.approx(1.0e-303, rel=2e-13)
    with pytest.raises(OverflowError, match="above"):
        dbm_to_watt(1.0e308)
    with pytest.raises(FloatingPointError, match="below"):
        dbm_to_watt(-1.0e308)


def test_paper_array_regime_is_enforced() -> None:
    with pytest.raises(ValueError, match="K <= M <= N"):
        ISACSystemModel(M=2, K=3, N=4)
    with pytest.raises(ValueError, match="K <= M <= N"):
        ISACSystemModel(M=5, K=1, N=4)


def test_seeded_synthetic_channel_is_reproducible_and_copy_safe() -> None:
    first = ISACSystemModel(M=5, K=2, N=6, seed=2026)
    second = ISACSystemModel(M=5, K=2, N=6, seed=2026)
    np.testing.assert_array_equal(first.get_csi(), second.get_csi())
    copied = first.get_csi()
    copied[0, 0] = 0.0
    assert first.get_csi()[0, 0] != 0.0


def test_steering_vectors_follow_paper_cosine_convention() -> None:
    model = ISACSystemModel(M=4, K=1, N=5)
    broadside = model.steering_vector_tx(np.pi / 2.0)
    endfire = model.steering_vector_tx(0.0)
    np.testing.assert_allclose(broadside, np.ones(4), atol=1e-14)
    np.testing.assert_allclose(endfire, (-1.0) ** np.arange(4), atol=1e-14)


@pytest.mark.parametrize("which", ["tx", "rx"])
def test_steering_derivative_matches_central_difference(which: str) -> None:
    model = ISACSystemModel(M=4, K=1, N=6)
    theta = 1.1
    step = 1e-6
    vector = getattr(model, f"steering_vector_{which}")
    derivative = getattr(model, f"steering_derivative_{which}")
    finite_difference = (vector(theta + step) - vector(theta - step)) / (2.0 * step)
    np.testing.assert_allclose(
        derivative(theta), finite_difference, rtol=2e-9, atol=2e-9
    )


def test_broadside_derivative_is_not_zero() -> None:
    model = ISACSystemModel(M=4, K=1, N=5)
    derivative = model.steering_derivative_tx(np.pi / 2.0)
    assert np.linalg.norm(derivative) > 1.0


def test_sinr_matches_hand_computation() -> None:
    model = ISACSystemModel(
        M=2,
        K=2,
        N=2,
        sigma_c_dbm=0.0,
    )
    model.set_csi(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex))
    W = np.array([[2.0, 0.5], [0.0, 1.0]], dtype=complex)
    expected = 4.0 / (1e-3 + 0.25)
    assert model.compute_sinr(0, W) == pytest.approx(expected)
    assert model.compute_total_power(W) == pytest.approx(5.25)


def test_system_sinr_uses_shared_weak_interference_path() -> None:
    model = ISACSystemModel(
        M=2,
        K=2,
        N=2,
        sigma_c_dbm=-270.0,
    )
    model.set_csi(np.array([[1.0, 0.0], [0.0, 1.0]]))
    W = np.array([[1.0, 1.0e-10], [0.0, 0.0]])
    assert model.compute_sinr(0, W) == pytest.approx(
        9.999999999000001e19, rel=3e-15
    )


def test_system_total_power_shares_explicit_range_contract() -> None:
    model = ISACSystemModel(M=1, K=1, N=1)
    assert model.compute_total_power(np.array([[1.0e150]])) == pytest.approx(
        1.0e300, rel=2e-15
    )
    with pytest.raises(OverflowError, match="squared norm exceeds"):
        model.compute_total_power(np.array([[1.0e200]]))
    with pytest.raises(FloatingPointError, match="squared norm underflows"):
        model.compute_total_power(np.array([[1.0e-200]]))


@pytest.mark.parametrize(
    ("keyword", "value", "exception"),
    (
        ("P_max_dbm", np.nan, ValueError),
        ("P0_dbm", np.inf, ValueError),
        ("sigma_c_dbm", 1.0e308, OverflowError),
        ("sigma_s_dbm", -1.0e308, FloatingPointError),
        ("wavelength", np.nan, ValueError),
        ("wavelength", np.inf, ValueError),
        ("d", np.nan, ValueError),
        ("d", np.inf, ValueError),
    ),
)
def test_model_rejects_nonphysical_or_unrepresentable_scalars(
    keyword: str,
    value: float,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        ISACSystemModel(**{keyword: value})


def test_invalid_shapes_fail_explicitly() -> None:
    model = ISACSystemModel(M=4, K=1, N=5)
    with pytest.raises(ValueError, match="shape"):
        model.set_csi(np.ones((2, 4)))
    with pytest.raises(ValueError, match="shape"):
        model.compute_sinr(0, np.ones((4, 2)))
    with pytest.raises(IndexError, match="outside"):
        model.get_channel(1)
