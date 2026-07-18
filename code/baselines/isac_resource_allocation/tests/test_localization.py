"""
Tests for Localization QoS module.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from ..src.system_model import ISACSystem
from ..src.localization_qos import LocalizationQoS


@pytest.fixture
def system():
    """Create ISAC system for testing."""
    rng = np.random.default_rng(42)
    return ISACSystem(Nt=32, Nr=32, Q=3, K=3, L=1, fc=30e9,
                      P_total=40.0, B_total=100e6, rng=rng)


@pytest.fixture
def localization_qos(system):
    """Create Localization QoS module."""
    return LocalizationQoS(system, w_d=1.0, w_theta=1.0)


def test_crb_decreases_with_power(localization_qos):
    """Test that CRB decreases with more power."""
    b = np.array([30e6, 30e6, 25e6])

    p_low = np.array([5.0, 5.0, 5.0])
    p_high = np.array([20.0, 20.0, 20.0])

    crb_low = localization_qos.compute_crb_range(p_low, b)
    crb_high = localization_qos.compute_crb_range(p_high, b)

    # Higher power should give lower CRB (better estimation)
    assert np.all(crb_low >= crb_high), \
        f"CRB should decrease with power: low={crb_low}, high={crb_high}"


def test_crb_decreases_with_bandwidth(localization_qos):
    """Test that range CRB decreases with more bandwidth."""
    p = np.array([10.0, 10.0, 10.0])

    b_low = np.array([10e6, 10e6, 10e6])
    b_high = np.array([50e6, 50e6, 50e6])

    crb_low = localization_qos.compute_crb_range(p, b_low)
    crb_high = localization_qos.compute_crb_range(p, b_high)

    # Higher bandwidth should give lower CRB (better range estimation)
    assert np.all(crb_low >= crb_high), \
        f"Range CRB should decrease with bandwidth: low={crb_low}, high={crb_high}"


def test_crb_angle_with_bandwidth(localization_qos):
    """Allocated bandwidth must not change the fixed angle-noise model."""
    p = np.array([10.0, 10.0, 10.0])

    b_low = np.array([10e6, 10e6, 10e6])
    b_high = np.array([50e6, 50e6, 50e6])

    crb_theta_low = localization_qos.compute_crb_angle(p, b_low)
    crb_theta_high = localization_qos.compute_crb_angle(p, b_high)

    np.testing.assert_array_equal(crb_theta_high, crb_theta_low)


def test_crb_combined(localization_qos):
    """Test combined CRB metric."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])

    rho = localization_qos.compute_crb_combined(p, b)

    # The score is dimensionless because each information term is multiplied
    # by the square of a same-unit reference scale.
    assert np.all(rho > 0), f"Combined CRB should be positive: {rho}"

    # Manual verification
    crb_d = localization_qos.compute_crb_range(p, b)
    crb_theta = localization_qos.compute_crb_angle(p, b)
    rho_manual = (
        localization_qos.w_d
        * localization_qos.range_reference_m**2
        / crb_d
        + localization_qos.w_theta
        * localization_qos.angle_reference_rad**2
        / crb_theta
    )

    np.testing.assert_allclose(rho, rho_manual, rtol=1e-5)


def test_localization_rmse(localization_qos):
    """Test RMSE computation."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])

    rmse_range, rmse_angle = localization_qos.compute_localization_rmse(p, b)

    # RMSE should be positive
    assert np.all(rmse_range > 0), f"Range RMSE should be positive: {rmse_range}"
    assert np.all(rmse_angle > 0), f"Angle RMSE should be positive: {rmse_angle}"

    # RMSE should equal sqrt(CRB)
    crb_d = localization_qos.compute_crb_range(p, b)
    crb_theta = localization_qos.compute_crb_angle(p, b)

    np.testing.assert_allclose(rmse_range, np.sqrt(crb_d), rtol=1e-5)
    np.testing.assert_allclose(rmse_angle, np.sqrt(crb_theta), rtol=1e-5)


def test_objective_sum(localization_qos):
    """Test sum objective."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])

    obj = localization_qos.compute_objective_sum(p, b)
    rho = localization_qos.compute_crb_combined(p, b)

    assert np.isclose(obj, np.sum(rho)), \
        f"Sum objective should equal sum(rho): {obj} vs {np.sum(rho)}"


def test_objective_proportional_fairness(localization_qos):
    """Test proportional fairness objective."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])

    obj = localization_qos.compute_objective_proportional_fairness(p, b)
    rho = localization_qos.compute_crb_combined(p, b)

    assert np.isclose(
        obj, np.sum(np.log(np.maximum(rho, np.finfo(float).tiny)))
    ), \
        f"Proportional fairness should equal sum(log(rho)): {obj}"


def test_fisher_information_matrix(localization_qos):
    """Test Fisher Information Matrix computation."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])

    fim = localization_qos.compute_fim(p, b)

    # FIM should be positive semi-definite
    for q in range(fim.shape[0]):
        eigenvalues = np.linalg.eigvalsh(fim[q])
        assert np.all(eigenvalues >= -1e-10), \
            f"FIM should be PSD for target {q}: eigenvalues={eigenvalues}"

    # Diagonal elements should be positive
    for q in range(fim.shape[0]):
        assert fim[q, 0, 0] > 0, f"FIM[0,0] should be positive for target {q}"
        assert fim[q, 1, 1] > 0, f"FIM[1,1] should be positive for target {q}"

    # Independent analytic construction, rather than calling either CRB
    # helper, verifies the declared information matrix constants.
    gain = (
        localization_qos.system.beta_sensing * localization_qos.system.rcs
    )
    range_oracle = (
        8.0
        * np.pi**2
        * p
        * gain
        * b
        / (localization_qos.system.N0 * localization_qos.c**2)
    )
    angle_snr = p * gain / (
        localization_qos.system.N0
        * localization_qos.angle_noise_bandwidth_hz
    )
    antennas = localization_qos.system.params.Nt
    angle_oracle = (
        angle_snr
        * antennas
        * (antennas**2 - 1)
        * np.pi**2
        * np.cos(localization_qos.system.target_angles) ** 2
        * localization_qos.d_lambda**2
        / 6.0
    )
    np.testing.assert_allclose(fim[:, 0, 0], range_oracle, rtol=1e-13)
    np.testing.assert_allclose(fim[:, 1, 1], angle_oracle, rtol=1e-13)
    inverse = np.stack([np.diag(np.linalg.inv(matrix)) for matrix in fim])
    expected_bound = np.column_stack(
        [
            localization_qos.compute_crb_range(p, b),
            localization_qos.compute_crb_angle(p, b),
        ]
    )
    np.testing.assert_allclose(inverse, expected_bound, rtol=1e-13)


def test_validate_localization_performance(localization_qos):
    """Test localization performance validation."""
    # High resources should meet performance requirements
    p_high = np.array([30.0, 30.0, 30.0])
    b_high = np.array([90e6, 90e6, 90e6])

    valid = localization_qos.validate_localization_performance(
        p_high, b_high, max_range_error=100.0, max_angle_error=1.0)
    assert valid, "High resources should meet relaxed performance requirements"

    # Low resources should not meet strict requirements
    p_low = np.array([0.1, 0.1, 0.1])
    b_low = np.array([1e6, 1e6, 1e6])

    valid = localization_qos.validate_localization_performance(
        p_low, b_low, max_range_error=0.01, max_angle_error=0.001)
    assert not valid, "Low resources should not meet strict performance requirements"


def test_zero_power_has_infinite_bounds_and_zero_information(localization_qos):
    power = np.zeros(3)
    bandwidth = np.full(3, 10e6)
    assert np.all(np.isinf(localization_qos.compute_crb_range(power, bandwidth)))
    assert np.all(np.isinf(localization_qos.compute_crb_angle(power, bandwidth)))
    np.testing.assert_array_equal(
        localization_qos.compute_crb_combined(power, bandwidth), np.zeros(3)
    )


def test_information_score_coefficients_are_exact(localization_qos):
    bandwidth = np.array([12e6, 18e6, 25e6])
    coefficient = localization_qos.compute_information_score_coefficients(
        bandwidth
    )
    power = np.array([2.0, 3.0, 5.0])
    score = localization_qos.compute_information_score(power, bandwidth)
    np.testing.assert_allclose(score, coefficient * power, rtol=2.0e-15)


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("w_d", -1.0),
        ("w_theta", np.nan),
        ("range_reference_m", 0.0),
        ("angle_reference_rad", -1.0),
        ("angle_noise_bandwidth_hz", np.inf),
        ("d_lambda", 0.0),
        ("d_lambda", 1.0e308),
        ("range_reference_m", 1.0e308),
    ],
)
def test_dimensionless_score_parameters_are_validated(system, keyword, value):
    with pytest.raises(ValueError):
        LocalizationQoS(system, **{keyword: value})


def test_at_least_one_information_weight_must_be_positive(system):
    with pytest.raises(ValueError, match="at least one"):
        LocalizationQoS(system, w_d=0.0, w_theta=0.0)


def test_unrepresentable_finite_information_raises_without_warning():
    system = ISACSystem(Nt=2, Nr=2, Q=1, K=1, L=1)
    system.N0 = 1.0
    system.beta_sensing = np.ones(1)
    system.rcs = np.ones(1)
    localization = LocalizationQoS(system)
    with pytest.raises(ValueError, match="outside the finite numerical domain"):
        localization.compute_information_components(
            np.array([1.0e308]), np.ones(1)
        )
