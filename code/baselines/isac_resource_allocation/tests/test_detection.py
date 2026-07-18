"""
Tests for Detection QoS module.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from scipy import stats

from ..src.system_model import ISACSystem
from ..src.detection_qos import DetectionQoS


@pytest.fixture
def system():
    """Create ISAC system for testing."""
    rng = np.random.default_rng(42)
    return ISACSystem(Nt=32, Nr=32, Q=3, K=3, L=1, fc=30e9,
                      P_total=40.0, B_total=100e6, rng=rng)


@pytest.fixture
def detection_qos(system):
    """Create Detection QoS module."""
    return DetectionQoS(system, Pfa=0.01)


def test_detection_prob_range(detection_qos):
    """Test that detection probability is in [0, 1]."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])

    P_D = detection_qos.compute_detection_probability(p, b)

    assert np.all(P_D >= 0), f"Detection probability below 0: {P_D}"
    assert np.all(P_D <= 1), f"Detection probability above 1: {P_D}"


def test_zero_snr_detection_probability_equals_false_alarm(detection_qos):
    """At zero signal, H1 reduces to H0 and P_D must equal P_FA."""
    p = np.zeros(3)
    b = np.full(3, 10e6)
    probability = detection_qos.compute_detection_probability(p, b)
    np.testing.assert_allclose(probability, detection_qos.Pfa, rtol=1e-12)


def test_scaled_central_chi_square_oracle(detection_qos):
    p = np.array([2.0, 3.0, 4.0])
    b = np.array([10e6, 20e6, 30e6])
    sigma = np.array([0.5, 1.5, 2.0])
    snr = p * detection_qos.system.beta_sensing * sigma / (
        detection_qos.system.N0 * b
    )
    expected = np.exp(np.log(detection_qos.Pfa) / (1.0 + snr))
    actual = detection_qos.compute_detection_probability(p, b, sigma)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize(
    "false_alarm",
    [1.0e-17, 1.0e-100, 1.0e-300, np.nextafter(0.0, 1.0)],
)
def test_small_false_alarm_probabilities_are_stable(system, false_alarm):
    """The accepted Pfa domain must not collapse through ``1 - Pfa``."""
    system.N0 = 1.0
    system.beta_sensing = np.ones(system.params.Q)
    system.rcs = np.ones(system.params.Q)
    detector = DetectionQoS(system, Pfa=false_alarm)
    snr = np.array([0.0, 1.0, 1.0e12])
    probability = detector.compute_detection_probability(
        snr, np.ones(system.params.Q)
    )
    expected = np.exp(np.log(false_alarm) / (1.0 + snr))
    np.testing.assert_allclose(probability, expected, rtol=2.0e-15, atol=0.0)
    assert np.all(probability > 0.0)
    assert np.all(np.diff(probability) > 0.0)


def test_threshold_uses_stable_inverse_survival_function(system):
    detector = DetectionQoS(system, Pfa=1.0e-300)
    np.testing.assert_allclose(
        detector._compute_threshold(),
        stats.chi2.isf(1.0e-300, df=2),
        rtol=0.0,
        atol=0.0,
    )


def test_detection_probability_is_monotone_in_snr(system):
    system.N0 = 1.0
    system.beta_sensing = np.ones(system.params.Q)
    system.rcs = np.ones(system.params.Q)
    detector = DetectionQoS(system, Pfa=0.01)
    snr = np.array([0.0, 1.0, 1.0e12])
    probability = detector.compute_detection_probability(
        snr, np.ones(system.params.Q)
    )
    assert probability[0] == pytest.approx(0.01)
    assert np.all(np.diff(probability) > 0.0)
    assert probability[-1] == pytest.approx(1.0, abs=5.0e-12)


def test_unrepresentable_finite_snr_inputs_raise_without_warning():
    system = ISACSystem(Nt=2, Nr=2, Q=1, K=1, L=1)
    system.N0 = 1.0
    system.beta_sensing = np.ones(1)
    system.rcs = np.ones(1)
    detector = DetectionQoS(system)
    with pytest.raises(ValueError, match="outside the finite numerical domain"):
        detector.compute_detection_probability(
            np.array([1.0e308]), np.ones(1), np.array([2.0])
        )


def test_detection_increases_with_power(detection_qos):
    """Test that detection probability increases with power."""
    b = np.array([30e6, 30e6, 25e6])

    p_low = np.array([5.0, 5.0, 5.0])
    p_high = np.array([20.0, 20.0, 20.0])

    P_D_low = detection_qos.compute_detection_probability(p_low, b)
    P_D_high = detection_qos.compute_detection_probability(p_high, b)

    # Higher power should give higher detection probability
    assert np.all(P_D_high >= P_D_low), \
        f"Detection probability should increase with power: low={P_D_low}, high={P_D_high}"


def test_detection_objective_maxmin(detection_qos):
    """Test max-min fairness objective."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])

    obj = detection_qos.compute_objective_maxmin(p, b)
    P_D = detection_qos.compute_detection_probability(p, b)

    assert np.isclose(obj, np.min(P_D)), \
        f"Max-min objective should equal min(P_D): {obj} vs {np.min(P_D)}"


def test_detection_objective_sum(detection_qos):
    """Test sum (comprehensiveness) objective."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])

    obj = detection_qos.compute_objective_sum(p, b)
    P_D = detection_qos.compute_detection_probability(p, b)

    assert np.isclose(obj, np.sum(P_D)), \
        f"Sum objective should equal sum(P_D): {obj} vs {np.sum(P_D)}"


def test_detection_gradient(detection_qos):
    """Test detection probability gradient."""
    p = np.array([10.0, 15.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])

    grad = detection_qos.detection_probability_gradient(p, b)

    assert grad.shape == p.shape, f"Gradient shape mismatch: {grad.shape} vs {p.shape}"
    # Gradient should be positive (more power → higher P_D)
    assert np.all(grad >= 0), f"Gradient should be non-negative: {grad}"


def test_detection_is_detectable(detection_qos):
    """Test detectability check."""
    # High power/bandwidth should make targets detectable
    p_high = np.array([30.0, 30.0, 30.0])
    b_high = np.array([90e6, 90e6, 90e6])

    detectable = detection_qos.is_detectable(p_high, b_high, threshold=0.5)
    assert np.all(detectable), "All targets should be detectable with high resources"

    # Low power should give lower detection probability than high power
    p_low = np.array([1.0, 1.0, 1.0])
    b_low = np.array([10e6, 10e6, 10e6])

    P_D_low = detection_qos.compute_detection_probability(p_low, b_low)
    P_D_high = detection_qos.compute_detection_probability(p_high, b_high)

    # High resources should give higher detection probability
    assert np.all(P_D_high >= P_D_low), \
        f"High resources should give higher detection probability: low={P_D_low}, high={P_D_high}"


def test_detection_with_different_rcs(detection_qos):
    """Test detection probability with different RCS values."""
    p = np.array([10.0, 10.0, 10.0])
    b = np.array([30e6, 30e6, 25e6])

    # Higher RCS should give higher detection probability
    sigma_low = np.array([1.0, 1.0, 1.0])
    sigma_high = np.array([10.0, 10.0, 10.0])

    P_D_low = detection_qos.compute_detection_probability(p, b, sigma_low)
    P_D_high = detection_qos.compute_detection_probability(p, b, sigma_high)

    assert np.all(P_D_high >= P_D_low), \
        f"Higher RCS should give higher detection probability: low={P_D_low}, high={P_D_high}"
