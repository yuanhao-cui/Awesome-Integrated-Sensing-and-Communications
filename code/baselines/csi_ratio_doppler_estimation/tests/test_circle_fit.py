"""Tests for circle fitting algorithms."""

import numpy as np
import pytest
from ..src.circle_fit import (
    circle_fit_error,
    fit_circle_iterative_weighted,
    fit_circle_kasa,
    least_squares_circle_fit,
)


def generate_circle_samples(center_A, center_B, radius, n_samples=100, noise_std=0.0):
    """Generate samples on a circle with optional noise."""
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    x = center_A + radius * np.cos(angles) + noise_std * np.random.randn(n_samples)
    y = center_B + radius * np.sin(angles) + noise_std * np.random.randn(n_samples)
    return x + 1j * y


def test_circle_fit_accuracy():
    """Circle fitting recovers center within tolerance."""
    # Test case 1: Centered at origin
    R = generate_circle_samples(0, 0, 1.0, n_samples=200, noise_std=0.0)
    A, B, r = least_squares_circle_fit(R)
    assert abs(A) < 1e-10, f"Center A={A} should be ~0"
    assert abs(B) < 1e-10, f"Center B={B} should be ~0"
    assert abs(r - 1.0) < 1e-10, f"Radius r={r} should be ~1"

    # Test case 2: Off-center
    R = generate_circle_samples(2.5, -1.3, 3.7, n_samples=300, noise_std=0.0)
    A, B, r = least_squares_circle_fit(R)
    assert abs(A - 2.5) < 1e-8, f"Center A={A} should be ~2.5"
    assert abs(B - (-1.3)) < 1e-8, f"Center B={B} should be ~-1.3"
    assert abs(r - 3.7) < 1e-8, f"Radius r={r} should be ~3.7"


def test_circle_fit_with_noise():
    """Circle fitting works with noisy samples."""
    true_A, true_B, true_r = 1.5, -0.8, 2.0
    noise_std = 0.05  # 5% noise relative to radius

    np.random.seed(42)
    R = generate_circle_samples(true_A, true_B, true_r,
                                n_samples=200, noise_std=noise_std)

    A, B, r = least_squares_circle_fit(R)

    # Should be within 5% of true values
    assert abs(A - true_A) / true_r < 0.05
    assert abs(B - true_B) / true_r < 0.05
    assert abs(r - true_r) / true_r < 0.05


def test_circle_fit_methods_agree():
    """All circle fitting methods give similar results for clean data."""
    R = generate_circle_samples(1.0, -2.0, 1.5, n_samples=150)

    A1, B1, r1 = least_squares_circle_fit(R)
    A2, B2, r2 = fit_circle_kasa(R)
    A3, B3, r3 = fit_circle_iterative_weighted(R)

    # All methods should agree within 1%
    for (A, B, r) in [(A2, B2, r2), (A3, B3, r3)]:
        assert abs(A - A1) / r1 < 0.01
        assert abs(B - B1) / r1 < 0.01
        assert abs(r - r1) / r1 < 0.01


def test_circle_fit_error():
    """Circle fit error computation is correct."""
    R = generate_circle_samples(0, 0, 1.0, n_samples=100)
    error = circle_fit_error(R, 0, 0, 1.0)
    assert error < 1e-10, f"Error for perfect circle should be ~0, got {error}"

    # With wrong center
    error_wrong = circle_fit_error(R, 0.1, 0, 1.0)
    assert error_wrong > error, "Wrong center should give larger error"


@pytest.mark.parametrize(
    "samples",
    [
        np.ones(8, dtype=complex),
        np.arange(8, dtype=float).astype(complex),
        np.array([0 + 0j, 1 + 0j]),
    ],
)
def test_degenerate_circle_samples_are_rejected(samples):
    with pytest.raises(ValueError, match="at least 3|degenerate"):
        least_squares_circle_fit(samples)


def test_circle_fit_error_validates_circle():
    with pytest.raises(ValueError, match="radius"):
        circle_fit_error(np.array([1 + 0j, 0 + 1j]), 0.0, 0.0, 0.0)


@pytest.mark.parametrize("scale", [1e-100, 1e-20, 1.0, 1e20, 1e100])
def test_circle_fit_is_scale_equivariant(scale):
    samples = scale * generate_circle_samples(0.25, -0.5, 1.5, n_samples=128)
    center_a, center_b, radius = least_squares_circle_fit(samples)
    np.testing.assert_allclose(
        [center_a, center_b, radius],
        scale * np.array([0.25, -0.5, 1.5]),
        rtol=2e-14,
        atol=0.0,
    )


@pytest.mark.parametrize("radius", [1e-4, 1e-6, 1e-8, 1e-9, 1e-10])
def test_small_circle_away_from_origin_is_resolved(radius):
    angles = np.linspace(0.0, 2.0 * np.pi, 100, endpoint=False)
    samples = 1.0 + 2.0j + radius * np.exp(1j * angles)
    center_a, center_b, fitted_radius = least_squares_circle_fit(samples)
    # The tightest achievable relative tolerance is limited by quantization
    # of the already-constructed binary64 samples around a magnitude-2 center.
    quantization_limit = 20.0 * np.finfo(float).eps * np.sqrt(5.0) / radius
    assert abs(fitted_radius - radius) / radius <= max(2e-12, quantization_limit)
    assert circle_fit_error(samples, center_a, center_b, fitted_radius) <= (
        5.0 * np.finfo(float).eps * np.sqrt(5.0)
    )


if __name__ == "__main__":
    test_circle_fit_accuracy()
    test_circle_fit_with_noise()
    test_circle_fit_methods_agree()
    test_circle_fit_error()
    print("All circle fit tests passed!")
