"""Tests for the local circle-phase rotation estimator."""

import numpy as np
import pytest
from ..src.signal_model import csi_with_doppler
from ..src.csi_ratio import compute_csi_ratio
from ..src.mobius_estimator import mobius_doppler_estimate


def generate_csi_ratio_samples(f_D, T_s, N, snr_db=np.inf):
    """Generate CSI-ratio samples with known Doppler frequency."""
    t = np.arange(N) * T_s
    H1, H2 = csi_with_doppler(t, f_D, snr_db=snr_db)
    R = compute_csi_ratio(H1, H2)
    return R, t


def test_mobius_doppler_estimate():
    """The adapter estimates an exact pure-rotation oracle."""
    T_s = 0.0005  # 2 kHz sampling
    N = 128
    f_D_true = 50.0  # Hz

    R, t = generate_csi_ratio_samples(f_D_true, T_s, N, snr_db=np.inf)
    result = mobius_doppler_estimate(R, T_s)

    assert abs(result['f_D'] - f_D_true) / f_D_true < 0.01, \
        f"Estimated f_D={result['f_D']:.2f} != true {f_D_true:.2f}"

    assert result['direction'] == 'unknown'
    assert result['rotation_sign'] == 1
    assert result['r_squared'] > 0.99, "Linear fit should be excellent for clean data"


def test_mobius_negative_doppler():
    """The local adapter preserves negative synthetic rotation sign."""
    T_s = 0.0005
    N = 128
    f_D_true = -30.0  # Receding target

    R, t = generate_csi_ratio_samples(f_D_true, T_s, N)
    result = mobius_doppler_estimate(R, T_s)

    assert abs(result['f_D'] - f_D_true) / abs(f_D_true) < 0.05, \
        f"Estimated f_D={result['f_D']:.2f} != true {f_D_true:.2f}"

    assert result['direction'] == 'unknown'
    assert result['rotation_sign'] == -1
    assert result['f_D'] < 0


def test_mobius_with_noise():
    """Mobius estimator is robust to moderate noise."""
    T_s = 0.0005
    N = 128
    f_D_true = 40.0
    snr_db = 15.0  # 15 dB SNR

    errors = []
    rng = np.random.default_rng(42)
    for _ in range(10):
        t = np.arange(N) * T_s
        H1, H2 = csi_with_doppler(t, f_D_true, snr_db=snr_db, rng=rng)
        R = compute_csi_ratio(H1, H2)
        result = mobius_doppler_estimate(R, T_s)
        errors.append(abs(result['f_D'] - f_D_true) / f_D_true)

    avg_error = np.mean(errors)
    assert avg_error < 0.15, f"Average error {avg_error:.2%} should be < 15% at 15 dB SNR"


def test_mobius_circle_fit_quality():
    """Circle fit quality metrics are reasonable."""
    T_s = 0.0005
    N = 128
    f_D_true = 60.0

    R, t = generate_csi_ratio_samples(f_D_true, T_s, N)
    result = mobius_doppler_estimate(R, T_s)

    # Circle fit should be excellent for synthetic data
    assert result['rms_error'] < 0.01, \
        f"RMS circle fit error {result['rms_error']:.6f} should be very small"

    assert result['r_squared'] > 0.95


def test_mobius_different_frequencies():
    """Mobius estimator works for various Doppler frequencies."""
    T_s = 0.0005
    N = 128

    test_frequencies = [10, 25, 50, 75, 100, 200, 500]

    for f_D in test_frequencies:
        R, t = generate_csi_ratio_samples(f_D, T_s, N)
        result = mobius_doppler_estimate(R, T_s)
        rel_error = abs(result['f_D'] - f_D) / f_D

        assert rel_error < 0.05, \
            f"f_D={f_D}: estimated {result['f_D']:.2f}, error {rel_error:.2%}"


def test_stationary_ratio_is_rejected():
    with pytest.raises(ValueError, match="degenerate"):
        mobius_doppler_estimate(np.ones(32, dtype=complex), 0.001)


def test_unknown_circle_method_is_rejected():
    ratio, _ = generate_csi_ratio_samples(25.0, 0.001, 64)
    with pytest.raises(ValueError, match="circle_method"):
        mobius_doppler_estimate(ratio, 0.001, circle_method="pratt")


@pytest.mark.parametrize("magnitude", [1e-100, 1e-20, 1.0, 1e20, 1e100])
def test_rotation_estimate_is_invariant_to_nonzero_complex_scaling(magnitude):
    ratio, _ = generate_csi_ratio_samples(37.0, 0.0005, 256)
    baseline = mobius_doppler_estimate(ratio, 0.0005)
    scaled = mobius_doppler_estimate(magnitude * np.exp(0.7j) * ratio, 0.0005)
    assert scaled["f_D"] == pytest.approx(baseline["f_D"], rel=1e-10)
    assert scaled["alias_limit_hz"] == 1000.0
    assert scaled["alias_ambiguous"] is True


def test_nonuniform_mobius_traversal_fails_validity_gate():
    time = np.arange(503, dtype=float)
    z = np.exp(1j * 2.0 * np.pi * 0.2 * time)
    ratio = z / (1.0 + 0.9999 * z)
    assert np.min(np.abs(1.0 + 0.9999 * z)) > 0.6
    with pytest.raises(ValueError, match=r"invalid.*R\^2"):
        mobius_doppler_estimate(ratio, 1.0)


@pytest.mark.parametrize("sample_interval", [1.0, 1e-3, 1e-9, 1e-12])
def test_regression_is_invariant_to_time_units(sample_interval):
    frequency = 0.05 / sample_interval
    ratio, _ = generate_csi_ratio_samples(frequency, sample_interval, 128)
    result = mobius_doppler_estimate(ratio, sample_interval)
    assert result["f_D"] == pytest.approx(frequency, rel=1e-10)


if __name__ == "__main__":
    test_mobius_doppler_estimate()
    test_mobius_negative_doppler()
    test_mobius_with_noise()
    test_mobius_circle_fit_quality()
    test_mobius_different_frequencies()
    print("All Mobius estimator tests passed!")
