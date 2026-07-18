"""Tests for CSI-ratio computation."""

import numpy as np
import pytest
from ..src.csi_ratio import (
    compute_csi_ratio,
    compute_csi_ratio_multi,
    compute_csi_ratio_robust,
)
from ..src.signal_model import csi_signal_model, csi_static_dynamic_model, csi_with_doppler


def test_csi_ratio_basic():
    """Test basic CSI-ratio computation."""
    H_m = np.array([1 + 1j, 2 + 2j, 3 + 3j])
    H_m1 = np.array([1 + 0j, 2 + 0j, 3 + 0j])

    R = compute_csi_ratio(H_m, H_m1)

    expected = np.array([1 + 1j, 1 + 1j, 1 + 1j])
    np.testing.assert_allclose(R, expected, atol=1e-10)


def test_maximum_finite_complex_components_have_unit_ratio():
    amplitude = 1.7e308
    value = np.array([amplitude + 1j * amplitude])
    np.testing.assert_array_equal(compute_csi_ratio(value, value), np.ones(1))
    robust, mask = compute_csi_ratio_robust(value, value)
    np.testing.assert_array_equal(robust, np.ones(1))
    np.testing.assert_array_equal(mask, np.ones(1, dtype=bool))


def test_csi_ratio_cancels_offset():
    """Verify CSI-ratio cancels CFO/TMO."""
    N = 100
    fs = 2000  # 2 kHz
    T_s = 1.0 / fs
    t = np.arange(N) * T_s

    f_D = 50.0  # Hz
    cfo_hz = 100.0  # 100 Hz CFO
    tmo_hz = 20.0   # 20 Hz TMO

    H1, H2 = csi_with_doppler(t, f_D, snr_db=np.inf, cfo_hz=cfo_hz, tmo_hz=tmo_hz)
    R = compute_csi_ratio(H1, H2)

    # R should be constant magnitude (no CFO/TMO effect)
    magnitudes = np.abs(R)
    np.testing.assert_allclose(magnitudes, magnitudes[0], rtol=1e-10,
                               err_msg="CSI-ratio magnitude varies (CFO/TMO not cancelled)")

    # Phase should increase linearly: angle(R) = 2π*f_D*t + const
    phase = np.unwrap(np.angle(R))
    # Slope should be 2π*f_D
    expected_slope = 2 * np.pi * f_D
    # Compute slope via linear regression
    slope, intercept = np.polyfit(t, phase, 1)
    assert abs(slope - expected_slope) < 1.0, \
        f"Phase slope {slope:.2f} != expected {expected_slope:.2f} (2π*{f_D})"


def test_csi_ratio_multi():
    """Test multi-antenna CSI-ratio computation."""
    N = 50
    M = 3  # 3 antennas
    H = np.random.randn(N, M) + 1j * np.random.randn(N, M)

    R = compute_csi_ratio_multi(H, ref_antenna=0)

    assert R.shape == (N, M - 1)
    np.testing.assert_allclose(R[:, 0], H[:, 1] / H[:, 0], atol=1e-10)
    np.testing.assert_allclose(R[:, 1], H[:, 2] / H[:, 0], atol=1e-10)

    ref_two = compute_csi_ratio_multi(H, ref_antenna=2)
    assert not np.allclose(R, ref_two)
    np.testing.assert_allclose(ref_two[:, 0], H[:, 0] / H[:, 2], atol=1e-10)


def test_zero_reference_csi_is_rejected():
    with pytest.raises(ValueError, match="zero/near-zero"):
        compute_csi_ratio(np.ones(8, dtype=complex), np.zeros(8, dtype=complex))


def test_unrepresentable_finite_quotient_is_rejected_without_warning():
    numerator = np.full(4, 1.0e308 + 0.0j)
    denominator = np.full(4, 1.0e-308 + 0.0j)
    with pytest.raises(ValueError, match="finite binary64"):
        compute_csi_ratio(numerator, denominator)


def test_robust_ratio_validates_and_preserves_complex_dtype():
    ratio, mask = compute_csi_ratio_robust(
        np.array([1.0, 2.0]),
        np.array([0.0, 1.0]),
        threshold_db=-30.0,
    )
    np.testing.assert_array_equal(mask, [False, True])
    assert np.issubdtype(ratio.dtype, np.complexfloating)
    np.testing.assert_array_equal(ratio, [0.0 + 0.0j, 2.0 + 0.0j])

    all_masked, all_mask = compute_csi_ratio_robust(
        np.ones(3), np.zeros(3), threshold_db=0.0
    )
    assert not np.any(all_mask)
    np.testing.assert_array_equal(all_masked, np.zeros(3, dtype=complex))

    with pytest.raises(ValueError, match="threshold_db"):
        compute_csi_ratio_robust(np.ones(3), np.ones(3), threshold_db=1.0)
    with pytest.raises(ValueError, match="equal shape"):
        compute_csi_ratio_robust(np.ones(3), np.ones(2))

    underflow_threshold, underflow_mask = compute_csi_ratio_robust(
        np.ones(2), np.array([1.0, 0.0]), threshold_db=-10000.0
    )
    np.testing.assert_array_equal(underflow_mask, [True, False])
    np.testing.assert_array_equal(underflow_threshold, [1.0 + 0.0j, 0.0 + 0.0j])


def test_robust_ratio_rejects_unrepresentable_retained_quotient():
    with pytest.raises(ValueError, match="finite binary64"):
        compute_csi_ratio_robust(
            np.full(3, 1.0e308),
            np.full(3, 1.0e-308),
            threshold_db=0.0,
        )


def test_csi_ratio_preserves_phase_difference():
    """CSI-ratio should preserve the phase difference between antennas."""
    N = 100
    t = np.arange(N) * 0.0005  # 2 kHz

    phase_diff = 0.5  # radians
    H1 = np.exp(1j * (2 * np.pi * 30 * t + phase_diff))
    H2 = np.exp(1j * 2 * np.pi * 30 * t)

    R = compute_csi_ratio(H1, H2)

    # All samples should have the same phase = phase_diff
    phases = np.angle(R)
    np.testing.assert_allclose(phases, phase_diff, atol=1e-10)


def test_static_dynamic_model_ratio_is_mobius_and_cancels_shared_offset():
    t = np.arange(200) * 0.001
    static = np.array([1.0 + 0.2j, 0.8 - 0.1j])
    dynamic = np.array([0.35 - 0.15j, -0.2 + 0.25j])
    H = csi_static_dynamic_model(
        t,
        17.0,
        static,
        dynamic,
        shared_offset_hz=123.0,
    )
    ratio = compute_csi_ratio(H[:, 0], H[:, 1])
    z = np.exp(1j * 2 * np.pi * 17.0 * t)
    expected = (static[0] + dynamic[0] * z) / (static[1] + dynamic[1] * z)
    np.testing.assert_allclose(ratio, expected, atol=1e-12)
    assert np.ptp(np.unwrap(np.angle(ratio))) > 1.0


def test_general_model_doppler_does_not_cancel_from_ratio():
    t = np.arange(256) * 0.001
    H = csi_signal_model(t, v_r=0.4)
    ratio = compute_csi_ratio(H[:, 1], H[:, 0])
    assert np.std(ratio) > 1e-3


def test_seeded_signal_noise_is_reproducible():
    t = np.arange(64) * 0.001
    first = csi_with_doppler(t, 20.0, snr_db=10.0, rng=np.random.default_rng(7))
    second = csi_with_doppler(t, 20.0, snr_db=10.0, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


@pytest.mark.parametrize(
    "builder",
    [
        lambda: csi_static_dynamic_model(
            np.array([0.0, 1.0, 2.0]),
            1e308,
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
        ),
        lambda: csi_with_doppler(np.array([0.0, 1.0, 2.0]), 1e308),
        lambda: csi_signal_model(np.array([0.0, 1.0, 2.0]), d0=1e200),
        lambda: csi_signal_model(
            np.array([0.0, 1.0, 2.0]), f_c=1e308, v_r=1e308
        ),
    ],
)
def test_unreliable_numeric_domains_are_rejected_without_overflow(builder):
    with pytest.raises(ValueError, match="phase|path scale|Doppler"):
        builder()


@pytest.mark.parametrize("snr_db", [-1e308, 1e308])
def test_noise_rejects_unrepresentable_linear_snr(snr_db):
    with pytest.raises(ValueError, match="linear SNR"):
        csi_with_doppler(
            np.array([0.0, 0.01, 0.02]),
            1.0,
            snr_db=snr_db,
            rng=np.random.default_rng(3),
        )


if __name__ == "__main__":
    test_csi_ratio_basic()
    test_csi_ratio_cancels_offset()
    test_csi_ratio_multi()
    test_csi_ratio_preserves_phase_difference()
    print("All CSI-ratio tests passed!")
