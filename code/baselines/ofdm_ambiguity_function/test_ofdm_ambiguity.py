"""
Test suite for OFDM Ambiguity Function Analysis

Tests cover:
1. Core ambiguity function properties
2. OFDM signal generation
3. LFM signal comparison
4. Resolution analysis
5. PAPR computation
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .ofdm_ambiguity import (
    generate_ofdm_signal,
    compute_ambiguity_function,
    compute_ambiguity_function_ofdm,
    generate_lfm_signal,
    compute_range_resolution,
    compute_doppler_resolution,
    compute_papr,
    compute_3db_resolution,
    plot_ambiguity_contour,
    _qam_modulate
)


class TestAmbiguityFunctionProperties:
    """Test fundamental ambiguity function properties."""

    def test_ambiguity_peak_at_origin(self):
        """The normalized ambiguity power is exactly one at the origin."""
        # Generate test signal
        signal = generate_ofdm_signal(
            n_subcarriers=32,
            cp_len=8,
            rng=np.random.default_rng(101),
        )
        # Compute ambiguity function at origin
        tau_range = np.array([0])
        nu_range = np.array([0])
        af = compute_ambiguity_function(signal, tau_range, nu_range)

        np.testing.assert_allclose(af[0, 0], 1.0, atol=1e-12)

    @pytest.mark.parametrize("amplitude", [1e-200, 1e-160, 1.0, 1e160, 1e200])
    def test_normalized_metrics_are_amplitude_scale_invariant(self, amplitude):
        signal = amplitude * np.array([1.0, 1.0j, -1.0], dtype=complex)
        af = compute_ambiguity_function(signal, np.array([0]), np.array([0.0]))
        np.testing.assert_allclose(af, np.ones((1, 1)), rtol=0, atol=5e-16)
        np.testing.assert_allclose(compute_papr(signal), 1.0, rtol=0, atol=5e-16)

    def test_maximum_finite_complex_components_do_not_overflow_magnitude(self):
        amplitude = 1.7e308
        signal = np.array(
            [amplitude + 1j * amplitude, -amplitude + 1j * amplitude]
        )
        ambiguity = compute_ambiguity_function(
            signal, np.array([0]), np.array([0.0])
        )
        np.testing.assert_allclose(ambiguity, np.ones((1, 1)), rtol=0, atol=0)
        np.testing.assert_allclose(compute_papr(signal), 1.0, rtol=0, atol=0)

    @pytest.mark.parametrize("delay", [-1, 1])
    def test_correlation_cancellation_preserves_representable_power_tail(
        self, delay
    ):
        signal = np.array([1e40, 1e-40, 1e-40, -1e40], dtype=complex)
        ambiguity = compute_ambiguity_function(
            signal,
            np.array([delay]),
            np.array([0.0]),
        )[0, 0]
        oracle = 2.5e-321
        assert ambiguity > 0.0
        assert abs(ambiguity - oracle) <= 2 * np.nextafter(0.0, 1.0)

    def test_delay_grid_rejects_fractional_or_duplicate_samples(self):
        test_signal = np.ones(8)
        with pytest.raises(ValueError, match="integer"):
            compute_ambiguity_function(
                test_signal,
                np.array([0.0, 0.5]),
                np.array([0.0]),
            )
        with pytest.raises(ValueError, match="duplicate"):
            compute_ambiguity_function(
                test_signal,
                np.array([0, 0]),
                np.array([0.0]),
            )

    @pytest.mark.parametrize("delay", [1e308, -1e308, float(2**63)])
    def test_unrepresentable_integer_delay_is_explicitly_rejected(self, delay):
        with pytest.raises(ValueError, match="sample-delay domain"):
            compute_ambiguity_function(
                np.ones(4), np.array([delay]), np.array([0.0])
            )

    def test_doppler_axis_is_hz(self):
        """At fs=4 Hz, a 1 Hz shift completes one cycle across four samples."""
        af = compute_ambiguity_function(
            np.ones(4),
            np.array([0]),
            np.array([0.0, 1.0]),
            fs=4.0,
        )
        np.testing.assert_allclose(af[:, 0], [1.0, 0.0], atol=1e-12)

    @pytest.mark.parametrize(
        "frequency, sample_rate",
        [(1e308, 1.0), (1.0, 5e-324), (1e308, 5e-324)],
    )
    def test_finite_extreme_doppler_inputs_are_reduced_modulo_fs(
        self, frequency, sample_rate
    ):
        signal = np.array([1.0, 0.5j, -0.25, 0.75j])
        actual = compute_ambiguity_function(
            signal, np.array([0]), np.array([frequency]), fs=sample_rate
        )
        reduced = np.remainder(frequency, sample_rate)
        expected = compute_ambiguity_function(
            signal, np.array([0]), np.array([reduced]), fs=sample_rate
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)

    def test_doppler_is_periodic_modulo_sample_rate(self):
        signal = np.array([1.0, 0.5j, -0.25, 0.75j])
        sample_rate = 3.0
        frequencies = np.array([0.2, 0.2 + 7.0 * sample_rate])
        af = compute_ambiguity_function(
            signal, np.array([-1, 0, 1]), frequencies, fs=sample_rate
        )
        np.testing.assert_allclose(af[0], af[1], rtol=0, atol=2e-15)

    def test_default_contour_levels_are_runnable(self, tmp_path):
        tau = np.arange(-2, 3)
        nu = np.linspace(-0.2, 0.2, 5)
        af = compute_ambiguity_function(np.ones(8), tau, nu)
        output = tmp_path / "contour.png"
        plot_ambiguity_contour(af, tau, nu, save_path=str(output))
        assert output.stat().st_size > 0

    def test_ambiguity_symmetry(self):
        """|χ(τ,ν)| = |χ(-τ,-ν)|* for ambiguity function."""
        signal = generate_ofdm_signal(
            n_subcarriers=16,
            cp_len=4,
            rng=np.random.default_rng(102),
        )

        # Small range for quick test
        tau_range = np.array([-2, -1, 0, 1, 2])
        nu_range = np.array([-0.1, 0, 0.1])

        af = compute_ambiguity_function(signal, tau_range, nu_range)

        # Check symmetry: af[ν, τ] ≈ af[-ν, -τ]
        for i, nu in enumerate(nu_range):
            for j, tau in enumerate(tau_range):
                # Find corresponding indices for (-τ, -ν)
                neg_nu_idx = len(nu_range) - 1 - i
                neg_tau_idx = len(tau_range) - 1 - j

                # Due to symmetry property
                if 0 <= neg_nu_idx < len(nu_range) and 0 <= neg_tau_idx < len(tau_range):
                    np.testing.assert_allclose(
                        af[i, j], af[neg_nu_idx, neg_tau_idx],
                        rtol=1e-12, atol=1e-14,
                        err_msg=f"Symmetry failed at ν={nu}, τ={tau}"
                    )

    def test_delay_resolution(self):
        """Resolution should match 1/BW theoretically."""
        bandwidths = [10e6, 20e6, 50e6]

        for bw in bandwidths:
            theoretical_res = compute_range_resolution(bw)
            expected_res = 3e8 / (2 * bw)  # c/(2B)

            np.testing.assert_allclose(
                theoretical_res, expected_res,
                rtol=1e-10,
                err_msg=f"Range resolution wrong for BW={bw}"
            )

    def test_doppler_resolution(self):
        """Doppler resolution should match 1/T."""
        coherent_times = [1e-3, 10e-3, 100e-3]

        for T in coherent_times:
            doppler_res = compute_doppler_resolution(T)
            expected_res = 1.0 / T

            np.testing.assert_allclose(
                doppler_res, expected_res,
                rtol=1e-10,
                err_msg=f"Doppler resolution wrong for T={T}"
            )

    def test_resolution_helpers_accept_arrays(self):
        np.testing.assert_allclose(
            compute_range_resolution(np.array([10e6, 20e6])),
            np.array([15.0, 7.5]),
        )
        np.testing.assert_allclose(
            compute_doppler_resolution(np.array([0.01, 0.02])),
            np.array([100.0, 50.0]),
        )

    def test_range_resolution_handles_large_finite_bandwidth(self):
        assert compute_range_resolution(1e308) == pytest.approx(1.5e-300)

    @pytest.mark.parametrize(
        "call",
        [
            lambda: compute_range_resolution(5e-324),
            lambda: compute_doppler_resolution(5e-324),
        ],
    )
    def test_unrepresentable_resolution_is_explicitly_rejected(self, call):
        with pytest.raises(ValueError, match="finite binary64"):
            call()


class TestOFDMSignalGeneration:
    """Test OFDM signal generation."""

    def test_ofdm_signal_length(self):
        """Signal length should be N + CP."""
        n_subcarriers = 64
        cp_len = 16

        signal = generate_ofdm_signal(
            n_subcarriers,
            cp_len,
            rng=np.random.default_rng(103),
        )

        assert len(signal) == n_subcarriers + cp_len

    def test_ofdm_cyclic_prefix(self):
        """Cyclic prefix should be copy of tail."""
        n_subcarriers = 64
        cp_len = 16

        signal = generate_ofdm_signal(
            n_subcarriers,
            cp_len,
            rng=np.random.default_rng(104),
        )

        # CP should match last CP samples of data
        np.testing.assert_array_almost_equal(
            signal[:cp_len], signal[-cp_len:],
            decimal=10,
            err_msg="Cyclic prefix mismatch"
        )

    def test_ofdm_different_modulations(self):
        """Different modulation orders should work."""
        mod_orders = [2, 4, 16, 64]

        for mod_order in mod_orders:
            signal = generate_ofdm_signal(
                n_subcarriers=32,
                cp_len=8,
                mod_order=mod_order,
                rng=np.random.default_rng(1000 + mod_order),
            )

            assert len(signal) == 40
            assert np.iscomplexobj(signal)
            assert not np.any(np.isnan(signal))

    def test_seeded_ofdm_generation_is_repeatable(self):
        first = generate_ofdm_signal(32, 8, rng=np.random.default_rng(2026))
        second = generate_ofdm_signal(32, 8, rng=np.random.default_rng(2026))
        np.testing.assert_array_equal(first, second)

    def test_default_ofdm_af_axes_are_unique_and_include_origin(self):
        _, tau, nu = compute_ambiguity_function_ofdm(
            n_subcarriers=16,
            cp_len=4,
            n_tau_points=101,
            n_nu_points=101,
            rng=np.random.default_rng(4),
        )
        assert len(np.unique(tau)) == len(tau)
        assert len(np.unique(nu)) == len(nu)
        assert 0 in tau and np.any(nu == 0)

    def test_ofdm_cp_effect(self):
        """CP shouldn't affect the ambiguity function peak."""
        n_subcarriers = 32

        # Signal with CP
        signal_with_cp = generate_ofdm_signal(
            n_subcarriers,
            cp_len=8,
            rng=np.random.default_rng(105),
        )

        # Signal without CP (same symbols, no prefix)
        symbols = np.random.randn(n_subcarriers) + 1j * np.random.randn(n_subcarriers)
        signal_no_cp = np.fft.ifft(symbols) * np.sqrt(n_subcarriers)

        # Compute ambiguity at origin
        tau_0 = np.array([0])
        nu_0 = np.array([0])

        af_with_cp = compute_ambiguity_function(signal_with_cp, tau_0, nu_0)
        af_no_cp = compute_ambiguity_function(signal_no_cp, tau_0, nu_0)

        # Both should have positive peaks
        assert af_with_cp[0, 0] > 0
        assert af_no_cp[0, 0] > 0

    def test_invalid_cp_and_modulation_are_rejected(self):
        with pytest.raises(ValueError, match="cp_len"):
            generate_ofdm_signal(n_subcarriers=8, cp_len=9)
        with pytest.raises(ValueError, match="square power-of-two"):
            generate_ofdm_signal(n_subcarriers=8, cp_len=2, mod_order=8)


class TestLFMSignal:
    """Test LFM (chirp) signal generation."""

    def test_lfm_signal_length(self):
        """LFM length should be fs * pulse_width."""
        fs = 40e6
        pulse_width = 10e-6

        signal = generate_lfm_signal(
            bandwidth=20e6,
            pulse_width=pulse_width,
            fs=fs
        )

        expected_length = int(fs * pulse_width)
        assert len(signal) == expected_length

    def test_lfm_constant_amplitude(self):
        """LFM should have constant amplitude."""
        signal = generate_lfm_signal(bandwidth=20e6, pulse_width=10e-6)

        amplitude = np.abs(signal)

        # All amplitudes should be 1.0
        np.testing.assert_array_almost_equal(
            amplitude, np.ones_like(amplitude),
            decimal=10,
            err_msg="LFM amplitude not constant"
        )

    def test_lfm_sweep_is_centered_about_zero_frequency(self):
        fs = 40e6
        lfm = generate_lfm_signal(10e6, 10e-6, fs)
        instantaneous_frequency = (
            np.angle(lfm[1:] * np.conj(lfm[:-1])) * fs / (2 * np.pi)
        )
        np.testing.assert_allclose(
            instantaneous_frequency[0],
            -instantaneous_frequency[-1],
            rtol=1e-12,
            atol=1e-6,
        )

    def test_lfm_rejects_aliased_requested_sweep(self):
        with pytest.raises(ValueError, match="must not exceed fs"):
            generate_lfm_signal(41e6, 10e-6, 40e6)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            (
                {"bandwidth": 1.0, "pulse_width": 1e308, "fs": 1e308},
                "pulse_width \\* fs",
            ),
            (
                {
                    "bandwidth": 1.0,
                    "pulse_width": 1001.0,
                    "fs": 1.0,
                    "max_samples": 1000,
                },
                "max_samples",
            ),
        ],
    )
    def test_lfm_allocation_domain_is_checked_before_construction(
        self, kwargs, message
    ):
        with pytest.raises(ValueError, match=message):
            generate_lfm_signal(**kwargs)

    def test_normalized_delay_cuts_have_bounded_sidelobes(self):
        """For this sampled realization, off-origin values stay below one."""
        # Generate signals
        ofdm = generate_ofdm_signal(
            n_subcarriers=64,
            cp_len=16,
            rng=np.random.default_rng(106),
        )
        lfm = generate_lfm_signal(bandwidth=20e6, pulse_width=10e-6, fs=40e6)

        # Compute autocorrelation (delay cut of ambiguity function)
        tau_range = np.arange(-100, 101)  # Wider range
        nu_zero = np.array([0])

        af_ofdm = compute_ambiguity_function(ofdm, tau_range, nu_zero)
        af_lfm = compute_ambiguity_function(lfm, tau_range, nu_zero)

        # Normalize to peak
        af_ofdm_norm = af_ofdm / np.max(af_ofdm)
        af_lfm_norm = af_lfm / np.max(af_lfm)

        # Find sidelobes (excluding main lobe around center)
        center_idx = len(tau_range) // 2
        main_lobe_width = 5  # samples

        ofdm_sidelobes = np.concatenate([
            af_ofdm_norm[0, :center_idx - main_lobe_width],
            af_ofdm_norm[0, center_idx + main_lobe_width + 1:]
        ])
        lfm_sidelobes = np.concatenate([
            af_lfm_norm[0, :center_idx - main_lobe_width],
            af_lfm_norm[0, center_idx + main_lobe_width + 1:]
        ])

        assert np.max(ofdm_sidelobes) < 1.0, "OFDM sidelobes should be < main peak"
        assert np.max(lfm_sidelobes) < 1.0, "LFM sidelobes should be < main peak"


class TestResolutionAnalysis:
    """Test resolution properties."""

    def test_different_subcarriers(self):
        """More subcarriers should give better Doppler resolution."""
        n_subcarriers_list = [16, 32, 64]

        for n_sub in n_subcarriers_list:
            # Doppler resolution is inversely proportional to observation time
            # For OFDM: observation time ≈ n_subcarriers / bandwidth
            # More subcarriers → longer symbol → better Doppler resolution

            symbol_duration = n_sub / 20e6  # Assuming 20 MHz bandwidth
            doppler_res = compute_doppler_resolution(symbol_duration)

            # Verify resolution improves with more subcarriers
            if n_sub > n_subcarriers_list[0]:
                prev_duration = n_subcarriers_list[n_subcarriers_list.index(n_sub) - 1] / 20e6
                prev_doppler_res = compute_doppler_resolution(prev_duration)

                # Doppler resolution should improve (get smaller)
                assert doppler_res < prev_doppler_res, \
                    "Doppler resolution should improve with more subcarriers"

    def test_range_resolution_from_af(self):
        """3dB resolution from AF should match theoretical."""
        bandwidth = 20e6
        fs = 40e6

        # Generate signal
        lfm = generate_lfm_signal(bandwidth=bandwidth, pulse_width=10e-6, fs=fs)

        # Compute delay cut of ambiguity function
        tau_range = np.arange(-100, 101)
        nu_range = np.array([0])

        af = compute_ambiguity_function(lfm, tau_range, nu_range)

        # Extract 3dB resolution
        resolution_samples = compute_3db_resolution(tau_range, af[0, :])

        assert np.isfinite(resolution_samples)
        resolution_time = resolution_samples / fs
        resolution_meters = 3e8 * resolution_time / 2  # Round-trip
        theoretical_res = compute_range_resolution(bandwidth)

        # The half-power width and c/(2B) are different conventions but have
        # the same reciprocal-bandwidth scale for this rectangular chirp.
        assert resolution_meters < 2 * theoretical_res

    def test_3db_width_uses_nearest_main_lobe_crossings(self):
        tau = np.arange(-6, 7, dtype=float)
        af_cut = np.zeros_like(tau)
        af_cut[tau == 0] = 1.0
        af_cut[np.abs(tau) == 1] = 0.75
        af_cut[np.abs(tau) == 2] = 0.25
        af_cut[np.abs(tau) == 5] = 0.8  # remote grating lobes
        np.testing.assert_allclose(compute_3db_resolution(tau, af_cut), 3.0)


class TestPAPRComputation:
    """Test PAPR calculations."""

    def test_papr_computation(self):
        """OFDM PAPR > LFM PAPR."""
        # Generate signals
        ofdm = generate_ofdm_signal(
            n_subcarriers=64,
            cp_len=16,
            rng=np.random.default_rng(42),
        )
        lfm = generate_lfm_signal(bandwidth=20e6, pulse_width=10e-6)

        ofdm_papr = compute_papr(ofdm)
        lfm_papr = compute_papr(lfm)

        # LFM has constant amplitude → PAPR = 1
        np.testing.assert_allclose(lfm_papr, 1.0, rtol=1e-10,
                                   err_msg="LFM PAPR should be 1.0")

        independent_ratio = np.max(np.abs(ofdm) ** 2) / np.mean(np.abs(ofdm) ** 2)
        np.testing.assert_allclose(ofdm_papr, independent_ratio)
        assert ofdm_papr >= lfm_papr

    def test_papr_values(self):
        """PAPR values should be in reasonable range."""
        # Test various OFDM configurations
        configs = [
            (16, 4),   # Small OFDM
            (64, 16),  # Standard OFDM
            (256, 32), # Large OFDM
        ]

        for n_sub, cp_len in configs:
            signal = generate_ofdm_signal(
                n_sub,
                cp_len,
                rng=np.random.default_rng(2000 + n_sub),
            )
            papr = compute_papr(signal)

            # PAPR should be >= 1
            assert papr >= 1.0, f"PAPR should be >= 1 for N={n_sub}"

            # For typical OFDM, PAPR shouldn't exceed ~20
            assert papr < 30, f"PAPR too high for N={n_sub}"


class TestQAMModulation:
    """Test QAM modulation functions."""

    def test_bpsk_modulation(self):
        """BPSK should map 0→-1, 1→+1."""
        bits = np.array([0, 1, 0, 1])
        symbols = _qam_modulate(bits, 2)

        expected = np.array([-1, 1, -1, 1])
        np.testing.assert_array_almost_equal(symbols, expected)

    def test_qpsk_modulation(self):
        """QPSK symbols should have unit power."""
        # Enumerate the complete constellation instead of using a random
        # finite sample.  Every QPSK point has unit energy.
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
        symbols = _qam_modulate(bits, 4)

        # Average power should be 1
        avg_power = np.mean(np.abs(symbols)**2)
        np.testing.assert_allclose(avg_power, 1.0, rtol=0.1)

    def test_16qam_modulation(self):
        """16-QAM symbols should have unit power."""
        # Exercise every bit pattern exactly once.  A random sample of 100
        # symbols can legitimately deviate by more than 10% from the ensemble
        # mean, which made this purported correctness test flaky.
        bits = np.array([
            bit
            for symbol_index in range(16)
            for bit in (
                (symbol_index >> 0) & 1,
                (symbol_index >> 1) & 1,
                (symbol_index >> 2) & 1,
                (symbol_index >> 3) & 1,
            )
        ])
        symbols = _qam_modulate(bits, 16)

        # Average power should be 1
        avg_power = np.mean(np.abs(symbols)**2)
        np.testing.assert_allclose(avg_power, 1.0, rtol=0, atol=1e-12)

    def test_64qam_complete_constellation_has_unit_average_power(self):
        bits_per_symbol = 6
        bits = np.array([
            bit
            for symbol_index in range(64)
            for bit in (
                (symbol_index >> np.arange(bits_per_symbol - 1, -1, -1)) & 1
            )
        ])
        symbols = _qam_modulate(bits, 64)
        np.testing.assert_allclose(np.mean(np.abs(symbols) ** 2), 1.0, atol=1e-12)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_subcarrier_ofdm(self):
        """Single subcarrier OFDM should work."""
        signal = generate_ofdm_signal(
            n_subcarriers=1,
            cp_len=0,
            rng=np.random.default_rng(107),
        )

        assert len(signal) == 1
        assert not np.isnan(signal[0])

    def test_zero_cp(self):
        """Zero CP should work."""
        signal = generate_ofdm_signal(
            n_subcarriers=32,
            cp_len=0,
            rng=np.random.default_rng(108),
        )

        assert len(signal) == 32
        assert not np.any(np.isnan(signal))

    def test_very_short_signal(self):
        """Very short signals should work."""
        signal = np.array([1.0, 0.5, -0.5])

        tau_range = np.array([0])
        nu_range = np.array([0])

        af = compute_ambiguity_function(signal, tau_range, nu_range)

        assert af.shape == (1, 1)
        assert af[0, 0] > 0

    @pytest.mark.parametrize(
        "call",
        [
            lambda: compute_ambiguity_function(np.zeros(8), np.array([0]), np.array([0])),
            lambda: compute_papr(np.zeros(8)),
            lambda: compute_range_resolution(0),
            lambda: compute_doppler_resolution(-1),
        ],
    )
    def test_invalid_physical_inputs_are_rejected(self, call):
        with pytest.raises(ValueError):
            call()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
