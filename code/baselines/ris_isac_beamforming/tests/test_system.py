"""Tests for RIS-ISAC system model."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ..src.system_model import RIS_ISAC_System


class TestSystemModel:
    """System model unit tests."""

    def setup_method(self):
        self.system = RIS_ISAC_System(M=4, K=2, L=30, seed=42)

    def test_ris_unit_modulus(self):
        """Test that all RIS phase elements have unit modulus |θ_l| = 1."""
        theta = self.system.theta
        magnitudes = np.abs(theta)
        np.testing.assert_allclose(magnitudes, 1.0, atol=1e-10,
                                   err_msg="RIS phases must have |θ_l| = 1")

    def test_ris_unit_modulus_after_set(self):
        """Test unit modulus after setting arbitrary phases."""
        theta_new = 3.0 * np.exp(1j * np.linspace(0, np.pi, self.system.L))
        self.system.set_ris_phases(theta_new)
        magnitudes = np.abs(self.system.theta)
        np.testing.assert_allclose(magnitudes, 1.0, atol=1e-10)

    def test_channel_dimensions(self):
        """Test that channel matrices have correct shapes."""
        ch = self.system.channels
        M, K, L = self.system.M, self.system.K, self.system.L

        assert ch["H_BR"].shape == (L, M), f"H_BR shape: {ch['H_BR'].shape}"
        assert ch["G"].shape == (K, L), f"G shape: {ch['G'].shape}"
        assert ch["h_d"].shape == (K, M), f"h_d shape: {ch['h_d'].shape}"
        assert ch["a_bs"].shape == (M,), f"a_bs shape: {ch['a_bs'].shape}"
        assert ch["a_ris"].shape == (L,), f"a_ris shape: {ch['a_ris'].shape}"

    def test_target_array_responses_share_one_geometry(self):
        """BS and RIS steering vectors must use the same target angle."""
        ch = self.system.channels
        phi = ch["target_angle_rad"]
        expected_bs = np.exp(
            1j * np.pi * np.arange(self.system.M) * np.sin(phi)
        )
        expected_ris = np.exp(
            1j * np.pi * np.arange(self.system.L) * np.sin(phi)
        )
        np.testing.assert_allclose(ch["a_bs"], expected_bs, atol=1e-12)
        np.testing.assert_allclose(ch["a_ris"], expected_ris, atol=1e-12)

    def test_matched_filter_uses_hermitian_channel_convention(self):
        system = RIS_ISAC_System(M=2, K=1, L=1, noise_power=1.0, seed=1)
        system.channels["H_BR"] = np.zeros((1, 2), dtype=complex)
        system.channels["G"] = np.zeros((1, 1), dtype=complex)
        system.channels["h_d"] = np.array([[1.0, 1.0j]])
        h = system.effective_channel(0)
        matched = h / np.linalg.norm(h)
        wrong_conjugated = h.conj() / np.linalg.norm(h)
        assert system.compute_sinr(0, matched, np.empty((2, 0))) == pytest.approx(2.0)
        assert system.compute_sinr(
            0, wrong_conjugated, np.empty((2, 0))
        ) == pytest.approx(0.0, abs=1e-15)

    def test_compute_sinr_matches_manual_expression(self):
        rng = np.random.default_rng(7)
        W = rng.normal(size=(self.system.M, self.system.K)) + 1j * rng.normal(
            size=(self.system.M, self.system.K)
        )
        k = 1
        h = self.system.effective_channel(k)
        expected = np.abs(h.conj() @ W[:, k]) ** 2 / (
            np.abs(h.conj() @ W[:, 0]) ** 2 + self.system.noise_power
        )
        assert self.system.compute_sinr(k, W[:, k], W[:, :1]) == pytest.approx(expected)

    def test_zero_ris_phase_is_rejected(self):
        theta = np.ones(self.system.L, dtype=complex)
        theta[0] = 0
        with pytest.raises(ValueError, match="zero-magnitude"):
            self.system.set_ris_phases(theta)

    def test_ris_diagonal_matrix(self):
        """Test RIS diagonal matrix construction."""
        Theta = self.system.ris_diagonal_matrix()
        assert Theta.shape == (self.system.L, self.system.L)
        # Check it's diagonal
        off_diag = Theta - np.diag(np.diag(Theta))
        np.testing.assert_allclose(off_diag, 0, atol=1e-10)
        # Check diagonal entries are unit-modulus
        np.testing.assert_allclose(np.abs(np.diag(Theta)), 1.0, atol=1e-10)

    def test_effective_channel_shape(self):
        """Test effective channel has correct shape for each user."""
        for k in range(self.system.K):
            h_eff = self.system.effective_channel(k)
            assert h_eff.shape == (self.system.M,)

    def test_sum_rate_positive(self):
        """Test that sum rate is positive for valid beamforming."""
        M, K = self.system.M, self.system.K
        P_max = self.system.P_max
        # Simple beamforming: equal power allocation
        W = np.random.randn(M, K) + 1j * np.random.randn(M, K)
        W *= np.sqrt(P_max / K) / np.linalg.norm(W, axis=0)
        rate = self.system.compute_sum_rate(W)
        assert rate > 0, f"Sum rate should be positive, got {rate}"

    def test_sum_rate_preserves_representable_sub_epsilon_snr(self):
        """The rate at SINR=1e-20 must not round to zero."""

        system = RIS_ISAC_System(
            M=1, K=1, L=1, noise_power=1.0, seed=17
        )
        system.channels["H_BR"] = np.zeros((1, 1), dtype=complex)
        system.channels["G"] = np.zeros((1, 1), dtype=complex)
        system.channels["h_d"] = np.array([[1.0e-10 + 0.0j]])
        expected = 1.4426950408889633e-20
        assert system.compute_sum_rate(np.ones((1, 1), dtype=complex)) \
            == pytest.approx(expected, rel=3e-16, abs=0.0)

    def test_equal_huge_signal_and_interference_have_unit_sinr(self):
        """A representable ratio must not inherit overflow from either power."""

        system = RIS_ISAC_System(
            M=1, K=2, L=1, noise_power=1.0, seed=17
        )
        system.channels["H_BR"] = np.zeros((1, 1), dtype=complex)
        system.channels["G"] = np.zeros((2, 1), dtype=complex)
        system.channels["h_d"] = np.full((2, 1), 1.0e100 + 0.0j)
        desired = np.array([1.0e100 + 0.0j])
        interference = np.array([[1.0e100 + 0.0j]])
        assert system.compute_sinr(0, desired, interference) == 1.0
        W = np.array([[1.0e100 + 0.0j, 1.0e100 + 0.0j]])
        assert system.compute_sum_rate(W) == 2.0

    @pytest.mark.parametrize("swap_surface_rows", [False, True])
    def test_exact_path_cancellation_preserves_tiny_tail(
        self, swap_surface_rows
    ):
        """Direct/RIS paths may cancel across the full binary64 exponent range."""

        system = RIS_ISAC_System(
            M=1, K=1, L=2, noise_power=1.0, seed=17
        )
        surface = np.array([1.0e280 + 0.0j, 1.0e-60 + 0.0j])
        H_BR = np.array([[-1.0 + 0.0j], [1.0 + 0.0j]])
        if swap_surface_rows:
            surface = surface[::-1]
            H_BR = H_BR[::-1]
        system.channels["h_d"] = np.array([[1.0e280 + 0.0j]])
        system.channels["G"] = surface.reshape(1, 2)
        system.channels["a_bs"] = np.array([1.0e280 + 0.0j])
        system.channels["a_ris"] = surface.copy()
        system.channels["H_BR"] = H_BR
        system.set_ris_phases(np.ones(2, dtype=complex))
        expected = 1.0e-120
        beam = np.ones(1, dtype=complex)
        assert system.effective_channel(0)[0] == 1.0e-60 + 0.0j
        assert system.compute_sinr(0, beam, np.empty((1, 0))) \
            == pytest.approx(expected, rel=2e-15, abs=0.0)
        assert system.compute_snr_sensing(beam[:, None]) \
            == pytest.approx(expected, rel=2e-15, abs=0.0)

    def test_sensing_ratio_is_scale_safe_when_projection_square_overflows(self):
        system = RIS_ISAC_System(
            M=1, K=1, L=1, noise_power=1.0e200, seed=17
        )
        system.channels["a_bs"] = np.array([1.0e100 + 0.0j])
        system.channels["a_ris"] = np.zeros(1, dtype=complex)
        system.channels["H_BR"] = np.zeros((1, 1), dtype=complex)
        snr = system.compute_snr_sensing(
            np.array([[1.0e100 + 0.0j]])
        )
        assert snr == pytest.approx(1.0e200, rel=2e-15)

    @pytest.mark.parametrize(
        "W",
        [
            np.array([[1.0 + 0.0j, -1.0 + 0.0j]]),
            np.array([[1.0 + 0.0j, 1.0 + 0.0j]]),
        ],
    )
    def test_independent_stream_sensing_power_does_not_coherently_collapse(
        self, W
    ):
        """Opposite and equal unit streams both carry two units of power."""

        system = RIS_ISAC_System(
            M=1, K=2, L=1, noise_power=1.0, seed=17
        )
        system.channels["a_bs"] = np.ones(1, dtype=complex)
        system.channels["a_ris"] = np.zeros(1, dtype=complex)
        system.channels["H_BR"] = np.zeros((1, 1), dtype=complex)
        assert system.compute_snr_sensing(W) == pytest.approx(2.0)

    def test_sensing_snr_matches_streamwise_manual_expression(self):
        rng = np.random.default_rng(20260718)
        W = rng.normal(size=(self.system.M, self.system.K)) + 1j * rng.normal(
            size=(self.system.M, self.system.K)
        )
        theta = self.system.ris_diagonal_matrix()
        sensing_channel = (
            self.system.channels["a_bs"]
            + self.system.channels["a_ris"].T
            @ theta
            @ self.system.channels["H_BR"]
        )
        expected = sum(
            abs(np.vdot(sensing_channel, W[:, stream])) ** 2
            for stream in range(self.system.K)
        ) / self.system.noise_power
        assert self.system.compute_snr_sensing(W) == pytest.approx(
            expected, rel=5e-13
        )

    def test_sensing_snr_requires_the_complete_stream_matrix(self):
        with pytest.raises(ValueError, match="finite matrix"):
            self.system.compute_snr_sensing(np.ones(self.system.M))

    @pytest.mark.parametrize(
        "phase", [1.0e308 + 1.0e308j, 1.0e-320 + 1.0e-320j]
    )
    def test_phase_normalization_is_scale_safe(self, phase):
        system = RIS_ISAC_System(M=1, K=1, L=1, seed=17)
        system.set_ris_phases(np.array([phase]))
        assert abs(system.theta[0]) == pytest.approx(1.0, rel=2e-15)

    @pytest.mark.parametrize(
        ("threshold_db", "error"),
        [
            (1.0e308, OverflowError),
            (-1.0e308, FloatingPointError),
        ],
    )
    def test_unrepresentable_db_threshold_is_explicit(
        self, threshold_db, error
    ):
        with pytest.raises(error):
            RIS_ISAC_System(
                M=1, K=1, L=1, sinr_thresh_dB=threshold_db
            )

    def test_power_constraint(self):
        """Test power constraint: Σ||w_k||² ≤ P_max."""
        M, K = self.system.M, self.system.K
        P_max = self.system.P_max
        W = np.random.randn(M, K) + 1j * np.random.randn(M, K)
        W *= np.sqrt(P_max / K) / np.linalg.norm(W, axis=0)
        total_power = np.sum(np.linalg.norm(W, axis=0) ** 2)
        assert total_power <= P_max * 1.01, f"Power {total_power} exceeds P_max {P_max}"

    def test_reset_channels(self):
        """Test channel regeneration with new seed."""
        H_BR_old = self.system.channels["H_BR"].copy()
        self.system.reset_channels(seed=123)
        H_BR_new = self.system.channels["H_BR"]
        # Should be different (with high probability)
        assert not np.allclose(H_BR_old, H_BR_new)
