"""Tests for RIS phase shift optimization."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ..src.system_model import RIS_ISAC_System
from ..src.ris_phase import RISPhaseOptimizer
from ..src.numerics import (
    optimal_single_stream_sensing_phases,
    stable_triangle_sensing_snr,
)


class TestRISPhase:
    """RIS phase optimizer tests."""

    def setup_method(self):
        self.system = RIS_ISAC_System(M=4, K=2, L=30, seed=42)
        self.ris_opt = RISPhaseOptimizer(self.system)

    def test_unit_modulus_after_rate_opt(self):
        """Test RIS phases maintain unit modulus after rate optimization."""
        M, K = self.system.M, self.system.K
        rng = np.random.default_rng(9)
        W = rng.normal(size=(M, K)) + 1j * rng.normal(size=(M, K))
        W *= np.sqrt(self.system.P_max / K) / np.linalg.norm(W, axis=0)
        theta = self.ris_opt.optimize_for_rate(W)
        np.testing.assert_allclose(np.abs(theta), 1.0, atol=1e-10)

    def test_unit_modulus_after_snr_opt(self):
        """Test RIS phases maintain unit modulus after SNR optimization."""
        M, K = self.system.M, self.system.K
        W = np.zeros((M, K), dtype=complex)
        theta, snr = self.ris_opt.optimize_for_snr(W)
        np.testing.assert_allclose(np.abs(theta), 1.0, atol=1e-10)
        assert snr >= 0, "SNR should be non-negative"

    def test_unit_modulus_after_joint_opt(self):
        """Test RIS phases maintain unit modulus after joint optimization."""
        M, K = self.system.M, self.system.K
        rng = np.random.default_rng(10)
        W = rng.normal(size=(M, K)) + 1j * rng.normal(size=(M, K))
        W *= np.sqrt(self.system.P_max / K) / np.linalg.norm(W, axis=0)
        theta = self.ris_opt.optimize_joint(W, sensing_weight=0.5)
        np.testing.assert_allclose(np.abs(theta), 1.0, atol=1e-10)

    def test_output_length(self):
        """Test output has correct length."""
        M, K = self.system.M, self.system.K
        rng = np.random.default_rng(11)
        W = rng.normal(size=(M, K)) + 1j * rng.normal(size=(M, K))
        theta = self.ris_opt.optimize_for_rate(W)
        assert theta.shape == (self.system.L,)

    @pytest.mark.parametrize("seed", range(30))
    def test_snr_update_never_decreases_physical_snr(self, seed):
        """Every grid-coordinate update must retain or improve physical SNR."""
        system = RIS_ISAC_System(M=4, K=2, L=10, seed=seed)
        optimizer = RISPhaseOptimizer(system)
        rng = np.random.default_rng(1000 + seed)
        W = rng.normal(size=(system.M, system.K)) + 1j * rng.normal(
            size=(system.M, system.K)
        )
        before = system.compute_snr_sensing(W)
        theta, after = optimizer.optimize_for_snr(W)
        np.testing.assert_allclose(np.abs(theta), 1.0, atol=1e-12)
        assert after >= before * (1.0 - 1e-12)

    def test_rate_update_never_decreases_rate(self):
        rng = np.random.default_rng(120)
        W = rng.normal(size=(self.system.M, self.system.K)) + 1j * rng.normal(
            size=(self.system.M, self.system.K)
        )
        before = self.system.compute_sum_rate(W)
        self.ris_opt.optimize_for_rate(W, max_sweeps=3)
        assert self.system.compute_sum_rate(W) >= before * (1.0 - 1e-12)

    def test_snr_update_uses_streamwise_power_counterexample(self):
        system = RIS_ISAC_System(M=1, K=2, L=1, noise_power=1.0, seed=9)
        system.channels["a_bs"] = np.ones(1, dtype=complex)
        system.channels["a_ris"] = np.zeros(1, dtype=complex)
        system.channels["H_BR"] = np.zeros((1, 1), dtype=complex)
        optimizer = RISPhaseOptimizer(system)
        for W in (
            np.array([[1.0, -1.0]], dtype=complex),
            np.array([[1.0, 1.0]], dtype=complex),
        ):
            _, achieved = optimizer.optimize_for_snr(W)
            assert achieved == pytest.approx(2.0)

    def test_multistream_coordinate_update_attains_analytic_maximum(self):
        """One phase coordinate uses the cross-stream covariance coefficient."""

        system = RIS_ISAC_System(M=2, K=2, L=1, noise_power=1.0, seed=9)
        system.channels["a_bs"] = np.ones(2, dtype=complex)
        system.channels["a_ris"] = np.ones(1, dtype=complex)
        system.channels["H_BR"] = np.array([[1.0, 1.0j]])
        system.set_ris_phases(np.ones(1, dtype=complex))
        W = np.eye(2, dtype=complex)

        theta, achieved = RISPhaseOptimizer(system).optimize_for_snr(W)

        expected_theta = np.exp(-0.25j * np.pi)
        expected_snr = 4.0 + 2.0 * np.sqrt(2.0)
        assert theta[0] == pytest.approx(expected_theta, abs=2e-15)
        assert achieved == pytest.approx(expected_snr, rel=2e-15)

    def test_triangle_alignment_helper_is_explicitly_single_stream(self):
        theta = optimal_single_stream_sensing_phases(
            np.array([1.0j]),
            np.ones(1, dtype=complex),
            np.ones((1, 1), dtype=complex),
            np.ones(1, dtype=complex),
            np.ones(1, dtype=complex),
        )
        assert theta[0] == pytest.approx(1.0j, abs=2e-15)

    @pytest.mark.parametrize(
        "common_scale",
        (1.0e-20, 1.0e-8, 1.0, 1.0e8, 1.0e20 * np.exp(0.37j)),
    )
    def test_snr_phase_solution_is_common_scale_invariant(
        self, common_scale
    ):
        """No absolute cutoff may alter the scalar phase optimum."""

        seed = 20260717
        rng = np.random.default_rng(seed)
        W = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))

        reference = RIS_ISAC_System(M=4, K=2, L=12, noise_power=1.0, seed=seed)
        reference_theta, reference_snr = RISPhaseOptimizer(
            reference
        ).optimize_for_snr(W)

        scaled = RIS_ISAC_System(M=4, K=2, L=12, noise_power=1.0, seed=seed)
        scaled.channels["a_bs"] *= common_scale
        scaled.channels["H_BR"] *= common_scale
        theta, achieved = RISPhaseOptimizer(scaled).optimize_for_snr(W)

        np.testing.assert_allclose(theta, reference_theta, rtol=0.0, atol=2e-14)
        oracle = reference_snr * abs(common_scale) ** 2
        assert oracle > 0.0
        np.testing.assert_allclose(achieved, oracle, rtol=5e-13, atol=0.0)

    def test_snr_phase_oracle_at_1e_minus_20_channel_scale(self):
        """Lock the former absolute-1e-15-cutoff counterexample."""

        seed = 431
        reference = RIS_ISAC_System(
            M=3, K=2, L=7, noise_power=1.0, seed=seed
        )
        system = RIS_ISAC_System(M=3, K=2, L=7, noise_power=1.0, seed=seed)
        system.channels["a_bs"] *= 1.0e-20
        system.channels["H_BR"] *= 1.0e-20
        rng = np.random.default_rng(seed)
        W = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
        _, reference_snr = RISPhaseOptimizer(reference).optimize_for_snr(W)
        before = system.compute_snr_sensing(W)
        _, achieved = RISPhaseOptimizer(system).optimize_for_snr(W)
        assert achieved >= before * (1.0 - 2e-14)
        np.testing.assert_allclose(
            achieved, reference_snr * 1.0e-40, rtol=5e-13, atol=0.0
        )

    def test_triangle_oracle_is_safe_when_individual_powers_overflow(self):
        system = RIS_ISAC_System(
            M=1, K=1, L=1, noise_power=1.0e200, seed=19
        )
        system.channels["a_bs"] = np.array([1.0e100 + 0.0j])
        system.channels["a_ris"] = np.ones(1, dtype=complex)
        system.channels["H_BR"] = np.array([[1.0e100 + 0.0j]])
        W = np.array([[1.0e100 + 0.0j]])
        oracle = stable_triangle_sensing_snr(
            system.channels["a_bs"],
            system.channels["a_ris"],
            system.channels["H_BR"],
            W[:, 0],
            system.noise_power,
        )
        _, achieved = RISPhaseOptimizer(system).optimize_for_snr(W)
        assert oracle == pytest.approx(4.0e200, rel=3e-15)
        assert achieved == pytest.approx(oracle, rel=3e-15)

    @pytest.mark.parametrize("noise", [0.0, -1.0, np.nan, np.inf])
    def test_triangle_oracle_rejects_invalid_noise(self, noise):
        with pytest.raises(ValueError, match="noise power"):
            stable_triangle_sensing_snr(
                np.ones(1, dtype=complex),
                np.ones(1, dtype=complex),
                np.ones((1, 1), dtype=complex),
                np.ones(1, dtype=complex),
                noise,
            )
