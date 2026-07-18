"""Tests for SNR-constrained solver."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ..src.system_model import RIS_ISAC_System
from ..src.snr_constraint import SNRConstrainedSolver
from ..src import snr_constraint


class TestSNRConstrained:
    """SNR-constrained solver tests."""

    def setup_method(self):
        self.system = RIS_ISAC_System(M=4, K=2, L=30, seed=42)
        self.solver = SNRConstrainedSolver(self.system, snr_min_dB=5.0, max_iter=10)

    def test_snr_constraint_output_keys(self):
        """Test solver output contains expected keys."""
        result = self.solver.solve()
        expected_keys = {"W", "theta", "sum_rate", "snr_sensing", "converged", "iterations"}
        assert expected_keys.issubset(result.keys())

    def test_snr_constraint_beamforming_shape(self):
        """Test beamforming matrix has correct shape."""
        result = self.solver.solve()
        M, K = self.system.M, self.system.K
        assert result["W"].shape == (M, K)

    def test_snr_constraint_ris_unit_modulus(self):
        """Test RIS phases from SNR solver satisfy unit modulus."""
        result = self.solver.solve()
        theta = result["theta"]
        np.testing.assert_allclose(np.abs(theta), 1.0, atol=1e-10)

    def test_snr_constraint_sensing_channel(self):
        """Test sensing channel computation."""
        h_s = self.solver._compute_sensing_channel()
        assert h_s.shape == (self.system.M,)
        assert h_s.dtype == complex

    def test_snr_constraint_positive_rate(self):
        """Test sum rate is positive."""
        result = self.solver.solve()
        assert result["sum_rate"] > 0, f"Sum rate: {result['sum_rate']}"

    def test_affine_covariance_bound_is_a_conservative_snr_certificate(self):
        """The SCA affine bound certifies the full stream covariance power."""

        thresholds = np.full(self.system.K, self.system.sinr_thresh)
        phase_beamformers = self.solver._initial_beamformers()
        self.solver.ris_optimizer.optimize_for_snr(phase_beamformers)
        W_reference = self.solver._feasible_sca_reference(thresholds)
        W, _ = self.solver._solve_beamforming_socp(
            thresholds, W_reference
        )
        h_s = self.solver._compute_sensing_channel()
        reference_projections = h_s.conj() @ W_reference
        projections = h_s.conj() @ W
        physical_snr = float(
            np.sum(np.abs(projections) ** 2) / self.system.noise_power
        )
        affine_lower_snr = float(
            (
                2.0
                * np.real(np.vdot(reference_projections, projections))
                - np.vdot(
                    reference_projections, reference_projections
                ).real
            )
            / self.system.noise_power
        )
        squared_distance = float(
            np.vdot(
                projections - reference_projections,
                projections - reference_projections,
            ).real
            / self.system.noise_power
        )
        assert physical_snr - affine_lower_snr == pytest.approx(
            squared_distance, rel=5e-12, abs=1e-9
        )
        assert physical_snr >= affine_lower_snr * (1.0 - 1e-12)
        assert affine_lower_snr >= self.solver.snr_min * (1.0 - 5e-4)
        assert self.system.compute_snr_sensing(W) == pytest.approx(
            physical_snr, rel=5e-13
        )

    def test_snr_constraint_history(self):
        """Test optimization history is recorded."""
        result = self.solver.solve()
        assert len(result["history"]) > 0
        assert len(result["history"]) <= self.solver.max_iter
        powers = np.asarray(result["power_history"])
        assert len(powers) == len(result["history"])
        assert np.all(np.diff(powers) <= 1e-8 * powers[:-1])

    def test_failed_subproblem_does_not_drop_snr_constraint(self, monkeypatch):
        """A failed constrained solve must be reported, never relaxed."""
        calls = 0

        def fail(_problem):
            nonlocal calls
            calls += 1
            raise RuntimeError("injected solver failure")

        monkeypatch.setattr(snr_constraint, "_solve_problem", fail)
        thresholds = np.full(self.system.K, self.system.sinr_thresh)
        with pytest.raises(RuntimeError, match="was not relaxed"):
            self.solver._solve_beamforming_socp(
                thresholds, self.solver._initial_beamformers()
            )
        assert calls == 1

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"snr_min_dB": np.nan}, "snr_min_dB"),
            ({"max_iter": 0}, "max_iter"),
            ({"max_iter": True}, "max_iter"),
            ({"tol": 0.0}, "tol"),
        ],
    )
    def test_invalid_solver_controls_are_rejected(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            SNRConstrainedSolver(self.system, **kwargs)

    def test_invalid_sinr_vector_is_rejected(self):
        with pytest.raises(ValueError, match="sinr_thresholds"):
            self.solver._solve_beamforming_socp(
                np.array([1.0, np.nan]),
                self.solver._initial_beamformers(),
            )

    def test_invalid_sensing_reference_is_rejected(self):
        thresholds = np.full(self.system.K, self.system.sinr_thresh)
        with pytest.raises(ValueError, match="W_reference"):
            self.solver._solve_beamforming_socp(
                thresholds, np.full((self.system.M, self.system.K), np.nan)
            )

    def test_zero_sensing_reference_is_rejected(self):
        thresholds = np.full(self.system.K, self.system.sinr_thresh)
        with pytest.raises(RuntimeError, match="zero covariance power"):
            self.solver._solve_beamforming_socp(
                thresholds, np.zeros((self.system.M, self.system.K))
            )

    @pytest.mark.parametrize(
        ("threshold_db", "error"),
        [
            (1.0e308, OverflowError),
            (-1.0e308, FloatingPointError),
        ],
    )
    def test_unrepresentable_sensing_threshold_is_explicit(
        self, threshold_db, error
    ):
        with pytest.raises(error):
            SNRConstrainedSolver(self.system, snr_min_dB=threshold_db)

    def test_iteration_limit_is_not_false_convergence(self):
        result = SNRConstrainedSolver(
            self.system,
            snr_min_dB=5.0,
            max_iter=1,
        ).solve()
        assert result["iterations"] == 1
        assert not result["converged"]
