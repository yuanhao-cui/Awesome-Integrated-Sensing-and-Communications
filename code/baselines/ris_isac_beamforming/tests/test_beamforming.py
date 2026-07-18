"""Tests for BS beamforming optimization."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ..src.system_model import RIS_ISAC_System
from ..src.beamforming import BeamformingOptimizer


class TestBeamforming:
    """Beamforming optimizer tests."""

    def setup_method(self):
        self.system = RIS_ISAC_System(M=4, K=2, L=30, seed=42)
        self.bf_opt = BeamformingOptimizer(self.system)

    def test_beamforming_shape(self):
        """Test beamforming output has correct shape."""
        sinr_thresh = np.full(self.system.K, self.system.sinr_thresh)
        W, obj = self.bf_opt.solve_full_power_feasible(sinr_thresh)
        M, K = self.system.M, self.system.K
        assert W.shape == (M, K), f"Expected ({M}, {K}), got {W.shape}"

    def test_power_constraint(self):
        """Test beamforming satisfies power constraint."""
        sinr_thresh = np.full(self.system.K, self.system.sinr_thresh)
        W, _ = self.bf_opt.solve_full_power_feasible(sinr_thresh)
        total_power = np.sum(np.linalg.norm(W, axis=0) ** 2)
        assert total_power <= self.system.P_max * 1.05

    def test_effective_channels_retrieval(self):
        """Test effective channel retrieval."""
        H_eff = self.bf_opt._get_effective_channels()
        assert H_eff.shape == (self.system.K, self.system.M)
        assert H_eff.dtype == complex

    def test_min_power_beamforming(self):
        """Test minimum power beamforming returns feasible solution."""
        sinr_thresh = np.full(self.system.K, 1.0)  # Low threshold
        W, min_power = self.bf_opt.solve_min_power(sinr_thresh)
        assert W.shape == (self.system.M, self.system.K)
        if min_power < float("inf"):
            assert min_power > 0

    def test_sum_rate_positive_after_feasible_scaling(self):
        """Test sum rate is positive after feasible full-power scaling."""
        sinr_thresh = np.full(self.system.K, self.system.sinr_thresh)
        W, _ = self.bf_opt.solve_full_power_feasible(sinr_thresh)
        rate = self.system.compute_sum_rate(W)
        assert rate > 0, f"Sum rate should be positive, got {rate}"

    def test_beamforming_nonzero(self):
        """Test beamforming vectors are non-zero."""
        sinr_thresh = np.full(self.system.K, self.system.sinr_thresh)
        W, _ = self.bf_opt.solve_full_power_feasible(sinr_thresh)
        norms = np.linalg.norm(W, axis=0)
        assert np.all(norms > 1e-10), "All beamforming vectors should be non-zero"

    @pytest.mark.parametrize("invalid", [np.nan, np.inf, -1.0])
    def test_invalid_sinr_threshold_is_rejected(self, invalid):
        thresholds = np.ones(self.system.K)
        thresholds[0] = invalid
        with pytest.raises(ValueError, match="sinr_thresholds"):
            self.bf_opt.solve_min_power(thresholds)
