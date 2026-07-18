"""
Tests for Quadratic Transform
==============================

Tests for the exact Eq. (14) log-SINR quadratic transform.

Reference: Zou et al., IEEE Trans. Commun., 2024 (Eq. 14)
"""

import numpy as np
import pytest

from ..src.ee_metrics import compute_sum_rate
from ..src.system_model import ISACSystemModel
from ..src.quadratic_transform import (
    quadratic_transform_objective,
    optimize_t,
    compute_sum_rate_quadratic,
)


class TestQuadraticTransform:
    """Test suite for quadratic transform."""

    @pytest.fixture
    def system_data(self):
        """Create test system data."""
        model = ISACSystemModel(M=8, K=3, N=10, seed=42)
        H = model.get_csi()
        sigma_c2 = model.sigma_c2
        rng = np.random.default_rng(202407)
        W = rng.standard_normal((8, 3)) + 1j * rng.standard_normal((8, 3))
        W *= 0.1  # Scale down
        return H, W, sigma_c2

    def test_optimize_t_dimensions(self, system_data):
        """Test optimal t has correct dimensions."""
        H, W, sigma_c2 = system_data
        t = optimize_t(H, W, sigma_c2)
        assert t.shape == (3,)
        assert np.iscomplexobj(t)

    def test_optimize_t_formula(self, system_data):
        """Test optimal t satisfies the closed-form solution."""
        H, W, sigma_c2 = system_data
        t = optimize_t(H, W, sigma_c2)

        K = H.shape[0]
        for k in range(K):
            h_k = H[k, :]
            hw_k = h_k.conj() @ W[:, k]
            interference_plus_noise = sigma_c2 + sum(
                np.abs(h_k.conj() @ W[:, j]) ** 2
                for j in range(K)
                if j != k
            )
            expected_t = hw_k / interference_plus_noise
            np.testing.assert_allclose(t[k], expected_t, rtol=1e-12)

    def test_quadratic_transform_objective(self, system_data):
        """Test quadratic transform objective is finite."""
        H, W, sigma_c2 = system_data
        t = optimize_t(H, W, sigma_c2)
        obj = quadratic_transform_objective(H, W, t, sigma_c2)
        assert np.isfinite(obj)

    def test_transform_equals_sum_rate_at_optimal_t(self, system_data):
        """Eq. (14) must equal the direct sum rate at the Eq. (15) t."""
        H, W, sigma_c2 = system_data
        t = optimize_t(H, W, sigma_c2)
        qt_obj = quadratic_transform_objective(H, W, t, sigma_c2)

        # Direct sum rate computation
        sum_rate = 0.0
        K = H.shape[0]
        for k in range(K):
            h_k = H[k, :]
            signal = np.abs(h_k.conj() @ W[:, k]) ** 2
            interference = sum(
                np.abs(h_k.conj() @ W[:, j]) ** 2 for j in range(K) if j != k
            )
            sinr_k = signal / (sigma_c2 + interference)
            sum_rate += np.log2(1 + sinr_k)

        np.testing.assert_allclose(qt_obj, sum_rate, rtol=1e-12, atol=1e-12)

    def test_nonoptimal_t_is_a_strict_lower_bound(self, system_data):
        """Moving away from Eq. (15) must lower the transformed rate."""
        H, W, sigma_c2 = system_data
        optimal_t = optimize_t(H, W, sigma_c2)
        optimal_value = quadratic_transform_objective(
            H, W, optimal_t, sigma_c2
        )
        perturbed_value = quadratic_transform_objective(
            H, W, 0.5 * optimal_t, sigma_c2
        )
        assert perturbed_value < optimal_value

    def test_compute_sum_rate_quadratic(self, system_data):
        """Test sum rate computation via quadratic transform."""
        H, W, sigma_c2 = system_data
        sum_rate = compute_sum_rate_quadratic(H, W, sigma_c2)
        direct_rate = 0.0
        for user in range(H.shape[0]):
            desired = abs(H[user].conj() @ W[:, user]) ** 2
            interference = sum(
                abs(H[user].conj() @ W[:, other]) ** 2
                for other in range(H.shape[0])
                if other != user
            )
            direct_rate += np.log2(1.0 + desired / (sigma_c2 + interference))
        np.testing.assert_allclose(sum_rate, direct_rate, rtol=1e-12)

    def test_quadratic_transform_preserves_sub_epsilon_snr(self):
        """The exact transform must retain the rate at SINR=1e-20."""

        H = np.array([[1.0e-10 + 0.0j]])
        W = np.array([[1.0 + 0.0j]])
        expected = 1.4426950408889633e-20
        assert compute_sum_rate_quadratic(H, W, 1.0) == pytest.approx(
            expected, rel=6e-16, abs=0.0
        )

    def test_reduced_optimum_avoids_transient_quadratic_overflow(self):
        H = np.array([[1.0e154 + 0.0j]])
        W = np.array([[1.0 + 0.0j]])
        expected = float(np.log1p(1.0e308) / np.log(2.0))
        transformed = compute_sum_rate_quadratic(H, W, 1.0)
        direct = compute_sum_rate(H, W, 1.0)
        np.testing.assert_allclose(transformed, expected, rtol=2e-15, atol=0)
        np.testing.assert_array_equal(transformed, direct)

    def test_invalid_shape_fails_explainably(self, system_data):
        """Shape errors must not leak through as opaque NumPy failures."""
        H, W, sigma_c2 = system_data
        with pytest.raises(ValueError, match="W must have shape"):
            optimize_t(H, W[:-1], sigma_c2)

    def test_t_k_zero_when_no_signal(self):
        """Test t_k is zero when signal is zero."""
        H = np.array([[1, 0], [0, 1]], dtype=complex)  # 2x2
        W = np.zeros((2, 2), dtype=complex)  # Zero beamforming
        sigma_c2 = 1.0

        t = optimize_t(H, W, sigma_c2)
        np.testing.assert_allclose(t, 0.0, atol=1e-15)
