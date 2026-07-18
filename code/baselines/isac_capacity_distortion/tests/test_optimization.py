"""Tests and independent numerical oracles for local optimizers."""

from __future__ import annotations

import numpy as np
import pytest

from ..src.optimization import (
    covariance_shaping_surrogate,
    isotropic_covariance,
    make_semiunitary_waveform,
    sample_gaussian_waveform,
    sample_row_semiunitary,
    water_filling_covariance,
)
from ..src.system_model import compute_rate


class TestIsotropicCovariance:
    def test_value_trace_and_dtype(self) -> None:
        covariance = isotropic_covariance(1.25, 4)
        np.testing.assert_array_equal(covariance, 1.25 * np.eye(4))
        assert covariance.dtype == np.complex128
        assert np.trace(covariance) == 5

    @pytest.mark.parametrize("power", [-1, np.nan, np.inf])
    def test_rejects_invalid_power(self, power: float) -> None:
        with pytest.raises(ValueError, match="power_per_tx"):
            isotropic_covariance(power, 2)

    @pytest.mark.parametrize("antennas", [0, -1, 1.5, True])
    def test_rejects_invalid_dimension(self, antennas: object) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            isotropic_covariance(1, antennas)  # type: ignore[arg-type]


class TestWaterFilling:
    def test_diagonal_solution_matches_closed_form(self) -> None:
        channel = np.diag([2.0, 1.0])
        covariance = water_filling_covariance(1.0, channel, sigma_c2=1.0)
        expected = np.diag([1.375, 0.625])
        np.testing.assert_allclose(covariance, expected, rtol=0, atol=2e-15)

    def test_high_noise_deactivates_weak_mode(self) -> None:
        covariance = water_filling_covariance(
            1.0,
            np.diag([2.0, 1.0]),
            sigma_c2=100,
        )
        np.testing.assert_allclose(covariance, np.diag([2.0, 0.0]), atol=1e-14)

    def test_zero_channel_uses_deterministic_isotropic_tie_break(self) -> None:
        covariance = water_filling_covariance(2.0, np.zeros((2, 3)), 1.0)
        np.testing.assert_array_equal(covariance, 2 * np.eye(3))

    def test_zero_power_returns_zero(self) -> None:
        covariance = water_filling_covariance(0, np.ones((2, 3)), 1)
        np.testing.assert_array_equal(covariance, np.zeros((3, 3)))

    def test_dynamic_range_safe_water_filling(self) -> None:
        channel = np.diag([2e-200, 1e-200])
        covariance = water_filling_covariance(1, channel, 1e-300)
        np.testing.assert_allclose(covariance, np.diag([2.0, 0.0]), atol=0)
        rate = compute_rate(covariance, channel, 1e-300)
        assert rate > 0
        np.testing.assert_allclose(rate, np.log1p(8e-100), rtol=2e-15)

    @pytest.mark.parametrize("exponent", [155, 158, 160, 161])
    def test_subnormal_gain_water_filling_avoids_reciprocal_overflow(
        self,
        exponent: int,
    ) -> None:
        scale = 10.0 ** (-exponent)
        channel = np.diag([2 * scale, scale])
        covariance = water_filling_covariance(1, channel, 1)
        np.testing.assert_allclose(covariance, np.diag([2.0, 0.0]), atol=0)
        assert compute_rate(covariance, channel, 1) > 0

    def test_rejects_unrepresentable_total_power_budget(self) -> None:
        with pytest.raises(ValueError, match="total power budget"):
            water_filling_covariance(1e308, np.eye(2), 1)
        with pytest.raises(ValueError, match="total power budget"):
            covariance_shaping_surrogate(0.5, 1e308, np.eye(2), 1)

    def test_dense_simplex_grid_oracle(self) -> None:
        channel = np.diag([2.0, 0.75])
        sigma = 0.8
        covariance = water_filling_covariance(1.0, channel, sigma)
        solver_power = float(covariance[0, 0].real)
        grid = np.linspace(0, 2, 200_001)
        gains = np.array([4.0, 0.75**2]) / sigma
        rates = np.log1p(gains[0] * grid) + np.log1p(gains[1] * (2 - grid))
        grid_power = float(grid[int(np.argmax(rates))])
        assert abs(solver_power - grid_power) <= 1.1 * float(grid[1] - grid[0])

    def test_random_psd_competitors_do_not_beat_solution(self) -> None:
        rng = np.random.default_rng(19)
        channel = (
            rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3))
        ) / np.sqrt(2)
        optimum = water_filling_covariance(1.0, channel, 0.7)
        optimum_rate = compute_rate(optimum, channel, 0.7)
        for _ in range(1_000):
            factor = (
                rng.standard_normal((3, 3))
                + 1j * rng.standard_normal((3, 3))
            ) / np.sqrt(2)
            competitor = factor @ factor.conj().T
            competitor *= 3 / np.trace(competitor).real
            assert compute_rate(competitor, channel, 0.7) <= optimum_rate + 1e-12

    def test_rejects_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="two-dimensional"):
            water_filling_covariance(1, np.ones(2), 1)
        with pytest.raises(ValueError, match="finite"):
            water_filling_covariance(1, np.array([[np.nan]]), 1)
        with pytest.raises(ValueError, match="sigma_c2"):
            water_filling_covariance(1, np.eye(2), 0)


class TestCovarianceSurrogate:
    def test_endpoints_are_explicit_reference_solutions(self) -> None:
        channel = np.array([[1.0, 0.2], [0.1, 0.5]])
        np.testing.assert_allclose(
            covariance_shaping_surrogate(0, 1, channel, 0.4),
            np.eye(2),
        )
        np.testing.assert_allclose(
            covariance_shaping_surrogate(1, 1, channel, 0.4),
            water_filling_covariance(1, channel, 0.4),
        )

    def test_interior_solution_matches_dense_grid(self) -> None:
        alpha = 0.37
        channel = np.diag([2.0, 0.75])
        sigma = 0.8
        covariance = covariance_shaping_surrogate(alpha, 1, channel, sigma)
        solver_power = float(covariance[0, 0].real)
        count = 200_001
        step = 2 / (count + 1)
        grid = np.linspace(step, 2 - step, count)
        gains = np.array([4.0, 0.75**2]) / sigma
        objective = (
            -(1 - alpha) * (np.log(grid) + np.log(2 - grid))
            - alpha
            * (np.log1p(gains[0] * grid) + np.log1p(gains[1] * (2 - grid)))
        )
        grid_power = float(grid[int(np.argmin(objective))])
        assert abs(solver_power - grid_power) <= 1.1 * step

    def test_extreme_channel_quadratic_root_is_scale_safe(self) -> None:
        covariance = covariance_shaping_surrogate(
            0.5,
            1.0,
            np.array([[1.0e154]], dtype=np.complex128),
            1.0,
        )
        np.testing.assert_array_equal(covariance, np.ones((1, 1)))

    def test_power_gain_product_need_not_be_representable(self) -> None:
        covariance = covariance_shaping_surrogate(
            0.5,
            1.0,
            np.diag([1.0e154, 0.0]).astype(np.complex128),
            1.0,
        )
        np.testing.assert_allclose(
            covariance,
            np.diag([4 / 3, 2 / 3]),
            rtol=0,
            atol=2e-15,
        )

    @pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9, 0.999999])
    def test_kkt_stationarity_and_constraints(self, alpha: float) -> None:
        channel = np.diag([3.0, 1.2, 0.25])
        sigma = 0.6
        covariance = covariance_shaping_surrogate(alpha, 1, channel, sigma)
        powers = np.real(np.diag(covariance))
        gains = np.diag(channel) ** 2 / sigma
        stationarity = (
            (1 - alpha) / powers + alpha * gains / (1 + gains * powers)
        )
        assert float(np.ptp(stationarity)) < 3e-12
        np.testing.assert_allclose(np.trace(covariance), 3, atol=2e-12)
        assert float(np.min(np.linalg.eigvalsh(covariance))) > 0

    def test_rotated_channel_solution_commutes_with_gram(self) -> None:
        rng = np.random.default_rng(3)
        channel = (
            rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        ) / np.sqrt(2)
        covariance = covariance_shaping_surrogate(0.4, 1, channel, 0.9)
        gram = channel.conj().T @ channel
        assert np.linalg.norm(covariance @ gram - gram @ covariance) < 2e-13

    @pytest.mark.parametrize("alpha", [-0.1, 1.1, np.nan])
    def test_rejects_invalid_alpha(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            covariance_shaping_surrogate(alpha, 1, np.eye(2), 1)


class TestWaveforms:
    def test_row_semiunitarity_and_seed_repeatability(self) -> None:
        first = sample_row_semiunitary(3, 8, np.random.default_rng(101))
        second = sample_row_semiunitary(3, 8, np.random.default_rng(101))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first @ first.conj().T, np.eye(3), atol=1e-14)

    def test_semiunitary_waveform_has_exact_sample_covariance(self) -> None:
        basis = np.eye(4, 2, dtype=np.complex128)
        waveform, q_rows = make_semiunitary_waveform(
            1.5,
            4,
            7,
            np.random.default_rng(6),
            basis=basis,
        )
        covariance = waveform @ waveform.conj().T / 7
        expected = 1.5 * 4 / 2 * basis @ basis.conj().T
        np.testing.assert_allclose(covariance, expected, atol=2e-15)
        np.testing.assert_allclose(q_rows @ q_rows.conj().T, np.eye(2), atol=1e-14)

    def test_gaussian_waveform_seed_repeatability_and_shape(self) -> None:
        first = sample_gaussian_waveform(0.5, 3, 9, np.random.default_rng(8))
        second = sample_gaussian_waveform(0.5, 3, 9, np.random.default_rng(8))
        np.testing.assert_array_equal(first, second)
        assert first.shape == (3, 9)

    def test_waveform_input_validation(self) -> None:
        with pytest.raises(ValueError, match="rows cannot exceed"):
            sample_row_semiunitary(3, 2, np.random.default_rng(1))
        with pytest.raises(TypeError, match="Generator"):
            sample_row_semiunitary(2, 3, 1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="orthonormal"):
            make_semiunitary_waveform(
                1,
                2,
                3,
                np.random.default_rng(1),
                basis=np.ones((2, 2)),
            )
