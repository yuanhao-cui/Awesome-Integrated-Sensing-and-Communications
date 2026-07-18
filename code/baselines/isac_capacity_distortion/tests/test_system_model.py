"""Tests for validated Gaussian-channel and information primitives."""

from __future__ import annotations

import numpy as np
import pytest

from ..src.system_model import (
    GaussianISACChannel,
    angle_to_channel,
    angle_to_hfunc,
    compute_bfim,
    compute_crb,
    compute_phi_angle,
    compute_rate,
    compute_rate_per_symbol,
    make_uniform_linear_array,
)


def _sensing_channel(eta: np.ndarray) -> np.ndarray:
    return float(eta[0]) * np.eye(2, dtype=np.complex128)


def _channel(seed: int = 7) -> GaussianISACChannel:
    return GaussianISACChannel(
        np.eye(2),
        _sensing_channel,
        sigma_c2=0.4,
        sigma_s2=0.7,
        M=2,
        Nc=2,
        Ns=2,
        T=6,
        rng=np.random.default_rng(seed),
    )


class TestGaussianISACChannel:
    def test_shapes_and_copies(self) -> None:
        model = _channel()
        returned = model.comm_channel()
        returned[0, 0] = 9
        assert model.comm_channel()[0, 0] == 1
        assert model.sensing_channel(np.array([2.0])).shape == (2, 2)
        assert model.generate_noise(3).shape == (3, 6)
        assert model.comm_receive(np.zeros((2, 6))).shape == (2, 6)
        assert model.sense_receive(np.zeros((2, 6)), np.array([1.0])).shape == (
            2,
            6,
        )

    def test_seeded_observations_are_bitwise_repeatable(self) -> None:
        first = _channel(2026)
        second = _channel(2026)
        waveform = np.arange(12).reshape(2, 6)
        np.testing.assert_array_equal(
            first.comm_receive(waveform),
            second.comm_receive(waveform),
        )
        np.testing.assert_array_equal(
            first.sense_receive(waveform, np.array([0.3])),
            second.sense_receive(waveform, np.array([0.3])),
        )

    def test_complex_noise_convention(self) -> None:
        model = GaussianISACChannel(
            np.ones((1, 1)),
            lambda _eta: np.ones((1, 1)),
            1,
            1,
            1,
            1,
            1,
            200_000,
            rng=np.random.default_rng(123),
        )
        samples = model.generate_noise(1)
        assert abs(float(np.mean(samples).real)) < 0.01
        assert abs(float(np.mean(samples).imag)) < 0.01
        assert abs(float(np.mean(np.abs(samples) ** 2)) - 1) < 0.01

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("M", 0),
            ("Nc", 1.5),
            ("Ns", True),
            ("T", -1),
            ("sigma_c2", 0),
            ("sigma_s2", np.inf),
        ],
    )
    def test_rejects_invalid_scalar_configuration(
        self,
        field: str,
        value: object,
    ) -> None:
        arguments: dict[str, object] = {
            "Hc": np.eye(2),
            "Hs_func": _sensing_channel,
            "sigma_c2": 1,
            "sigma_s2": 1,
            "M": 2,
            "Nc": 2,
            "Ns": 2,
            "T": 3,
        }
        arguments[field] = value
        with pytest.raises(ValueError):
            GaussianISACChannel(**arguments)

    def test_rejects_bad_channel_shape_and_rng(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            GaussianISACChannel(
                np.eye(3),
                _sensing_channel,
                1,
                1,
                2,
                2,
                2,
                3,
            )
        with pytest.raises(TypeError, match="Generator"):
            GaussianISACChannel(
                np.eye(2),
                _sensing_channel,
                1,
                1,
                2,
                2,
                2,
                3,
                rng=3,  # type: ignore[arg-type]
            )

    def test_rejects_bad_waveform_and_sensing_output(self) -> None:
        model = _channel()
        with pytest.raises(ValueError, match="shape"):
            model.comm_receive(np.zeros((2, 5)))
        bad_model = GaussianISACChannel(
            np.eye(2),
            lambda _eta: np.eye(3),
            1,
            1,
            2,
            2,
            2,
            3,
        )
        with pytest.raises(ValueError, match="Hs_func"):
            bad_model.sensing_channel(np.array([0.0]))


class TestInformation:
    def test_bfim_matches_diagonal_formula(self) -> None:
        covariance = np.diag([0.25, 2.0])
        prior = np.diag([1.0, 3.0])
        actual = compute_bfim(covariance, 7, 0.5, Jp=prior)
        np.testing.assert_allclose(actual, np.diag([4.5, 31.0]), rtol=0, atol=0)

    def test_custom_phi_is_scaled_once(self) -> None:
        covariance = np.eye(2)
        actual = compute_bfim(
            covariance,
            4,
            2,
            phi_func=lambda matrix: 3 * matrix,
        )
        np.testing.assert_allclose(actual, 6 * np.eye(2))

    @pytest.mark.parametrize("T", [0, -1, 1.5, True])
    def test_bfim_rejects_invalid_interval(self, T: object) -> None:
        with pytest.raises(ValueError, match="T"):
            compute_bfim(np.eye(2), T, 1)  # type: ignore[arg-type]

    @pytest.mark.parametrize("variance", [0, -1, np.nan, np.inf])
    def test_bfim_rejects_invalid_noise(self, variance: float) -> None:
        with pytest.raises(ValueError, match="sigma_s2"):
            compute_bfim(np.eye(2), 1, variance)

    @pytest.mark.parametrize(
        "covariance",
        [
            np.ones((2, 3)),
            np.array([[1, 1], [0, 1]]),
            np.diag([1.0, -0.01]),
            np.array([[np.nan]]),
        ],
    )
    def test_bfim_rejects_invalid_covariance(self, covariance: np.ndarray) -> None:
        with pytest.raises(ValueError):
            compute_bfim(covariance, 2, 1)

    def test_bfim_rejects_invalid_phi_and_prior(self) -> None:
        with pytest.raises(ValueError, match="Phi"):
            compute_bfim(np.eye(2), 2, 1, phi_func=lambda _matrix: np.ones((2, 3)))
        with pytest.raises(ValueError, match="same shape"):
            compute_bfim(np.eye(2), 2, 1, Jp=np.eye(3))
        with pytest.raises(ValueError, match="Hermitian"):
            compute_bfim(
                np.eye(2),
                2,
                1,
                Jp=np.array([[1, 1], [0, 1]]),
            )
        with pytest.raises(ValueError, match="positive semidefinite"):
            compute_bfim(np.eye(2), 2, 1, Jp=np.diag([1.0, -0.1]))

    def test_bfim_handles_large_hermitian_scale_without_intermediate_overflow(
        self,
    ) -> None:
        actual = compute_bfim(np.array([[1e308]]), 1, 1)
        np.testing.assert_array_equal(actual, np.array([[1e308]]))

    def test_bfim_rejects_unrepresentable_observation_information(self) -> None:
        with pytest.raises(ValueError, match="BFIM exceeds"):
            compute_bfim(np.array([[1e308]]), 2, np.nextafter(0.0, 1.0))
        with pytest.raises(ValueError, match="T is outside"):
            compute_bfim(np.eye(1), 10**1000, 1.0)

    def test_bfim_combines_large_scale_ratio_with_weak_information(self) -> None:
        actual = compute_bfim(
            np.array([[1e-308]]),
            2,
            np.nextafter(0.0, 1.0),
        )
        expected = 2e-308 / np.nextafter(0.0, 1.0)
        np.testing.assert_allclose(actual, np.array([[expected]]), rtol=5e-15)

    def test_crb_matches_analytic_formula(self) -> None:
        crb = compute_crb(bfim=np.diag([4.5, 31.0]))
        np.testing.assert_allclose(crb, 1 / 4.5 + 1 / 31.0, rtol=1e-15)

    def test_crb_singular_is_infinite(self) -> None:
        assert np.isinf(compute_crb(bfim=np.diag([1.0, 0.0])))

    def test_crb_weak_positive_information_is_finite(self) -> None:
        np.testing.assert_allclose(
            compute_crb(bfim=np.array([[1e-13]])),
            1e13,
            rtol=0,
            atol=0,
        )

    def test_crb_beyond_representable_range_is_infinite(self) -> None:
        assert np.isinf(compute_crb(bfim=np.array([[1e-320]])))
        assert np.isinf(compute_crb(bfim=np.diag([1e-308, 1e-308])))

    def test_crb_rejects_negative_information(self) -> None:
        with pytest.raises(ValueError, match="positive semidefinite"):
            compute_crb(bfim=np.array([[-1e-13]]))
        with pytest.raises(ValueError, match="Rx"):
            compute_crb()


class TestRate:
    def test_siso_rate_matches_shannon_formula(self) -> None:
        channel = np.array([[1.2 - 0.4j]])
        covariance = np.array([[0.7]])
        actual = compute_rate(covariance, channel, 0.3)
        expected = np.log1p((1.2**2 + 0.4**2) * 0.7 / 0.3)
        np.testing.assert_allclose(actual, expected, rtol=1e-15)

    def test_identity_mimo_rate(self) -> None:
        actual = compute_rate(2 * np.eye(3), np.eye(3), 0.5)
        np.testing.assert_allclose(actual, 3 * np.log(5), rtol=1e-15)

    def test_zero_covariance_has_zero_rate(self) -> None:
        assert compute_rate(np.zeros((2, 2)), np.ones((1, 2)), 1) == 0

    def test_rate_normalizes_before_channel_energy_product(self) -> None:
        """Representable H/sqrt(sigma) energy must not underflow via H^H H."""

        channel = np.diag([2e-200, 1e-200])
        covariance = np.diag([2.0, 0.0])
        actual = compute_rate(covariance, channel, 1e-300)
        expected = np.log1p(8e-100)
        assert actual > 0
        np.testing.assert_allclose(actual, expected, rtol=2e-15)

    @pytest.mark.parametrize("exponent", [155, 158, 160, 161])
    def test_rate_preserves_representable_subnormal_energy(
        self,
        exponent: int,
    ) -> None:
        scale = 10.0 ** (-exponent)
        channel = np.diag([2 * scale, scale])
        actual = compute_rate(np.diag([2.0, 0.0]), channel, 1.0)
        expected = np.log1p(8 * scale**2)
        assert actual > 0
        np.testing.assert_allclose(actual, expected, rtol=0.01, atol=0)

    def test_rate_rejects_unrepresentable_normalized_channel(self) -> None:
        with pytest.raises(ValueError, match="floating-point range"):
            compute_rate(np.eye(1), np.array([[1e308]]), 1e-300)

    @pytest.mark.parametrize("variance", [0, -1, np.nan])
    def test_rate_rejects_invalid_noise(self, variance: float) -> None:
        with pytest.raises(ValueError, match="sigma_c2"):
            compute_rate(np.eye(2), np.eye(2), variance)

    def test_rate_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError, match="incompatible"):
            compute_rate(np.eye(2), np.eye(3), 1)
        with pytest.raises(ValueError, match="Hermitian"):
            compute_rate(np.array([[1, 1], [0, 1]]), np.eye(2), 1)
        with pytest.raises(ValueError, match="finite"):
            compute_rate(np.eye(2), np.array([[np.nan, 0]]), 1)

    def test_waveform_rate_uses_xxh_over_interval(self) -> None:
        waveform = np.array([[1, 1j, -1, -1j]], dtype=np.complex128)
        actual = compute_rate_per_symbol(waveform, np.array([[2.0]]), 0.5)
        np.testing.assert_allclose(actual, np.log(9), rtol=1e-15)

    def test_waveform_rate_avoids_unnecessary_xxh_overflow(self) -> None:
        actual = compute_rate_per_symbol(
            np.array([[1e200 + 0j]]),
            np.array([[1e-200 + 0j]]),
            1.0,
        )
        np.testing.assert_allclose(actual, np.log(2.0), rtol=1e-15)

    @pytest.mark.parametrize(
        ("channel", "waveform"),
        [
            ([1e280, 1.0, 1e280], [1.0, 1e-60, -1.0]),
            ([1.0, 1e280, 1e280], [1e-60, 1.0, -1.0]),
            ([1e280, 1e280, 1.0], [1.0, -1.0, 1e-60]),
        ],
    )
    def test_waveform_rate_preserves_permuted_cross_scale_cancellation(
        self,
        channel: list[float],
        waveform: list[float],
    ) -> None:
        actual = compute_rate_per_symbol(
            np.asarray(waveform, dtype=np.complex128).reshape(3, 1),
            np.asarray(channel, dtype=np.complex128).reshape(1, 3),
            1.0,
        )
        expected = np.log1p(1e-120)
        assert actual > 0.0
        np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0)


class TestAngleModel:
    def test_steering_norm_and_channel_shape(self) -> None:
        steering = make_uniform_linear_array(5)(np.deg2rad(20))
        np.testing.assert_allclose(np.linalg.norm(steering), np.sqrt(5))
        assert angle_to_channel(np.deg2rad(20), 5, 3).shape == (3, 5)
        assert angle_to_hfunc(5, 3)(np.array([np.deg2rad(20)])).shape == (3, 5)

    def test_angle_information_matches_finite_difference(self) -> None:
        covariance = np.diag([0.2, 0.5, 0.8, 1.1])
        theta = np.deg2rad(27)
        epsilon = 1e-6
        derivative = (
            angle_to_channel(theta + epsilon, 4, 3)
            - angle_to_channel(theta - epsilon, 4, 3)
        ) / (2 * epsilon)
        expected = 2 * np.real(
            np.trace(derivative.conj().T @ derivative @ covariance)
        )
        actual = compute_phi_angle(covariance, 9, theta, 4, 3)[0, 0].real
        np.testing.assert_allclose(actual, expected, rtol=1e-8)

    def test_angle_information_scales_linearly_with_covariance(self) -> None:
        covariance = np.diag([0.2, 0.5, 0.8])
        first = compute_phi_angle(covariance, 2, 0.3, 3, 2)
        second = compute_phi_angle(4 * covariance, 2, 0.3, 3, 2)
        np.testing.assert_allclose(second, 4 * first, rtol=1e-14)

    def test_angle_helpers_reject_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="M"):
            make_uniform_linear_array(0)
        with pytest.raises(ValueError, match="non-negative"):
            make_uniform_linear_array(2, -0.5)
        with pytest.raises(ValueError, match="exactly one"):
            angle_to_hfunc(2, 2)(np.array([0.1, 0.2]))
        with pytest.raises(ValueError, match="shape"):
            compute_phi_angle(np.eye(3), 2, 0.1, 2, 2)
        with pytest.raises(ValueError, match="finite angle-information domain"):
            compute_phi_angle(
                np.eye(2),
                1,
                0.0,
                2,
                2,
                d_tx=1e308,
                d_rx=1e308,
            )
