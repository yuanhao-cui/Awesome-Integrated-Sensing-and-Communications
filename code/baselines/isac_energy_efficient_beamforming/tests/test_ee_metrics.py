"""Independent numerical oracles for the published model equations."""

from decimal import Decimal, localcontext
from itertools import permutations

import numpy as np
import pytest

from ..src.ee_metrics import (
    compute_crb,
    compute_ee_c,
    compute_ee_s,
    compute_sinr,
    compute_sum_rate,
    compute_total_power,
    point_target_information_terms,
)
from ..src.system_model import ISACSystemModel


def _fixture() -> tuple[ISACSystemModel, np.ndarray, float]:
    model = ISACSystemModel(
        M=3,
        K=2,
        N=4,
        L=8,
        sigma_c_dbm=-20.0,
        sigma_s_dbm=-10.0,
        seed=11,
    )
    rng = np.random.default_rng(90210)
    W = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    W *= np.sqrt(0.2 / np.vdot(W, W).real)
    return model, W, 1.03


def test_sum_rate_and_ee_match_direct_equations() -> None:
    model, W, _ = _fixture()
    rates = []
    for k in range(model.K):
        projections = model.H[k].conj() @ W
        signal = abs(projections[k]) ** 2
        interference = sum(
            abs(projections[j]) ** 2 for j in range(model.K) if j != k
        )
        rates.append(np.log2(1.0 + signal / (model.sigma_c2 + interference)))
    expected_rate = float(sum(rates))
    expected_ee = expected_rate / (
        np.vdot(W, W).real / model.epsilon + model.P0
    )
    assert compute_sum_rate(model.H, W, model.sigma_c2) == pytest.approx(
        expected_rate, rel=1e-13
    )
    assert compute_ee_c(
        model.H, W, model.sigma_c2, model.epsilon, model.P0
    ) == pytest.approx(expected_ee, rel=1e-13)


def test_sum_rate_preserves_representable_sub_epsilon_snr() -> None:
    """The rate of SINR=1e-20 must not round to zero via ``1 + SINR``."""

    H = np.array([[1.0e-10 + 0.0j]])
    W = np.array([[1.0 + 0.0j]])
    expected = 1.4426950408889633e-20
    assert compute_sum_rate(H, W, sigma_c2=1.0) == pytest.approx(
        expected, rel=3e-16, abs=0.0
    )


def test_weak_interference_is_not_cancelled_by_desired_power() -> None:
    """Lock the independent gate's desired-subtraction counterexample."""

    h = np.array([1.0, 0.0])
    W = np.array([[1.0, 1.0e-10], [0.0, 0.0]])
    oracle = 9.999999999000001e19
    assert compute_sinr(0, h, W, 1.0e-30) == pytest.approx(
        oracle, rel=3e-16
    )


def test_exact_cancellation_tail_survives_every_antenna_permutation() -> None:
    """Lock the independent gate's greater-than-324-decade counterexample."""

    h = np.array([1.0e280, 1.0e280, 1.0e-60])
    W = np.array([[1.0], [-1.0], [1.0]])
    sinr_oracle = 1.0e-120
    rate_oracle = 1.4426950408889635e-120
    for permutation in permutations(range(3)):
        indices = np.asarray(permutation)
        assert compute_sinr(0, h[indices], W[indices], 1.0) \
            == pytest.approx(sinr_oracle, rel=3e-15, abs=0.0)
        assert compute_sum_rate(h[indices][None, :], W[indices], 1.0) \
            == pytest.approx(rate_oracle, rel=3e-15, abs=0.0)


@pytest.mark.parametrize("tail_phase", (1.0, -1.0, 1.0j, -1.0j))
def test_complex_cancellation_tail_preserves_phase_and_power(
    tail_phase: complex,
) -> None:
    """Exact real/imaginary buckets retain signed and quadrature tails."""

    h = np.array(
        [
            1.0e280 + 2.0e280j,
            1.0e280 + 2.0e280j,
            1.0e-60 + 2.0e-60j,
        ]
    )
    W = np.array([[1.0], [-1.0], [tail_phase]], dtype=complex)
    for permutation in permutations(range(3)):
        indices = np.asarray(permutation)
        assert compute_sinr(0, h[indices], W[indices], 1.0) \
            == pytest.approx(5.0e-120, rel=3e-15, abs=0.0)


def test_eighty_decade_buckets_cancel_before_tail_conversion() -> None:
    """Five exact cancellation buckets must not erase the final tail."""

    entries: list[float] = []
    weights: list[float] = []
    for exponent in (280, 200, 120, 40, -40):
        value = float(10.0**exponent)
        entries.extend((value, value))
        weights.extend((1.0, -1.0))
    entries.append(1.0e-60)
    weights.append(1.0)
    h = np.asarray(entries)
    W = np.asarray(weights)[:, None]
    rng = np.random.default_rng(60120)
    for _ in range(50):
        permutation = rng.permutation(h.size)
        assert compute_sinr(
            0, h[permutation], W[permutation], 1.0
        ) == pytest.approx(1.0e-120, rel=3e-15, abs=0.0)


def test_sum_rate_retains_weak_interference_with_orthogonal_csi() -> None:
    """The former false 99.6578-bit result is 66.4386 bit/s/Hz."""

    H = np.eye(2)
    W = np.array([[1.0, 1.0e-10], [0.0, 0.0]])
    oracle = 66.43856189760298
    assert compute_sum_rate(H, W, 1.0e-30) == pytest.approx(
        oracle, rel=3e-15
    )


def test_sinr_is_invariant_to_stream_permutation() -> None:
    """The desired index may move, but the excluded stream must follow it."""

    h = np.array([1.0 - 2.0j, -0.5 + 0.25j, 3.0j])
    W = np.array(
        [
            [1.0e80, 1.0e-60j, 2.0, -3.0e20j],
            [2.0e80j, 2.0e-60, -1.0j, 4.0e20],
            [-1.0e80, -3.0e-60j, 0.5, 2.0e20j],
        ],
        dtype=complex,
    )
    k = 2
    reference = compute_sinr(k, h, W, 1.0e-90)
    permutation = np.array([2, 0, 3, 1])
    permuted_k = int(np.flatnonzero(permutation == k)[0])
    assert compute_sinr(
        permuted_k, h, W[:, permutation], 1.0e-90
    ) == pytest.approx(reference, rel=2e-15)


def test_sinr_handles_projection_powers_outside_binary64() -> None:
    """Unrepresentable individual powers may still have a finite ratio."""

    h = np.array([1.0e200, 0.0])
    W = np.array([[1.0e200, 1.0e200], [0.0, 0.0]])
    assert compute_sinr(0, h, W, 1.0) == pytest.approx(1.0)

    high_rate_W = np.array([[1.0e200], [0.0]])
    with pytest.raises(OverflowError, match="SINR exceeds"):
        compute_sinr(0, h, high_rate_W, 1.0)
    assert compute_sum_rate(h[None, :], high_rate_W, 1.0) == pytest.approx(
        800.0 * np.log2(10.0), rel=2e-15
    )


def test_sinr_reports_positive_output_underflow() -> None:
    h = np.array([1.0e-200])
    W = np.array([[1.0e-200]])
    with pytest.raises(FloatingPointError, match="SINR underflows"):
        compute_sinr(0, h, W, 1.0)
    with pytest.raises(FloatingPointError, match="efficiency underflows"):
        compute_sum_rate(h[None, :], W, 1.0)


def test_random_high_dynamic_sinr_matches_decimal_oracle() -> None:
    """Direct exclusion agrees with 220-digit arithmetic over 240 decades."""

    rng = np.random.default_rng(20260718)
    worst_relative_error = 0.0
    with localcontext() as context:
        context.prec = 220
        for _ in range(80):
            h = rng.uniform(-2.0, 2.0, 5) * np.power(
                10.0, rng.integers(-120, 121, 5)
            )
            W = rng.uniform(-2.0, 2.0, (5, 4)) * np.power(
                10.0, rng.integers(-120, 121, (5, 4))
            )
            noise = float(10.0 ** rng.integers(-240, 241))
            k = int(rng.integers(0, 4))
            projections = [
                sum(
                    Decimal.from_float(float(h[row]))
                    * Decimal.from_float(float(W[row, stream]))
                    for row in range(5)
                )
                for stream in range(4)
            ]
            denominator = Decimal.from_float(noise) + sum(
                projection * projection
                for stream, projection in enumerate(projections)
                if stream != k
            )
            oracle_decimal = projections[k] * projections[k] / denominator
            oracle = float(oracle_decimal)
            if oracle == 0.0 or not np.isfinite(oracle):
                continue
            actual = compute_sinr(k, h, W, noise)
            relative_error = abs(actual - oracle) / oracle
            worst_relative_error = max(worst_relative_error, relative_error)
            assert actual == pytest.approx(oracle, rel=3e-13)
    assert worst_relative_error < 3e-13


def test_random_bucket_cancellations_match_decimal_oracle() -> None:
    """Random signed cancellation pairs retain independently computed tails."""

    rng = np.random.default_rng(324080)
    worst_relative_error = 0.0
    checked = 0
    with localcontext() as context:
        context.prec = 700
        for _ in range(80):
            h_entries: list[float] = []
            beam_entries: list[float] = []
            for exponent in (280, 200, 120, 40, -40):
                channel = float(rng.uniform(0.5, 1.5) * 10.0**exponent)
                beam = float(rng.uniform(-2.0, 2.0))
                h_entries.extend((channel, channel))
                beam_entries.extend((beam, -beam))
            tail_channel = float(rng.uniform(0.5, 1.5) * 1.0e-60)
            tail_beam = float(rng.uniform(0.5, 1.5))
            h_entries.append(tail_channel)
            beam_entries.append(tail_beam)
            permutation = rng.permutation(len(h_entries))
            h = np.asarray(h_entries)[permutation]
            W = np.asarray(beam_entries)[permutation, None]
            projection = sum(
                Decimal.from_float(float(channel))
                * Decimal.from_float(float(beam))
                for channel, beam in zip(h, W[:, 0], strict=True)
            )
            oracle = float(projection * projection)
            actual = compute_sinr(0, h, W, 1.0)
            relative_error = abs(actual - oracle) / oracle
            worst_relative_error = max(worst_relative_error, relative_error)
            checked += 1
            assert actual == pytest.approx(oracle, rel=3e-15, abs=0.0)
    assert checked == 80
    assert worst_relative_error < 3e-15


def test_random_complex_bucket_cancellations_match_decimal_oracle() -> None:
    """Signed complex phases agree with an exact-decimal independent path."""

    rng = np.random.default_rng(324081)
    worst_relative_error = 0.0
    with localcontext() as context:
        context.prec = 700
        for _ in range(40):
            h_entries: list[complex] = []
            beam_rows: list[list[complex]] = []
            for exponent in (280, 200, 120, 40, -40):
                channel = complex(
                    rng.uniform(-1.5, 1.5) * 10.0**exponent,
                    rng.uniform(-1.5, 1.5) * 10.0**exponent,
                )
                beams = [
                    complex(rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0))
                    for _ in range(2)
                ]
                h_entries.extend((channel, channel))
                beam_rows.extend((beams, [-value for value in beams]))
            h_entries.append(
                complex(
                    rng.uniform(0.5, 1.5) * 1.0e-60,
                    rng.uniform(0.5, 1.5) * 1.0e-60,
                )
            )
            beam_rows.append(
                [
                    complex(rng.uniform(0.5, 1.5), rng.uniform(-1.0, 1.0))
                    for _ in range(2)
                ]
            )
            permutation = rng.permutation(len(h_entries))
            h = np.asarray(h_entries)[permutation]
            W = np.asarray(beam_rows)[permutation]

            projection_powers: list[Decimal] = []
            for stream in range(2):
                real = Decimal(0)
                imaginary = Decimal(0)
                for channel, beam in zip(h, W[:, stream], strict=True):
                    channel_real = Decimal.from_float(float(channel.real))
                    channel_imag = Decimal.from_float(float(channel.imag))
                    beam_real = Decimal.from_float(float(beam.real))
                    beam_imag = Decimal.from_float(float(beam.imag))
                    real += channel_real * beam_real + channel_imag * beam_imag
                    imaginary += (
                        channel_real * beam_imag - channel_imag * beam_real
                    )
                projection_powers.append(
                    real * real + imaginary * imaginary
                )
            oracle = float(
                projection_powers[0]
                / (Decimal(1) + projection_powers[1])
            )
            actual = compute_sinr(0, h, W, 1.0)
            relative_error = abs(actual - oracle) / oracle
            worst_relative_error = max(worst_relative_error, relative_error)
            assert actual == pytest.approx(oracle, rel=4e-15, abs=0.0)
    assert worst_relative_error < 4e-15


def test_total_power_is_scale_safe_and_has_explicit_output_domain() -> None:
    normalized = np.array([[0.6 + 0.8j]])
    assert compute_total_power(normalized) == pytest.approx(1.0, rel=2e-15)
    assert compute_total_power(1.0e150 * normalized) == pytest.approx(
        1.0e300, rel=2e-15
    )
    assert compute_total_power(np.zeros((2, 2))) == 0.0
    with pytest.raises(OverflowError, match="squared norm exceeds"):
        compute_total_power(np.array([[1.0e200]]))
    with pytest.raises(FloatingPointError, match="squared norm underflows"):
        compute_total_power(np.array([[1.0e-200]]))


def test_covariance_crb_matches_explicit_snapshot_fim() -> None:
    model, W, theta = _fixture()
    a_t = model.steering_vector_tx(theta)
    a_r = model.steering_vector_rx(theta)
    da_t = model.steering_derivative_tx(theta)
    da_r = model.steering_derivative_rx(theta)

    # The first K unnormalised DFT rows satisfy S S^H = L I_K exactly.
    rows = np.arange(model.K)[:, None]
    columns = np.arange(model.L)[None, :]
    symbols = np.exp(2j * np.pi * rows * columns / model.L)
    X = W @ symbols
    response = np.outer(a_r, a_t.conj())
    derivative = np.outer(da_r, a_t.conj()) + np.outer(a_r, da_t.conj())
    g = (response @ X).reshape(-1, order="F")
    g_dot = (derivative @ X).reshape(-1, order="F")
    effective = np.vdot(g_dot, g_dot).real - abs(np.vdot(g, g_dot)) ** 2 / np.vdot(g, g).real
    expected = model.sigma_s2 / (2.0 * effective)

    actual = compute_crb(
        W, a_t, a_r, da_t, da_r, model.sigma_s2, model.L
    )
    assert actual == pytest.approx(expected, rel=2e-12)


def test_information_terms_match_explicit_snapshots() -> None:
    model, W, theta = _fixture()
    a_t = model.steering_vector_tx(theta)
    a_r = model.steering_vector_rx(theta)
    da_t = model.steering_derivative_tx(theta)
    da_r = model.steering_derivative_rx(theta)
    signal, derivative, cross = point_target_information_terms(
        W, a_t, a_r, da_t, da_r, model.L
    )
    assert signal > 0.0
    assert derivative > 0.0
    assert abs(cross) ** 2 <= signal * derivative * (1.0 + 1e-12)


def test_crb_has_required_power_noise_snapshot_scaling() -> None:
    model, W, theta = _fixture()
    vectors = (
        model.steering_vector_tx(theta),
        model.steering_vector_rx(theta),
        model.steering_derivative_tx(theta),
        model.steering_derivative_rx(theta),
    )
    base = compute_crb(W, *vectors, model.sigma_s2, model.L)
    assert compute_crb(2.0 * W, *vectors, model.sigma_s2, model.L) == pytest.approx(
        base / 4.0, rel=1e-12
    )
    assert compute_crb(W, *vectors, 3.0 * model.sigma_s2, model.L) == pytest.approx(
        3.0 * base, rel=1e-12
    )
    assert compute_crb(W, *vectors, model.sigma_s2, 2 * model.L) == pytest.approx(
        base / 2.0, rel=1e-12
    )


@pytest.mark.parametrize(
    "beam_scale",
    (1.0e-12, 1.0e-9, 1.0e-8, 1.0e-4, 1.0, 1.0e4, 1.0e8, 1.0e12),
)
def test_crb_identifiability_is_invariant_over_beam_decades(
    beam_scale: float,
) -> None:
    """A common beam scale changes CRB by exactly its inverse square."""

    model, W, theta = _fixture()
    vectors = (
        model.steering_vector_tx(theta),
        model.steering_vector_rx(theta),
        model.steering_derivative_tx(theta),
        model.steering_derivative_rx(theta),
    )
    base = compute_crb(W, *vectors, model.sigma_s2, model.L)
    assert base == pytest.approx(3.802580952136013e-7, rel=3e-13)
    scaled = compute_crb(
        beam_scale * W, *vectors, model.sigma_s2, model.L
    )
    assert np.isfinite(scaled)
    assert scaled == pytest.approx(base / beam_scale**2, rel=3e-13)


def test_low_scale_crb_matches_scaled_explicit_snapshot_fim() -> None:
    """The former false-``inf`` counterexample remains identifiable."""

    model, W, theta = _fixture()
    scale = 1.0e-8
    scaled_W = scale * W
    a_t = model.steering_vector_tx(theta)
    a_r = model.steering_vector_rx(theta)
    da_t = model.steering_derivative_tx(theta)
    da_r = model.steering_derivative_rx(theta)
    rows = np.arange(model.K)[:, None]
    columns = np.arange(model.L)[None, :]
    symbols = np.exp(2j * np.pi * rows * columns / model.L)
    X = scaled_W @ symbols
    response = np.outer(a_r, a_t.conj())
    derivative = np.outer(da_r, a_t.conj()) + np.outer(
        a_r, da_t.conj()
    )
    g = (response @ X).reshape(-1, order="F")
    g_dot = (derivative @ X).reshape(-1, order="F")
    g_direction = g / np.linalg.norm(g)
    residual = g_dot - g_direction * np.vdot(g_direction, g_dot)
    explicit_crb = model.sigma_s2 / (
        2.0 * np.vdot(residual, residual).real
    )
    actual = compute_crb(
        scaled_W, a_t, a_r, da_t, da_r, model.sigma_s2, model.L
    )
    assert actual == pytest.approx(3.802580952136013e9, rel=3e-13)
    assert actual == pytest.approx(explicit_crb, rel=3e-13)


def test_crb_reports_binary64_output_range_failures_explicitly() -> None:
    """Representability failures must not masquerade as unidentifiability."""

    a_t = np.array([1.0, 0.0])
    a_r = np.array([1.0, 0.0])
    da_t = np.zeros(2)
    da_r = np.array([0.0, 1.0])
    smallest = np.nextafter(0.0, 1.0)
    with pytest.raises(OverflowError, match="CRB exceeds"):
        compute_crb(
            np.array([[smallest], [0.0]]),
            a_t,
            a_r,
            da_t,
            da_r,
            sigma_s2=1.0,
            L=1,
        )
    with pytest.raises(FloatingPointError, match="CRB underflows"):
        compute_crb(
            np.array([[1.0e308], [0.0]]),
            a_t,
            a_r,
            da_t,
            da_r,
            sigma_s2=1.0,
            L=1,
        )


def test_unidentifiable_zero_waveform_returns_infinity() -> None:
    model, W, theta = _fixture()
    vectors = (
        model.steering_vector_tx(theta),
        model.steering_vector_rx(theta),
        model.steering_derivative_tx(theta),
        model.steering_derivative_rx(theta),
    )
    assert np.isinf(
        compute_crb(np.zeros_like(W), *vectors, model.sigma_s2, model.L)
    )


def test_sensing_ee_matches_definition() -> None:
    model, W, theta = _fixture()
    vectors = (
        model.steering_vector_tx(theta),
        model.steering_vector_rx(theta),
        model.steering_derivative_tx(theta),
        model.steering_derivative_rx(theta),
    )
    crb = compute_crb(W, *vectors, model.sigma_s2, model.L)
    expected = 1.0 / (
        crb
        * model.L
        * (np.vdot(W, W).real / model.epsilon + model.P0)
    )
    assert compute_ee_s(
        W,
        *vectors,
        model.sigma_s2,
        model.L,
        model.epsilon,
        model.P0,
    ) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("keyword", "value"),
    (("sigma_s2", 0.0), ("alpha_abs", 0.0)),
)
def test_crb_rejects_nonphysical_scalars(keyword: str, value: float) -> None:
    model, W, theta = _fixture()
    arguments = {
        "W": W,
        "a_t": model.steering_vector_tx(theta),
        "a_r": model.steering_vector_rx(theta),
        "da_t": model.steering_derivative_tx(theta),
        "da_r": model.steering_derivative_rx(theta),
        "sigma_s2": model.sigma_s2,
        "L": model.L,
        "alpha_abs": 1.0,
    }
    arguments[keyword] = value
    with pytest.raises(ValueError, match=keyword):
        compute_crb(**arguments)
