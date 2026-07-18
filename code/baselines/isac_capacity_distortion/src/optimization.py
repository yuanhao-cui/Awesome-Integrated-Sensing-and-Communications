"""Deterministic optimizers for the local capacity-distortion surrogate."""

from __future__ import annotations

import math

import numpy as np

from .system_model import _noise_normalized_channel, _positive_integer


Array = np.ndarray


def _positive_product_ratio(
    numerators: tuple[float, ...],
    denominators: tuple[float, ...],
) -> float:
    """Evaluate a positive product ratio without overflowing intermediates."""

    if any(value == 0 for value in numerators):
        return 0.0
    mantissa = 1.0
    exponent = 0
    for value in numerators:
        fraction, power = math.frexp(value)
        mantissa *= fraction
        exponent += power
    for value in denominators:
        fraction, power = math.frexp(value)
        mantissa /= fraction
        exponent -= power
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError:
        return float("inf")


def _power_budget(
    power_per_tx: float,
    transmit_antennas: int,
) -> tuple[float, float]:
    """Validate per-antenna power and a representable total trace budget."""

    power = float(power_per_tx)
    if not np.isfinite(power) or power < 0:
        raise ValueError("power_per_tx must be non-negative and finite")
    if power > np.finfo(float).max / transmit_antennas:
        raise ValueError("the total power budget exceeds the floating-point range")
    return power, power * transmit_antennas


def _channel_modes(Hc: Array, sigma_c2: float) -> tuple[Array, Array]:
    """Return noise-normalized squared singular values and right modes."""

    normalized = _noise_normalized_channel(Hc, sigma_c2)
    _, singular_values, right_vectors_h = np.linalg.svd(
        normalized,
        full_matrices=True,
    )
    if singular_values.size:
        square_limit = float(np.sqrt(np.finfo(float).max))
        if float(np.max(singular_values)) > square_limit:
            raise ValueError(
                "the noise-normalized channel energy exceeds the "
                "floating-point range"
            )
    gains = np.zeros(normalized.shape[1], dtype=float)
    with np.errstate(under="ignore"):
        gains[: singular_values.size] = singular_values**2
    return gains, right_vectors_h.conj().T


def isotropic_covariance(power_per_tx: float, transmit_antennas: int) -> Array:
    """Return ``power_per_tx * I`` under the local power convention."""

    antennas = _positive_integer(transmit_antennas, "transmit_antennas")
    power, _ = _power_budget(power_per_tx, antennas)
    return power * np.eye(antennas, dtype=np.complex128)


def water_filling_covariance(
    power_per_tx: float,
    Hc: Array,
    sigma_c2: float = 1.0,
) -> Array:
    """Maximize Gaussian MIMO rate by exact eigenmode water filling.

    The feasible set is ``Rx >= 0`` with
    ``trace(Rx) = power_per_tx * Hc.shape[1]``.  If ``Hc`` is identically
    zero, every feasible covariance is optimal and the isotropic covariance is
    returned deterministically.
    """

    channel = np.asarray(Hc, dtype=np.complex128)
    if channel.ndim != 2 or 0 in channel.shape:
        raise ValueError("Hc must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(channel)):
        raise ValueError("Hc must contain only finite values")
    transmit_antennas = channel.shape[1]
    per_tx, budget = _power_budget(power_per_tx, transmit_antennas)
    if budget == 0:
        return np.zeros((transmit_antennas, transmit_antennas), dtype=np.complex128)

    gains, eigenvectors = _channel_modes(channel, sigma_c2)
    positive = np.flatnonzero(gains > 0)
    if positive.size == 0:
        return isotropic_covariance(per_tx, transmit_antennas)

    ordered = positive[np.argsort(gains[positive])[::-1]]
    active_count = 1
    for candidate in range(2, ordered.size + 1):
        weakest_gain = gains[ordered[candidate - 1]]
        stronger_gains = gains[ordered[: candidate - 1]]
        activation_cost = float(
            np.sum(1.0 - weakest_gain / stronger_gains)
        )
        if activation_cost == 0:
            active_count = candidate
            continue
        if budget <= 1 or weakest_gain <= np.finfo(float).max / budget:
            with np.errstate(under="ignore"):
                available = budget * weakest_gain
        else:
            available = float("inf")
        if available > activation_cost:
            active_count = candidate
        else:
            break

    powers = np.zeros(transmit_antennas, dtype=float)
    active_modes = ordered[:active_count]
    active_gains = gains[active_modes]
    if active_count == 1:
        powers[active_modes[0]] = budget
    elif np.all(active_gains == active_gains[0]):
        powers[active_modes] = budget / active_count
    else:
        gain_scale = active_gains[0]
        relative_inverse = gain_scale / active_gains
        correction = (
            float(np.mean(relative_inverse)) - relative_inverse
        ) / gain_scale
        powers[active_modes] = budget / active_count + correction
        roundoff = (
            128
            * np.finfo(float).eps
            * max(budget, np.finfo(float).tiny)
        )
        if float(np.min(powers[active_modes])) < -roundoff:
            raise RuntimeError("water-filling active-set powers became negative")
        powers[active_modes] = np.maximum(powers[active_modes], 0.0)

    residual = budget - float(np.sum(powers))
    powers[active_modes[0]] += residual
    if powers[active_modes[0]] < 0:
        raise RuntimeError("water-filling post-correction produced negative power")

    covariance = eigenvectors @ np.diag(powers) @ eigenvectors.conj().T
    return (covariance + covariance.conj().T) / 2


def covariance_shaping_surrogate(
    alpha: float,
    power_per_tx: float,
    Hc: Array,
    sigma_c2: float = 1.0,
) -> Array:
    r"""Solve the repository's explicit log-determinant surrogate.

    For ``0 <= alpha <= 1``, this function minimizes

    .. math::

       -(1-\alpha)\log\det R_X
       -\alpha\log\det(I + H_cR_XH_c^H/\sigma_c^2)

    over PSD covariances with
    ``trace(Rx) = power_per_tx * number_of_transmit_antennas``.

    This is *not* the paper's general CRB objective.  It is an internal,
    strictly defined educational surrogate.  For ``0 < alpha < 1`` its KKT
    equations are solved in the eigenbasis of ``Hc^H Hc`` by monotone
    bisection.  The endpoints are the isotropic and water-filling solutions.
    """

    weight = float(alpha)
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("alpha must be in the closed interval [0, 1]")
    channel = np.asarray(Hc, dtype=np.complex128)
    if channel.ndim != 2 or 0 in channel.shape:
        raise ValueError("Hc must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(channel)):
        raise ValueError("Hc must contain only finite values")
    transmit_antennas = channel.shape[1]
    per_tx, budget = _power_budget(power_per_tx, transmit_antennas)

    if budget == 0 or weight == 0:
        return isotropic_covariance(per_tx, transmit_antennas)
    if weight == 1:
        return water_filling_covariance(per_tx, channel, sigma_c2)

    gains, eigenvectors = _channel_modes(channel, sigma_c2)
    sensing_weight = 1 - weight

    def powers_at(dual_value: float) -> Array:
        powers = np.empty(transmit_antennas, dtype=float)
        null_modes = gains == 0
        powers[null_modes] = sensing_weight / dual_value

        active = ~null_modes
        active_gains = gains[active]
        active_powers = np.empty_like(active_gains)

        # Work with t = budget * gain only through exponent-scaled ratios.
        # Both t and the usual quadratic discriminant may overflow even when
        # the KKT power fraction is finite.  Each branch below uses a ratio in
        # [0, 1] and the second branch is rationalized to avoid cancellation.
        for index, gain in enumerate(active_gains):
            gain_to_dual = _positive_product_ratio(
                (budget, float(gain)),
                (dual_value,),
            )
            if gain_to_dual > 1:
                dual_to_gain = _positive_product_ratio(
                    (dual_value,),
                    (budget, float(gain)),
                )
                root = np.sqrt(
                    (1 - dual_to_gain) ** 2
                    + 4 * dual_to_gain * sensing_weight
                )
                active_powers[index] = (
                    1 - dual_to_gain + root
                ) / (2 * dual_value)
            else:
                root = np.sqrt(
                    (1 - gain_to_dual) ** 2
                    + 4 * gain_to_dual * sensing_weight
                )
                active_powers[index] = (
                    2 * sensing_weight / dual_value
                ) / (1 - gain_to_dual + root)
        powers[active] = active_powers
        return powers

    dual_low = np.finfo(float).tiny
    dual_high = 1.0
    while float(np.sum(powers_at(dual_high))) > 1:
        dual_high *= 2
        if not np.isfinite(dual_high):
            raise RuntimeError("failed to bracket the covariance-shaping dual")

    for _ in range(200):
        dual_mid = (dual_low + dual_high) / 2
        if float(np.sum(powers_at(dual_mid))) > 1:
            dual_low = dual_mid
        else:
            dual_high = dual_mid

    fractions = powers_at(dual_high)
    residual = 1 - float(np.sum(fractions))
    fractions[int(np.argmax(fractions))] += residual
    if np.any(fractions <= 0):
        raise RuntimeError("interior surrogate solver returned non-positive power")
    with np.errstate(under="ignore"):
        powers = budget * fractions
    if np.any(powers <= 0):
        raise ValueError(
            "the positive interior covariance is below the floating-point range"
        )

    covariance = eigenvectors @ np.diag(powers) @ eigenvectors.conj().T
    return (covariance + covariance.conj().T) / 2


def sample_row_semiunitary(
    rows: int,
    columns: int,
    rng: np.random.Generator,
) -> Array:
    """Sample a complex row-semiunitary matrix using Gaussian QR."""

    row_count = _positive_integer(rows, "rows")
    column_count = _positive_integer(columns, "columns")
    if row_count > column_count:
        raise ValueError("rows cannot exceed columns")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    gaussian = (
        rng.standard_normal((column_count, row_count))
        + 1j * rng.standard_normal((column_count, row_count))
    ) / np.sqrt(2)
    q_columns, triangular = np.linalg.qr(gaussian, mode="reduced")

    diagonal = np.diag(triangular)
    phases = np.ones_like(diagonal)
    nonzero = np.abs(diagonal) > 0
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    q_columns = q_columns * phases
    return q_columns.conj().T


def sample_gaussian_waveform(
    power_per_tx: float,
    transmit_antennas: int,
    interval: int,
    rng: np.random.Generator,
) -> Array:
    """Sample columns independently from ``CN(0, power_per_tx I)``."""

    covariance = isotropic_covariance(power_per_tx, transmit_antennas)
    symbols = _positive_integer(interval, "interval")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    standard = (
        rng.standard_normal((covariance.shape[0], symbols))
        + 1j * rng.standard_normal((covariance.shape[0], symbols))
    ) / np.sqrt(2)
    return np.sqrt(float(power_per_tx)) * standard


def make_semiunitary_waveform(
    power_per_tx: float,
    transmit_antennas: int,
    interval: int,
    rng: np.random.Generator,
    basis: Array | None = None,
) -> tuple[Array, Array]:
    """Construct a waveform with an exactly prescribed isotropic subspace.

    If ``basis`` has ``r`` orthonormal columns, the returned sample covariance
    is ``(power_per_tx * M / r) basis basis^H`` and its trace is
    ``power_per_tx * M``.
    """

    antennas = _positive_integer(transmit_antennas, "transmit_antennas")
    symbols = _positive_integer(interval, "interval")
    power = float(power_per_tx)
    if not np.isfinite(power) or power < 0:
        raise ValueError("power_per_tx must be non-negative and finite")

    if basis is None:
        rank = min(antennas, symbols)
        subspace = np.eye(antennas, rank, dtype=np.complex128)
    else:
        subspace = np.asarray(basis, dtype=np.complex128)
        if subspace.ndim != 2 or subspace.shape[0] != antennas:
            raise ValueError("basis must have one row per transmit antenna")
        rank = subspace.shape[1]
        if rank < 1 or rank > min(antennas, symbols):
            raise ValueError("basis rank must lie between one and min(M, T)")
        if not np.all(np.isfinite(subspace)):
            raise ValueError("basis must contain only finite values")
        if not np.allclose(
            subspace.conj().T @ subspace,
            np.eye(rank),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("basis columns must be orthonormal")

    q_rows = sample_row_semiunitary(rank, symbols, rng)
    scale = np.sqrt(symbols * power * antennas / rank)
    return scale * subspace @ q_rows, q_rows
