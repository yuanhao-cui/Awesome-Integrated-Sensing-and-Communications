"""Scale-safe binary64 primitives for communication metrics.

The helpers in this module accumulate every complex channel projection as an
exact integer times a power of two. Only after all positive/negative terms
have cancelled is the projection converted to a normalized binary power.
Physical powers then remain mantissa/exponent pairs until a dimensionless
ratio or representable output is required. This avoids squaring a large
projection, subtracting a strong desired stream from total received power, and
discarding a small residual before large terms cancel.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
import math

import numpy as np


_LOG_2 = math.log(2.0)
_MAX_LOG = math.log(np.finfo(float).max)
_MIN_LOG = math.log(np.nextafter(0.0, 1.0))

_BinaryMagnitude = tuple[float, int]


def _component_scale(value: np.ndarray) -> float:
    """Return a finite componentwise complex scale without ``abs`` overflow."""

    return float(
        max(
            np.max(np.abs(value.real), initial=0.0),
            np.max(np.abs(value.imag), initial=0.0),
        )
    )


def _normalize_complex(value: np.ndarray, scale: float) -> np.ndarray:
    """Normalize components without subnormal complex-division overflow."""

    with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
        return value.real / scale + 1j * (value.imag / scale)


def _positive_product_representation(
    *values: float,
) -> _BinaryMagnitude | None:
    """Represent a nonnegative product as ``mantissa * 2**exponent``."""

    mantissa = 1.0
    exponent = 0
    for raw_value in values:
        value = float(raw_value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("binary product factors must be finite and nonnegative")
        if value == 0.0:
            return None
        factor_mantissa, factor_exponent = math.frexp(value)
        mantissa *= factor_mantissa
        mantissa, adjustment = math.frexp(mantissa)
        exponent += factor_exponent + adjustment
    return mantissa, exponent


def _float_product_term(first: float, second: float) -> tuple[int, int] | None:
    """Return the exact binary product ``integer * 2**exponent``."""

    first = float(first)
    second = float(second)
    if first == 0.0 or second == 0.0:
        return None
    first_numerator, first_denominator = first.as_integer_ratio()
    second_numerator, second_denominator = second.as_integer_ratio()
    first_denominator_exponent = first_denominator.bit_length() - 1
    second_denominator_exponent = second_denominator.bit_length() - 1
    return (
        first_numerator * second_numerator,
        -(first_denominator_exponent + second_denominator_exponent),
    )


def _exact_binary_sum(terms: list[tuple[int, int]]) -> tuple[int, int]:
    """Sum signed binary terms exactly and independently of their order."""

    if not terms:
        return 0, 0
    common_exponent = min(exponent for _, exponent in terms)
    total = sum(
        integer << (exponent - common_exponent)
        for integer, exponent in terms
    )
    if total == 0:
        return 0, 0
    trailing_bit = abs(total) & -abs(total)
    trailing_zeros = trailing_bit.bit_length() - 1
    return total >> trailing_zeros, common_exponent + trailing_zeros


def _integer_binary_representation(
    positive_integer: int,
    exponent: int,
) -> _BinaryMagnitude:
    """Normalize an exact positive binary integer without float overflow."""

    if positive_integer <= 0:
        raise ValueError("positive_integer must be positive")
    bit_count = positive_integer.bit_length()
    retained_bits = min(bit_count, 64)
    retained = positive_integer >> (bit_count - retained_bits)
    mantissa = math.ldexp(float(retained), -retained_bits)
    normalized_exponent = exponent + bit_count
    if mantissa == 1.0:
        mantissa = 0.5
        normalized_exponent += 1
    return mantissa, normalized_exponent


def _exact_projection_power(
    h_k: np.ndarray,
    column: np.ndarray,
) -> _BinaryMagnitude | None:
    """Return ``|h_k^H column|^2`` after exact binary complex summation."""

    real_terms: list[tuple[int, int]] = []
    imaginary_terms: list[tuple[int, int]] = []
    for channel, beam in zip(h_k, column, strict=True):
        products = (
            (real_terms, channel.real, beam.real, 1),
            (real_terms, channel.imag, beam.imag, 1),
            (imaginary_terms, channel.real, beam.imag, 1),
            (imaginary_terms, channel.imag, beam.real, -1),
        )
        for destination, first, second, sign in products:
            term = _float_product_term(first, second)
            if term is not None:
                integer, exponent = term
                destination.append((sign * integer, exponent))

    real_integer, real_exponent = _exact_binary_sum(real_terms)
    imaginary_integer, imaginary_exponent = _exact_binary_sum(
        imaginary_terms
    )
    power_terms: list[tuple[int, int]] = []
    if real_integer != 0:
        power_terms.append((real_integer * real_integer, 2 * real_exponent))
    if imaginary_integer != 0:
        power_terms.append(
            (
                imaginary_integer * imaginary_integer,
                2 * imaginary_exponent,
            )
        )
    if not power_terms:
        return None
    power_integer, power_exponent = _exact_binary_sum(power_terms)
    return _integer_binary_representation(power_integer, power_exponent)


def _representation_log(value: _BinaryMagnitude) -> float:
    mantissa, exponent = value
    return math.log(mantissa) + exponent * _LOG_2


def _larger_representation(
    values: list[_BinaryMagnitude],
) -> _BinaryMagnitude:
    return max(values, key=lambda value: (value[1], value[0]))


def _relative_to(
    value: _BinaryMagnitude,
    reference: _BinaryMagnitude,
) -> float:
    """Return an overflow-free power ratio no greater than one."""

    mantissa, exponent = value
    reference_mantissa, reference_exponent = reference
    return math.ldexp(
        mantissa / reference_mantissa,
        exponent - reference_exponent,
    )


@dataclass(frozen=True)
class _SINRGeometry:
    has_signal: bool
    signal_scaled: float
    denominator_scaled: float
    log_sinr: float


def _sinr_geometry(
    k: int,
    h_k: np.ndarray,
    W: np.ndarray,
    noise_power: float,
) -> _SINRGeometry:
    """Build a cancellation-free, scale-normalized SINR representation."""

    noise_power = float(noise_power)
    if not np.isfinite(noise_power) or noise_power <= 0.0:
        raise ValueError("noise_power must be finite and positive")
    h_k = np.asarray(h_k, dtype=complex)
    W = np.asarray(W, dtype=complex)
    if W.ndim != 2 or h_k.shape != (W.shape[0],):
        raise ValueError("h_k must match the antenna dimension of W")
    if not 0 <= k < W.shape[1]:
        raise IndexError("k is outside the user dimension of W")
    if not np.all(np.isfinite(h_k)) or not np.all(np.isfinite(W)):
        raise ValueError("h_k and W must be finite")

    noise_representation = _positive_product_representation(noise_power)
    if noise_representation is None:  # pragma: no cover - guarded above
        raise AssertionError("positive noise must have a representation")

    projection_powers = [
        _exact_projection_power(h_k, W[:, stream])
        for stream in range(W.shape[1])
    ]

    signal = projection_powers[k]
    denominator_terms = [noise_representation]
    denominator_terms.extend(
        power
        for stream, power in enumerate(projection_powers)
        if stream != k and power is not None
    )
    denominator_reference = _larger_representation(denominator_terms)
    denominator_own_scale = math.fsum(
        _relative_to(power, denominator_reference)
        for power in denominator_terms
    )
    denominator_log = (
        _representation_log(denominator_reference)
        + math.log(denominator_own_scale)
    )
    if signal is None:
        return _SINRGeometry(False, 0.0, 1.0, -math.inf)

    overall_reference = _larger_representation(
        [signal, *denominator_terms]
    )
    signal_scaled = _relative_to(signal, overall_reference)
    denominator_scaled = math.fsum(
        _relative_to(power, overall_reference)
        for power in denominator_terms
    )
    return _SINRGeometry(
        True,
        signal_scaled,
        denominator_scaled,
        _representation_log(signal) - denominator_log,
    )


def stable_sinr(
    k: int,
    h_k: np.ndarray,
    W: np.ndarray,
    noise_power: float,
) -> float:
    """Return SINR without desired/interference cancellation or square overflow.

    Exact zero desired projection returns zero. A positive mathematical SINR
    outside binary64 is reported explicitly rather than rounded to zero or
    infinity.
    """

    geometry = _sinr_geometry(k, h_k, W, noise_power)
    if not geometry.has_signal:
        return 0.0
    if geometry.log_sinr > _MAX_LOG:
        raise OverflowError("the finite SINR exceeds the binary64 range")
    if geometry.log_sinr < _MIN_LOG:
        raise FloatingPointError(
            "the positive SINR underflows the binary64 range"
        )
    if geometry.denominator_scaled == 0.0:
        raise OverflowError("the finite SINR exceeds the binary64 range")
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        value = float(
            np.divide(geometry.signal_scaled, geometry.denominator_scaled)
        )
    if np.isinf(value):
        raise OverflowError("the finite SINR exceeds the binary64 range")
    if value == 0.0:
        raise FloatingPointError(
            "the positive SINR underflows the binary64 range"
        )
    return value


def stable_spectral_efficiency(
    k: int,
    h_k: np.ndarray,
    W: np.ndarray,
    noise_power: float,
) -> float:
    """Return ``log2(1 + SINR)`` over the declared binary64 rate domain."""

    geometry = _sinr_geometry(k, h_k, W, noise_power)
    if not geometry.has_signal:
        return 0.0
    if geometry.log_sinr < _MIN_LOG:
        raise FloatingPointError(
            "the positive spectral efficiency underflows the binary64 range"
        )

    if geometry.denominator_scaled > 0.0:
        with np.errstate(over="ignore", under="ignore", invalid="raise"):
            sinr = float(
                np.divide(
                    geometry.signal_scaled,
                    geometry.denominator_scaled,
                )
            )
        if np.isfinite(sinr) and sinr > 0.0:
            return float(np.log1p(sinr) / _LOG_2)

    if geometry.log_sinr > 0.0:
        with np.errstate(under="ignore"):
            natural_rate = geometry.log_sinr + np.log1p(
                np.exp(-geometry.log_sinr)
            )
    else:
        with np.errstate(under="ignore"):
            natural_rate = np.log1p(np.exp(geometry.log_sinr))
    rate = float(natural_rate / _LOG_2)
    if not np.isfinite(rate) or rate <= 0.0:
        raise FloatingPointError("spectral efficiency is outside binary64")
    return rate


def stable_squared_norm(value: np.ndarray) -> float:
    """Return ``sum(abs(value)**2)`` or an explicit range error."""

    value = np.asarray(value, dtype=complex)
    if not np.all(np.isfinite(value)):
        raise ValueError("value must contain only finite entries")
    scale = _component_scale(value)
    if scale == 0.0:
        return 0.0
    normalized = _normalize_complex(value, scale)
    scaled_sum = math.fsum(
        component * component
        for component in chain(
            normalized.real.reshape(-1),
            normalized.imag.reshape(-1),
        )
    )
    representation = _positive_product_representation(
        scale, scale, scaled_sum
    )
    if representation is None:  # pragma: no cover - scale and sum are positive
        raise AssertionError("positive norm must have a representation")
    log_value = _representation_log(representation)
    if log_value > _MAX_LOG:
        raise OverflowError("the finite squared norm exceeds binary64")
    if log_value < _MIN_LOG:
        raise FloatingPointError("the positive squared norm underflows binary64")
    mantissa, exponent = representation
    result = math.ldexp(mantissa, exponent)
    if not np.isfinite(result):  # pragma: no cover - guarded in log domain
        raise OverflowError("the finite squared norm exceeds binary64")
    if result == 0.0:  # pragma: no cover - guarded in log domain
        raise FloatingPointError("the positive squared norm underflows binary64")
    return result


__all__ = [
    "stable_sinr",
    "stable_spectral_efficiency",
    "stable_squared_norm",
]
