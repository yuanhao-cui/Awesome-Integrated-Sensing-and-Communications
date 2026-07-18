"""Scale-safe communication and sensing projections for the RIS surrogate.

Every finite binary64 channel/beam coefficient is converted to an exact
integer times a power of two. Direct and RIS-reflected complex products are
then accumulated exactly before an independent-stream power, phase, SINR, or
SNR is rounded back to binary64. This makes the metric path permutation
invariant and prevents overflow, underflow, or premature loss of cancellation
residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_LOG_2 = math.log(2.0)
_LOG_10 = math.log(10.0)
_MAX_LOG = math.log(np.finfo(float).max)
_MIN_LOG = math.log(np.nextafter(0.0, 1.0))

_ExactReal = tuple[int, int]
_ExactComplex = tuple[_ExactReal, _ExactReal]
_BinaryMagnitude = tuple[float, int]
_ZERO_REAL: _ExactReal = (0, 0)
_ZERO_COMPLEX: _ExactComplex = (_ZERO_REAL, _ZERO_REAL)


def _normalize_real(integer: int, exponent: int) -> _ExactReal:
    if integer == 0:
        return _ZERO_REAL
    trailing_bit = abs(integer) & -abs(integer)
    trailing_zeros = trailing_bit.bit_length() - 1
    return integer >> trailing_zeros, exponent + trailing_zeros


def _real_from_float(value: float) -> _ExactReal:
    value = float(value)
    if value == 0.0:
        return _ZERO_REAL
    numerator, denominator = value.as_integer_ratio()
    return _normalize_real(numerator, -(denominator.bit_length() - 1))


def _real_add(first: _ExactReal, second: _ExactReal) -> _ExactReal:
    first_integer, first_exponent = first
    second_integer, second_exponent = second
    if first_integer == 0:
        return second
    if second_integer == 0:
        return first
    common_exponent = min(first_exponent, second_exponent)
    total = (
        first_integer << (first_exponent - common_exponent)
    ) + (second_integer << (second_exponent - common_exponent))
    return _normalize_real(total, common_exponent)


def _real_negate(value: _ExactReal) -> _ExactReal:
    return -value[0], value[1]


def _real_multiply(first: _ExactReal, second: _ExactReal) -> _ExactReal:
    if first[0] == 0 or second[0] == 0:
        return _ZERO_REAL
    return _normalize_real(first[0] * second[0], first[1] + second[1])


def _complex_from_value(value: complex) -> _ExactComplex:
    value = complex(value)
    return _real_from_float(value.real), _real_from_float(value.imag)


def _complex_add(
    first: _ExactComplex,
    second: _ExactComplex,
) -> _ExactComplex:
    return _real_add(first[0], second[0]), _real_add(first[1], second[1])


def _complex_conjugate(value: _ExactComplex) -> _ExactComplex:
    return value[0], _real_negate(value[1])


def _complex_multiply(
    first: _ExactComplex,
    second: _ExactComplex,
) -> _ExactComplex:
    real = _real_add(
        _real_multiply(first[0], second[0]),
        _real_negate(_real_multiply(first[1], second[1])),
    )
    imaginary = _real_add(
        _real_multiply(first[0], second[1]),
        _real_multiply(first[1], second[0]),
    )
    return real, imaginary


def _complex_product(*values: complex) -> _ExactComplex:
    result = _complex_from_value(1.0 + 0.0j)
    for value in values:
        result = _complex_multiply(result, _complex_from_value(value))
    return result


def _integer_binary_representation(
    positive_integer: int,
    exponent: int,
) -> _BinaryMagnitude:
    bit_count = positive_integer.bit_length()
    retained_bits = min(bit_count, 64)
    retained = positive_integer >> (bit_count - retained_bits)
    mantissa = math.ldexp(float(retained), -retained_bits)
    normalized_exponent = exponent + bit_count
    if mantissa == 1.0:
        mantissa = 0.5
        normalized_exponent += 1
    return mantissa, normalized_exponent


def _real_to_float(value: _ExactReal, name: str) -> float:
    """Round one exact binary real to binary64 or report its range failure."""

    integer, exponent = value
    if integer == 0:
        return 0.0
    representation = _integer_binary_representation(abs(integer), exponent)
    log_magnitude = _representation_log(representation)
    if log_magnitude > _MAX_LOG:
        raise OverflowError(f"the finite {name} exceeds the binary64 range")
    if log_magnitude < _MIN_LOG:
        raise FloatingPointError(
            f"the nonzero {name} underflows the binary64 range"
        )
    try:
        rounded = math.ldexp(representation[0], representation[1])
    except OverflowError as error:
        raise OverflowError(
            f"the finite {name} exceeds the binary64 range"
        ) from error
    if rounded == 0.0:
        raise FloatingPointError(
            f"the nonzero {name} underflows the binary64 range"
        )
    return math.copysign(rounded, integer)


def _complex_to_value(value: _ExactComplex, name: str) -> complex:
    return complex(
        _real_to_float(value[0], f"real part of {name}"),
        _real_to_float(value[1], f"imaginary part of {name}"),
    )


def _complex_power(value: _ExactComplex) -> _BinaryMagnitude | None:
    real, imaginary = value
    real_square = _real_multiply(real, real)
    imaginary_square = _real_multiply(imaginary, imaginary)
    total = _real_add(real_square, imaginary_square)
    if total[0] == 0:
        return None
    return _integer_binary_representation(total[0], total[1])


def _sum_binary_powers(
    powers: list[_BinaryMagnitude],
) -> _BinaryMagnitude | None:
    """Return an exact-scale representation of a sum of non-negative powers."""

    if not powers:
        return None
    reference = _largest(powers)
    scaled_sum = math.fsum(_relative(power, reference) for power in powers)
    mantissa, adjustment = math.frexp(reference[0] * scaled_sum)
    return mantissa, reference[1] + adjustment


def _representation_log(value: _BinaryMagnitude) -> float:
    return math.log(value[0]) + value[1] * _LOG_2


def _largest(values: list[_BinaryMagnitude]) -> _BinaryMagnitude:
    return max(values, key=lambda value: (value[1], value[0]))


def _relative(
    value: _BinaryMagnitude,
    reference: _BinaryMagnitude,
) -> float:
    return math.ldexp(
        value[0] / reference[0], value[1] - reference[1]
    )


def _positive_representation(value: float) -> _BinaryMagnitude:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("noise power must be finite and positive")
    return math.frexp(value)


def db_to_linear(value_db: float, name: str) -> float:
    """Convert a finite dB ratio to a positive representable linear ratio."""

    value_db = float(value_db)
    if not np.isfinite(value_db):
        raise ValueError(f"{name} must be finite")
    log_value = value_db * (_LOG_10 / 10.0)
    if log_value > _MAX_LOG:
        raise OverflowError(f"{name} converts above the binary64 range")
    if log_value < _MIN_LOG:
        raise FloatingPointError(
            f"{name} converts below the positive binary64 range"
        )
    try:
        result = math.exp(log_value)
    except OverflowError as error:
        raise OverflowError(
            f"{name} converts above the binary64 range"
        ) from error
    if not math.isfinite(result):
        raise OverflowError(f"{name} converts above the binary64 range")
    if result == 0.0:
        raise FloatingPointError(
            f"{name} converts below the positive binary64 range"
        )
    return result


def normalize_unit_phases(theta: np.ndarray) -> np.ndarray:
    """Normalize finite nonzero complex entries without scale overflow."""

    theta = np.asarray(theta, dtype=complex)
    if not np.all(np.isfinite(theta)):
        raise ValueError("RIS phases must be finite")
    component_scale = np.maximum(np.abs(theta.real), np.abs(theta.imag))
    if np.any(component_scale == 0.0):
        raise ValueError("RIS phases cannot contain zero-magnitude entries")
    real = theta.real / component_scale
    imaginary = theta.imag / component_scale
    magnitude = np.hypot(real, imaginary)
    return real / magnitude + 1j * (imaginary / magnitude)


@dataclass(frozen=True)
class _RatioGeometry:
    has_signal: bool
    signal_scaled: float
    denominator_scaled: float
    log_ratio: float


def _ratio_geometry(
    signal: _BinaryMagnitude | None,
    denominator_terms: list[_BinaryMagnitude],
) -> _RatioGeometry:
    denominator_reference = _largest(denominator_terms)
    denominator_own_scale = math.fsum(
        _relative(value, denominator_reference)
        for value in denominator_terms
    )
    denominator_log = _representation_log(
        denominator_reference
    ) + math.log(denominator_own_scale)
    if signal is None:
        return _RatioGeometry(False, 0.0, 1.0, -math.inf)
    overall_reference = _largest([signal, *denominator_terms])
    return _RatioGeometry(
        True,
        _relative(signal, overall_reference),
        math.fsum(
            _relative(value, overall_reference)
            for value in denominator_terms
        ),
        _representation_log(signal) - denominator_log,
    )


def _ratio_value(geometry: _RatioGeometry, name: str) -> float:
    if not geometry.has_signal:
        return 0.0
    if geometry.log_ratio > _MAX_LOG:
        raise OverflowError(f"the finite {name} exceeds the binary64 range")
    if geometry.log_ratio < _MIN_LOG:
        raise FloatingPointError(
            f"the positive {name} underflows the binary64 range"
        )
    if geometry.denominator_scaled == 0.0:
        raise OverflowError(f"the finite {name} exceeds the binary64 range")
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        value = float(
            np.divide(geometry.signal_scaled, geometry.denominator_scaled)
        )
    if np.isinf(value):
        raise OverflowError(f"the finite {name} exceeds the binary64 range")
    if value == 0.0:
        raise FloatingPointError(
            f"the positive {name} underflows the binary64 range"
        )
    return value


def _spectral_efficiency(geometry: _RatioGeometry) -> float:
    if not geometry.has_signal:
        return 0.0
    if geometry.log_ratio < _MIN_LOG:
        raise FloatingPointError(
            "the positive spectral efficiency underflows the binary64 range"
        )
    if geometry.denominator_scaled > 0.0:
        with np.errstate(over="ignore", under="ignore", invalid="raise"):
            ratio = float(
                np.divide(
                    geometry.signal_scaled,
                    geometry.denominator_scaled,
                )
            )
        if np.isfinite(ratio) and ratio > 0.0:
            return float(np.log1p(ratio) / _LOG_2)
    if geometry.log_ratio > 0.0:
        with np.errstate(under="ignore"):
            natural_rate = geometry.log_ratio + np.log1p(
                np.exp(-geometry.log_ratio)
            )
    else:
        with np.errstate(under="ignore"):
            natural_rate = np.log1p(np.exp(geometry.log_ratio))
    return float(natural_rate / _LOG_2)


def _validate_ris_inputs(
    direct_channel: np.ndarray,
    surface_channel: np.ndarray,
    H_BR: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    direct_channel = np.asarray(direct_channel, dtype=complex)
    surface_channel = np.asarray(surface_channel, dtype=complex)
    H_BR = np.asarray(H_BR, dtype=complex)
    theta = np.asarray(theta, dtype=complex)
    if direct_channel.ndim != 1 or surface_channel.ndim != 1:
        raise ValueError("direct and surface channels must be vectors")
    if H_BR.shape != (surface_channel.size, direct_channel.size):
        raise ValueError("H_BR dimensions must match the channel vectors")
    if theta.shape != surface_channel.shape:
        raise ValueError("theta must match the surface-channel dimension")
    arrays = (direct_channel, surface_channel, H_BR, theta)
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("all channel and phase inputs must be finite")
    return direct_channel, surface_channel, H_BR, theta


def _ris_projection(
    direct_channel: np.ndarray,
    surface_channel: np.ndarray,
    H_BR: np.ndarray,
    theta: np.ndarray,
    beam: np.ndarray,
) -> _ExactComplex:
    beam = np.asarray(beam, dtype=complex)
    if beam.shape != direct_channel.shape or not np.all(np.isfinite(beam)):
        raise ValueError("beam must be a finite vector matching the direct channel")
    total = _ZERO_COMPLEX
    for direct, beam_entry in zip(direct_channel, beam, strict=True):
        total = _complex_add(
            total,
            _complex_product(np.conj(direct), beam_entry),
        )
    for surface_index, surface in enumerate(surface_channel):
        for antenna_index, beam_entry in enumerate(beam):
            total = _complex_add(
                total,
                _complex_product(
                    np.conj(surface),
                    np.conj(theta[surface_index]),
                    np.conj(H_BR[surface_index, antenna_index]),
                    beam_entry,
                ),
            )
    return total


def stable_effective_channel(
    direct_channel: np.ndarray,
    surface_channel: np.ndarray,
    H_BR: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Return the effective channel with exact path summation per antenna."""

    direct_channel, surface_channel, H_BR, theta = _validate_ris_inputs(
        direct_channel, surface_channel, H_BR, theta
    )
    result = np.empty(direct_channel.shape, dtype=complex)
    for antenna in range(direct_channel.size):
        basis = np.zeros(direct_channel.shape, dtype=complex)
        basis[antenna] = 1.0
        # _ris_projection returns h^H e_m = conj(h_m).
        exact_component = _complex_conjugate(
            _ris_projection(
                direct_channel,
                surface_channel,
                H_BR,
                theta,
                basis,
            )
        )
        result[antenna] = _complex_to_value(
            exact_component, "effective-channel component"
        )
    return result


def _link_geometry(
    direct_channel: np.ndarray,
    surface_channel: np.ndarray,
    H_BR: np.ndarray,
    theta: np.ndarray,
    desired_beam: np.ndarray,
    interference_beams: np.ndarray,
    noise_power: float,
) -> _RatioGeometry:
    direct_channel, surface_channel, H_BR, theta = _validate_ris_inputs(
        direct_channel, surface_channel, H_BR, theta
    )
    desired_beam = np.asarray(desired_beam, dtype=complex)
    interference_beams = np.asarray(interference_beams, dtype=complex)
    if desired_beam.shape != direct_channel.shape:
        raise ValueError("desired_beam must match the antenna dimension")
    if (
        interference_beams.ndim != 2
        or interference_beams.shape[0] != direct_channel.size
    ):
        raise ValueError("interference_beams has an invalid antenna dimension")
    if not np.all(np.isfinite(desired_beam)) or not np.all(
        np.isfinite(interference_beams)
    ):
        raise ValueError("beamformers must be finite")
    signal = _complex_power(
        _ris_projection(
            direct_channel,
            surface_channel,
            H_BR,
            theta,
            desired_beam,
        )
    )
    denominator_terms = [_positive_representation(noise_power)]
    denominator_terms.extend(
        power
        for stream in range(interference_beams.shape[1])
        if (
            power := _complex_power(
                _ris_projection(
                    direct_channel,
                    surface_channel,
                    H_BR,
                    theta,
                    interference_beams[:, stream],
                )
            )
        )
        is not None
    )
    return _ratio_geometry(signal, denominator_terms)


def stable_link_sinr(
    direct_channel: np.ndarray,
    surface_channel: np.ndarray,
    H_BR: np.ndarray,
    theta: np.ndarray,
    desired_beam: np.ndarray,
    interference_beams: np.ndarray,
    noise_power: float,
) -> float:
    """Return a scale-safe communication SINR for one RIS-assisted user."""

    return _ratio_value(
        _link_geometry(
            direct_channel,
            surface_channel,
            H_BR,
            theta,
            desired_beam,
            interference_beams,
            noise_power,
        ),
        "SINR",
    )


def stable_link_rate(
    direct_channel: np.ndarray,
    surface_channel: np.ndarray,
    H_BR: np.ndarray,
    theta: np.ndarray,
    desired_beam: np.ndarray,
    interference_beams: np.ndarray,
    noise_power: float,
) -> float:
    """Return scale-safe ``log2(1 + SINR)`` for one user."""

    return _spectral_efficiency(
        _link_geometry(
            direct_channel,
            surface_channel,
            H_BR,
            theta,
            desired_beam,
            interference_beams,
            noise_power,
        )
    )


def stable_sensing_snr(
    a_bs: np.ndarray,
    a_ris: np.ndarray,
    H_BR: np.ndarray,
    theta: np.ndarray,
    beamformers: np.ndarray,
    noise_power: float,
) -> float:
    """Return independent-stream sensing SNR without coherent stream collapse.

    For columns ``w_k`` this evaluates

    ``sum_k |h_s^H w_k|^2 / noise_power``.

    A one-dimensional input is treated as one stream only.  Distinct data
    streams are never added before taking magnitudes.
    """

    a_bs, a_ris, H_BR, theta = _validate_ris_inputs(
        a_bs, a_ris, H_BR, theta
    )
    beamformers = np.asarray(beamformers, dtype=complex)
    if beamformers.ndim == 1:
        beamformers = beamformers[:, None]
    if (
        beamformers.ndim != 2
        or beamformers.shape[0] != a_bs.size
        or not np.all(np.isfinite(beamformers))
    ):
        raise ValueError(
            "beamformers must be a finite matrix with one row per antenna"
        )
    stream_powers = [
        power
        for stream in range(beamformers.shape[1])
        if (
            power := _complex_power(
                _ris_projection(
                    a_bs,
                    a_ris,
                    H_BR,
                    theta,
                    beamformers[:, stream],
                )
            )
        )
        is not None
    ]
    signal = _sum_binary_powers(stream_powers)
    return _ratio_value(
        _ratio_geometry(signal, [_positive_representation(noise_power)]),
        "sensing SNR",
    )


def stable_squared_norm(value: np.ndarray) -> float:
    """Return an exact-sum squared norm or an explicit binary64 range error."""

    value = np.asarray(value, dtype=complex)
    if not np.all(np.isfinite(value)):
        raise ValueError("value must contain only finite entries")
    powers = [
        power
        for entry in value.reshape(-1)
        if (power := _complex_power(_complex_from_value(entry))) is not None
    ]
    if not powers:
        return 0.0
    reference = _largest(powers)
    scaled_sum = math.fsum(_relative(power, reference) for power in powers)
    log_norm = _representation_log(reference) + math.log(scaled_sum)
    if log_norm > _MAX_LOG:
        raise OverflowError("the finite squared norm exceeds the binary64 range")
    if log_norm < _MIN_LOG:
        raise FloatingPointError(
            "the positive squared norm underflows the binary64 range"
        )
    mantissa, adjustment = math.frexp(reference[0] * scaled_sum)
    try:
        result = math.ldexp(mantissa, reference[1] + adjustment)
    except OverflowError as error:
        raise OverflowError(
            "the finite squared norm exceeds the binary64 range"
        ) from error
    if result == 0.0:
        raise FloatingPointError(
            "the positive squared norm underflows the binary64 range"
        )
    return result


def _sensing_terms(
    a_bs: np.ndarray,
    a_ris: np.ndarray,
    H_BR: np.ndarray,
    beam: np.ndarray,
) -> tuple[_ExactComplex, list[_ExactComplex]]:
    beam = np.asarray(beam, dtype=complex)
    if beam.shape != a_bs.shape or not np.all(np.isfinite(beam)):
        raise ValueError("beam must be a finite vector matching a_bs")
    direct = _ZERO_COMPLEX
    for channel, beam_entry in zip(a_bs, beam, strict=True):
        direct = _complex_add(
            direct,
            _complex_product(np.conj(channel), beam_entry),
        )
    coefficients: list[_ExactComplex] = []
    for surface_index, surface in enumerate(a_ris):
        coefficient = _ZERO_COMPLEX
        for antenna_index, beam_entry in enumerate(beam):
            coefficient = _complex_add(
                coefficient,
                _complex_product(
                    np.conj(surface),
                    np.conj(H_BR[surface_index, antenna_index]),
                    beam_entry,
                ),
            )
        coefficients.append(coefficient)
    return direct, coefficients


def sensing_coordinate_phase_candidate(
    a_bs: np.ndarray,
    a_ris: np.ndarray,
    H_BR: np.ndarray,
    beamformers: np.ndarray,
    current_theta: np.ndarray,
    element: int,
) -> complex:
    """Return the exact one-coordinate maximizer for streamwise sensing power.

    With all phases except ``theta_l`` fixed, stream ``k`` has projection

    ``u_k + conj(theta_l) c_lk``.

    The phase-dependent part of the summed power is
    ``2 Re{conj(theta_l) C_l}``, where
    ``C_l = sum_k conj(u_k) c_lk``.  Hence ``theta_l = exp(j angle(C_l))`` is a
    global maximizer of this one-coordinate subproblem.  All path products and
    the cross-stream sum forming ``C_l`` are accumulated in the exact binary
    representation used by the metric oracle.
    """

    a_bs = np.asarray(a_bs, dtype=complex)
    a_ris = np.asarray(a_ris, dtype=complex)
    H_BR = np.asarray(H_BR, dtype=complex)
    current_theta = np.asarray(current_theta, dtype=complex)
    _validate_ris_inputs(a_bs, a_ris, H_BR, current_theta)
    beamformers = np.asarray(beamformers, dtype=complex)
    if (
        beamformers.ndim != 2
        or beamformers.shape[0] != a_bs.size
        or not np.all(np.isfinite(beamformers))
    ):
        raise ValueError(
            "beamformers must be a finite matrix with one row per antenna"
        )
    if not isinstance(element, (int, np.integer)) or not (
        0 <= int(element) < a_ris.size
    ):
        raise IndexError("element is outside the RIS phase-vector range")
    element = int(element)

    cross_coefficient = _ZERO_COMPLEX
    for stream in range(beamformers.shape[1]):
        direct, coefficients = _sensing_terms(
            a_bs, a_ris, H_BR, beamformers[:, stream]
        )
        residual = direct
        for other, coefficient in enumerate(coefficients):
            if other == element:
                continue
            residual = _complex_add(
                residual,
                _complex_multiply(
                    _complex_conjugate(
                        _complex_from_value(current_theta[other])
                    ),
                    coefficient,
                ),
            )
        cross_coefficient = _complex_add(
            cross_coefficient,
            _complex_multiply(
                _complex_conjugate(residual),
                coefficients[element],
            ),
        )
    phase = _exact_phase(cross_coefficient)
    if phase is None:
        return complex(current_theta[element])
    return complex(np.exp(1j * phase))


def _exact_phase(value: _ExactComplex) -> float | None:
    real, imaginary = value
    if real[0] == 0 and imaginary[0] == 0:
        return None
    components: list[_BinaryMagnitude] = []
    if real[0] != 0:
        components.append(
            _integer_binary_representation(abs(real[0]), real[1])
        )
    if imaginary[0] != 0:
        components.append(
            _integer_binary_representation(abs(imaginary[0]), imaginary[1])
        )
    reference = _largest(components)

    def scaled_component(component: _ExactReal) -> float:
        if component[0] == 0:
            return 0.0
        magnitude = _integer_binary_representation(
            abs(component[0]), component[1]
        )
        return math.copysign(_relative(magnitude, reference), component[0])

    return math.atan2(
        scaled_component(imaginary), scaled_component(real)
    )


def optimal_single_stream_sensing_phases(
    a_bs: np.ndarray,
    a_ris: np.ndarray,
    H_BR: np.ndarray,
    beam: np.ndarray,
    current_theta: np.ndarray,
) -> np.ndarray:
    """Return the exact triangle-alignment solution for one fixed beam.

    This helper is a single-stream analytic oracle.  It is not an optimizer for
    the independent multi-stream power ``sum_k |h_s^H w_k|^2``.
    """

    a_bs = np.asarray(a_bs, dtype=complex)
    a_ris = np.asarray(a_ris, dtype=complex)
    H_BR = np.asarray(H_BR, dtype=complex)
    current_theta = np.asarray(current_theta, dtype=complex)
    _validate_ris_inputs(a_bs, a_ris, H_BR, current_theta)
    direct, coefficients = _sensing_terms(a_bs, a_ris, H_BR, beam)
    reference_phase = _exact_phase(direct)
    if reference_phase is None:
        reference_phase = 0.0
    theta = current_theta.copy()
    for index, coefficient in enumerate(coefficients):
        coefficient_phase = _exact_phase(coefficient)
        if coefficient_phase is not None:
            theta[index] = np.exp(
                1j * (coefficient_phase - reference_phase)
            )
    return theta


def stable_triangle_sensing_snr(
    a_bs: np.ndarray,
    a_ris: np.ndarray,
    H_BR: np.ndarray,
    beam: np.ndarray,
    noise_power: float,
) -> float:
    """Return the single-stream triangle-inequality upper-bound SNR safely."""

    a_bs = np.asarray(a_bs, dtype=complex)
    a_ris = np.asarray(a_ris, dtype=complex)
    H_BR = np.asarray(H_BR, dtype=complex)
    dummy_theta = np.ones(a_ris.shape, dtype=complex)
    _validate_ris_inputs(a_bs, a_ris, H_BR, dummy_theta)
    noise_representation = _positive_representation(noise_power)
    direct, coefficients = _sensing_terms(a_bs, a_ris, H_BR, beam)
    powers = [
        power
        for value in (direct, *coefficients)
        if (power := _complex_power(value)) is not None
    ]
    if not powers:
        return 0.0
    power_reference = _largest(powers)

    def relative_magnitude(power: _BinaryMagnitude) -> float:
        mantissa_ratio = power[0] / power_reference[0]
        exponent_delta = power[1] - power_reference[1]
        if exponent_delta % 2:
            mantissa_ratio *= 2.0
            exponent_delta -= 1
        return math.ldexp(math.sqrt(mantissa_ratio), exponent_delta // 2)

    scaled_magnitude_sum = math.fsum(
        relative_magnitude(power) for power in powers
    )
    numerator_mantissa, adjustment = math.frexp(
        power_reference[0] * scaled_magnitude_sum**2
    )
    numerator = (
        numerator_mantissa,
        power_reference[1] + adjustment,
    )
    return _ratio_value(
        _ratio_geometry(numerator, [noise_representation]),
        "triangle-bound sensing SNR",
    )


__all__ = [
    "db_to_linear",
    "normalize_unit_phases",
    "optimal_single_stream_sensing_phases",
    "sensing_coordinate_phase_candidate",
    "stable_effective_channel",
    "stable_link_rate",
    "stable_link_sinr",
    "stable_sensing_snr",
    "stable_squared_norm",
    "stable_triangle_sensing_snr",
]
