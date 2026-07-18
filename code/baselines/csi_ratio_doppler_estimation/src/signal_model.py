"""Synthetic CSI generators used by the Doppler-estimator tests.

The general toy model is a static-plus-dynamic narrowband channel

``H_m(t) = (s_m + d_m exp(j 2 pi f_D t)) exp(j 2 pi f_o t) + n_m(t)``.

Consequently, an antenna ratio is the Mobius trajectory

``R_mr(t) = (s_m + d_m z(t)) / (s_r + d_r z(t))``.

The common offset ``f_o`` cancels in the noiseless ratio.  This is an
educational short-window model, not a hardware trace or a complete
reimplementation of the cited paper's propagation model.
"""

from typing import Optional, Tuple

import numpy as np


# Beyond this bound, binary64 no longer retains enough fractional-cycle
# resolution for this educational phase oracle.  Rejecting such inputs is
# preferable to silently returning a platform-dependent argument reduction.
_MAX_RELIABLE_PHASE_CYCLES = float(2**40)


def _validate_time(t: np.ndarray) -> np.ndarray:
    """Return a validated one-dimensional time axis."""
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 3 or not np.all(np.isfinite(t)):
        raise ValueError("t must be a finite one-dimensional array with at least 3 samples")
    if np.any(t[1:] <= t[:-1]):
        raise ValueError("t must be strictly increasing")
    return t


def _rotation_from_frequency(
    t: np.ndarray,
    frequency_hz: float,
    *,
    label: str,
) -> np.ndarray:
    """Construct ``exp(j 2 pi f t)`` inside an explicit binary64 domain."""
    max_abs_time = float(np.max(np.abs(t)))
    if (
        frequency_hz != 0.0
        and max_abs_time != 0.0
        and np.log(abs(frequency_hz)) + np.log(max_abs_time)
        > np.log(_MAX_RELIABLE_PHASE_CYCLES)
    ):
        raise ValueError(
            f"{label} phase exceeds the reliable binary64 cycle domain"
        )
    cycles = frequency_hz * t
    if not np.all(np.isfinite(cycles)):
        raise ValueError(f"{label} phase cycles must be finite")
    reduced_cycles = np.remainder(cycles, 1.0)
    return np.exp(1j * (2.0 * np.pi * reduced_cycles))


def _constant_rotation(phase_radians: float) -> complex:
    """Construct a constant phase after a reliability-domain check."""
    cycles = phase_radians / (2.0 * np.pi)
    if not np.isfinite(cycles) or abs(cycles) > _MAX_RELIABLE_PHASE_CYCLES:
        raise ValueError("phase_offset exceeds the reliable binary64 cycle domain")
    return complex(np.exp(1j * (2.0 * np.pi * np.remainder(cycles, 1.0))))


def _sum_frequencies(first_hz: float, second_hz: float) -> float:
    """Add two finite frequencies without first overflowing binary64."""
    if (
        np.signbit(first_hz) == np.signbit(second_hz)
        and abs(first_hz) > np.finfo(float).max - abs(second_hz)
    ):
        raise ValueError("combined frequency is outside the finite binary64 domain")
    combined = first_hz + second_hz
    if not np.isfinite(combined):
        raise ValueError("combined frequency must be finite")
    return combined


def _finite_product(*factors: float, label: str) -> float:
    """Multiply finite scalars while rejecting overflow before it occurs."""
    nonzero = [factor for factor in factors if factor != 0.0]
    if not nonzero:
        return 0.0
    log_magnitude = sum(np.log(abs(factor)) for factor in nonzero)
    if log_magnitude > np.log(np.finfo(float).max):
        raise ValueError(f"{label} is outside the finite binary64 domain")
    product = float(np.prod(np.asarray(factors, dtype=float)))
    if not np.isfinite(product):
        raise ValueError(f"{label} must be finite")
    if product == 0.0:
        raise ValueError(f"{label} is below the representable binary64 domain")
    return product


def _add_complex_noise(
    H: np.ndarray,
    snr_db: float,
    rng: Optional[np.random.Generator],
) -> np.ndarray:
    """Add independent circular complex Gaussian noise at the requested SNR."""
    if np.isposinf(snr_db):
        return H
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite or positive infinity")
    generator = np.random.default_rng() if rng is None else rng
    if not isinstance(generator, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    H = np.asarray(H, dtype=complex)
    if not np.all(np.isfinite(H)):
        raise ValueError("H must be finite before adding noise")
    with np.errstate(over="ignore", invalid="ignore"):
        magnitudes = np.abs(H)
    signal_scale = float(np.max(magnitudes))
    if not np.isfinite(signal_scale):
        raise ValueError("H magnitude must be representable in binary64")
    if signal_scale <= 0:
        raise ValueError("cannot define SNR for an all-zero channel")
    normalized = H / signal_scale
    normalized_power = float(np.mean(np.abs(normalized) ** 2))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        snr_linear = float(np.power(10.0, snr_db / 10.0))
    if not np.isfinite(snr_linear) or snr_linear <= 0:
        raise ValueError("snr_db must map to a finite positive linear SNR")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized_noise_std = float(
            np.sqrt(normalized_power / snr_linear / 2.0)
        )
        noise_scale = signal_scale * normalized_noise_std
    if not np.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("requested noise scale is not representable in binary64")
    with np.errstate(over="ignore", invalid="ignore"):
        noise = noise_scale * (
            generator.standard_normal(H.shape)
            + 1j * generator.standard_normal(H.shape)
        )
        noisy = H + noise
    if not np.all(np.isfinite(noisy)):
        raise ValueError("requested SNR produces a non-finite noisy channel")
    return noisy


def csi_static_dynamic_model(
    t: np.ndarray,
    f_D: float,
    static_coefficients: np.ndarray,
    dynamic_coefficients: np.ndarray,
    shared_offset_hz: float = 0.0,
    snr_db: float = np.inf,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a transparent static-plus-dynamic multi-antenna CSI toy model.

    ``f_D`` is the signed rotation frequency under the module's positive
    complex-exponential convention.  It is not, by itself, an
    approaching/receding label.
    """
    t = _validate_time(t)
    if not np.isfinite(f_D) or not np.isfinite(shared_offset_hz):
        raise ValueError("f_D and shared_offset_hz must be finite")
    static = np.asarray(static_coefficients, dtype=complex)
    dynamic = np.asarray(dynamic_coefficients, dtype=complex)
    if static.ndim != 1 or static.size < 2 or static.shape != dynamic.shape:
        raise ValueError("coefficient arrays must have equal one-dimensional shape (M >= 2)")
    if not np.all(np.isfinite(static)) or not np.all(np.isfinite(dynamic)):
        raise ValueError("coefficient arrays must be finite")

    z = _rotation_from_frequency(t, f_D, label="Doppler")[:, None]
    shared = _rotation_from_frequency(
        t, shared_offset_hz, label="shared-offset"
    )[:, None]
    with np.errstate(over="ignore", invalid="ignore"):
        H = (static[None, :] + dynamic[None, :] * z) * shared
    if not np.all(np.isfinite(H)):
        raise ValueError("coefficients produce a non-finite channel")
    return _add_complex_noise(H, snr_db, rng)


def csi_signal_model(
    t: np.ndarray,
    f_c: float = 5.8e9,
    c: float = 3e8,
    d0: float = 5.0,
    v_r: float = 0.0,
    antenna_positions: Optional[np.ndarray] = None,
    snr_db: float = np.inf,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate illustrative multi-antenna CSI with a Mobius ratio trajectory.

    The default coefficients represent two distinct illustrative arrival
    angles and a dynamic-path amplitude 0.6 times the static-path amplitude.
    They are fixed during the observation window.  ``v_r`` is converted using
    ``f_D = 2 f_c v_r / c`` under this module's phase convention; interpreting
    its sign physically requires a separately declared geometry convention.
    """
    t = _validate_time(t)
    if not np.isfinite(f_c) or f_c <= 0 or not np.isfinite(c) or c <= 0:
        raise ValueError("f_c and c must be positive and finite")
    if not np.isfinite(d0) or d0 <= 0 or not np.isfinite(v_r):
        raise ValueError("d0 must be positive and v_r must be finite")
    wavelength = c / f_c
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("c / f_c must be representable as a positive wavelength")
    if antenna_positions is None:
        antenna_positions = np.array([0.0, wavelength / 2.0, wavelength])
    positions = np.asarray(antenna_positions, dtype=float)
    if positions.ndim != 1 or positions.size < 2 or not np.all(np.isfinite(positions)):
        raise ValueError("antenna_positions must be a finite 1-D array with M >= 2")

    max_position = float(np.max(np.abs(positions)))
    if (
        wavelength < np.finfo(float).max / _MAX_RELIABLE_PHASE_CYCLES
        and max_position > wavelength * _MAX_RELIABLE_PHASE_CYCLES
    ):
        raise ValueError("antenna spatial phase exceeds the reliable binary64 domain")
    spatial_cycles = positions / wavelength
    if not np.all(np.isfinite(spatial_cycles)):
        raise ValueError("antenna spatial phase cycles must be finite")

    smallest_subnormal = np.finfo(float).smallest_subnormal
    minimum_d0 = np.exp(-0.5 * np.log(np.finfo(float).max))
    maximum_d0 = np.exp(0.5 * (np.log(0.6) - np.log(smallest_subnormal)))
    if not minimum_d0 <= d0 <= maximum_d0:
        raise ValueError("d0 produces an unrepresentable inverse-square path scale")
    inverse_distance = 1.0 / d0
    path_scale = inverse_distance * inverse_distance
    dynamic_scale = 0.6 * path_scale
    if (
        not np.isfinite(path_scale)
        or path_scale <= 0
        or dynamic_scale <= 0
    ):
        raise ValueError("d0 produces an unrepresentable inverse-square path scale")

    static_cycles = -spatial_cycles * np.sin(np.deg2rad(10.0))
    dynamic_cycles = -spatial_cycles * np.sin(np.deg2rad(-35.0))
    static = path_scale * np.exp(
        1j * 2.0 * np.pi * np.remainder(static_cycles, 1.0)
    )
    dynamic = dynamic_scale * np.exp(
        1j * 2.0 * np.pi * np.remainder(dynamic_cycles, 1.0)
    )

    carrier_ratio = f_c / c
    if not np.isfinite(carrier_ratio) or carrier_ratio <= 0:
        raise ValueError("f_c / c must be representable and positive")
    doppler_hz = _finite_product(
        2.0,
        carrier_ratio,
        v_r,
        label="Doppler frequency",
    ) if v_r != 0.0 else 0.0
    return csi_static_dynamic_model(
        t,
        f_D=doppler_hz,
        static_coefficients=static,
        dynamic_coefficients=dynamic,
        shared_offset_hz=50.01,
        snr_db=snr_db,
        rng=rng,
    )


def csi_with_doppler(
    t: np.ndarray,
    f_D: float,
    snr_db: float = np.inf,
    amplitude_ratio: float = 1.0,
    phase_offset: float = 0.0,
    cfo_hz: float = 0.0,
    tmo_hz: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a pure-rotation ratio for estimator unit tests.

    The construction deliberately enforces
    ``H1/H2 = amplitude_ratio * exp(j(2 pi f_D t + phase_offset))``.
    It is useful as an exact numerical oracle, but it is not the general
    static-plus-dynamic CSI model and does not establish paper parity.
    """
    t = _validate_time(t)
    finite_values = (f_D, amplitude_ratio, phase_offset, cfo_hz, tmo_hz)
    if not all(np.isfinite(value) for value in finite_values):
        raise ValueError("frequency, amplitude, and phase parameters must be finite")
    if amplitude_ratio <= 0:
        raise ValueError("amplitude_ratio must be positive")

    shared_frequency = _sum_frequencies(cfo_hz, tmo_hz)
    shared = _rotation_from_frequency(t, shared_frequency, label="shared-offset")
    doppler = _rotation_from_frequency(t, f_D, label="Doppler")
    offset = _constant_rotation(phase_offset)
    H1 = amplitude_ratio * shared * doppler * offset
    H2 = shared
    H = _add_complex_noise(np.column_stack([H1, H2]), snr_db, rng)
    return H[:, 0], H[:, 1]
