"""CSI-ratio computation.

The basic quotient is
    R(t) = H_m(t) / H_{m+1}(t)

It cancels only factors that are genuinely common and multiplicative across
the selected antennas. Independent noise and antenna-dependent errors remain.

Properties:
- R(t) is a Mobius transform of z(t) = exp(j*2π*f_D*t)
- As z(t) traces the unit circle, R(t) traces a circle in complex plane
- Time order and traversal rate, not circle geometry alone, carry the observed
  rotation-frequency information
"""

import numpy as np
from typing import Tuple


def _validate_csi_pair(
    H_m: np.ndarray, H_m1: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return equal, finite one-dimensional complex CSI arrays."""
    numerator = np.asarray(H_m, dtype=complex)
    denominator = np.asarray(H_m1, dtype=complex)
    if numerator.shape != denominator.shape or numerator.ndim != 1:
        raise ValueError("CSI inputs must be one-dimensional arrays with equal shape")
    if (
        numerator.size == 0
        or not np.all(np.isfinite(numerator))
        or not np.all(np.isfinite(denominator))
    ):
        raise ValueError("CSI inputs must be non-empty and finite")
    return numerator, denominator


def _component_scaled_magnitudes(value: np.ndarray) -> tuple[np.ndarray, float]:
    """Return relative magnitudes without overflowing ``abs(complex)``."""
    scale = float(
        max(
            np.max(np.abs(value.real), initial=0.0),
            np.max(np.abs(value.imag), initial=0.0),
        )
    )
    if scale == 0.0:
        return np.zeros(value.shape, dtype=float), scale
    with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
        real = value.real / scale
        imag = value.imag / scale
    return np.hypot(real, imag), scale


def _finite_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Use component-scaled Smith division and reject unrepresentable output."""
    quotient = np.empty(numerator.shape, dtype=np.complex128)
    for index in np.ndindex(numerator.shape):
        numerator_value = complex(numerator[index])
        denominator_value = complex(denominator[index])
        scale = max(
            abs(numerator_value.real),
            abs(numerator_value.imag),
            abs(denominator_value.real),
            abs(denominator_value.imag),
        )
        if scale == 0.0 or denominator_value == 0.0:
            raise ValueError("CSI quotient denominator must be nonzero")
        a = numerator_value.real / scale
        b = numerator_value.imag / scale
        c = denominator_value.real / scale
        d = denominator_value.imag / scale
        try:
            if abs(c) >= abs(d):
                relative = d / c
                divisor = c + d * relative
                real = (a + b * relative) / divisor
                imag = (b - a * relative) / divisor
            else:
                relative = c / d
                divisor = d + c * relative
                real = (a * relative + b) / divisor
                imag = (b * relative - a) / divisor
        except (OverflowError, ZeroDivisionError) as error:
            raise ValueError(
                "CSI quotient is outside the finite binary64 domain"
            ) from error
        value = complex(real, imag)
        if not np.isfinite(value) or (
            value == 0.0 and numerator_value != 0.0
        ):
            raise ValueError("CSI quotient is outside the finite binary64 domain")
        quotient[index] = value
    return quotient


def compute_csi_ratio(H_m: np.ndarray, H_m1: np.ndarray) -> np.ndarray:
    """
    Compute CSI-ratio between two adjacent receive antennas.

    Computes the elementary quotient
        R(t_k) = H_m(t_k) / H_{m+1}(t_k)

    Parameters
    ----------
    H_m : np.ndarray
        CSI samples from antenna m, shape (N,) complex.
    H_m1 : np.ndarray
        CSI samples from antenna m+1, shape (N,) complex.

    Returns
    -------
    R : np.ndarray
        CSI-ratio samples, shape (N,) complex.

    Notes
    -----
    - If H_m1 has very small magnitude, the ratio will have large values.
      Consider adding a small regularization or clipping.
    - The ratio cancels multiplicative phase terms common to both antennas.
    """
    H_m, H_m1 = _validate_csi_pair(H_m, H_m1)
    relative_magnitudes, scale = _component_scaled_magnitudes(H_m1)
    if scale == 0 or np.any(
        relative_magnitudes
        <= np.finfo(float).eps * float(np.max(relative_magnitudes))
    ):
        raise ValueError(
            "Reference CSI contains zero/near-zero samples; use "
            "compute_csi_ratio_robust to mask them"
        )
    return _finite_divide(H_m, H_m1)


def compute_csi_ratio_multi(H: np.ndarray, ref_antenna: int = 0) -> np.ndarray:
    """
    Compute CSI-ratios for all antenna pairs.

    Parameters
    ----------
    H : np.ndarray
        CSI matrix, shape (N, M) where N = time samples, M = antennas.
    ref_antenna : int
        Reference antenna index for forming ratios. Default: 0.

    Returns
    -------
    R : np.ndarray
        CSI-ratios relative to ``ref_antenna``, shape (N, M-1).
        Columns follow increasing antenna index with the reference omitted.
    """
    H = np.asarray(H, dtype=complex)
    if H.ndim != 2 or H.shape[1] < 2:
        raise ValueError("H must have shape (samples, antennas) with at least 2 antennas")
    _, M = H.shape
    if not 0 <= ref_antenna < M:
        raise IndexError(f"ref_antenna must be in [0, {M})")
    other_antennas = [index for index in range(M) if index != ref_antenna]
    return np.column_stack(
        [compute_csi_ratio(H[:, index], H[:, ref_antenna]) for index in other_antennas]
    )


def compute_csi_ratio_robust(
    H_m: np.ndarray,
    H_m1: np.ndarray,
    threshold_db: float = -30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute CSI-ratio with robustness to low-SNR samples.

    Filters out samples where |H_m1| is below a threshold relative
    to the maximum magnitude.

    Parameters
    ----------
    H_m : np.ndarray
        CSI samples from antenna m, shape (N,) complex.
    H_m1 : np.ndarray
        CSI samples from antenna m+1, shape (N,) complex.
    threshold_db : float
        Threshold in dB below max magnitude. Samples with |H_m1|
        below this are excluded. Default: -30 dB.

    Returns
    -------
    R : np.ndarray
        CSI-ratio samples (filtered).
    mask : np.ndarray
        Boolean mask indicating which samples were kept, shape (N,).
    """
    H_m, H_m1 = _validate_csi_pair(H_m, H_m1)
    if not np.isfinite(threshold_db) or threshold_db > 0:
        raise ValueError("threshold_db must be finite and no greater than 0 dB")
    relative_magnitudes, _ = _component_scaled_magnitudes(H_m1)
    max_abs = np.max(relative_magnitudes)
    threshold_linear = max_abs * 10 ** (threshold_db / 20)

    mask = (relative_magnitudes > 0) & (
        relative_magnitudes >= threshold_linear
    )
    R = np.zeros(H_m.shape, dtype=complex)
    if np.any(mask):
        R[mask] = _finite_divide(H_m[mask], H_m1[mask])

    return R, mask
