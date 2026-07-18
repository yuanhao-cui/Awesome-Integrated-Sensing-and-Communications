"""Auditable energy-efficiency and point-target CRB equations.

The CRB is evaluated from the real-parameter Fisher information matrix for
``Y = alpha a_r(theta) a_t(theta)^H X + Z`` with the complex reflection
coefficient treated as an unknown nuisance parameter.  This general two-array
form avoids silently identifying the paper's M- and N-element steering vectors.
It reduces to the familiar Schur-complement expression and includes the
``2 L`` information factor used by constraint (17) in the published method.
"""

from __future__ import annotations

import numpy as np

from .numerics import stable_sinr, stable_spectral_efficiency, stable_squared_norm

_IDENTIFIABILITY_RTOL = 128.0 * np.finfo(float).eps


def _divide_complex_by_positive_scale(
    value: np.ndarray, scale: float
) -> np.ndarray:
    """Divide componentwise without the subnormal complex-division failure.

    NumPy's generic complex division forms a squared complex denominator;
    that denominator can underflow even when ``value / scale`` is bounded by
    one. Dividing the real and imaginary components separately avoids that
    spurious overflow.
    """

    with np.errstate(divide="raise", invalid="raise", over="raise", under="ignore"):
        return value.real / scale + 1j * (value.imag / scale)


def _require_positive(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _validate_beamforming(H: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H = np.asarray(H, dtype=complex)
    W = np.asarray(W, dtype=complex)
    if H.ndim != 2 or W.ndim != 2:
        raise ValueError("H and W must be two-dimensional")
    if H.shape != (W.shape[1], W.shape[0]):
        raise ValueError("H must have shape (K, M) and W shape (M, K)")
    if not np.all(np.isfinite(H)) or not np.all(np.isfinite(W)):
        raise ValueError("H and W must be finite")
    return H, W


def compute_sinr(k: int, h_k: np.ndarray, W: np.ndarray, sigma_c2: float) -> float:
    """Evaluate (2) for user ``k`` with direct stream exclusion.

    Interference is formed only from columns ``j != k``. No desired-power
    subtraction is used, and normalized binary power representations prevent
    projection squares from overflowing or underflowing prematurely.
    """

    sigma_c2 = _require_positive("sigma_c2", sigma_c2)
    return stable_sinr(k, h_k, W, sigma_c2)


def compute_sum_rate(H: np.ndarray, W: np.ndarray, sigma_c2: float) -> float:
    """Return the spectral-efficiency sum in bit/s/Hz.

    Each stream uses the same cancellation-free geometry as
    :func:`compute_sinr`. ``log1p`` retains sub-epsilon SINR, while a
    log-domain softplus covers finite rates whose linear SINR exceeds
    binary64.
    """

    H, W = _validate_beamforming(H, W)
    return float(
        sum(
            stable_spectral_efficiency(k, H[k], W, sigma_c2)
            for k in range(H.shape[0])
        )
    )


def compute_total_power(W: np.ndarray) -> float:
    """Return radiated power ``tr(W W^H)``."""

    W = np.asarray(W, dtype=complex)
    if W.ndim != 2 or not np.all(np.isfinite(W)):
        raise ValueError("W must be a finite two-dimensional array")
    return stable_squared_norm(W)


def compute_ee_c(
    H: np.ndarray,
    W: np.ndarray,
    sigma_c2: float,
    epsilon: float,
    P0: float,
) -> float:
    """Evaluate communication energy efficiency (4), in bit/J/Hz."""

    if not 0.0 < float(epsilon) <= 1.0:
        raise ValueError("epsilon must lie in (0, 1]")
    if not np.isfinite(P0) or P0 < 0.0:
        raise ValueError("P0 must be finite and non-negative")
    denominator = compute_total_power(W) / float(epsilon) + float(P0)
    if denominator <= 0.0:
        raise ValueError("total consumed power must be positive")
    return compute_sum_rate(H, W, sigma_c2) / denominator


def point_target_information_terms(
    W: np.ndarray,
    a_t: np.ndarray,
    a_r: np.ndarray,
    da_t: np.ndarray,
    da_r: np.ndarray,
    L: int,
) -> tuple[float, float, complex]:
    """Return signal energy, derivative energy, and their inner product.

    The terms are computed from ``R_x = W W^H`` and are exactly equal to
    their explicit ``L``-snapshot counterparts when ``S S^H = L I``.
    """

    W = np.asarray(W, dtype=complex)
    a_t = np.asarray(a_t, dtype=complex)
    a_r = np.asarray(a_r, dtype=complex)
    da_t = np.asarray(da_t, dtype=complex)
    da_r = np.asarray(da_r, dtype=complex)
    if W.ndim != 2:
        raise ValueError("W must be two-dimensional")
    if a_t.shape != (W.shape[0],) or da_t.shape != a_t.shape:
        raise ValueError("a_t and da_t must match W's transmit dimension")
    if a_r.ndim != 1 or da_r.shape != a_r.shape:
        raise ValueError("a_r and da_r must be matching vectors")
    if not isinstance(L, int) or isinstance(L, bool) or L <= 0:
        raise ValueError("L must be a positive integer")
    arrays = (W, a_t, a_r, da_t, da_r)
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("all array inputs must be finite")

    response = np.outer(a_r, a_t.conj())
    derivative = np.outer(da_r, a_t.conj()) + np.outer(a_r, da_t.conj())
    covariance = W @ W.conj().T
    signal_energy = L * np.trace(response @ covariance @ response.conj().T)
    derivative_energy = L * np.trace(derivative @ covariance @ derivative.conj().T)
    cross = L * np.trace(response.conj().T @ derivative @ covariance)
    return float(np.real(signal_energy)), float(np.real(derivative_energy)), complex(cross)


def _point_target_crb_log_information(
    W: np.ndarray,
    a_t: np.ndarray,
    a_r: np.ndarray,
    da_t: np.ndarray,
    da_r: np.ndarray,
    sigma_s2: float,
    L: int,
    alpha_abs: float,
) -> float | None:
    """Return ``log(J_theta)`` using scale-free projection geometry.

    A direct Schur subtraction,
    ``||g_dot||^2 - |g^H g_dot|^2 / ||g||^2``, loses all significant
    digits when the nuisance component is large.  It also makes a fixed
    absolute identifiability threshold depend on the arbitrary common scale
    of ``W``.  This helper instead normalizes every multiplicative scale,
    projects the derivative onto the orthogonal complement of the response,
    and restores the physical scale only in the logarithmic domain.

    ``None`` denotes a physically or numerically unidentifiable direction.
    A finite log-information value may still imply a CRB outside binary64;
    the public function reports that representability failure explicitly.
    """

    W = np.asarray(W, dtype=complex)
    a_t = np.asarray(a_t, dtype=complex)
    a_r = np.asarray(a_r, dtype=complex)
    da_t = np.asarray(da_t, dtype=complex)
    da_r = np.asarray(da_r, dtype=complex)
    if W.ndim != 2:
        raise ValueError("W must be two-dimensional")
    if a_t.shape != (W.shape[0],) or da_t.shape != a_t.shape:
        raise ValueError("a_t and da_t must match W's transmit dimension")
    if a_r.ndim != 1 or da_r.shape != a_r.shape:
        raise ValueError("a_r and da_r must be matching vectors")
    if not isinstance(L, int) or isinstance(L, bool) or L <= 0:
        raise ValueError("L must be a positive integer")
    arrays = (W, a_t, a_r, da_t, da_r)
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("all array inputs must be finite")

    w_scale = float(np.max(np.abs(W), initial=0.0))
    a_t_scale = float(np.max(np.abs(a_t), initial=0.0))
    a_r_scale = float(np.max(np.abs(a_r), initial=0.0))
    da_t_scale = float(np.max(np.abs(da_t), initial=0.0))
    da_r_scale = float(np.max(np.abs(da_r), initial=0.0))
    if w_scale == 0.0 or a_t_scale == 0.0 or a_r_scale == 0.0:
        return None
    if not all(
        np.isfinite(value)
        for value in (w_scale, a_t_scale, a_r_scale, da_t_scale, da_r_scale)
    ):
        raise FloatingPointError("array magnitudes exceed the binary64 domain")

    W_unit = _divide_complex_by_positive_scale(W, w_scale)
    a_t_unit = _divide_complex_by_positive_scale(a_t, a_t_scale)
    a_r_unit = _divide_complex_by_positive_scale(a_r, a_r_scale)
    response_unit = np.outer(a_r_unit, a_t_unit.conj())

    derivative_log_scales: list[float] = []
    if da_r_scale > 0.0:
        derivative_log_scales.append(np.log(da_r_scale) + np.log(a_t_scale))
    if da_t_scale > 0.0:
        derivative_log_scales.append(np.log(a_r_scale) + np.log(da_t_scale))
    if not derivative_log_scales:
        return None
    derivative_log_scale = max(derivative_log_scales)
    derivative_unit = np.zeros(
        (a_r.shape[0], a_t.shape[0]), dtype=complex
    )
    if da_r_scale > 0.0:
        with np.errstate(under="ignore"):
            weight = np.exp(
                np.log(da_r_scale)
                + np.log(a_t_scale)
                - derivative_log_scale
            )
        derivative_unit += weight * np.outer(
            _divide_complex_by_positive_scale(da_r, da_r_scale),
            a_t_unit.conj(),
        )
    if da_t_scale > 0.0:
        with np.errstate(under="ignore"):
            weight = np.exp(
                np.log(a_r_scale)
                + np.log(da_t_scale)
                - derivative_log_scale
            )
        derivative_unit += weight * np.outer(
            a_r_unit,
            _divide_complex_by_positive_scale(da_t, da_t_scale).conj(),
        )

    with np.errstate(under="ignore"):
        response_projection = (response_unit @ W_unit).reshape(-1)
        derivative_projection = (derivative_unit @ W_unit).reshape(-1)
    if not np.all(np.isfinite(response_projection)) or not np.all(
        np.isfinite(derivative_projection)
    ):
        raise FloatingPointError("projected target response is outside binary64")

    response_norm = float(np.linalg.norm(response_projection))
    response_reference = float(
        np.linalg.norm(response_unit, ord="fro")
        * np.linalg.norm(W_unit, ord="fro")
    )
    if (
        response_norm == 0.0
        or response_reference == 0.0
        or response_norm <= _IDENTIFIABILITY_RTOL * response_reference
    ):
        return None

    derivative_norm = float(np.linalg.norm(derivative_projection))
    if derivative_norm == 0.0:
        return None
    response_direction = response_projection / response_norm
    nuisance_projection = np.vdot(response_direction, derivative_projection)
    residual = derivative_projection - response_direction * nuisance_projection
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm <= _IDENTIFIABILITY_RTOL * derivative_norm:
        return None

    log_effective = (
        np.log(float(L))
        + 2.0 * np.log(w_scale)
        + 2.0 * derivative_log_scale
        + 2.0 * np.log(residual_norm)
    )
    return float(
        np.log(2.0)
        + 2.0 * np.log(alpha_abs)
        - np.log(sigma_s2)
        + log_effective
    )


def compute_crb_point_target(
    W: np.ndarray,
    a_t: np.ndarray,
    a_r: np.ndarray,
    da_t: np.ndarray,
    da_r: np.ndarray,
    sigma_s2: float,
    L: int,
    alpha_abs: float = 1.0,
) -> float:
    """Return the angle CRB with unknown complex target reflectivity.

    The nuisance parameter is eliminated by a Schur complement:

    ``J_theta = 2 |alpha|^2 / sigma_s^2 * (||g_dot||^2
    - |g^H g_dot|^2 / ||g||^2)``.

    ``inf`` is returned only for a physically or numerically unidentifiable
    angle under the declared relative geometry tolerance. A finite CRB outside
    binary64 is reported separately with ``OverflowError`` or
    ``FloatingPointError``; it never masquerades as unidentifiability.
    """

    sigma_s2 = _require_positive("sigma_s2", sigma_s2)
    alpha_abs = _require_positive("alpha_abs", alpha_abs)
    log_information = _point_target_crb_log_information(
        W,
        a_t,
        a_r,
        da_t,
        da_r,
        sigma_s2,
        L,
        alpha_abs,
    )
    if log_information is None:
        return float("inf")
    log_crb = -log_information
    max_log = float(np.log(np.finfo(float).max))
    min_log = float(np.log(np.nextafter(0.0, 1.0)))
    if log_crb > max_log:
        raise OverflowError("the finite CRB exceeds the binary64 range")
    if log_crb < min_log:
        raise FloatingPointError("the positive CRB underflows the binary64 range")
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        crb = float(np.exp(log_crb))
    if crb == 0.0:
        raise FloatingPointError("the positive CRB underflows the binary64 range")
    return crb


def compute_crb(
    W: np.ndarray,
    a_t: np.ndarray,
    a_r: np.ndarray,
    da_t: np.ndarray,
    da_r: np.ndarray,
    sigma_s2: float,
    L: int,
    alpha_abs: float = 1.0,
) -> float:
    """Alias for :func:`compute_crb_point_target`."""

    return compute_crb_point_target(
        W, a_t, a_r, da_t, da_r, sigma_s2, L, alpha_abs
    )


def compute_ee_s(
    W: np.ndarray,
    a_t: np.ndarray,
    a_r: np.ndarray,
    da_t: np.ndarray,
    da_r: np.ndarray,
    sigma_s2: float,
    L: int,
    epsilon: float,
    P0: float,
    alpha_abs: float = 1.0,
) -> float:
    """Evaluate the point-target sensing EE definition (33)."""

    crb = compute_crb(
        W, a_t, a_r, da_t, da_r, sigma_s2, L, alpha_abs
    )
    if not np.isfinite(crb):
        return 0.0
    if not 0.0 < float(epsilon) <= 1.0:
        raise ValueError("epsilon must lie in (0, 1]")
    if not np.isfinite(P0) or P0 < 0.0:
        raise ValueError("P0 must be finite and non-negative")
    energy = L * (compute_total_power(W) / float(epsilon) + float(P0))
    if energy <= 0.0:
        raise ValueError("consumed energy must be positive")
    return float(1.0 / (crb * energy))


__all__ = [
    "compute_crb",
    "compute_crb_point_target",
    "compute_ee_c",
    "compute_ee_s",
    "compute_sinr",
    "compute_sum_rate",
    "compute_total_power",
    "point_target_information_terms",
]
