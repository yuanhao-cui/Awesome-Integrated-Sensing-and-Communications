"""Quadratic transform used by the communication-rate objective.

For user ``k`` let ``a_k = h_k^H w_k`` and let ``B_k`` contain noise and
*inter-user* interference (but not the desired signal).  Equation (14) of
Zou et al. rewrites the rate exactly as

    log2(1 + |a_k|^2 / B_k)
      = max_t log2(1 + 2 Re(t* a_k) - |t|^2 B_k).

The maximizer is ``t = a_k / B_k``.  The logarithm is essential: the inner
quadratic expression alone is an SINR surrogate, not a rate.  This module
only evaluates the exact transform for fixed beamformers; optimization over
beamformers belongs in a constrained solver that keeps vector and lifted SDR
variables distinct.

Reference: J. Zou et al., "Energy-Efficient Beamforming Design for
Integrated Sensing and Communications Systems," IEEE TCOM, 2024, Eq. (14),
DOI 10.1109/TCOMM.2024.3369696.
"""

from __future__ import annotations

import numpy as np

from .numerics import stable_spectral_efficiency


def _validate_inputs(
    H: np.ndarray,
    W: np.ndarray,
    sigma_c2: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate and normalize arrays shared by the transform helpers."""
    H = np.asarray(H, dtype=np.complex128)
    W = np.asarray(W, dtype=np.complex128)
    if H.ndim != 2 or W.ndim != 2:
        raise ValueError("H and W must both be two-dimensional arrays")
    users, antennas = H.shape
    if W.shape != (antennas, users):
        raise ValueError(
            f"W must have shape {(antennas, users)} for H shape {H.shape}; "
            f"received {W.shape}"
        )
    if not np.isfinite(sigma_c2) or sigma_c2 <= 0:
        raise ValueError("sigma_c2 must be finite and strictly positive")
    if not np.all(np.isfinite(H)) or not np.all(np.isfinite(W)):
        raise ValueError("H and W must contain only finite values")
    return H, W, float(sigma_c2)


def _interference_plus_noise(
    H: np.ndarray,
    W: np.ndarray,
    sigma_c2: float,
    user: int,
) -> float:
    """Return B_k: noise plus interference, excluding desired power."""
    h_k = H[user]
    interference = sum(
        abs(h_k.conj() @ W[:, other]) ** 2
        for other in range(H.shape[0])
        if other != user
    )
    return float(sigma_c2 + interference)


def optimize_t(
    H: np.ndarray,
    W: np.ndarray,
    sigma_c2: float,
) -> np.ndarray:
    """Return the closed-form Eq. (15) auxiliary variables for fixed ``W``."""
    H, W, sigma_c2 = _validate_inputs(H, W, sigma_c2)
    t = np.empty(H.shape[0], dtype=np.complex128)
    for user in range(H.shape[0]):
        desired_amplitude = H[user].conj() @ W[:, user]
        denominator = _interference_plus_noise(H, W, sigma_c2, user)
        t[user] = desired_amplitude / denominator
    return t


def quadratic_transform_objective(
    H: np.ndarray,
    W: np.ndarray,
    t: np.ndarray,
    sigma_c2: float,
) -> float:
    """Evaluate the transformed sum rate in bits/s/Hz.

    For arbitrary auxiliary variables this is a lower bound whenever every
    logarithm argument is positive.  At :func:`optimize_t`, it is equal to the
    direct sum rate up to floating-point error.
    """
    H, W, sigma_c2 = _validate_inputs(H, W, sigma_c2)
    t = np.asarray(t, dtype=np.complex128)
    if t.shape != (H.shape[0],):
        raise ValueError(f"t must have shape {(H.shape[0],)}; received {t.shape}")
    if not np.all(np.isfinite(t)):
        raise ValueError("t must contain only finite values")

    transformed_rate = 0.0
    for user in range(H.shape[0]):
        desired_amplitude = H[user].conj() @ W[:, user]
        denominator = _interference_plus_noise(H, W, sigma_c2, user)
        transformed_sinr = (
            2.0 * np.real(np.conj(t[user]) * desired_amplitude)
            - abs(t[user]) ** 2 * denominator
        )
        transformed_sinr = float(transformed_sinr)
        if transformed_sinr <= -1.0:
            raise ValueError(
                f"invalid auxiliary variable for user {user}: "
                f"log2 argument is {1.0 + transformed_sinr:.6g}"
            )
        transformed_rate += np.log1p(transformed_sinr) / np.log(2.0)
    return float(transformed_rate)


def compute_sum_rate_quadratic(
    H: np.ndarray,
    W: np.ndarray,
    sigma_c2: float,
) -> float:
    """Compute the exact fixed-``W`` optimum of Eqs. (14)-(15).

    Substituting the Eq. (15) maximizer into Eq. (14) is algebraically the
    direct ``log2(1 + SINR)`` rate.  Evaluating that reduced identity with the
    scale-safe projection primitive avoids transient ``2|a|^2/B`` and
    ``|a|^2/B`` terms that can individually overflow although their difference
    and final logarithm are representable.  The explicit arbitrary-``t``
    objective remains available for ordinary-range lower-bound diagnostics.
    """
    H, W, sigma_c2 = _validate_inputs(H, W, sigma_c2)
    return float(
        sum(
            stable_spectral_efficiency(user, H[user], W, sigma_c2)
            for user in range(H.shape[0])
        )
    )
