"""Evaluation helpers for the local surrogate (legacy filename).

Nothing in this module is asserted to be an inner or outer bound from the
reference paper.  The historical filename is retained to avoid an unnecessary
filesystem-level compatibility break.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from .optimization import (
    covariance_shaping_surrogate,
    isotropic_covariance,
    water_filling_covariance,
)
from .system_model import compute_crb, compute_rate


Array = np.ndarray


def evaluate_reference_endpoints(
    Hc: Array,
    T: int,
    sigma_c2: float,
    sigma_s2: float,
    power_per_tx: float,
    phi_func: Callable[[Array], Array] | None = None,
    Jp: Array | None = None,
) -> dict[str, float | Array]:
    """Evaluate isotropic and rate-maximizing covariance references.

    The isotropic covariance is only a local reference.  It is not described
    as sensing-optimal for an arbitrary ``phi_func``.
    """

    channel = np.asarray(Hc, dtype=np.complex128)
    if channel.ndim != 2 or 0 in channel.shape:
        raise ValueError("Hc must be a non-empty two-dimensional matrix")
    isotropic = isotropic_covariance(power_per_tx, channel.shape[1])
    water_filled = water_filling_covariance(power_per_tx, channel, sigma_c2)
    return {
        "isotropic_covariance": isotropic,
        "isotropic_rate": compute_rate(isotropic, channel, sigma_c2),
        "isotropic_crb": compute_crb(
            isotropic,
            T,
            sigma_s2,
            phi_func=phi_func,
            Jp=Jp,
        ),
        "water_filling_covariance": water_filled,
        "water_filling_rate": compute_rate(water_filled, channel, sigma_c2),
        "water_filling_crb": compute_crb(
            water_filled,
            T,
            sigma_s2,
            phi_func=phi_func,
            Jp=Jp,
        ),
    }


def evaluate_surrogate_curve(
    alphas: Iterable[float],
    Hc: Array,
    T: int,
    sigma_c2: float,
    sigma_s2: float,
    power_per_tx: float,
    phi_func: Callable[[Array], Array] | None = None,
    Jp: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Evaluate the explicit log-determinant surrogate over ``alphas``.

    Returns ``(crb, rate, covariance)`` arrays.  The last array has shape
    ``(len(alphas), M, M)``.  This helper performs no interpolation and makes
    no region-bound or paper-result claim.
    """

    weights = np.asarray(tuple(alphas), dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("alphas must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(weights)) or np.any((weights < 0) | (weights > 1)):
        raise ValueError("every alpha must lie in [0, 1]")

    channel = np.asarray(Hc, dtype=np.complex128)
    covariances = []
    crbs = []
    rates = []
    for weight in weights:
        covariance = covariance_shaping_surrogate(
            float(weight),
            power_per_tx,
            channel,
            sigma_c2,
        )
        covariances.append(covariance)
        crbs.append(
            compute_crb(
                covariance,
                T,
                sigma_s2,
                phi_func=phi_func,
                Jp=Jp,
            )
        )
        rates.append(compute_rate(covariance, channel, sigma_c2))
    return np.asarray(crbs), np.asarray(rates), np.asarray(covariances)
