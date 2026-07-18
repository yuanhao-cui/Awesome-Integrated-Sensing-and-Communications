"""Algebraic circle fitting for complex CSI-ratio samples.

The fitted parameters solve

``min sum_i (x_i**2 + y_i**2 - 2 A x_i - 2 B y_i - C)**2``

and ``r = sqrt(C + A**2 + B**2)``.  This is an algebraic residual,
not the geometric-distance objective.
"""

from typing import Tuple

import numpy as np


def _circle_system(R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validate samples and return the algebraic design system."""
    samples = np.asarray(R, dtype=complex)
    if samples.ndim != 1 or samples.size < 3 or not np.all(np.isfinite(samples)):
        raise ValueError("R must be a finite one-dimensional array with at least 3 samples")
    x = samples.real
    y = samples.imag
    design = np.column_stack([2.0 * x, 2.0 * y, np.ones(samples.size)])
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("circle fit is degenerate; samples must not be constant or collinear")
    target = x**2 + y**2
    return design, target


def _normalize_samples(R: np.ndarray) -> Tuple[np.ndarray, complex, float]:
    """Center by the sample mean and scale by spread before fitting.

    Scaling only by ``max(abs(R))`` leaves the algebraic radius calculation
    ill-conditioned when a small identifiable circle is far from the origin.
    The two-stage scaling also keeps the mean calculation finite for large
    inputs.
    """
    samples = np.asarray(R, dtype=complex)
    if samples.ndim != 1 or samples.size < 3 or not np.all(np.isfinite(samples)):
        raise ValueError("R must be a finite one-dimensional array with at least 3 samples")
    global_scale = float(np.max(np.abs(samples)))
    if global_scale <= 0:
        raise ValueError("circle fit is degenerate; samples must not be constant")
    globally_scaled = samples / global_scale
    offset_scaled = complex(np.mean(globally_scaled))
    centered = globally_scaled - offset_scaled
    spread_scaled = float(np.max(np.abs(centered)))
    if spread_scaled <= np.finfo(float).eps:
        raise ValueError("circle fit is degenerate; samples must not be constant")
    offset = offset_scaled * global_scale
    spread = spread_scaled * global_scale
    return centered / spread_scaled, offset, spread


def _parameters_from_solution(theta: np.ndarray) -> Tuple[float, float, float]:
    A, B, C = (float(value) for value in theta)
    radius_squared = C + A**2 + B**2
    scale = max(abs(C), A**2 + B**2, np.finfo(float).tiny)
    if not np.isfinite(radius_squared) or radius_squared <= np.finfo(float).eps * scale:
        raise ValueError("circle fit has zero or invalid radius")
    return A, B, float(np.sqrt(radius_squared))


def least_squares_circle_fit(R: np.ndarray) -> Tuple[float, float, float]:
    """Fit a circle by unweighted algebraic least squares."""
    samples, offset, scale = _normalize_samples(R)
    design, target = _circle_system(samples)
    theta, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise ValueError("circle fit is rank deficient")
    A, B, radius = _parameters_from_solution(theta)
    return (
        offset.real + A * scale,
        offset.imag + B * scale,
        radius * scale,
    )


def fit_circle_kasa(R: np.ndarray) -> Tuple[float, float, float]:
    """Alias the same algebraic fit commonly called Kasa's method.

    This function is retained to make the method name explicit.  It is
    mathematically the same objective as :func:`least_squares_circle_fit`.
    """
    return least_squares_circle_fit(R)


def fit_circle_iterative_weighted(
    R: np.ndarray,
    max_iter: int = 50,
    tolerance: float = 1e-10,
) -> Tuple[float, float, float]:
    """Iteratively reweight the algebraic residual by inverse radial distance.

    This local refinement is not a Pratt or Taubin constrained algebraic fit;
    the descriptive name is intentional.
    """
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    samples, offset, sample_scale = _normalize_samples(R)
    design, target = _circle_system(samples)
    theta, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise ValueError("circle fit is rank deficient")
    A, B, radius = _parameters_from_solution(theta)

    for _ in range(max_iter):
        distances = np.abs(samples - (A + 1j * B))
        scale = max(float(np.max(distances)), np.finfo(float).tiny)
        weights = 1.0 / np.maximum(distances, np.finfo(float).eps * scale)
        sqrt_weights = np.sqrt(weights / np.sum(weights))
        weighted_design = design * sqrt_weights[:, None]
        weighted_target = target * sqrt_weights
        theta, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
        if rank < 3:
            raise ValueError("weighted circle fit is rank deficient")
        A_new, B_new, radius_new = _parameters_from_solution(theta)
        if max(abs(A_new - A), abs(B_new - B), abs(radius_new - radius)) < tolerance:
            return (
                offset.real + A_new * sample_scale,
                offset.imag + B_new * sample_scale,
                radius_new * sample_scale,
            )
        A, B, radius = A_new, B_new, radius_new
    return (
        offset.real + A * sample_scale,
        offset.imag + B * sample_scale,
        radius * sample_scale,
    )


def circle_fit_error(R: np.ndarray, A: float, B: float, r: float) -> float:
    """Return the RMS geometric radial residual for a supplied circle."""
    samples = np.asarray(R, dtype=complex)
    if samples.ndim != 1 or samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("R must be a non-empty finite one-dimensional array")
    if not all(np.isfinite(value) for value in (A, B, r)) or r <= 0:
        raise ValueError("circle center must be finite and radius must be positive")
    center = A + 1j * B
    centered = samples - center
    scale = max(float(np.max(np.abs(centered))), r)
    distances = np.abs(centered / scale)
    return float(np.sqrt(np.mean((distances - r / scale) ** 2)) * scale)
