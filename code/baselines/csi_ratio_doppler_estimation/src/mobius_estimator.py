"""Circle-phase rotation estimator for CSI-ratio trajectories.

This is a paper-inspired educational adapter: it fits the algebraic circle,
centres the samples, unwraps their observed angle, and fits a weighted line.
A general Mobius map traverses its circle non-uniformly, so the fitted slope is
an observation-window rotation proxy rather than an exact paper-algorithm
reproduction.

The observed rotation sign cannot universally be mapped to physical
approaching/receding direction: that mapping also depends on the complex
exponential convention and on static/dynamic path dominance.
"""

from typing import Dict, Tuple

import numpy as np

from .circle_fit import (
    circle_fit_error,
    fit_circle_iterative_weighted,
    fit_circle_kasa,
    least_squares_circle_fit,
)


def mobius_doppler_estimate(
    R: np.ndarray,
    T_s: float,
    circle_method: str = "least_squares",
    unwrap_phases: bool = True,
    min_angular_coverage_rad: float = np.pi,
    min_r_squared: float = 0.95,
) -> Dict[str, object]:
    """Estimate the signed angular-rotation rate of a ratio trajectory.

    ``f_D`` is retained as a compatibility key.  It is the signed observed
    circle-rotation proxy in hertz, not a universal physical direction label.
    ``direction`` is therefore always ``"unknown"`` and ``rotation_sign``
    reports only the sign of the fitted trajectory rotation.
    """
    samples = np.asarray(R, dtype=complex)
    if samples.ndim != 1 or samples.size < 4 or not np.all(np.isfinite(samples)):
        raise ValueError("R must be a finite one-dimensional array with at least 4 samples")
    if not np.isfinite(T_s) or T_s <= 0:
        raise ValueError("T_s must be positive and finite")
    if not isinstance(unwrap_phases, (bool, np.bool_)):
        raise TypeError("unwrap_phases must be boolean")
    if not np.isfinite(min_angular_coverage_rad) or min_angular_coverage_rad <= 0:
        raise ValueError("min_angular_coverage_rad must be positive and finite")
    if not np.isfinite(min_r_squared) or not 0 <= min_r_squared <= 1:
        raise ValueError("min_r_squared must lie in [0, 1]")

    fitters = {
        "least_squares": least_squares_circle_fit,
        "kasa": fit_circle_kasa,
        "iterative_weighted": fit_circle_iterative_weighted,
    }
    if circle_method not in fitters:
        choices = ", ".join(sorted(fitters))
        raise ValueError(f"unknown circle_method {circle_method!r}; choose one of: {choices}")
    A, B, radius = fitters[circle_method](samples)

    centered = (samples - (A + 1j * B)) / radius
    if radius <= 0 or not np.all(np.isfinite(centered)):
        raise ValueError("ratio trajectory is stationary or has a degenerate circle")
    angles = np.angle(centered)
    if unwrap_phases:
        angles = np.unwrap(angles)
    times = np.arange(samples.size, dtype=float) * T_s
    beta_0, beta_1, r_squared = _weighted_linear_regression(
        times, angles, np.abs(centered)
    )
    angular_coverage = float(np.ptp(angles))
    if angular_coverage < min_angular_coverage_rad or r_squared < min_r_squared:
        raise ValueError(
            "circle-phase trajectory is invalid for frequency estimation: "
            f"angular coverage={angular_coverage:.6g} rad "
            f"(minimum {min_angular_coverage_rad:.6g}), "
            f"R^2={r_squared:.6g} (minimum {min_r_squared:.6g})"
        )
    rotation_frequency = float(beta_1 / (2.0 * np.pi))
    sign_tolerance = np.finfo(float).eps / T_s
    rotation_sign = 0 if abs(rotation_frequency) <= sign_tolerance else int(
        np.sign(rotation_frequency)
    )

    return {
        "f_D": rotation_frequency,
        "rotation_frequency_hz": rotation_frequency,
        "f_D_magnitude": abs(rotation_frequency),
        "rotation_sign": rotation_sign,
        "direction": "unknown",
        "alias_limit_hz": 0.5 / T_s,
        "alias_ambiguous": True,
        "valid": True,
        "angular_coverage_rad": angular_coverage,
        "min_angular_coverage_rad": min_angular_coverage_rad,
        "min_r_squared": min_r_squared,
        "center_A": A,
        "center_B": B,
        "radius": radius,
        "beta_0": beta_0,
        "beta_1": beta_1,
        "r_squared": r_squared,
        "rms_error": circle_fit_error(samples, A, B, radius),
    }


def _weighted_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float, float]:
    """Fit ``y = beta_0 + beta_1 x`` with validated non-negative weights."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if x.ndim != 1 or x.size < 2 or x.shape != y.shape or x.shape != weights.shape:
        raise ValueError("x, y, and weights must be equal 1-D arrays with at least 2 samples")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("regression inputs must be finite")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or np.sum(weights) <= 0:
        raise ValueError("weights must be finite, non-negative, and have positive sum")

    normalized = weights / np.sum(weights)
    x_span = float(np.max(x) - np.min(x))
    if x_span <= np.finfo(float).tiny:
        raise ValueError("regression time axis has zero numerical variance")
    x_scaled = (x - np.min(x)) / x_span
    x_scaled_mean = float(np.sum(normalized * x_scaled))
    y_mean = float(np.sum(normalized * y))
    x_centered = x_scaled - x_scaled_mean
    y_centered = y - y_mean
    s_xx = float(np.sum(normalized * x_centered**2))
    if s_xx <= np.finfo(float).eps:
        raise ValueError("regression time axis has zero numerical variance")
    scaled_slope = float(np.sum(normalized * x_centered * y_centered) / s_xx)
    beta_1 = scaled_slope / x_span
    beta_0 = float(y_mean - beta_1 * np.sum(normalized * x))

    prediction = beta_0 + beta_1 * x
    ss_res = float(np.sum(normalized * (y - prediction) ** 2))
    ss_tot = float(np.sum(normalized * y_centered**2))
    r_squared = 1.0 if ss_tot <= np.finfo(float).eps else 1.0 - ss_res / ss_tot
    return beta_0, beta_1, float(r_squared)
