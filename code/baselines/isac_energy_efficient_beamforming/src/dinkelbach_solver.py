"""Exact Dinkelbach reference for a declared single-user model slice.

This is not an implementation of the paper's multi-user Algorithm 1.  It
solves the scientifically auditable restriction in which the beam direction is
fixed and only scalar radiated power is optimized.  For one user, every inner
Dinkelbach problem is concave and has the closed-form optimizer implemented
below.  A dense-grid oracle is provided as an independent numerical check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ee_metrics import compute_crb, compute_ee_c, compute_sum_rate
from .system_model import ISACSystemModel


class InfeasibleReferenceProblem(ValueError):
    """Raised when explicit SINR/CRB requirements exceed the power budget."""


@dataclass(frozen=True)
class DinkelbachIteration:
    iteration: int
    lambda_before: float
    power_watt: float
    subtractive_residual: float
    energy_efficiency: float


@dataclass(frozen=True)
class DinkelbachResult:
    W: np.ndarray
    power_watt: float
    ee_c: float
    sum_rate: float
    sinr: float
    crb: float
    n_iterations: int
    converged: bool
    residual: float
    history: tuple[DinkelbachIteration, ...]

    @property
    def total_power(self) -> float:
        """Backward-readable alias for radiated power."""

        return self.power_watt

    @property
    def obj_history(self) -> list[float]:
        """Return EE values without pretending they are paper-figure data."""

        return [item.energy_efficiency for item in self.history]


def _unit_direction(model: ISACSystemModel, direction: np.ndarray | None) -> np.ndarray:
    if model.K != 1:
        raise ValueError("the validated reference slice requires K=1")
    if direction is None:
        direction = model.get_channel(0)
    direction = np.asarray(direction, dtype=complex)
    if direction.shape != (model.M,) or not np.all(np.isfinite(direction)):
        raise ValueError(f"direction must be a finite vector of shape {(model.M,)}")
    norm = float(np.linalg.norm(direction))
    if norm <= np.finfo(float).tiny:
        raise ValueError("direction must be non-zero")
    return direction / norm


def _beam(direction: np.ndarray, power_watt: float) -> np.ndarray:
    return np.sqrt(max(float(power_watt), 0.0)) * direction[:, None]


class SingleUserPowerDinkelbach:
    """Optimize (4) over scalar power for a fixed unit beam direction."""

    def __init__(
        self,
        model: ISACSystemModel,
        direction: np.ndarray | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-10,
    ) -> None:
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        self.model = model
        self.direction = _unit_direction(model, direction)
        self.max_iterations = max_iterations
        self.tolerance = float(tolerance)

    def _sensing_vectors(
        self, target_angle_deg: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        theta = np.deg2rad(float(target_angle_deg))
        if not np.isfinite(theta):
            raise ValueError("target_angle_deg must be finite")
        return (
            self.model.steering_vector_tx(theta),
            self.model.steering_vector_rx(theta),
            self.model.steering_derivative_tx(theta),
            self.model.steering_derivative_rx(theta),
        )

    def feasible_power_interval(
        self,
        target_angle_deg: float = 90.0,
        crb_max: float | None = None,
        gamma_min: float | None = None,
        alpha_abs: float = 1.0,
    ) -> tuple[float, float]:
        """Return the exact scalar feasible interval ``[p_min, P_max]``."""

        gain = float(np.abs(self.model.get_channel(0).conj() @ self.direction) ** 2)
        if gain <= np.finfo(float).tiny:
            raise InfeasibleReferenceProblem("the fixed direction has zero channel gain")
        p_min = 0.0
        if gamma_min is not None:
            gamma_min = float(gamma_min)
            if not np.isfinite(gamma_min) or gamma_min < 0.0:
                raise ValueError("gamma_min must be finite and non-negative")
            p_min = max(p_min, gamma_min * self.model.sigma_c2 / gain)
        if crb_max is not None:
            crb_max = float(crb_max)
            if not np.isfinite(crb_max) or crb_max <= 0.0:
                raise ValueError("crb_max must be finite and positive")
            a_t, a_r, da_t, da_r = self._sensing_vectors(target_angle_deg)
            unit_crb = compute_crb(
                _beam(self.direction, 1.0),
                a_t,
                a_r,
                da_t,
                da_r,
                self.model.sigma_s2,
                self.model.L,
                alpha_abs,
            )
            if not np.isfinite(unit_crb):
                raise InfeasibleReferenceProblem(
                    "the fixed direction contains no identifiable angle information"
                )
            p_min = max(p_min, unit_crb / crb_max)
        if p_min > self.model.P_max * (1.0 + 64.0 * np.finfo(float).eps):
            raise InfeasibleReferenceProblem(
                f"minimum feasible power {p_min:.12g} W exceeds "
                f"P_max={self.model.P_max:.12g} W"
            )
        return min(p_min, self.model.P_max), self.model.P_max

    def _inner_power(self, lambda_value: float, p_min: float, p_max: float) -> float:
        gain = float(np.abs(self.model.get_channel(0).conj() @ self.direction) ** 2)
        noise_over_gain = self.model.sigma_c2 / gain
        if lambda_value <= 0.0:
            return p_max
        stationary = self.model.epsilon / (lambda_value * np.log(2.0)) - noise_over_gain
        return float(np.clip(stationary, p_min, p_max))

    def solve(
        self,
        target_angle_deg: float = 90.0,
        crb_max: float | None = None,
        gamma_min: float | None = None,
        alpha_abs: float = 1.0,
    ) -> DinkelbachResult:
        """Solve the fixed-direction fractional program and postvalidate it."""

        p_min, p_max = self.feasible_power_interval(
            target_angle_deg, crb_max, gamma_min, alpha_abs
        )
        lambda_value = 0.0
        history: list[DinkelbachIteration] = []
        converged = False
        power = p_max
        residual = float("inf")

        for iteration in range(1, self.max_iterations + 1):
            power = self._inner_power(lambda_value, p_min, p_max)
            W = _beam(self.direction, power)
            numerator = compute_sum_rate(self.model.H, W, self.model.sigma_c2)
            denominator = power / self.model.epsilon + self.model.P0
            residual = numerator - lambda_value * denominator
            ee = numerator / denominator
            history.append(
                DinkelbachIteration(iteration, lambda_value, power, residual, ee)
            )
            residual_scale = max(1.0, abs(numerator), abs(lambda_value * denominator))
            if abs(residual) <= self.tolerance * residual_scale:
                converged = True
                break
            lambda_value = ee

        W = _beam(self.direction, power)
        sinr = self.model.compute_sinr(0, W)
        a_t, a_r, da_t, da_r = self._sensing_vectors(target_angle_deg)
        crb = compute_crb(
            W,
            a_t,
            a_r,
            da_t,
            da_r,
            self.model.sigma_s2,
            self.model.L,
            alpha_abs,
        )
        power_tolerance = 128.0 * np.finfo(float).eps * max(1.0, p_max)
        if not converged:
            raise RuntimeError("Dinkelbach iteration did not meet its residual tolerance")
        if power < p_min - power_tolerance or power > p_max + power_tolerance:
            raise RuntimeError("returned power violates the certified feasible interval")
        if gamma_min is not None and sinr + 1e-10 < float(gamma_min):
            raise RuntimeError("returned beam violates gamma_min")
        if crb_max is not None and crb > float(crb_max) * (1.0 + 1e-10):
            raise RuntimeError("returned beam violates crb_max")

        return DinkelbachResult(
            W=W,
            power_watt=power,
            ee_c=compute_ee_c(
                self.model.H,
                W,
                self.model.sigma_c2,
                self.model.epsilon,
                self.model.P0,
            ),
            sum_rate=compute_sum_rate(self.model.H, W, self.model.sigma_c2),
            sinr=sinr,
            crb=crb,
            n_iterations=len(history),
            converged=converged,
            residual=residual,
            history=tuple(history),
        )

    def dense_grid_oracle(
        self,
        target_angle_deg: float = 90.0,
        crb_max: float | None = None,
        gamma_min: float | None = None,
        alpha_abs: float = 1.0,
        n_points: int = 200_001,
    ) -> tuple[float, float, float]:
        """Return ``(power, EE, grid_spacing)`` from an independent grid search."""

        if not isinstance(n_points, int) or n_points < 2:
            raise ValueError("n_points must be an integer of at least two")
        p_min, p_max = self.feasible_power_interval(
            target_angle_deg, crb_max, gamma_min, alpha_abs
        )
        powers = np.linspace(p_min, p_max, n_points)
        gain = float(np.abs(self.model.get_channel(0).conj() @ self.direction) ** 2)
        rates = np.log1p(gain * powers / self.model.sigma_c2) / np.log(2.0)
        efficiencies = rates / (powers / self.model.epsilon + self.model.P0)
        index = int(np.argmax(efficiencies))
        spacing = float((p_max - p_min) / (n_points - 1))
        return float(powers[index]), float(efficiencies[index]), spacing


__all__ = [
    "DinkelbachIteration",
    "DinkelbachResult",
    "InfeasibleReferenceProblem",
    "SingleUserPowerDinkelbach",
]
