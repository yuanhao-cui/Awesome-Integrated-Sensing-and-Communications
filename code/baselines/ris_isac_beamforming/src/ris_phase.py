"""Deterministic unit-modulus phase updates for the local RIS surrogate."""

from __future__ import annotations

import numpy as np

from .numerics import sensing_coordinate_phase_candidate
from .system_model import RIS_ISAC_System


class RISPhaseOptimizer:
    """Coordinate phase updates with physical-objective monotonicity checks."""

    def __init__(self, system: RIS_ISAC_System):
        self.system = system
        self.L = system.L

    def _coordinate_ascent(
        self,
        objective,
        candidates_per_element: int,
        max_sweeps: int,
        tolerance: float = 1e-12,
    ) -> np.ndarray:
        if candidates_per_element < 2 or max_sweeps < 1:
            raise ValueError("grid size and max_sweeps must be positive")
        theta = self.system.theta.copy()
        self.system.set_ris_phases(theta)
        current = float(objective())
        if not np.isfinite(current):
            raise RuntimeError("initial RIS objective is not finite")
        grid = np.exp(
            1j
            * np.linspace(
                0.0,
                2.0 * np.pi,
                candidates_per_element,
                endpoint=False,
            )
        )

        for _ in range(max_sweeps):
            sweep_start = current
            for element in range(self.L):
                best_phase = theta[element]
                best_value = current
                for candidate in grid:
                    trial = theta.copy()
                    trial[element] = candidate
                    self.system.set_ris_phases(trial)
                    value = float(objective())
                    improvement_scale = max(
                        abs(value),
                        abs(best_value),
                        np.nextafter(0.0, 1.0),
                    )
                    if value > best_value + tolerance * improvement_scale:
                        best_value = value
                        best_phase = candidate
                theta[element] = best_phase
                self.system.set_ris_phases(theta)
                current = best_value
            sweep_scale = max(
                abs(current),
                abs(sweep_start),
                np.nextafter(0.0, 1.0),
            )
            if current <= sweep_start + tolerance * sweep_scale:
                break

        self.system.set_ris_phases(theta)
        final = float(objective())
        final_scale = max(
            abs(final), abs(current), np.nextafter(0.0, 1.0)
        )
        if final + tolerance * final_scale < current:
            raise RuntimeError("RIS coordinate update failed its monotonicity check")
        return theta.copy()

    def optimize_for_rate(
        self,
        W: np.ndarray,
        candidates_per_element: int = 16,
        max_sweeps: int = 50,
    ) -> np.ndarray:
        """Monotonically increase the local surrogate's sum rate."""

        W = np.asarray(W, dtype=complex)
        if W.shape != (self.system.M, self.system.K):
            raise ValueError(
                f"W must have shape {(self.system.M, self.system.K)}"
            )
        before = self.system.compute_sum_rate(W)
        theta = self._coordinate_ascent(
            lambda: self.system.compute_sum_rate(W),
            candidates_per_element,
            max_sweeps,
        )
        after = self.system.compute_sum_rate(W)
        if after + 1e-10 < before:
            raise RuntimeError("rate-oriented RIS update decreased sum rate")
        return theta

    def optimize_for_snr(
        self,
        W: np.ndarray,
        max_sweeps: int = 20,
        tolerance: float = 1e-12,
    ) -> tuple[np.ndarray, float]:
        """Monotonically improve independent-stream sensing power.

        The objective is

        ``sum_k |h_s(theta)^H w_k|^2 / sigma^2``.

        A single phase generally cannot align the reflected terms of every
        independent stream simultaneously, so no joint global optimum is
        claimed.  Holding all other phases fixed reduces one coordinate to
        ``2 Re{conj(theta_l) C_l}``; its exact maximizer is the phase of the
        cross-stream coefficient ``C_l``.  Each candidate is post-evaluated
        with the physical streamwise SNR and rolled back if binary64 phase
        rounding would decrease it.  Thus every accepted coordinate and the
        complete update are non-decreasing.
        """

        W = np.asarray(W, dtype=complex)
        if W.shape != (self.system.M, self.system.K):
            raise ValueError(
                f"W must have shape {(self.system.M, self.system.K)}"
            )
        if not isinstance(max_sweeps, (int, np.integer)) or max_sweeps < 1:
            raise ValueError("max_sweeps must be a positive integer")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite")
        before = self.system.compute_snr_sensing(W)
        current = before
        theta = self.system.theta.copy()
        for _ in range(max_sweeps):
            sweep_start = current
            for element in range(self.L):
                candidate = sensing_coordinate_phase_candidate(
                    self.system.channels["a_bs"],
                    self.system.channels["a_ris"],
                    self.system.channels["H_BR"],
                    W,
                    theta,
                    element,
                )
                trial = theta.copy()
                trial[element] = candidate
                self.system.set_ris_phases(trial)
                value = self.system.compute_snr_sensing(W)
                if value >= current:
                    theta = self.system.theta.copy()
                    current = value
                else:
                    self.system.set_ris_phases(theta)
            scale = max(
                abs(current),
                abs(sweep_start),
                np.nextafter(0.0, 1.0),
            )
            if current <= sweep_start + tolerance * scale:
                break

        self.system.set_ris_phases(theta)
        achieved = self.system.compute_snr_sensing(W)
        if achieved < before:
            raise RuntimeError("sensing-oriented RIS update decreased SNR")
        return self.system.theta.copy(), achieved

    def optimize_joint(
        self,
        W: np.ndarray,
        sensing_weight: float = 0.5,
        candidates_per_element: int = 12,
        max_sweeps: int = 30,
    ) -> np.ndarray:
        """Increase a declared normalized rate/SNR scalarization.

        Raw rate and SNR have incompatible units and scales.  Each term is
        therefore normalized by its value at the input phases before applying
        the user-supplied dimensionless weight.
        """

        if not 0.0 <= sensing_weight <= 1.0:
            raise ValueError("sensing_weight must lie in [0, 1]")
        W = np.asarray(W, dtype=complex)
        if W.shape != (self.system.M, self.system.K):
            raise ValueError(
                f"W must have shape {(self.system.M, self.system.K)}"
            )
        rate_scale = max(self.system.compute_sum_rate(W), 1e-15)
        snr_scale = max(self.system.compute_snr_sensing(W), 1e-15)

        def objective() -> float:
            rate = self.system.compute_sum_rate(W) / rate_scale
            snr = self.system.compute_snr_sensing(W) / snr_scale
            return (1.0 - sensing_weight) * rate + sensing_weight * snr

        return self._coordinate_ascent(
            objective,
            candidates_per_element,
            max_sweeps,
        )


__all__ = ["RISPhaseOptimizer"]
