"""Direct fixed-phase SOCP for the educational RIS feasibility surrogate."""

import numpy as np
import cvxpy as cp
from typing import Optional, Tuple

from .system_model import RIS_ISAC_System
from .numerics import stable_squared_norm

# Prefer deterministic open-source solvers that are actually installed.  MOSEK
# remains usable when explicitly installed, but merely importing the CVXPY
# constant does not mean that a licensed solver is available.
_SOLVER_PREF = ("CLARABEL", "SCS", "MOSEK")


def _solve_problem(prob: cp.Problem) -> float:
    """Solve a CVXPY problem with solver fallback.

    Tries installed solvers in a stable order and accepts only a reported
    optimum.  An inaccurate or failed solve is surfaced to the caller; the
    caller must never drop a scientific constraint to make the run succeed.

    Args:
        prob: CVXPY problem to solve.

    Returns:
        Optimal objective value.
    """
    installed = set(cp.installed_solvers())
    attempts = []
    for solver in _SOLVER_PREF:
        if solver not in installed:
            continue
        try:
            prob.solve(solver=solver, verbose=False)
            attempts.append(f"{solver}:{prob.status}")
            if prob.status == cp.OPTIMAL:
                return prob.value
        except (cp.error.SolverError, ImportError) as exc:
            attempts.append(f"{solver}:{type(exc).__name__}")
            continue
    detail = ", ".join(attempts) or "no supported solver installed"
    raise RuntimeError(f"Convex subproblem did not reach an optimum ({detail})")


class BeamformingOptimizer:
    """Educational downlink beamforming feasibility optimizer.

    The minimum-power problem is represented directly as an SOCP after fixing
    each desired-signal phase. ``solve_full_power_feasible`` is a power-scaled
    feasible-beamformer heuristic, not a sum-rate optimizer.

    Attributes:
        system: RIS_ISAC_System instance.
        M: Number of BS antennas.
        K: Number of users.
        P_max: Power budget.
    """

    def __init__(self, system: RIS_ISAC_System):
        """Initialize beamforming optimizer.

        Args:
            system: The RIS-ISAC system model (with fixed RIS phases).
        """
        self.system = system
        self.M = system.M
        self.K = system.K
        self.P_max = system.P_max

    def _get_effective_channels(self) -> np.ndarray:
        """Get effective channels for all users.

        Returns:
            Matrix of shape (K, M) whose rows store the column-channel
            coefficients returned by ``effective_channel``.
        """
        H_eff = np.zeros((self.K, self.M), dtype=complex)
        for k in range(self.K):
            H_eff[k, :] = self.system.effective_channel(k)
        return H_eff

    def solve_min_power(
        self,
        sinr_thresholds: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Minimum-power beamforming satisfying SINR constraints.

        With ``Im(h_k^H w_k)=0``, each SINR constraint has a standard
        second-order-cone representation.  This avoids rank-relaxation and
        lifted-variable ambiguity.

        Args:
            sinr_thresholds: SINR thresholds γ_k (K,).

        Returns:
            Tuple of (W, min_power) where W is (M, K) beamforming matrix.
        """
        H_eff = self._get_effective_channels()
        sigma2 = self.system.noise_power

        thresholds = np.asarray(sinr_thresholds, dtype=float)
        if (
            thresholds.shape != (self.K,)
            or not np.all(np.isfinite(thresholds))
            or np.any(thresholds < 0)
        ):
            raise ValueError(f"sinr_thresholds must have shape ({self.K},)")

        W_var = cp.Variable((self.M, self.K), complex=True)
        constraints = [cp.sum_squares(cp.abs(W_var)) <= self.P_max]
        for k in range(self.K):
            h_k = H_eff[k, :]
            desired = h_k.conj() @ W_var[:, k]
            interference = [
                h_k.conj() @ W_var[:, j]
                for j in range(self.K)
                if j != k
            ]
            constraints.extend([
                cp.imag(desired) == 0,
                cp.real(desired)
                >= np.sqrt(thresholds[k])
                * cp.norm(cp.hstack(interference + [np.sqrt(sigma2)]), 2),
            ])

        objective = cp.sum_squares(cp.abs(W_var))
        problem = cp.Problem(cp.Minimize(objective), constraints)
        min_power = _solve_problem(problem)
        if W_var.value is None:
            raise RuntimeError("SOCP solver returned no beamforming solution")
        W = np.asarray(W_var.value)
        self._validate_physical_solution(W, thresholds)
        return W, float(min_power)

    def solve_full_power_feasible(
        self,
        sinr_thresholds: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Return a full-power feasible-beamformer heuristic.

        The method solves the minimum-power SOCP and scales all beams by a
        common factor.  Scaling preserves interference ratios and improves the
        noise-limited SINRs.  It does not optimize the exact sum-rate objective.

        Args:
            sinr_thresholds: Optional SINR constraints γ_k (K,).
        Returns:
            Tuple of (W, sum_rate).
        """
        thresholds = (
            np.zeros(self.K)
            if sinr_thresholds is None
            else np.asarray(sinr_thresholds, dtype=float)
        )
        W, used_power = self.solve_min_power(thresholds)
        if used_power <= 0:
            raise RuntimeError("Minimum-power solve produced a zero solution")
        W *= np.sqrt(self.P_max / used_power)
        self._validate_physical_solution(W, thresholds)
        return W, self.system.compute_sum_rate(W)

    def _validate_physical_solution(
        self, W: np.ndarray, sinr_thresholds: np.ndarray, tolerance: float = 5e-4
    ) -> None:
        """Validate recovered/returned vectors against physical constraints."""
        if W.shape != (self.M, self.K) or not np.all(np.isfinite(W)):
            raise RuntimeError("Beamforming solution has invalid shape or values")
        power = stable_squared_norm(W)
        if power > self.P_max * (1.0 + tolerance):
            raise RuntimeError(
                f"Beamforming power constraint violated: {power:.6g} > {self.P_max:.6g}"
            )
        for k, threshold in enumerate(sinr_thresholds):
            interferers = np.delete(W, k, axis=1)
            sinr = self.system.compute_sinr(k, W[:, k], interferers)
            if sinr < threshold * (1.0 - tolerance):
                raise RuntimeError(
                    "Beamforming solution infeasible: "
                    f"user {k} SINR {sinr:.6g} < {threshold:.6g}"
                )
