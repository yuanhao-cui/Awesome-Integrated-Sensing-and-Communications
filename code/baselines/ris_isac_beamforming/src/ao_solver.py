"""Unified interface for the supported RIS feasibility surrogate."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .snr_constraint import SNRConstrainedSolver
from .system_model import RIS_ISAC_System
from .numerics import stable_squared_norm


class AlternatingOptimizationSolver:
    """Dispatch the supported SNR-constrained feasibility iteration.

    The earlier ``problem_type='crb'`` path was removed because its scalar
    derivative proxy was not the paper's two-angle, nuisance-RCS Fisher
    information model.  Unsupported scientific models fail explicitly.
    """

    def __init__(
        self,
        system: RIS_ISAC_System,
        problem_type: Literal["snr"] = "snr",
        snr_min_dB: float = 5.0,
        max_iter: int = 50,
        tol: float = 1e-4,
        **unsupported: object,
    ) -> None:
        if problem_type != "snr":
            raise ValueError(
                "Only problem_type='snr' is supported; the former CRB proxy "
                "was not the cited paper's CRB model"
            )
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise TypeError(f"unsupported arguments for the SNR surrogate: {names}")
        self.system = system
        self.problem_type = problem_type
        self.max_iter = max_iter
        self.tol = tol
        self._solver = SNRConstrainedSolver(
            system,
            snr_min_dB=snr_min_dB,
            max_iter=max_iter,
            tol=tol,
        )

    def solve(self) -> dict:
        """Return a post-validated local-surrogate solution."""

        return self._solver.solve()

    def evaluate(self, W: np.ndarray, theta: np.ndarray) -> dict:
        """Evaluate power, rate, SINR, and sensing SNR for one tuple."""

        W = np.asarray(W, dtype=complex)
        if W.shape != (self.system.M, self.system.K):
            raise ValueError(
                f"W must have shape {(self.system.M, self.system.K)}"
            )
        self.system.set_ris_phases(theta)
        sinrs = []
        for user in range(self.system.K):
            interferers = np.delete(W, user, axis=1)
            sinrs.append(
                self.system.compute_sinr(user, W[:, user], interferers)
            )
        return {
            "sum_rate": self.system.compute_sum_rate(W),
            "snr_sensing": self.system.compute_snr_sensing(W),
            "power_used": stable_squared_norm(W),
            "sinr_per_user": np.asarray(sinrs),
        }


__all__ = ["AlternatingOptimizationSolver"]
