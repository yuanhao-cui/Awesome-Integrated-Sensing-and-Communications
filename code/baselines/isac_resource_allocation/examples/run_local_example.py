"""Run one deterministic, local resource-allocation diagnostic.

This example exercises the repository's educational heuristic.  It does not
reproduce a paper figure or establish optimality.
"""

import os
import sys

import numpy as np

BASELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASELINE_DIR)

from src.ao_solver import AOSolver  # noqa: E402
from src.system_model import ISACSystem  # noqa: E402


def main() -> None:
    """Solve and print a deterministic synthetic detection allocation."""
    system = ISACSystem(
        Nt=32,
        Nr=32,
        Q=3,
        K=3,
        L=1,
        fc=30e9,
        P_total=40.0,
        B_total=100e6,
        rng=np.random.default_rng(42),
    )
    threshold = 1e6
    result = AOSolver(
        system,
        qos_type="detection",
        fairness="maxmin",
        max_iter=20,
    ).solve(Gamma_c=threshold)
    if result.comm_rates is None or result.detection_probs is None:
        raise RuntimeError("solver omitted required diagnostics")
    if result.diagnostics is None or result.objective_history is None:
        raise RuntimeError("solver omitted feasibility/convergence diagnostics")
    if not result.diagnostics["best_feasible_returned"]:
        raise RuntimeError("solver did not certify its returned feasible iterate")
    if np.any(result.comm_rates < threshold):
        raise RuntimeError("example result violates its rate threshold")

    print(f"iterations={result.iterations}, strict_convergence={result.converged}")
    print(f"phase_one={result.diagnostics['phase_one']}")
    print(f"objective_history={result.objective_history.tolist()}")
    print(f"minimum detection probability={np.min(result.detection_probs):.6f}")
    print(f"minimum communication rate={np.min(result.comm_rates) / 1e6:.6f} Mbit/s")
    print(f"power sum={np.sum(result.p):.6f} W")
    print(f"bandwidth sum={np.sum(result.b) / 1e6:.6f} MHz")


if __name__ == "__main__":
    main()
