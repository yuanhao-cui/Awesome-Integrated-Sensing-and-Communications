"""Feasibility-first alternating optimizer for the local teaching surrogate."""

from dataclasses import dataclass
from typing import Dict, Optional
import warnings

import numpy as np
from scipy.optimize import minimize

from .comm_rate import CommunicationRate
from .detection_qos import DetectionQoS
from .fairness import FairnessType
from .localization_qos import LocalizationQoS
from .system_model import ISACSystem
from .tracking_qos import TrackingQoS


@dataclass
class AOResult:
    """Best feasible iterate returned by the local alternating solver."""

    p: np.ndarray
    b: np.ndarray
    objective: float
    iterations: int
    converged: bool
    detection_probs: Optional[np.ndarray] = None
    localization_rho: Optional[np.ndarray] = None
    tracking_pcrb: Optional[np.ndarray] = None
    comm_rates: Optional[np.ndarray] = None
    diagnostics: Optional[Dict[str, object]] = None
    objective_history: Optional[np.ndarray] = None


class AOSolver:
    """Return a feasible, monotonic local iterate without global claims.

    Every power and bandwidth step evaluates the same public sensing objective
    as :meth:`_compute_current_objective`.  A failed numerical subproblem keeps
    its feasible starting point and prevents a ``converged`` claim.  The solver
    stores the best feasible iterate and never returns a lower-objective point.
    """

    def __init__(
        self,
        system: ISACSystem,
        qos_type: str = "detection",
        fairness: str = "maxmin",
        max_iter: int = 50,
        tol: float = 1.0e-4,
    ):
        """Initialize the local solver."""
        if qos_type not in {"detection", "localization", "tracking"}:
            raise ValueError(
                "qos_type must be 'detection', 'localization', or 'tracking'"
            )
        if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be positive and finite")
        self.system = system
        self.qos_type = qos_type
        self.fairness_type = self._coerce_supported_fairness(fairness)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.min_bandwidth_fraction = 1.0e-9

        self.detection_qos = DetectionQoS(system)
        self.localization_qos = LocalizationQoS(system)
        self.tracking_qos = TrackingQoS(
            system, localization_qos=self.localization_qos
        )
        self.comm_rate = CommunicationRate(system)
        self._phase_one_status = "not run"
        self._power_status = "not run"
        self._bandwidth_status = "not run"
        self._power_success = False
        self._bandwidth_success = False

    @staticmethod
    def _coerce_supported_fairness(fairness: object) -> FairnessType:
        """Normalize and restrict the implemented objective aggregations."""
        try:
            normalized = (
                fairness
                if isinstance(fairness, FairnessType)
                else FairnessType(fairness)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("fairness must be 'maxmin' or 'sum'") from error
        if normalized not in (FairnessType.MAXMIN, FairnessType.SUM):
            raise ValueError("AOSolver implements only 'maxmin' and 'sum' fairness")
        return normalized

    @staticmethod
    def _run_slsqp(*args: object, **kwargs: object) -> object:
        """Run SLSQP while containing its benign bound-clipping warning.

        SciPy 1.13 can propose a floating-point trial value infinitesimally
        outside a declared bound, then clip it before evaluating the supplied
        function.  That implementation detail emits a ``RuntimeWarning`` even
        though the candidate passed back to this class is clipped.  Suppress
        only that exact upstream warning; every returned candidate is still
        checked explicitly for feasibility and objective non-decrease.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Values in x were outside bounds during a minimize step, "
                    "clipping to bounds"
                ),
                category=RuntimeWarning,
            )
            return minimize(*args, **kwargs)

    def _minimum_rate_power(
        self, bandwidth: np.ndarray, beta: np.ndarray, gamma: float
    ) -> np.ndarray:
        """Invert the local Shannon rate with an explicit numerical margin."""
        bandwidth = np.asarray(bandwidth, dtype=float)
        beta = np.asarray(beta, dtype=float)
        if bandwidth.shape != beta.shape:
            raise ValueError("bandwidth and beta must have the same shape")
        if np.any(bandwidth <= 0.0) or not np.all(np.isfinite(bandwidth)):
            raise ValueError("bandwidth allocations must be positive and finite")
        if np.any(beta <= 0.0) or not np.all(np.isfinite(beta)):
            raise ValueError("channel gains must be positive and finite")
        if not np.isfinite(gamma) or gamma < 0.0:
            raise ValueError("gamma must be non-negative and finite")
        exponent = np.log(2.0) * gamma / bandwidth
        with np.errstate(over="ignore", invalid="ignore"):
            required = (
                np.expm1(exponent) * self.system.N0 * bandwidth / beta
            )
        required = required * (1.0 + 1.0e-7) + np.finfo(float).tiny
        return required

    def _communication_rates(self, p: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return all local per-user communication rates."""
        sensing_count = self.system.params.Q
        communication_count = self.system.params.K
        return np.concatenate(
            [
                self.comm_rate.compute_rate(
                    p[sensing_count : sensing_count + communication_count],
                    b[sensing_count : sensing_count + communication_count],
                    "comm",
                ),
                self.comm_rate.compute_rate(
                    p[sensing_count + communication_count :],
                    b[sensing_count + communication_count :],
                    "isac",
                ),
            ]
        )

    def _minimum_communication_power(
        self, bandwidth: np.ndarray, gamma: float
    ) -> np.ndarray:
        """Return the full vector containing only minimum user powers."""
        sensing_count = self.system.params.Q
        communication_count = self.system.params.K
        power = np.zeros(self.system.total_objects, dtype=float)
        power[sensing_count : sensing_count + communication_count] = (
            self._minimum_rate_power(
                bandwidth[sensing_count : sensing_count + communication_count],
                self.system.beta_comm,
                gamma,
            )
        )
        power[sensing_count + communication_count :] = self._minimum_rate_power(
            bandwidth[sensing_count + communication_count :],
            self.system.beta_isac,
            gamma,
        )
        return power

    def _bandwidth_supports_rate_floor(
        self, bandwidth: np.ndarray, gamma: float
    ) -> bool:
        """Check whether minimum user powers fit within the total budget."""
        required = self._minimum_communication_power(bandwidth, gamma)
        return bool(
            np.all(np.isfinite(required))
            and np.sum(required) <= self.system.params.P_total
        )

    def _phase_one_bandwidth(self, gamma: float) -> np.ndarray:
        """Find a rate-feasible bandwidth start when uniform allocation fails."""
        total_count = self.system.total_objects
        sensing_count = self.system.params.Q
        user_count = total_count - sensing_count
        total_bandwidth = self.system.params.B_total
        floor = self.min_bandwidth_fraction
        user_budget = 1.0 - sensing_count * floor
        if user_count < 1 or user_budget <= user_count * floor:
            raise RuntimeError("Phase-I bandwidth problem has no positive simplex")

        user_gains = np.concatenate(
            [self.system.beta_comm, self.system.beta_isac]
        )
        initial = np.full(user_count, user_budget / user_count)

        def objective(user_fraction: np.ndarray) -> float:
            required = self._minimum_rate_power(
                user_fraction * total_bandwidth, user_gains, gamma
            )
            if not np.all(np.isfinite(required)):
                return np.finfo(float).max
            return float(np.sum(required) / max(self.system.params.P_total, 1.0))

        result = self._run_slsqp(
            objective,
            initial,
            method="SLSQP",
            bounds=[(floor, user_budget)] * user_count,
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda value: np.sum(value) - user_budget,
                }
            ],
            options={"maxiter": 500, "ftol": 1.0e-12, "disp": False},
        )
        candidate = result.x if result.success else initial
        fraction = np.concatenate(
            [np.full(sensing_count, floor, dtype=float), candidate]
        )
        fraction[np.argmax(fraction)] += 1.0 - np.sum(fraction)
        bandwidth = fraction * total_bandwidth
        if not self._bandwidth_supports_rate_floor(bandwidth, gamma):
            raise RuntimeError(
                "Phase-I did not find a bandwidth iterate whose minimum-rate "
                "powers fit the total power budget"
            )
        self._phase_one_status = (
            "optimized feasible start" if result.success else "feasible equal-user start"
        )
        return bandwidth

    def _initialize_feasible_bandwidth(self, gamma: float) -> np.ndarray:
        """Prefer uniform bandwidth and invoke Phase-I only when necessary."""
        uniform = np.full(
            self.system.total_objects,
            self.system.params.B_total / self.system.total_objects,
        )
        if self._bandwidth_supports_rate_floor(uniform, gamma):
            self._phase_one_status = "uniform start is rate feasible"
            return uniform
        return self._phase_one_bandwidth(gamma)

    def _power_objective_coefficients(self, bandwidth: np.ndarray) -> np.ndarray:
        """Return exact fixed-bandwidth coefficients when the score is linear."""
        sensing_count = self.system.params.Q
        sensing_bandwidth = np.asarray(bandwidth[:sensing_count], dtype=float)
        if self.qos_type == "detection":
            return (
                self.system.beta_sensing
                * self.system.rcs
                / (self.system.N0 * sensing_bandwidth)
            )
        if self.qos_type == "localization":
            return self.localization_qos.compute_information_score_coefficients(
                sensing_bandwidth
            )
        raise ValueError("tracking has no declared linear power coefficient")

    def _correct_power_budget(self, power: np.ndarray) -> np.ndarray:
        """Apply the final floating-point correction to a positive sensing entry."""
        power = np.asarray(power, dtype=float).copy()
        sensing_count = self.system.params.Q
        target_index = int(np.argmax(power[:sensing_count]))
        power[target_index] += self.system.params.P_total - np.sum(power)
        tolerance = 32.0 * np.finfo(float).eps * max(
            self.system.params.P_total, 1.0
        )
        if power[target_index] < 0.0 and power[target_index] >= -tolerance:
            power[target_index] = 0.0
            second_index = int(np.argmax(power[:sensing_count]))
            power[second_index] += self.system.params.P_total - np.sum(power)
        if np.any(power < 0.0) or not np.all(np.isfinite(power)):
            raise RuntimeError("power correction produced an invalid allocation")
        return power

    def _numeric_power_step(
        self,
        base_power: np.ndarray,
        bandwidth: np.ndarray,
        remaining: float,
        initial_p: Optional[np.ndarray],
    ) -> np.ndarray:
        """Optimize a non-linear true sensing objective on the power simplex."""
        sensing_count = self.system.params.Q
        if initial_p is None or np.sum(initial_p[:sensing_count]) <= 0.0:
            start = np.full(sensing_count, remaining / sensing_count)
        else:
            start = np.asarray(initial_p[:sensing_count], dtype=float)
            start = remaining * start / np.sum(start)

        def full_power(sensing_power: np.ndarray) -> np.ndarray:
            candidate = base_power.copy()
            candidate[:sensing_count] = sensing_power
            return self._correct_power_budget(candidate)

        start_power = full_power(start)
        start_objective = self._compute_current_objective(start_power, bandwidth)
        scale = max(abs(start_objective), 1.0)

        result = self._run_slsqp(
            lambda value: -self._compute_current_objective(
                full_power(value), bandwidth
            )
            / scale,
            start,
            method="SLSQP",
            bounds=[(0.0, remaining)] * sensing_count,
            constraints=[
                {"type": "eq", "fun": lambda value: np.sum(value) - remaining}
            ],
            options={"maxiter": 500, "ftol": 1.0e-12, "disp": False},
        )
        if not result.success:
            self._power_success = False
            self._power_status = (
                "SLSQP failed; retained feasible power start "
                f"({result.message})"
            )
            return start_power
        candidate = full_power(result.x)
        candidate_objective = self._compute_current_objective(candidate, bandwidth)
        if candidate_objective < start_objective:
            self._power_success = False
            self._power_status = "SLSQP decreased objective; retained feasible power start"
            return start_power
        self._power_success = True
        self._power_status = "optimized true-objective power iterate"
        return candidate

    def _solve_power_subproblem(
        self,
        b: np.ndarray,
        Gamma_c: float,
        initial_p: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Update power using the public sensing objective at fixed bandwidth."""
        bandwidth = np.asarray(b, dtype=float)
        expected = (self.system.total_objects,)
        if bandwidth.shape != expected or not np.all(np.isfinite(bandwidth)):
            raise ValueError(f"b must be a finite vector with shape {expected}")
        if np.any(bandwidth <= 0.0):
            raise ValueError("all bandwidth entries must be positive")

        power = self._minimum_communication_power(bandwidth, Gamma_c)
        remaining = self.system.params.P_total - float(np.sum(power))
        feasibility_tolerance = 32.0 * np.finfo(float).eps * max(
            self.system.params.P_total, 1.0
        )
        if remaining < -feasibility_tolerance:
            raise RuntimeError(
                "the current fixed-bandwidth iterate cannot meet the rate floor "
                "within the power budget"
            )
        remaining = max(remaining, 0.0)
        fairness = self._coerce_supported_fairness(self.fairness_type)

        if self.qos_type == "detection" and fairness == FairnessType.MAXMIN:
            coefficient = self._power_objective_coefficients(bandwidth)
            power[: self.system.params.Q] = remaining / (
                coefficient * np.sum(1.0 / coefficient)
            )
            self._power_success = True
            self._power_status = "exact monotone max-min detector power step"
            return self._correct_power_budget(power)

        if self.qos_type == "localization":
            coefficient = self._power_objective_coefficients(bandwidth)
            if fairness == FairnessType.MAXMIN:
                power[: self.system.params.Q] = remaining / (
                    coefficient * np.sum(1.0 / coefficient)
                )
                self._power_status = "exact linear max-min information-score step"
            else:
                power[int(np.argmax(coefficient))] = remaining
                self._power_status = "exact linear sum information-score step"
            self._power_success = True
            return self._correct_power_budget(power)

        return self._numeric_power_step(
            power, bandwidth, remaining, initial_p
        )

    def _solve_bandwidth_subproblem(
        self,
        p: np.ndarray,
        Gamma_c: float,
        initial_b: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Optimize the public objective at fixed power, retaining feasibility."""
        power = np.asarray(p, dtype=float)
        expected = (self.system.total_objects,)
        if (
            power.shape != expected
            or np.any(power < 0.0)
            or not np.all(np.isfinite(power))
        ):
            raise ValueError(
                f"p must be a finite non-negative vector with shape {expected}"
            )
        total_bandwidth = self.system.params.B_total
        if initial_b is None:
            start_fraction = np.full(
                self.system.total_objects, 1.0 / self.system.total_objects
            )
        else:
            initial_b = np.asarray(initial_b, dtype=float)
            if (
                initial_b.shape != expected
                or np.any(initial_b <= 0.0)
                or not np.all(np.isfinite(initial_b))
            ):
                raise ValueError(
                    f"initial_b must be positive and finite with shape {expected}"
                )
            start_fraction = initial_b / np.sum(initial_b)
        start_bandwidth = start_fraction * total_bandwidth
        start_rates = self._communication_rates(power, start_bandwidth)
        if np.any(start_rates < Gamma_c):
            raise RuntimeError("bandwidth subproblem received an infeasible start")

        start_objective = self._compute_current_objective(power, start_bandwidth)
        scale = max(abs(start_objective), 1.0)

        def bandwidth_from_fraction(fraction: np.ndarray) -> np.ndarray:
            candidate = np.asarray(fraction, dtype=float).copy()
            candidate[np.argmax(candidate)] += 1.0 - np.sum(candidate)
            return candidate * total_bandwidth

        def rate_residual(fraction: np.ndarray) -> np.ndarray:
            rates = self._communication_rates(
                power, bandwidth_from_fraction(fraction)
            )
            return (rates - Gamma_c) / max(abs(Gamma_c), 1.0)

        result = self._run_slsqp(
            lambda value: -self._compute_current_objective(
                power, bandwidth_from_fraction(value)
            )
            / scale,
            start_fraction,
            method="SLSQP",
            bounds=[
                (self.min_bandwidth_fraction, 1.0)
            ]
            * self.system.total_objects,
            constraints=[
                {"type": "eq", "fun": lambda value: np.sum(value) - 1.0},
                {"type": "ineq", "fun": rate_residual},
            ],
            options={"maxiter": 500, "ftol": 1.0e-12, "disp": False},
        )
        if not result.success:
            self._bandwidth_success = False
            self._bandwidth_status = (
                "SLSQP failed; retained feasible bandwidth start "
                f"({result.message})"
            )
            return start_bandwidth

        candidate = bandwidth_from_fraction(result.x)
        rates = self._communication_rates(power, candidate)
        objective = self._compute_current_objective(power, candidate)
        if np.any(rates < Gamma_c):
            self._bandwidth_success = False
            self._bandwidth_status = (
                "SLSQP returned an infeasible point; retained feasible bandwidth start"
            )
            return start_bandwidth
        if objective < start_objective:
            self._bandwidth_success = False
            self._bandwidth_status = (
                "SLSQP decreased objective; retained feasible bandwidth start"
            )
            return start_bandwidth
        self._bandwidth_success = True
        self._bandwidth_status = "optimized feasible true-objective bandwidth iterate"
        return candidate

    def solve(
        self, Gamma_c: float = 1.0e6, initial_b: Optional[np.ndarray] = None
    ) -> AOResult:
        """Return the best feasible local iterate and auditable status flags."""
        if not np.isfinite(Gamma_c) or Gamma_c < 0.0:
            raise ValueError("Gamma_c must be non-negative and finite")
        if initial_b is None:
            bandwidth = self._initialize_feasible_bandwidth(Gamma_c)
        else:
            bandwidth = np.asarray(initial_b, dtype=float).copy()
            expected = (self.system.total_objects,)
            if (
                bandwidth.shape != expected
                or np.any(bandwidth <= 0.0)
                or not np.all(np.isfinite(bandwidth))
            ):
                raise ValueError(
                    f"initial_b must be positive and finite with shape {expected}"
                )
            if not np.isclose(
                np.sum(bandwidth),
                self.system.params.B_total,
                rtol=0.0,
                atol=1.0e-9 * max(self.system.params.B_total, 1.0),
            ):
                raise ValueError("initial_b must use the complete bandwidth budget")
            if not self._bandwidth_supports_rate_floor(bandwidth, Gamma_c):
                bandwidth = self._phase_one_bandwidth(Gamma_c)
                self._phase_one_status += " (replaced infeasible caller start)"
            else:
                self._phase_one_status = "caller start is rate feasible"

        previous_power = None
        previous_bandwidth = None
        previous_objective = None
        best_power = None
        best_bandwidth = None
        best_objective = -np.inf
        objective_history = []
        converged = False

        for iteration in range(1, self.max_iter + 1):
            power = self._solve_power_subproblem(
                bandwidth, Gamma_c, initial_p=previous_power
            )
            candidate_bandwidth = self._solve_bandwidth_subproblem(
                power, Gamma_c, initial_b=bandwidth
            )
            objective = float(
                self._compute_current_objective(power, candidate_bandwidth)
            )
            rates = self._communication_rates(power, candidate_bandwidth)
            if np.any(rates < Gamma_c) or not self.system.validate_allocations(
                power, candidate_bandwidth
            ):
                raise RuntimeError("an internal subproblem returned an infeasible iterate")

            if objective < best_objective:
                self._power_success = False
                self._bandwidth_success = False
                self._power_status += "; rejected objective-decreasing outer iterate"
                break
            best_power = power.copy()
            best_bandwidth = candidate_bandwidth.copy()
            best_objective = objective
            objective_history.append(objective)

            if previous_objective is not None:
                objective_change = abs(objective - previous_objective) / max(
                    abs(objective), abs(previous_objective), 1.0
                )
                allocation_change = max(
                    np.linalg.norm(power - previous_power)
                    / max(self.system.params.P_total, 1.0),
                    np.linalg.norm(candidate_bandwidth - previous_bandwidth)
                    / max(self.system.params.B_total, 1.0),
                )
                if (
                    self._power_success
                    and self._bandwidth_success
                    and objective_change <= self.tol
                    and allocation_change <= self.tol
                ):
                    converged = True
                    bandwidth = candidate_bandwidth
                    break

            previous_power = power.copy()
            previous_bandwidth = candidate_bandwidth.copy()
            previous_objective = objective
            bandwidth = candidate_bandwidth
            if not self._power_success or not self._bandwidth_success:
                break

        if best_power is None or best_bandwidth is None:
            raise RuntimeError("the solver did not construct a feasible iterate")

        power = best_power
        bandwidth = best_bandwidth
        sensing_count = self.system.params.Q
        sensing_power = power[:sensing_count]
        sensing_bandwidth = bandwidth[:sensing_count]
        detection_probabilities = None
        localization_score = None
        tracking_pcrb = None
        if self.qos_type == "detection":
            detection_probabilities = (
                self.detection_qos.compute_detection_probability(
                    sensing_power, sensing_bandwidth
                )
            )
        elif self.qos_type == "localization":
            localization_score = self.localization_qos.compute_information_score(
                sensing_power, sensing_bandwidth
            )
        else:
            tracking_pcrb = self.tracking_qos.compute_pcrb(
                sensing_power, sensing_bandwidth
            )

        communication_rates = self._communication_rates(power, bandwidth)
        if np.any(communication_rates < Gamma_c):
            raise RuntimeError("best iterate violates a communication-rate constraint")
        if not self.system.validate_allocations(power, bandwidth):
            raise RuntimeError("best iterate violates a resource budget")

        history = np.asarray(objective_history, dtype=float)
        if history.size > 1 and np.any(np.diff(history) < 0.0):
            raise RuntimeError("objective history is not monotonic")
        diagnostics: Dict[str, object] = {
            "phase_one": self._phase_one_status,
            "power_subproblem": self._power_status,
            "bandwidth_subproblem": self._bandwidth_status,
            "power_subproblem_success": self._power_success,
            "bandwidth_subproblem_success": self._bandwidth_success,
            "best_feasible_returned": True,
            "objective_history_monotonic": True,
            "convergence_requires_success_and_stability": True,
        }
        return AOResult(
            p=power,
            b=bandwidth,
            objective=best_objective,
            iterations=len(objective_history),
            converged=converged,
            detection_probs=detection_probabilities,
            localization_rho=localization_score,
            tracking_pcrb=tracking_pcrb,
            comm_rates=communication_rates,
            diagnostics=diagnostics,
            objective_history=history,
        )

    def _compute_current_objective(self, p: np.ndarray, b: np.ndarray) -> float:
        """Evaluate the single public sensing objective used by every step."""
        sensing_count = self.system.params.Q
        fairness = self._coerce_supported_fairness(self.fairness_type)
        sensing_power = np.asarray(p[:sensing_count], dtype=float)
        sensing_bandwidth = np.asarray(b[:sensing_count], dtype=float)
        if self.qos_type == "detection":
            values = self.detection_qos.compute_detection_probability(
                sensing_power, sensing_bandwidth
            )
            return float(
                np.min(values) if fairness == FairnessType.MAXMIN else np.sum(values)
            )
        if self.qos_type == "localization":
            values = self.localization_qos.compute_information_score(
                sensing_power, sensing_bandwidth
            )
            return float(
                np.min(values) if fairness == FairnessType.MAXMIN else np.sum(values)
            )
        if self.qos_type == "tracking":
            position_trace = self.tracking_qos.compute_pcrb_position_trace(
                sensing_power, sensing_bandwidth
            )
            return -float(
                np.max(position_trace)
                if fairness == FairnessType.MAXMIN
                else np.sum(position_trace)
            )
        raise ValueError(f"unknown QoS type: {self.qos_type}")

    def solve_multiple_qos(self, Gamma_c: float = 1.0e6) -> Dict[str, AOResult]:
        """Solve each declared local objective with an independent solver state."""
        return {
            qos_type: AOSolver(
                self.system,
                qos_type=qos_type,
                fairness=self.fairness_type,
                max_iter=self.max_iter,
                tol=self.tol,
            ).solve(Gamma_c)
            for qos_type in ("detection", "localization", "tracking")
        }
