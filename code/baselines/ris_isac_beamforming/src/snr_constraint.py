"""SNR-constrained feasibility iteration for the local RIS surrogate.

It alternates a conservative minimum-power SOCP for fixed phases with a
monotone coordinate update of the independent-stream sensing SNR. It is not
the paper's Algorithm 1 and makes no paper-optimality claim.
"""

import cvxpy as cp
import numpy as np

from .system_model import RIS_ISAC_System
from .beamforming import BeamformingOptimizer, _solve_problem
from .numerics import db_to_linear, stable_squared_norm
from .ris_phase import RISPhaseOptimizer


class SNRConstrainedSolver:
    """Find a locally feasible SNR/SINR/power tuple.

    Each returned tuple is post-validated in the original local surrogate.

    Attributes:
        system: RIS-ISAC system model.
        M: BS antenna count.
        K: User count.
        L: RIS element count.
        P_max: Power budget (W).
        noise_power: Noise variance (W).
        snr_min: Minimum radar SNR (linear).
        sinr_thresh: SINR threshold (linear).
        max_iter: Maximum AO iterations.
        tol: Convergence tolerance.
    """

    def __init__(
        self,
        system: RIS_ISAC_System,
        snr_min_dB: float = 5.0,
        max_iter: int = 50,
        tol: float = 1e-4,
    ):
        """Initialize SNR-constrained solver.

        Args:
            system: RIS-ISAC system instance.
            snr_min_dB: Minimum radar sensing SNR in dB (γ_min).
            max_iter: Maximum alternating optimization iterations.
            tol: Convergence tolerance.
        """
        if not np.isfinite(snr_min_dB):
            raise ValueError("snr_min_dB must be finite")
        if (
            not isinstance(max_iter, int)
            or isinstance(max_iter, bool)
            or max_iter < 1
        ):
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be positive and finite")
        self.system = system
        self.M = system.M
        self.K = system.K
        self.L = system.L
        self.P_max = system.P_max
        self.noise_power = system.noise_power

        self.snr_min_dB = snr_min_dB
        self.snr_min = db_to_linear(snr_min_dB, "snr_min_dB")

        self.sinr_thresh_dB = system.sinr_thresh_dB
        self.sinr_thresh = system.sinr_thresh

        self.max_iter = max_iter
        self.tol = tol

        self.bf_optimizer = BeamformingOptimizer(system)
        self.ris_optimizer = RISPhaseOptimizer(system)

    def _compute_sensing_channel(self) -> np.ndarray:
        """Compute the effective sensing channel vector h_s (M,).

        Returns:
            Sensing channel vector (M,).
        """
        H_BR = self.system.channels["H_BR"]
        a_bs = self.system.channels["a_bs"]
        a_ris = self.system.channels["a_ris"]
        Theta = self.system.ris_diagonal_matrix()
        return a_bs + a_ris.T @ Theta @ H_BR

    def solve(self) -> dict:
        """Run a safeguarded feasibility iteration.

        Algorithm:
            1. Improve the RIS phases by a monotone sensing-SNR grid sweep.
            2. Solve the fixed-phase minimum-power SOCP.
            3. Propose another monotone sensing-SNR phase update.
            4. Accept it only when the re-solved feasible transmit power does
               not increase; otherwise restore the last certified tuple.

        The safeguard gives a monotone, bounded power certificate for this
        local surrogate. It is not the alternating algorithm in the cited
        paper. ``converged`` only denotes termination of this declared
        safeguarded iteration.

        Returns:
            Dictionary with keys:
                'W': Optimal beamforming matrix (M, K).
                'theta': Optimal RIS phases (L,).
                'sum_rate': Achieved sum rate (bps/Hz).
                'snr_sensing': Achieved radar SNR (linear).
                'converged': Whether AO converged.
                'iterations': Number of AO iterations.
                'history': List of sum-rate values for accepted iterates.
                'power_history': Nonincreasing accepted transmit powers.
        """
        sinr_thresholds = np.full(self.K, self.sinr_thresh)

        # Initialize with matched-filter columns under the h^H w convention.
        W = self._initial_beamformers()

        # Improve the physical independent-stream sensing SNR initially.
        self.ris_optimizer.optimize_for_snr(W)

        W_reference = self._feasible_sca_reference(sinr_thresholds)
        W, _ = self._solve_beamforming_socp(
            sinr_thresholds, W_reference
        )
        power = stable_squared_norm(W)
        sum_rate = self.system.compute_sum_rate(W)
        history = [sum_rate]
        power_history = [power]
        converged = False

        for _ in range(1, self.max_iter):
            accepted_theta = self.system.theta.copy()
            accepted_W = W.copy()
            accepted_power = power
            accepted_rate = sum_rate

            self.ris_optimizer.optimize_for_snr(W)
            try:
                candidate_reference = self._feasible_sca_reference(
                    sinr_thresholds
                )
            except RuntimeError:
                self.system.set_ris_phases(accepted_theta)
                W = accepted_W
                power = accepted_power
                sum_rate = accepted_rate
                converged = True
                break
            candidate_W, _ = self._solve_beamforming_socp(
                sinr_thresholds, candidate_reference
            )
            candidate_power = stable_squared_norm(candidate_W)
            candidate_rate = self.system.compute_sum_rate(candidate_W)
            relative_change = abs(accepted_power - candidate_power) / max(
                accepted_power, 1e-15
            )

            if candidate_power > accepted_power * (1.0 + 1e-8):
                self.system.set_ris_phases(accepted_theta)
                W = accepted_W
                power = accepted_power
                sum_rate = accepted_rate
                converged = True
                break

            W = candidate_W
            power = candidate_power
            sum_rate = candidate_rate
            history.append(sum_rate)
            power_history.append(power)
            if relative_change < self.tol:
                converged = True
                break

        self.bf_optimizer._validate_physical_solution(W, sinr_thresholds)
        snr_sensing = self.system.compute_snr_sensing(W)
        if snr_sensing < self.snr_min * (1.0 - 5e-4):
            raise RuntimeError("Final AO iterate violates the sensing-SNR constraint")

        return {
            "W": W,
            "theta": self.system.theta.copy(),
            "sum_rate": sum_rate,
            "snr_sensing": snr_sensing,
            "converged": converged,
            "iterations": len(power_history),
            "history": history,
            "power_history": power_history,
        }

    def _initial_beamformers(self) -> np.ndarray:
        """Return deterministic full-power matched-filter reference columns."""

        W = np.zeros((self.M, self.K), dtype=complex)
        column_power = self.P_max / self.K
        for k in range(self.K):
            h_k = self.system.effective_channel(k)
            channel_norm = np.linalg.norm(h_k)
            if not np.isfinite(channel_norm) or channel_norm <= 0.0:
                raise RuntimeError(
                    f"user {k} has no finite nonzero effective channel"
                )
            W[:, k] = np.sqrt(column_power) * h_k / channel_norm
        return W

    def _feasible_sca_reference(
        self, sinr_thresholds: np.ndarray
    ) -> np.ndarray:
        """Return a fixed-phase reference feasible for every physical QoS.

        The communication-only minimum-power SOCP supplies the reference.  If
        its complete independent-stream sensing SNR is too small, all columns
        are scaled by the minimum common factor that meets the sensing target.
        Common scaling preserves or improves every SINR.  The power budget and
        all physical constraints are rechecked.  At this accepted reference
        the affine sensing lower bound is tight, so the subsequent SCA
        subproblem has a known feasible point without unnecessarily forcing a
        full-power solution.
        """

        W_reference, _ = self.bf_optimizer.solve_min_power(
            sinr_thresholds
        )
        reference_snr = self.system.compute_snr_sensing(W_reference)
        target_snr = self.snr_min * (1.0 + 1e-8)
        if reference_snr == 0.0:
            raise RuntimeError(
                "the communication-feasible SCA reference has zero "
                "independent-stream sensing power"
            )
        if reference_snr < target_snr:
            required_scale_squared = target_snr / reference_snr
            reference_power = stable_squared_norm(W_reference)
            if required_scale_squared > self.P_max / reference_power:
                raise RuntimeError(
                    "no power-feasible common scaling of the SCA reference "
                    "meets the independent-stream sensing-SNR threshold"
                )
            W_reference *= np.sqrt(required_scale_squared)
            self.bf_optimizer._validate_physical_solution(
                W_reference, sinr_thresholds
            )
            reference_snr = self.system.compute_snr_sensing(W_reference)
        if reference_snr < self.snr_min * (1.0 - 5e-4):
            raise RuntimeError(
                "no fixed-phase SCA reference satisfies the "
                "independent-stream sensing-SNR threshold"
            )
        return W_reference

    def _solve_beamforming_socp(
        self,
        sinr_thresholds: np.ndarray,
        W_reference: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Find a minimum-power beamformer with SINR and SNR constraints.

        This educational surrogate uses a conservative SOCP.  Desired-user
        projections have fixed phase.  For sensing, define

        ``z = h_s^H W`` and ``z_0 = h_s^H W_reference``.

        Convexity of the squared norm gives the global affine lower bound

        ``||z||_2^2 >= 2 Re{z_0^H z} - ||z_0||_2^2``.

        Requiring this lower bound to exceed ``gamma_s sigma^2`` is therefore
        a convex sufficient condition for the independent-stream covariance
        constraint ``h_s^H W W^H h_s >= gamma_s sigma^2``.  It contains no
        coherent sum of the data-stream beamformers.  The reference is the
        fixed-phase communication-feasible reference and the
        returned matrix is post-evaluated with the full covariance expression.

        Args:
            sinr_thresholds: SINR thresholds (K,).
            W_reference: Previous physical beamforming iterate (M, K).
        Returns:
            Tuple of (W, objective_value).
        """
        H_eff = self.bf_optimizer._get_effective_channels()
        sigma2 = self.system.noise_power
        h_s = self._compute_sensing_channel()
        thresholds = np.asarray(sinr_thresholds, dtype=float)
        if (
            thresholds.shape != (self.K,)
            or not np.all(np.isfinite(thresholds))
            or np.any(thresholds < 0)
        ):
            raise ValueError(
                f"sinr_thresholds must be a finite non-negative ({self.K},) vector"
            )
        W_reference = np.asarray(W_reference, dtype=complex)
        if (
            W_reference.shape != (self.M, self.K)
            or not np.all(np.isfinite(W_reference))
        ):
            raise ValueError(
                f"W_reference must be a finite {(self.M, self.K)} matrix"
            )
        sensing_reference = h_s.conj() @ W_reference
        sensing_reference_power = stable_squared_norm(sensing_reference)
        if sensing_reference_power == 0.0:
            raise RuntimeError(
                "the sensing linearization reference has zero covariance power"
            )
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

        sensing_projection = h_s.conj() @ W_var
        sensing_lower_bound = (
            2.0
            * cp.real(sensing_reference.conj() @ sensing_projection)
            - sensing_reference_power
        )
        constraints.append(
            sensing_lower_bound >= self.snr_min * sigma2
        )
        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(cp.abs(W_var))), constraints
        )
        try:
            _solve_problem(problem)
        except RuntimeError as exc:
            raise RuntimeError(
                "SNR-constrained beamforming subproblem failed; "
                "the sensing constraint was not relaxed"
            ) from exc
        if W_var.value is None:
            raise RuntimeError("SNR-constrained solver returned no beamformer")
        W = np.asarray(W_var.value)
        self.bf_optimizer._validate_physical_solution(W, thresholds)
        achieved_snr = self.system.compute_snr_sensing(W)
        if achieved_snr < self.snr_min * (1.0 - 5e-4):
            raise RuntimeError(
                "SNR-constrained physical solution is infeasible: "
                f"{achieved_snr:.6g} < {self.snr_min:.6g}"
            )
        return W, stable_squared_norm(W)
