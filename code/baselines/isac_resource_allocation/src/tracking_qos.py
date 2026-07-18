"""Auditable constant-velocity covariance-bound tracking surrogate."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .localization_qos import LocalizationQoS
from .system_model import ISACSystem


@dataclass
class TargetState:
    """Synthetic Cartesian position and velocity state."""

    position: np.ndarray
    velocity: np.ndarray


class TrackingQoS:
    """Constant-velocity prediction and range/angle covariance update.

    Only the trace of the 2x2 position block is used as a scalar objective.
    A full-state trace would add square metres to square metres per square
    second and is therefore intentionally not exposed as a QoS score.
    """

    def __init__(
        self,
        system: ISACSystem,
        dt: float = 0.1,
        process_noise_std: float = 0.5,
        localization_qos: Optional[LocalizationQoS] = None,
    ):
        """Initialize the motion model and shared measurement proxy."""
        self.system = system
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not np.isfinite(process_noise_std) or process_noise_std < 0.0:
            raise ValueError("process_noise_std must be non-negative and finite")
        self.dt = float(dt)
        self.process_noise_std = float(process_noise_std)
        self.localization_qos = localization_qos or LocalizationQoS(system)
        if self.localization_qos.system is not system:
            raise ValueError("localization_qos must use the same ISACSystem")
        # Reject finite parameters that cannot form a finite covariance in the
        # active floating-point domain, rather than failing during a later step.
        self._get_process_noise_cov()
        self.target_states = self._initialize_target_states()

    def _initialize_target_states(self) -> List[TargetState]:
        """Initialize seeded Cartesian target states from system geometry."""
        states = []
        for target_index in range(self.system.params.Q):
            distance = self.system.target_positions[target_index]
            angle = self.system.target_angles[target_index]
            position = np.array(
                [distance * np.cos(angle), distance * np.sin(angle)]
            )
            velocity = self.system.rng.uniform(-10.0, 10.0, size=2)
            states.append(TargetState(position=position, velocity=velocity))
        return states

    def _get_transition_matrix(self) -> np.ndarray:
        """Return the Cartesian constant-velocity transition matrix."""
        dt = self.dt
        return np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def _get_process_noise_cov(self) -> np.ndarray:
        """Return ``sigma_a^2 G G^T`` for interval-constant acceleration."""
        if self.process_noise_std == 0.0:
            return np.zeros((4, 4), dtype=float)
        dt = np.float64(self.dt)
        sigma = np.float64(self.process_noise_std)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            half_dt_squared = 0.5 * np.square(dt)
            acceleration_map = np.array(
                [
                    [half_dt_squared, 0.0],
                    [0.0, half_dt_squared],
                    [dt, 0.0],
                    [0.0, dt],
                ]
            )
            scaled_map = sigma * acceleration_map
            covariance = scaled_map @ scaled_map.T
        if not np.all(np.isfinite(covariance)):
            raise ValueError(
                "dt and process_noise_std must yield a finite process covariance"
            )
        return covariance

    @staticmethod
    def _compute_measurement_jacobian(state: np.ndarray) -> np.ndarray:
        """Return the range/azimuth Jacobian at a nonzero Cartesian state."""
        state = np.asarray(state, dtype=float)
        if state.shape != (4,) or not np.all(np.isfinite(state)):
            raise ValueError("state must be a finite four-vector")
        x_coord, y_coord = state[:2]
        radius = float(np.hypot(x_coord, y_coord))
        if not np.isfinite(radius):
            raise ValueError("state radius is outside the finite numerical domain")
        if radius <= 1.0e-10:
            raise ValueError("range/angle Jacobian is undefined at the origin")
        inverse_radius = 1.0 / radius
        inverse_radius_squared = inverse_radius**2
        return np.array(
            [
                [x_coord * inverse_radius, y_coord * inverse_radius, 0.0, 0.0],
                [
                    -y_coord * inverse_radius_squared,
                    x_coord * inverse_radius_squared,
                    0.0,
                    0.0,
                ],
            ]
        )

    def _validate_resources(
        self, p: np.ndarray, b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate per-target power and allocated-bandwidth vectors."""
        p = np.asarray(p, dtype=float)
        b = np.asarray(b, dtype=float)
        expected = (self.system.params.Q,)
        if p.shape != expected or b.shape != expected:
            raise ValueError(f"p and b must have shape {expected}")
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(b)):
            raise ValueError("p and b must contain only finite values")
        if np.any(p < 0.0) or np.any(b <= 0.0):
            raise ValueError("power must be non-negative and bandwidth positive")
        return p, b

    def _compute_measurement_covariance(
        self,
        power: float,
        bandwidth: float,
        target_index: int,
        angle: float,
    ) -> np.ndarray:
        """Return the same range/angle variance proxies as localization."""
        if not isinstance(target_index, (int, np.integer)) or not (
            0 <= target_index < self.system.params.Q
        ):
            raise IndexError("target_index is outside the sensing-target range")
        if not np.isfinite(power) or power <= 0.0:
            raise ValueError("power must be positive and finite")
        if not np.isfinite(bandwidth) or bandwidth <= 0.0:
            raise ValueError("bandwidth must be positive and finite")
        if not np.isfinite(angle):
            raise ValueError("angle must be finite")

        p_vector = np.zeros(self.system.params.Q, dtype=float)
        b_vector = np.ones(self.system.params.Q, dtype=float)
        angle_vector = np.asarray(self.system.target_angles, dtype=float).copy()
        p_vector[target_index] = power
        b_vector[target_index] = bandwidth
        angle_vector[target_index] = angle
        range_bound = self.localization_qos.compute_crb_range(
            p_vector, b_vector
        )[target_index]
        angle_bound = self.localization_qos.compute_crb_angle(
            p_vector, b_vector, angles=angle_vector
        )[target_index]
        covariance = np.diag([range_bound, angle_bound])
        if not np.all(np.isfinite(covariance)) or np.any(
            np.diag(covariance) <= 0.0
        ):
            raise RuntimeError("measurement covariance must be finite and positive")
        return covariance

    def compute_fim(
        self, p: np.ndarray, b: np.ndarray, state_idx: int = 0
    ) -> np.ndarray:
        """Return predicted-epoch measurement information for one target."""
        p, b = self._validate_resources(p, b)
        if not isinstance(state_idx, (int, np.integer)) or not (
            0 <= state_idx < self.system.params.Q
        ):
            raise IndexError("state_idx is outside the sensing-target range")
        if p[state_idx] == 0.0:
            return np.zeros((4, 4), dtype=float)
        state = self.target_states[state_idx]
        state_vector = np.concatenate([state.position, state.velocity])
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            predicted_state = self._get_transition_matrix() @ state_vector
        if not np.all(np.isfinite(predicted_state)):
            raise ValueError("predicted target state is outside the finite domain")
        jacobian = self._compute_measurement_jacobian(predicted_state)
        angle = float(np.arctan2(predicted_state[1], predicted_state[0]))
        covariance = self._compute_measurement_covariance(
            p[state_idx], b[state_idx], state_idx, angle
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            information = jacobian.T @ np.linalg.solve(covariance, jacobian)
        if not np.all(np.isfinite(information)):
            raise ValueError(
                "tracking measurement information is outside the finite domain"
            )
        return information

    @staticmethod
    def _validate_prior(prior: np.ndarray, target_count: int) -> np.ndarray:
        """Validate a batch of symmetric positive-definite prior covariances."""
        prior = np.asarray(prior, dtype=float)
        expected = (target_count, 4, 4)
        if prior.shape != expected or not np.all(np.isfinite(prior)):
            raise ValueError(f"prior_pcrb must be finite with shape {expected}")
        if not np.allclose(prior, np.swapaxes(prior, 1, 2), rtol=0.0, atol=1e-12):
            raise ValueError("each prior_pcrb matrix must be symmetric")
        if any(np.min(np.linalg.eigvalsh(matrix)) <= 0.0 for matrix in prior):
            raise ValueError("each prior_pcrb matrix must be positive definite")
        return prior

    def compute_pcrb(
        self,
        p: np.ndarray,
        b: np.ndarray,
        prior_pcrb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Perform one same-epoch covariance prediction/measurement update.

        The covariance and the target state are both predicted with ``F`` before
        the measurement Jacobian is evaluated.  The posterior uses the Joseph
        covariance form, avoiding an explicit inverse of the predicted
        covariance or posterior information matrix.
        """
        p, b = self._validate_resources(p, b)
        target_count = self.system.params.Q
        transition = self._get_transition_matrix()
        process_covariance = self._get_process_noise_cov()
        if prior_pcrb is None:
            prior = np.tile(
                np.diag([1.0e6, 1.0e6, 1.0e4, 1.0e4]),
                (target_count, 1, 1),
            )
        else:
            prior = self._validate_prior(prior_pcrb, target_count)

        posterior = np.zeros((target_count, 4, 4), dtype=float)
        identity = np.eye(4)
        for target_index in range(target_count):
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                predicted_covariance = (
                    transition @ prior[target_index] @ transition.T
                    + process_covariance
                )
            if not np.all(np.isfinite(predicted_covariance)):
                raise ValueError(
                    "predicted covariance is outside the finite numerical domain"
                )
            predicted_covariance = 0.5 * (
                predicted_covariance + predicted_covariance.T
            )
            if p[target_index] == 0.0:
                posterior[target_index] = predicted_covariance
                continue

            state = self.target_states[target_index]
            state_vector = np.concatenate([state.position, state.velocity])
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                predicted_state = transition @ state_vector
            if not np.all(np.isfinite(predicted_state)):
                raise ValueError(
                    "predicted target state is outside the finite numerical domain"
                )
            jacobian = self._compute_measurement_jacobian(predicted_state)
            predicted_angle = float(
                np.arctan2(predicted_state[1], predicted_state[0])
            )
            measurement_covariance = self._compute_measurement_covariance(
                p[target_index],
                b[target_index],
                target_index,
                predicted_angle,
            )
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                innovation_covariance = (
                    jacobian @ predicted_covariance @ jacobian.T
                    + measurement_covariance
                )
                kalman_gain = np.linalg.solve(
                    innovation_covariance,
                    jacobian @ predicted_covariance,
                ).T
                residual_map = identity - kalman_gain @ jacobian
                updated = (
                    residual_map @ predicted_covariance @ residual_map.T
                    + kalman_gain @ measurement_covariance @ kalman_gain.T
                )
            if not (
                np.all(np.isfinite(innovation_covariance))
                and np.all(np.isfinite(kalman_gain))
                and np.all(np.isfinite(updated))
            ):
                raise ValueError(
                    "tracking update is outside the finite numerical domain"
                )
            posterior[target_index] = 0.5 * (updated + updated.T)

        if not np.all(np.isfinite(posterior)):
            raise RuntimeError("posterior covariance contains non-finite values")
        return posterior

    def compute_pcrb_position_trace(
        self,
        p: np.ndarray,
        b: np.ndarray,
        prior_pcrb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return each target's 2x2 position-covariance trace in square metres."""
        pcrb = self.compute_pcrb(p, b, prior_pcrb)
        return np.trace(pcrb[:, :2, :2], axis1=1, axis2=2)

    def compute_pcrb_trace(
        self,
        p: np.ndarray,
        b: np.ndarray,
        prior_pcrb: Optional[np.ndarray] = None,
    ) -> float:
        """Return the sum of position-block traces only, in square metres."""
        return float(np.sum(self.compute_pcrb_position_trace(p, b, prior_pcrb)))

    def update_target_states(self) -> None:
        """Advance each synthetic state once under the declared motion model."""
        transition = self._get_transition_matrix()
        process_covariance = self._get_process_noise_cov()
        for target_index, state in enumerate(self.target_states):
            state_vector = np.concatenate([state.position, state.velocity])
            process_noise = self.system.rng.multivariate_normal(
                np.zeros(4), process_covariance, check_valid="raise"
            )
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                updated = transition @ state_vector + process_noise
            if not np.all(np.isfinite(updated)):
                raise ValueError(
                    "predicted target state is outside the finite numerical domain"
                )
            self.target_states[target_index].position = updated[:2]
            self.target_states[target_index].velocity = updated[2:]

    def simulate_tracking(
        self, p: np.ndarray, b: np.ndarray, num_steps: int = 50
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Simulate repeated prediction/update bounds and position traces."""
        p, b = self._validate_resources(p, b)
        if not isinstance(num_steps, (int, np.integer)) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        pcrb_history: List[np.ndarray] = []
        position_trace_history: List[float] = []
        prior_pcrb = None
        for _ in range(num_steps):
            pcrb = self.compute_pcrb(p, b, prior_pcrb)
            pcrb_history.append(pcrb.copy())
            position_trace_history.append(
                float(np.sum(np.trace(pcrb[:, :2, :2], axis1=1, axis2=2)))
            )
            prior_pcrb = pcrb
            self.update_target_states()
        return pcrb_history, position_trace_history

    def compute_tracking_error_bound(
        self,
        p: np.ndarray,
        b: np.ndarray,
        prior_pcrb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return position error-bound radii from the 2x2 position blocks."""
        return np.sqrt(self.compute_pcrb_position_trace(p, b, prior_pcrb))
