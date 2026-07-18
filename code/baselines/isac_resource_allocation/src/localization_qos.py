"""Dimensionally explicit local information-bound localization proxy."""

from typing import Optional, Tuple

import numpy as np

from .system_model import ISACSystem


class LocalizationQoS:
    """Evaluate a declared synthetic range/angle information model.

    This class is not an implementation of a paper-specific likelihood.  The
    range proxy uses allocated bandwidth, whereas the angle proxy uses a fixed
    noise-equivalent reference bandwidth.  Consequently, angle information is
    independent of allocated bandwidth.  This is an explicit local modeling
    choice and is not presented as a paper-equation implementation.
    """

    def __init__(
        self,
        system: ISACSystem,
        w_d: float = 1.0,
        w_theta: float = 1.0,
        range_reference_m: float = 1.0,
        angle_reference_rad: float = 1.0e-3,
        angle_noise_bandwidth_hz: float = 10.0e6,
        d_lambda: float = 0.5,
    ):
        """Initialize the local proxy and its dimensionless score scales."""
        self.system = system
        self.w_d = self._nonnegative_finite("w_d", w_d)
        self.w_theta = self._nonnegative_finite("w_theta", w_theta)
        if self.w_d == 0.0 and self.w_theta == 0.0:
            raise ValueError("at least one localization weight must be positive")
        self.range_reference_m = self._positive_finite(
            "range_reference_m", range_reference_m
        )
        self.angle_reference_rad = self._positive_finite(
            "angle_reference_rad", angle_reference_rad
        )
        self.angle_noise_bandwidth_hz = self._positive_finite(
            "angle_noise_bandwidth_hz", angle_noise_bandwidth_hz
        )
        self.d_lambda = self._positive_finite("d_lambda", d_lambda)
        for name, value in (
            ("range_reference_m", self.range_reference_m),
            ("angle_reference_rad", self.angle_reference_rad),
            ("d_lambda", self.d_lambda),
        ):
            with np.errstate(over="ignore", invalid="ignore"):
                squared = np.square(np.float64(value))
            if not np.isfinite(squared):
                raise ValueError(f"{name} squared must be finite")
        self.c = 3.0e8

    @staticmethod
    def _positive_finite(name: str, value: float) -> float:
        """Return a validated positive scalar."""
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
        return value

    @staticmethod
    def _nonnegative_finite(name: str, value: float) -> float:
        """Return a validated non-negative scalar."""
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be non-negative and finite")
        return value

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

    def _validate_angles(self, angles: Optional[np.ndarray]) -> np.ndarray:
        """Return a validated angle vector, defaulting to the system geometry."""
        if angles is None:
            angles = self.system.target_angles
        angles = np.asarray(angles, dtype=float)
        expected = (self.system.params.Q,)
        if angles.shape != expected or not np.all(np.isfinite(angles)):
            raise ValueError(f"angles must be finite with shape {expected}")
        return angles

    def compute_information_components(
        self,
        p: np.ndarray,
        b: np.ndarray,
        d_lambda: Optional[float] = None,
        *,
        angles: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return range information in 1/m^2 and angle information in 1/rad^2.

        The local assumptions are

        ``J_d = 8 pi^2 SNR_alloc b^2 / c^2`` with
        ``SNR_alloc = p g / (N0 b)``, and

        ``J_theta = SNR_angle Nt(Nt^2-1) pi^2 cos^2(theta) d^2 / 6``
        with ``SNR_angle = p g / (N0 B_angle_ref)``.

        ``B_angle_ref`` is fixed at construction, so changing an allocation's
        bandwidth does not silently change the assumed angle-noise model.
        """
        p, b = self._validate_resources(p, b)
        angles = self._validate_angles(angles)
        spacing = self.d_lambda if d_lambda is None else self._positive_finite(
            "d_lambda", d_lambda
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            gain = self.system.beta_sensing * self.system.rcs
            allocated_snr = p * gain / (self.system.N0 * b)
            range_information = (
                8.0 * np.pi**2 * allocated_snr * np.square(b) / self.c**2
            )

            angle_snr = p * gain / (
                self.system.N0 * self.angle_noise_bandwidth_hz
            )
            nt = self.system.params.Nt
            angle_information = (
                angle_snr
                * nt
                * (nt**2 - 1)
                * np.pi**2
                * np.square(np.cos(angles))
                * np.square(np.float64(spacing))
                / 6.0
            )
        if not (
            np.all(np.isfinite(range_information))
            and np.all(np.isfinite(angle_information))
        ):
            raise ValueError(
                "localization information is outside the finite numerical domain"
            )
        return range_information, angle_information

    @staticmethod
    def _reciprocal_bound(information: np.ndarray) -> np.ndarray:
        """Invert positive information and map zero information to infinity."""
        bound = np.full_like(information, np.inf, dtype=float)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            np.divide(1.0, information, out=bound, where=information > 0.0)
        return bound

    def compute_crb_range(self, p: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return the local range-variance lower-bound proxy in square metres."""
        range_information, _ = self.compute_information_components(p, b)
        return self._reciprocal_bound(range_information)

    def compute_crb_angle(
        self,
        p: np.ndarray,
        b: np.ndarray,
        d_lambda: Optional[float] = None,
        *,
        angles: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return the local angle-variance proxy in square radians.

        The allocated-bandwidth vector is validated for API consistency, but
        the angle model uses the fixed ``angle_noise_bandwidth_hz`` declared at
        construction and is therefore allocation-bandwidth invariant.
        """
        _, angle_information = self.compute_information_components(
            p, b, d_lambda=d_lambda, angles=angles
        )
        return self._reciprocal_bound(angle_information)

    def compute_information_score(
        self,
        p: np.ndarray,
        b: np.ndarray,
        *,
        angles: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return a dimensionless weighted range/angle information score."""
        range_information, angle_information = self.compute_information_components(
            p, b, angles=angles
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            score = (
                self.w_d
                * np.square(np.float64(self.range_reference_m))
                * range_information
                + self.w_theta
                * np.square(np.float64(self.angle_reference_rad))
                * angle_information
            )
        if not np.all(np.isfinite(score)):
            raise ValueError(
                "localization score is outside the finite numerical domain"
            )
        return score

    def compute_information_score_coefficients(self, b: np.ndarray) -> np.ndarray:
        """Return exact score-per-watt coefficients for fixed bandwidth."""
        b = np.asarray(b, dtype=float)
        ones = np.ones(self.system.params.Q, dtype=float)
        return self.compute_information_score(ones, b)

    def compute_crb_combined(self, p: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compatibility wrapper returning the dimensionless information score."""
        return self.compute_information_score(p, b)

    def compute_localization_rmse(
        self, p: np.ndarray, b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return range and angle RMSE lower-bound proxies."""
        return np.sqrt(self.compute_crb_range(p, b)), np.sqrt(
            self.compute_crb_angle(p, b)
        )

    def compute_objective_sum(self, p: np.ndarray, b: np.ndarray) -> float:
        """Return the sum of dimensionless information scores."""
        return float(np.sum(self.compute_information_score(p, b)))

    def compute_objective_proportional_fairness(
        self, p: np.ndarray, b: np.ndarray
    ) -> float:
        """Return a log-score diagnostic with a dimensionless argument."""
        score = self.compute_information_score(p, b)
        return float(np.sum(np.log(np.maximum(score, np.finfo(float).tiny))))

    def compute_objective_maxmin(self, p: np.ndarray, b: np.ndarray) -> float:
        """Return the minimum dimensionless information score."""
        return float(np.min(self.compute_information_score(p, b)))

    def compute_fim(self, p: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Construct the declared diagonal local information matrices.

        These matrices collect the two proxy-information components.  They are
        not presented as raw or expected Hessians of an unspecified likelihood.
        """
        range_information, angle_information = self.compute_information_components(
            p, b
        )
        fim = np.zeros((self.system.params.Q, 2, 2), dtype=float)
        fim[:, 0, 0] = range_information
        fim[:, 1, 1] = angle_information
        return fim

    def validate_localization_performance(
        self,
        p: np.ndarray,
        b: np.ndarray,
        max_range_error: float = 1.0,
        max_angle_error: float = 0.1,
    ) -> bool:
        """Check declared RMSE limits in metres and radians."""
        max_range_error = self._positive_finite(
            "max_range_error", max_range_error
        )
        max_angle_error = self._positive_finite(
            "max_angle_error", max_angle_error
        )
        rmse_range, rmse_angle = self.compute_localization_rmse(p, b)
        return bool(
            np.all(rmse_range <= max_range_error)
            and np.all(rmse_angle <= max_angle_error)
        )
