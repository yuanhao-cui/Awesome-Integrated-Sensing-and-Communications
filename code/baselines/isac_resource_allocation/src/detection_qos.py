"""Normalized scaled-central-chi-square detection QoS surrogate.

The declared local statistic has two degrees of freedom and, under ``H1``, a
central chi-square scale of ``1 + SNR``.  This is a teaching model rather than
a claim that a particular raw-measurement likelihood has been reproduced.
"""

from typing import Optional

import numpy as np
from scipy import stats

from .system_model import ISACSystem


class DetectionQoS:
    """Detection probabilities under a normalized two-DOF energy statistic."""

    def __init__(self, system: ISACSystem, Pfa: float = 0.01):
        """
        Initialize Detection QoS.

        Parameters
        ----------
        system : ISACSystem
            ISAC system model
        Pfa : float
            Probability of false alarm (default: 0.01)
        """
        self.system = system
        if not np.isfinite(Pfa) or not 0 < Pfa < 1:
            raise ValueError("Pfa must lie strictly between zero and one")
        self.Pfa = Pfa
        self._chi2 = stats.chi2(df=2)  # Central chi-squared with 2 DOF

    def _compute_threshold(self) -> float:
        """
        Compute detection threshold for given Pfa.

        Returns
        -------
        threshold : float
            Detection threshold
        """
        # ``isf`` avoids the catastrophic cancellation in ``1 - Pfa`` for
        # small, but representable, false-alarm probabilities.
        return self._chi2.isf(self.Pfa)

    def compute_detection_probability(self, p: np.ndarray, b: np.ndarray,
                                       sigma: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute detection probability for each sensing target.

        For normalized threshold ``eta = chi2.isf(Pfa, 2)``, compute
        ``P_D,q = chi2.sf(eta / (1 + SNR_q), 2)``.  With two degrees of
        freedom this is exactly ``Pfa ** (1 / (1 + SNR_q))``.

        where:
        - δ is the detection threshold (determined by Pfa)
        - N0 is the noise power spectral density
        - σ² is the target RCS
        - p_q is the transmit power for target q
        - S_q is the signal-to-noise ratio factor

        Parameters
        ----------
        p : np.ndarray
            Power allocation for sensing targets (Q,)
        b : np.ndarray
            Bandwidth allocation for sensing targets (Q,)
        sigma : np.ndarray, optional
            Target radar cross sections. If None, uses system defaults.

        Returns
        -------
        P_D : np.ndarray
            Detection probabilities for each target (Q,)
        """
        p = np.asarray(p, dtype=float)
        b = np.asarray(b, dtype=float)

        if sigma is None:
            sigma = self.system.rcs

        Q = self.system.params.Q
        sigma = np.asarray(sigma, dtype=float)
        if p.shape != (Q,) or b.shape != (Q,) or sigma.shape != (Q,):
            raise ValueError(f"p, b, and sigma must have shape ({Q},)")
        if not (
            np.all(np.isfinite(p))
            and np.all(np.isfinite(b))
            and np.all(np.isfinite(sigma))
        ):
            raise ValueError("power, bandwidth, and RCS must be finite")
        if np.any(p < 0) or np.any(b <= 0) or np.any(sigma < 0):
            raise ValueError("power/RCS must be non-negative and bandwidth positive")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            snr = (
                p
                * self.system.beta_sensing
                * sigma
                / (self.system.N0 * b)
            )
        if not np.all(np.isfinite(snr)):
            raise ValueError(
                "the normalized sensing SNR is outside the finite numerical domain"
            )

        # For two degrees of freedom, chi2.sf(x, 2) = exp(-x/2) and
        # chi2.isf(Pfa, 2) = -2 log(Pfa).  Evaluating the resulting closed
        # form directly remains accurate throughout the accepted Pfa range.
        return np.exp(np.log(self.Pfa) / (1.0 + snr))

    def compute_detection_prob_simplified(self, p: np.ndarray, b: np.ndarray,
                                          sigma: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compatibility wrapper for the normalized energy-detector model.

        This wrapper returns the same scaled-central-chi-square probability as
        :meth:`compute_detection_probability`.

        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
        sigma : np.ndarray, optional
            Target RCS (Q,)

        Returns
        -------
        P_D : np.ndarray
            Detection probabilities (Q,)
        """
        return self.compute_detection_probability(p, b, sigma)

    def compute_objective_maxmin(self, p: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the local max-min detection objective.

        max min_q P_D,q

        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)

        Returns
        -------
        objective : float
            Minimum detection probability across all targets
        """
        P_D = self.compute_detection_probability(p, b)
        return np.min(P_D)

    def compute_objective_sum(self, p: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the local sum-detection objective.

        max Σ_q P_D,q

        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)

        Returns
        -------
        objective : float
            Sum of detection probabilities
        """
        P_D = self.compute_detection_probability(p, b)
        return np.sum(P_D)

    def detection_probability_gradient(self, p: np.ndarray, b: np.ndarray,
                                        sigma: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute gradient of detection probability w.r.t. power.

        ∂P_D,q/∂p_q for gradient-based optimization.

        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
        sigma : np.ndarray, optional
            Target RCS (Q,)

        Returns
        -------
        grad : np.ndarray
            Gradient w.r.t. power (Q,)
        """
        p = np.asarray(p, dtype=float)
        b = np.asarray(b, dtype=float)
        sigma = self.system.rcs if sigma is None else np.asarray(sigma, dtype=float)
        probability = self.compute_detection_probability(p, b, sigma)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            coefficient = (
                self.system.beta_sensing * sigma / (self.system.N0 * b)
            )
            snr = p * coefficient
            denominator = np.square(1.0 + snr)
            gradient = (
                probability
                * (-np.log(self.Pfa))
                * coefficient
                / denominator
            )
        # An infinite squared denominator represents an asymptotically zero
        # derivative, whereas a non-finite coefficient is an invalid domain.
        gradient = np.where(np.isinf(denominator), 0.0, gradient)
        if not np.all(np.isfinite(coefficient)) or not np.all(
            np.isfinite(gradient)
        ):
            raise ValueError(
                "detection gradient is outside the finite numerical domain"
            )
        return gradient

    def is_detectable(self, p: np.ndarray, b: np.ndarray,
                      threshold: float = 0.9) -> np.ndarray:
        """
        Check if targets are detectable with given probability.

        Parameters
        ----------
        p : np.ndarray
            Power allocation (Q,)
        b : np.ndarray
            Bandwidth allocation (Q,)
        threshold : float
            Required detection probability

        Returns
        -------
        detectable : np.ndarray
            Boolean array indicating detectability (Q,)
        """
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must lie in [0, 1]")
        P_D = self.compute_detection_probability(p, b)
        return P_D >= threshold
