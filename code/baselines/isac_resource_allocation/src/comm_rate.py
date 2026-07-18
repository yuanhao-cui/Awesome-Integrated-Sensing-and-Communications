"""Shannon-form communication-rate helper for the synthetic system."""

import numpy as np
from typing import Optional, Tuple

from .numerics import stable_shannon_rates
from .system_model import ISACSystem


class CommunicationRate:
    """Compute local per-user bit-rate and efficiency diagnostics."""

    def __init__(self, system: ISACSystem):
        """
        Initialize Communication Rate.

        Parameters
        ----------
        system : ISACSystem
            ISAC system model
        """
        self.system = system

    def _combined_rates(
        self,
        p_comm: np.ndarray,
        b_comm: np.ndarray,
        p_isac: Optional[np.ndarray],
        b_isac: Optional[np.ndarray],
    ) -> np.ndarray:
        """Return comm rates plus an optional complete ISAC allocation pair."""
        if (p_isac is None) != (b_isac is None):
            raise ValueError("p_isac and b_isac must be provided together")
        rates = self.compute_rate(p_comm, b_comm, "comm")
        if p_isac is not None:
            rates = np.concatenate(
                [rates, self.compute_rate(p_isac, b_isac, "isac")]
            )
        return rates

    def compute_rate(self, p: np.ndarray, b: np.ndarray,
                     user_type: str = 'comm') -> np.ndarray:
        """
        Compute the local Shannon-form communication rate.

        R_k = b_k * log2(1 + SNR_k)

        where:
        - SNR_k = (p_k * β_k) / (N0 * b_k)

        Parameters
        ----------
        p : np.ndarray
            Power allocation
        b : np.ndarray
            Bandwidth allocation
        user_type : str
            'comm' for communication users, 'isac' for ISAC users

        Returns
        -------
        rates : np.ndarray
            Communication rates in bit/s
        """
        p = np.asarray(p, dtype=float)
        b = np.asarray(b, dtype=float)
        beta = self.system.beta_comm if user_type == 'comm' else self.system.beta_isac
        if user_type not in {'comm', 'isac'}:
            raise ValueError(f"Unknown user_type: {user_type}")
        if p.shape != beta.shape or b.shape != beta.shape:
            raise ValueError(
                f"{user_type} allocations must have shape {beta.shape}"
            )
        if np.any(p < 0) or np.any(b <= 0):
            raise ValueError("power must be non-negative and bandwidth positive")
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(b)):
            raise ValueError("allocations must contain only finite values")

        return stable_shannon_rates(p, b, beta, self.system.N0)

    def compute_sum_rate(self, p_comm: np.ndarray, b_comm: np.ndarray,
                         p_isac: Optional[np.ndarray] = None,
                         b_isac: Optional[np.ndarray] = None) -> float:
        """
        Compute sum communication rate.

        R_total = Σ_k R_k + Σ_l R_l (ISAC users)

        Parameters
        ----------
        p_comm : np.ndarray
            Power for communication users (K,)
        b_comm : np.ndarray
            Bandwidth for communication users (K,)
        p_isac : np.ndarray, optional
            Power for ISAC users (L,)
        b_isac : np.ndarray, optional
            Bandwidth for ISAC users (L,)

        Returns
        -------
        sum_rate : float
            Total communication rate
        """
        return float(np.sum(self._combined_rates(p_comm, b_comm, p_isac, b_isac)))

    def compute_min_rate(self, p_comm: np.ndarray, b_comm: np.ndarray,
                         p_isac: Optional[np.ndarray] = None,
                         b_isac: Optional[np.ndarray] = None) -> float:
        """
        Compute minimum communication rate (for fairness).

        min(R_k, R_l)

        Parameters
        ----------
        p_comm : np.ndarray
            Power for communication users (K,)
        b_comm : np.ndarray
            Bandwidth for communication users (K,)
        p_isac : np.ndarray, optional
            Power for ISAC users (L,)
        b_isac : np.ndarray, optional
            Bandwidth for ISAC users (L,)

        Returns
        -------
        min_rate : float
            Minimum communication rate
        """
        all_rates = self._combined_rates(p_comm, b_comm, p_isac, b_isac)
        return float(np.min(all_rates))

    def check_rate_constraints(self, p_comm: np.ndarray, b_comm: np.ndarray,
                               Gamma_c: float,
                               p_isac: Optional[np.ndarray] = None,
                               b_isac: Optional[np.ndarray] = None) -> Tuple[bool, np.ndarray]:
        """
        Check if communication rate constraints are satisfied.

        R_k ≥ Γc for all k

        Parameters
        ----------
        p_comm : np.ndarray
            Power for communication users (K,)
        b_comm : np.ndarray
            Bandwidth for communication users (K,)
        Gamma_c : float
            Rate threshold (bit/s)
        p_isac : np.ndarray, optional
            Power for ISAC users (L,)
        b_isac : np.ndarray, optional
            Bandwidth for ISAC users (L,)

        Returns
        -------
        satisfied : bool
            True if all rate constraints are met
        rates : np.ndarray
            Actual rates for all users
        """
        if not np.isfinite(Gamma_c) or Gamma_c < 0:
            raise ValueError("Gamma_c must be non-negative and finite")
        all_rates = self._combined_rates(p_comm, b_comm, p_isac, b_isac)
        satisfied = np.all(all_rates >= Gamma_c)

        return satisfied, all_rates

    def compute_spectral_efficiency(self, p: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute spectral efficiency (rate per unit bandwidth).

        η_k = R_k / b_k = log2(1 + SNR_k) [bps/Hz]

        Parameters
        ----------
        p : np.ndarray
            Power allocation
        b : np.ndarray
            Bandwidth allocation

        Returns
        -------
        eta : np.ndarray
            Spectral efficiency (bps/Hz)
        """
        rates = self.compute_rate(p, b, "comm")
        return rates / np.asarray(b, dtype=float)

    def compute_energy_efficiency(self, p: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute energy efficiency (rate per unit power).

        EE_k = R_k / p_k [bit/J]

        Parameters
        ----------
        p : np.ndarray
            Power allocation
        b : np.ndarray
            Bandwidth allocation

        Returns
        -------
        ee : np.ndarray
            Energy efficiency
        """
        rates = self.compute_rate(p, b, 'comm')
        p = np.asarray(p, dtype=float)
        efficiency = np.zeros_like(rates, dtype=float)
        np.divide(rates, p, out=efficiency, where=p > 0)
        return efficiency
