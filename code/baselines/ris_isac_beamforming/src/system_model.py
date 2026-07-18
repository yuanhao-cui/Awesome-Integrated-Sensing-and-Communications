"""Internally consistent synthetic RIS-ISAC system model.

Defines the physical layer model including:
- Received signal at users (communication)
- Reflected signal for radar sensing
- SINR, SNR, and rate computations

The model is an educational narrowband surrogate.  It uses a column-channel
convention throughout: :meth:`effective_channel` returns ``h_k`` and every
received projection is evaluated as ``h_k^H w`` using :func:`numpy.vdot`.
"""

import numpy as np
from typing import Optional
from .channel_model import RISChannelModel
from .numerics import (
    db_to_linear,
    normalize_unit_phases,
    stable_effective_channel,
    stable_link_rate,
    stable_link_sinr,
    stable_sensing_snr,
)


class RIS_ISAC_System:
    """RIS-ISAC system model for joint communication and sensing.

    Models a multi-antenna BS assisted by a passive RIS serving K
    single-antenna users while performing radar sensing of a target.

    Attributes:
        M: BS antennas.
        K: Single-antenna users.
        L: RIS elements.
        P_max: Maximum transmit power (W).
        noise_power: Noise variance (W).
        sinr_thresh_dB: SINR threshold in dB.
        channels: Dictionary of channel matrices.
        theta: RIS phase shift vector (L,) with |theta_l| = 1.
    """

    def __init__(
        self,
        M: int = 4,
        K: int = 2,
        L: int = 30,
        P_max: float = 10e-3,
        noise_power: float = 3.98e-12,
        sinr_thresh_dB: float = 10.0,
        seed: Optional[int] = None,
    ):
        """Initialize RIS-ISAC system.

        Args:
            M: Number of BS antennas.
            K: Number of single-antenna users.
            L: Number of RIS elements.
            P_max: Maximum transmit power in W (default 0.01 W).
            noise_power: Noise variance in W (default 3.98e-12 W).
            sinr_thresh_dB: Local communication SINR threshold in dB.
            seed: Random seed for channel generation.
        """
        for name, value in (("M", M), ("K", K), ("L", L)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(P_max) or P_max <= 0:
            raise ValueError("P_max must be finite and positive")
        if not np.isfinite(noise_power) or noise_power <= 0:
            raise ValueError("noise_power must be finite and positive")
        if not np.isfinite(sinr_thresh_dB):
            raise ValueError("sinr_thresh_dB must be finite")
        self.M = M
        self.K = K
        self.L = L
        self.P_max = P_max
        self.noise_power = noise_power
        self.sinr_thresh_dB = sinr_thresh_dB
        self.sinr_thresh = db_to_linear(
            sinr_thresh_dB, "sinr_thresh_dB"
        )

        # Generate channels
        channel_model = RISChannelModel(M=M, K=K, L=L, seed=seed)
        self.channels = channel_model.generate_all_channels()

        # Initialize RIS phases randomly
        self.theta = np.exp(1j * np.random.default_rng(seed).uniform(0, 2 * np.pi, L))

    def ris_diagonal_matrix(self) -> np.ndarray:
        """Construct RIS diagonal matrix Θ = diag(θ).

        Returns:
            Diagonal matrix Θ of shape (L, L).
        """
        return np.diag(self.theta)

    def effective_channel(self, user_idx: int) -> np.ndarray:
        """Compute effective channel for user k.

        The stored row coefficients are returned as a column-vector channel
        ``h_k``.  Received projections are always ``h_k^H w``.

        Args:
            user_idx: User index k (0-based).

        Returns:
            Effective channel vector of shape (M,).
        """
        if not 0 <= user_idx < self.K:
            raise IndexError(f"user_idx must be in [0, {self.K})")
        return stable_effective_channel(
            self.channels["h_d"][user_idx],
            self.channels["G"][user_idx],
            self.channels["H_BR"],
            self.theta,
        )

    def compute_sinr(
        self, user_idx: int, w_k: np.ndarray, W_interf: np.ndarray
    ) -> float:
        """Compute SINR for a user given beamforming vectors.

        SINR_k = |h_k^H w_k|^2 / (Σ_{j≠k} |h_k^H w_j|^2 + σ^2)

        Args:
            user_idx: Zero-based user index defining the effective channel.
            w_k: Beamforming vector for user k (M,).
            W_interf: Stacked interference beamformers (M, K-1).

        Returns:
            SINR value (linear scale).
        """
        if not 0 <= user_idx < self.K:
            raise IndexError(f"user_idx must be in [0, {self.K})")
        w_k = np.asarray(w_k, dtype=complex)
        W_interf = np.asarray(W_interf, dtype=complex)
        if w_k.shape != (self.M,):
            raise ValueError(f"w_k must have shape ({self.M},)")
        if W_interf.ndim != 2 or W_interf.shape[0] != self.M:
            raise ValueError(f"W_interf must have shape ({self.M}, J)")
        if not np.all(np.isfinite(w_k)) or not np.all(np.isfinite(W_interf)):
            raise ValueError("beamformers must be finite")
        return stable_link_sinr(
            self.channels["h_d"][user_idx],
            self.channels["G"][user_idx],
            self.channels["H_BR"],
            self.theta,
            w_k,
            W_interf,
            self.noise_power,
        )

    def compute_snr_sensing(self, W: np.ndarray) -> float:
        """Compute radar sensing SNR for independent unit-variance streams.

        SNR_s = h_s^H W W^H h_s / σ^2
              = Σ_k |h_s^H w_k|^2 / σ^2

        where ``h_s`` is the sensing channel through the RIS and the columns of
        ``W`` multiply mutually independent, unit-variance data symbols.

        Args:
            W: Beamforming matrix (M, K), one data stream per column.

        Returns:
            Sensing SNR (linear scale).
        """
        W = np.asarray(W, dtype=complex)
        if W.shape != (self.M, self.K) or not np.all(np.isfinite(W)):
            raise ValueError(
                f"W must be a finite matrix of shape {(self.M, self.K)}"
            )
        return stable_sensing_snr(
            self.channels["a_bs"],
            self.channels["a_ris"],
            self.channels["H_BR"],
            self.theta,
            W,
            self.noise_power,
        )

    def compute_sum_rate(self, W: np.ndarray) -> float:
        """Compute sum rate over all users.

        R_k = log2(1 + SINR_k), Sum rate = Σ_k R_k

        Args:
            W: Beamforming matrix of shape (M, K), columns are w_k.

        Returns:
            Sum rate in bits/s/Hz.
        """
        W = np.asarray(W, dtype=complex)
        if W.shape != (self.M, self.K) or not np.all(np.isfinite(W)):
            raise ValueError(f"W must be a finite matrix of shape {(self.M, self.K)}")
        sum_rate = 0.0
        for k in range(self.K):
            interferers = np.delete(W, k, axis=1)
            sum_rate += stable_link_rate(
                self.channels["h_d"][k],
                self.channels["G"][k],
                self.channels["H_BR"],
                self.theta,
                W[:, k],
                interferers,
                self.noise_power,
            )
        return float(sum_rate)

    def set_ris_phases(self, theta: np.ndarray) -> None:
        """Set RIS phase shifts with unit-modulus enforcement.

        Args:
            theta: Complex phase vector (L,). Will be normalized to |θ_l| = 1.
        """
        theta = np.asarray(theta, dtype=complex)
        if theta.shape != (self.L,):
            raise ValueError(f"theta must have shape ({self.L},)")
        self.theta = normalize_unit_phases(theta)

    def reset_channels(self, seed: Optional[int] = None) -> None:
        """Regenerate all channel matrices.

        Args:
            seed: New random seed (None for random).
        """
        cm = RISChannelModel(M=self.M, K=self.K, L=self.L, seed=seed)
        self.channels = cm.generate_all_channels()
