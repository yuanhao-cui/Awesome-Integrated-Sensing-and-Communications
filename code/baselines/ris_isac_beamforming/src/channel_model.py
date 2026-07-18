"""Seeded synthetic RIS-assisted channels for the educational surrogate.

Implements Rician fading channels for BS-RIS, RIS-user, RIS-target,
and direct BS-user/BS-target links.

The dimensions and effective-channel topology are inspired by Rang Liu et al.,
IEEE TWC 2024.  The geometry and normalization below are local assumptions,
not a reproduction of that paper's simulation channel.
"""

import numpy as np
from typing import Optional


class RISChannelModel:
    """Generates RIS-assisted channels with Rician fading.

    Attributes:
        M: Number of BS antennas.
        K: Number of single-antenna users.
        L: Number of RIS elements.
        K_r: Rician fading factor.
        rng: NumPy random generator.
    """

    def __init__(
        self,
        M: int,
        K: int,
        L: int,
        K_r: float = 3.0,
        seed: Optional[int] = None,
    ):
        """Initialize channel model.

        Args:
            M: BS antenna count.
            K: User count.
            L: RIS element count.
            K_r: Local linear-scale Rician factor (default 3).
            seed: Random seed for reproducibility.
        """
        for name, value in (("M", M), ("K", K), ("L", L)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(K_r) or K_r < 0:
            raise ValueError("K_r must be finite and non-negative")
        self.M = M
        self.K = K
        self.L = L
        self.K_r = K_r
        self.rng = np.random.default_rng(seed)

    def _rician_channel(self, n_rx: int, n_tx: int, d_los: np.ndarray) -> np.ndarray:
        """Generate a single Rician fading channel matrix.

        Implements the Rician model: H = sqrt(K_r/(K_r+1)) * H_los
                                       + sqrt(1/(K_r+1)) * H_nlos

        Args:
            n_rx: Number of receive antennas.
            n_tx: Number of transmit antennas.
            d_los: Deterministic LoS matrix (n_rx x n_tx).

        Returns:
            Channel matrix of shape (n_rx, n_tx), complex-valued.
        """
        # NLoS component: Rayleigh fading
        h_nlos = (
            self.rng.standard_normal((n_rx, n_tx))
            + 1j * self.rng.standard_normal((n_rx, n_tx))
        ) / np.sqrt(2)

        # Rician combination
        K = self.K_r
        h = np.sqrt(K / (K + 1)) * d_los + np.sqrt(1 / (K + 1)) * h_nlos
        return h

    def _uniform_linear_array_los(self, n1: int, n2: int) -> np.ndarray:
        """Generate LoS component based on uniform linear array (ULA) geometry.

        Implements array response vectors for far-field plane-wave propagation:
            a(φ) = [1, e^{jπ sin(φ)}, ..., e^{j(N-1)π sin(φ)}]^T

        Args:
            n1: Number of antennas at receiver side.
            n2: Number of antennas at transmitter side.

        Returns:
            LoS channel matrix (n1 x n2).
        """
        # Random AoA/AoD angles
        phi_rx = self.rng.uniform(-np.pi / 2, np.pi / 2)
        phi_tx = self.rng.uniform(-np.pi / 2, np.pi / 2)

        a_rx = np.exp(1j * np.pi * np.arange(n1) * np.sin(phi_rx))
        a_tx = np.exp(1j * np.pi * np.arange(n2) * np.sin(phi_tx))

        return np.outer(a_rx, a_tx.conj())

    def generate_bs_ris_channel(self) -> np.ndarray:
        """Generate the local BS-to-RIS coefficient matrix (L x M).

        Returns:
            Channel matrix H_BR of shape (L, M).
        """
        d_los = self._uniform_linear_array_los(self.L, self.M)
        return self._rician_channel(self.L, self.M, d_los)

    def generate_ris_user_channel(self) -> np.ndarray:
        """Generate local RIS-to-user coefficient rows G (K x L).

        Each row is composed as part of the column channel returned by
        :meth:`RIS_ISAC_System.effective_channel`; it is deliberately not
        labelled ``g_k^H`` because this surrogate uses a declared synthetic
        coefficient convention.

        Returns:
            Channel matrix G of shape (K, L).
        """
        G = np.zeros((self.K, self.L), dtype=complex)
        for k in range(self.K):
            d_los = self._uniform_linear_array_los(1, self.L)
            G[k : k + 1, :] = self._rician_channel(1, self.L, d_los)
        return G

    def generate_bs_user_channel(self) -> np.ndarray:
        """Generate local direct-channel coefficient rows h_d (K x M).

        Direct link from BS to each user.

        Returns:
            Channel matrix h_d of shape (K, M).
        """
        h_d = np.zeros((self.K, self.M), dtype=complex)
        for k in range(self.K):
            d_los = self._uniform_linear_array_los(1, self.M)
            h_d[k : k + 1, :] = self._rician_channel(1, self.M, d_los)
        return h_d

    def generate_ris_target_channel(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Generate local BS/RIS steering vectors for one synthetic target.

        Returns:
            BS vector, RIS vector, and the sampled angle in radians.
        """
        phi_target = self.rng.uniform(-np.pi / 6, np.pi / 6)
        # BS array response toward target
        a_bs = np.exp(1j * np.pi * np.arange(self.M) * np.sin(phi_target))
        # RIS array response toward target
        a_ris = np.exp(1j * np.pi * np.arange(self.L) * np.sin(phi_target))
        return a_bs, a_ris, float(phi_target)

    def generate_all_channels(self) -> dict:
        """Generate complete set of channels for ISAC system.

        Returns:
            Dictionary with keys:
                'H_BR': BS-to-RIS (L x M)
                'G': RIS-to-users (K x L)
                'h_d': BS-to-users direct (K x M)
                'a_bs': BS array response for radar (M,)
                'a_ris': RIS array response for radar (L,)
        """
        a_bs, a_ris, target_angle = self.generate_ris_target_channel()
        return {
            "H_BR": self.generate_bs_ris_channel(),
            "G": self.generate_ris_user_channel(),
            "h_d": self.generate_bs_user_channel(),
            "a_bs": a_bs,
            "a_ris": a_ris,
            "target_angle_rad": target_angle,
        }
