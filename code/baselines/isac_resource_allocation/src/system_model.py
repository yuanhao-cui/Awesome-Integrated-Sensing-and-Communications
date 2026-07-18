"""Synthetic ISAC system model used by the local allocation heuristic."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .numerics import stable_shannon_rates


@dataclass
class SystemParameters:
    """ISAC system parameters."""
    Nt: int = 32          # Number of transmit antennas
    Nr: int = 32          # Number of receive antennas
    Q: int = 3            # Number of sensing targets
    K: int = 3            # Number of communication users
    L: int = 1            # Number of ISAC users
    fc: float = 30e9      # Carrier frequency (Hz)
    P_total: float = 40.0 # Total transmit power (W)
    B_total: float = 100e6 # Total bandwidth (Hz)
    N0_dBm: float = -174.0 # Noise PSD (dBm/Hz)
    NF_dB: float = 10.0   # Noise figure (dB)

    @property
    def M(self) -> int:
        """Total number of objects."""
        return self.Q + self.K + self.L


class ISACSystem:
    """Synthetic system with sensing, communication, and joint-user objects."""

    def __init__(self, Nt: int = 32, Nr: int = 32, Q: int = 3, K: int = 3, L: int = 1,
                 fc: float = 30e9, P_total: float = 40.0, B_total: float = 100e6,
                 N0_dBm: float = -174.0, NF_dB: float = 10.0,
                 rng: Optional[np.random.Generator] = None):
        """
        Initialize ISAC system.

        Parameters
        ----------
        Nt : int
            Number of transmit antennas (default: 32)
        Nr : int
            Number of receive antennas (default: 32)
        Q : int
            Number of sensing targets (default: 3)
        K : int
            Number of communication users (default: 3)
        L : int
            Number of ISAC users (default: 1)
        fc : float
            Carrier frequency in Hz (default: 30 GHz)
        P_total : float
            Total transmit power in Watts (default: 40W)
        B_total : float
            Total bandwidth in Hz (default: 100 MHz)
        N0_dBm : float
            Noise power spectral density in dBm/Hz (default: -174 dBm/Hz)
        NF_dB : float
            Noise figure in dB (default: 10 dB)
        rng : np.random.Generator, optional
            Random number generator for reproducibility
        """
        positive_counts = {"Nt": Nt, "Nr": Nr, "Q": Q}
        nonnegative_counts = {"K": K, "L": L}
        for name, value in {**positive_counts, **nonnegative_counts}.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer")
        if any(value <= 0 for value in positive_counts.values()):
            raise ValueError("Nt, Nr, and Q must be positive")
        if any(value < 0 for value in nonnegative_counts.values()):
            raise ValueError("K and L must be non-negative")
        if K + L < 1:
            raise ValueError("at least one communication or joint user is required")
        for name, value in {
            "fc": fc,
            "P_total": P_total,
            "B_total": B_total,
        }.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isfinite(N0_dBm) or not np.isfinite(NF_dB):
            raise ValueError("N0_dBm and NF_dB must be finite")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")

        self.params = SystemParameters(
            int(Nt), int(Nr), int(Q), int(K), int(L),
            float(fc), float(P_total), float(B_total),
            float(N0_dBm), float(NF_dB),
        )
        self.rng = rng if rng is not None else np.random.default_rng(42)

        # Convert dBm/Hz to W/Hz and reject finite inputs that leave the
        # representable positive floating-point domain.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            self.N0 = float(
                np.power(10.0, (float(N0_dBm) + float(NF_dB)) / 10.0)
                / 1000.0
            )
        if not np.isfinite(self.N0) or self.N0 <= 0.0:
            raise ValueError(
                "N0_dBm and NF_dB must yield a finite positive noise PSD"
            )

        # Initialize channels
        self._init_channels()

    def _init_channels(self):
        """Initialize synthetic channels for testing."""
        p = self.params

        # Sensing target positions (randomized)
        self.target_positions = self.rng.uniform(10, 100, p.Q)  # Range 10-100m
        self.target_angles = self.rng.uniform(-np.pi/3, np.pi/3, p.Q)  # Azimuth angles

        # Communication user positions
        self.user_positions = self.rng.uniform(50, 200, p.K)  # Range 50-200m
        self.user_angles = self.rng.uniform(-np.pi/2, np.pi/2, p.K)

        # ISAC user positions (communication + sensing)
        self.isac_positions = self.rng.uniform(30, 150, p.L)
        self.isac_angles = self.rng.uniform(-np.pi/4, np.pi/4, p.L)
        # Preserve the historical single-user convenience attributes without
        # using them internally for multi-user computations.
        self.isac_position = (
            float(self.isac_positions[0]) if p.L > 0 else None
        )
        self.isac_angle = float(self.isac_angles[0]) if p.L > 0 else None

        # Local free-space-style path-loss proxy.
        self.alpha_sensing = self._compute_path_loss(self.target_positions)
        self.alpha_comm = self._compute_path_loss(self.user_positions)
        self.alpha_isac = self._compute_path_loss(self.isac_positions)

        # Radar cross sections for sensing targets
        self.rcs = self.rng.uniform(1.0, 10.0, p.Q)  # m^2

        # Channel gains (normalized)
        self.beta_sensing = self._compute_channel_gain(self.target_positions, self.target_angles)
        self.beta_comm = self._compute_channel_gain(self.user_positions, self.user_angles)
        self.beta_isac = self._compute_channel_gain(
            self.isac_positions, self.isac_angles
        )

    def _compute_path_loss(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute path loss in linear scale.

        Local path-loss proxy: PL = 32.4 + 20 log10(d) + 20 log10(fc_GHz) dB.
        """
        fc_GHz = self.params.fc / 1e9
        distances = np.asarray(distances, dtype=float)
        if np.any(distances <= 0.0) or not np.all(np.isfinite(distances)):
            raise ValueError("path-loss distances must be positive and finite")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            pl_dB = (
                32.4
                + 20.0 * np.log10(distances)
                + 20.0 * np.log10(fc_GHz)
            )
            path_gain = np.power(10.0, -pl_dB / 10.0)
        if not np.all(np.isfinite(path_gain)) or np.any(path_gain <= 0.0):
            raise ValueError(
                "fc and distance must yield finite positive path gains"
            )
        return path_gain

    def _compute_channel_gain(self, distances: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """
        Compute channel gain including path loss and small-scale fading.

        β = α * |h|^2 where h ~ CN(0, I)
        """
        # Small-scale fading: Rayleigh
        h = (self.rng.normal(0, 1/np.sqrt(2), (len(distances), self.params.Nt)) +
             1j * self.rng.normal(0, 1/np.sqrt(2), (len(distances), self.params.Nt)))

        # Array response
        d_lambda = 0.5  # Half-wavelength spacing
        array_response = np.exp(1j * 2 * np.pi * d_lambda * np.outer(np.sin(angles),
                                                                       np.arange(self.params.Nt)))

        # Combined channel
        h_combined = h * array_response

        # Path loss
        alpha = self._compute_path_loss(distances)

        # Channel gain: β = α * ||h||^2 / Nt
        return alpha * np.linalg.norm(h_combined, axis=1)**2 / self.params.Nt

    def get_channel_matrix(self, idx: int, target_type: str = 'sensing') -> np.ndarray:
        """
        Get channel matrix for a specific target/user.

        Parameters
        ----------
        idx : int
            Index of target or user
        target_type : str
            'sensing', 'comm', or 'isac'

        Returns
        -------
        h : np.ndarray
            Channel vector of shape (Nt,)
        """
        if target_type == 'sensing':
            h = (self.rng.normal(0, 1/np.sqrt(2), self.params.Nt) +
                 1j * self.rng.normal(0, 1/np.sqrt(2), self.params.Nt))
            return h * np.sqrt(self.beta_sensing[idx])
        elif target_type == 'comm':
            h = (self.rng.normal(0, 1/np.sqrt(2), self.params.Nt) +
                 1j * self.rng.normal(0, 1/np.sqrt(2), self.params.Nt))
            return h * np.sqrt(self.beta_comm[idx])
        elif target_type == 'isac':
            h = (self.rng.normal(0, 1/np.sqrt(2), self.params.Nt) +
                 1j * self.rng.normal(0, 1/np.sqrt(2), self.params.Nt))
            return h * np.sqrt(self.beta_isac[idx])
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

    def get_snr(self, power: np.ndarray, bandwidth: np.ndarray) -> np.ndarray:
        """
        Compute SNR for each sensing target.

        SNR_q = (p_q * β_q * σ_q) / (N0 * b_q)

        Parameters
        ----------
        power : np.ndarray
            Power allocation (Q,)
        bandwidth : np.ndarray
            Bandwidth allocation (Q,)

        Returns
        -------
        snr : np.ndarray
            SNR for each target (Q,)
        """
        power = np.asarray(power, dtype=float)
        bandwidth = np.asarray(bandwidth, dtype=float)
        expected = (self.params.Q,)
        if power.shape != expected or bandwidth.shape != expected:
            raise ValueError(f"power and bandwidth must have shape {expected}")
        if (
            np.any(power < 0.0)
            or np.any(bandwidth <= 0.0)
            or not np.all(np.isfinite(power))
            or not np.all(np.isfinite(bandwidth))
        ):
            raise ValueError(
                "power must be finite and non-negative; bandwidth must be "
                "finite and positive"
            )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            snr = (
                power * self.beta_sensing * self.rcs
                / (self.N0 * bandwidth)
            )
        if not np.all(np.isfinite(snr)):
            raise ValueError("sensing SNR is outside the finite numerical domain")
        return snr

    def get_comm_snr(self, power: np.ndarray, bandwidth: np.ndarray) -> np.ndarray:
        """
        Compute SNR for communication users.

        SNR_k = (p_k * β_k) / (N0 * b_k)

        Parameters
        ----------
        power : np.ndarray
            Power allocation (K,)
        bandwidth : np.ndarray
            Bandwidth allocation (K,)

        Returns
        -------
        snr : np.ndarray
            SNR for each user (K,)
        """
        power = np.asarray(power, dtype=float)
        bandwidth = np.asarray(bandwidth, dtype=float)
        expected = (self.params.K,)
        if power.shape != expected or bandwidth.shape != expected:
            raise ValueError(f"power and bandwidth must have shape {expected}")
        if (
            np.any(power < 0.0)
            or np.any(bandwidth <= 0.0)
            or not np.all(np.isfinite(power))
            or not np.all(np.isfinite(bandwidth))
        ):
            raise ValueError(
                "power must be finite and non-negative; bandwidth must be "
                "finite and positive"
            )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            snr = power * self.beta_comm / (self.N0 * bandwidth)
        if not np.all(np.isfinite(snr)):
            raise ValueError(
                "communication SNR is outside the finite numerical domain"
            )
        return snr

    def validate_allocations(self, p: np.ndarray, b: np.ndarray) -> bool:
        """
        Validate power and bandwidth allocations.

        Parameters
        ----------
        p : np.ndarray
            Power allocation vector (M,) = [p_sensing, p_comm, p_isac]
        b : np.ndarray
            Bandwidth allocation vector (M,) = [b_sensing, b_comm, b_isac]

        Returns
        -------
        valid : bool
            True if allocations satisfy constraints
        """
        p = np.asarray(p)
        b = np.asarray(b)

        if p.shape != (self.params.M,) or b.shape != (self.params.M,):
            return False
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(b)):
            return False

        # Bandwidth is strictly positive because every SNR divides by it.
        if np.any(p < 0) or np.any(b <= 0):
            return False

        # Budget constraints
        if not np.isclose(np.sum(p), self.params.P_total, rtol=1e-3):
            return False
        if not np.isclose(np.sum(b), self.params.B_total, rtol=1e-3):
            return False

        return True

    def compute_communication_rate(self, p: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute communication rate for each user.

        R_k = b_k * log2(1 + SNR_k) [bit/s]

        Parameters
        ----------
        p : np.ndarray
            Power allocation (K + L,)
        b : np.ndarray
            Bandwidth allocation (K + L,)

        Returns
        -------
        rates : np.ndarray
            Communication rates (K + L,)
        """
        p = np.asarray(p, dtype=float)
        b = np.asarray(b, dtype=float)
        expected = self.params.K + self.params.L
        if p.shape != (expected,) or b.shape != (expected,):
            raise ValueError(f"p and b must have shape ({expected},)")
        if np.any(p < 0) or np.any(b <= 0):
            raise ValueError("power must be non-negative and bandwidth positive")
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(b)):
            raise ValueError("p and b must contain only finite values")

        beta = np.concatenate([self.beta_comm, self.beta_isac])
        return stable_shannon_rates(p, b, beta, self.N0)

    def compute_sensing_snr(self, p_sensing: np.ndarray, b_sensing: np.ndarray) -> np.ndarray:
        """
        Compute sensing SNR for each target.

        SNR_q = (p_q * β_q * σ_q) / (N0 * b_q)

        Parameters
        ----------
        p_sensing : np.ndarray
            Power allocation for sensing targets (Q,)
        b_sensing : np.ndarray
            Bandwidth allocation for sensing targets (Q,)

        Returns
        -------
        snr : np.ndarray
            Sensing SNR for each target (Q,)
        """
        return self.get_snr(p_sensing, b_sensing)

    @property
    def total_objects(self) -> int:
        """Total number of objects (M = Q + K + L)."""
        return self.params.M
