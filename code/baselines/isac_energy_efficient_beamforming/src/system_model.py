"""Deterministic system model for the validated energy-efficiency slice.

The steering convention follows (6)--(7) of Zou et al., IEEE TCOM 2024:
``exp(-j 2 pi d m cos(theta) / wavelength)``.  The public manuscript does
not specify a channel distribution or receiver-noise value for its numerical
figures, so this module labels its seeded CN(0, 1) channel as synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .numerics import stable_sinr, stable_squared_norm


def dbm_to_watt(value_dbm: float) -> float:
    """Convert finite dBm to a positive representable binary64 watt value."""

    value_dbm = float(value_dbm)
    if not np.isfinite(value_dbm):
        raise ValueError("value_dbm must be finite")
    log_watt = (value_dbm / 10.0 - 3.0) * math.log(10.0)
    if log_watt > math.log(np.finfo(float).max):
        raise OverflowError("dBm value converts above the binary64 watt range")
    if log_watt < math.log(np.nextafter(0.0, 1.0)):
        raise FloatingPointError(
            "dBm value converts below the positive binary64 watt range"
        )
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        result = float(np.exp(log_watt))
    if not np.isfinite(result):
        raise OverflowError("dBm value converts above the binary64 watt range")
    if result == 0.0:
        raise FloatingPointError(
            "dBm value converts below the positive binary64 watt range"
        )
    return result


@dataclass(frozen=True)
class PaperParameterProvenance:
    """Publicly reported parameters and explicit local assumptions."""

    source_doi: str = "10.1109/TCOMM.2024.3369696"
    public_parameters: tuple[str, ...] = (
        "N=20 receive antennas",
        "L=30 snapshots",
        "P_max=30 dBm",
        "epsilon=0.35",
        "theta=90 degrees",
    )
    local_assumptions: tuple[str, ...] = (
        "single user",
        "seeded CN(0,1) communication channel",
        "communication and sensing noise powers are explicit inputs",
        "unit target reflection magnitude",
    )


class ISACSystemModel:
    """ISAC model used by the equation-level reference implementation.

    This class intentionally enforces the paper's ``K <= M <= N`` array
    regime.  Power-valued noise inputs use dBm and are converted to watts;
    this avoids the former ambiguous ``*_db`` convention.
    """

    def __init__(
        self,
        M: int = 16,
        K: int = 1,
        N: int = 20,
        P_max_dbm: float = 30.0,
        P0_dbm: float = 30.0,
        epsilon: float = 0.35,
        sigma_c_dbm: float = -80.0,
        sigma_s_dbm: float = -80.0,
        L: int = 30,
        wavelength: float = 1.0,
        d: float = 0.5,
        seed: int | None = 0,
    ) -> None:
        for name, value in (("M", M), ("K", K), ("N", N), ("L", L)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not K <= M <= N:
            raise ValueError("the reference model requires K <= M <= N")
        if not 0.0 < epsilon <= 1.0:
            raise ValueError("epsilon must lie in (0, 1]")
        if not np.isfinite(wavelength) or wavelength <= 0.0:
            raise ValueError("wavelength must be finite and positive")
        if not np.isfinite(d) or d <= 0.0:
            raise ValueError("d must be finite and positive")

        self.M = M
        self.K = K
        self.N = N
        self.L = L
        self.epsilon = float(epsilon)
        self.wavelength = float(wavelength)
        self.d = float(d)
        self.P_max = dbm_to_watt(P_max_dbm)
        self.P0 = dbm_to_watt(P0_dbm)
        self.sigma_c2 = dbm_to_watt(sigma_c_dbm)
        self.sigma_s2 = dbm_to_watt(sigma_s_dbm)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.H = self._draw_synthetic_channels()

    def _draw_synthetic_channels(self) -> np.ndarray:
        return (
            self.rng.standard_normal((self.K, self.M))
            + 1j * self.rng.standard_normal((self.K, self.M))
        ) / np.sqrt(2.0)

    def _steering(self, size: int, theta_rad: float) -> np.ndarray:
        if not np.isfinite(theta_rad):
            raise ValueError("theta_rad must be finite")
        indices = np.arange(size, dtype=float)
        phase = (
            -2.0
            * np.pi
            * self.d
            * indices
            * np.cos(theta_rad)
            / self.wavelength
        )
        return np.exp(1j * phase)

    def _steering_derivative(self, size: int, theta_rad: float) -> np.ndarray:
        steering = self._steering(size, theta_rad)
        indices = np.arange(size, dtype=float)
        coefficient = (
            1j
            * 2.0
            * np.pi
            * self.d
            * indices
            * np.sin(theta_rad)
            / self.wavelength
        )
        return coefficient * steering

    def steering_vector_tx(self, theta_rad: float) -> np.ndarray:
        """Return the M-element transmit steering vector."""

        return self._steering(self.M, theta_rad)

    def steering_vector_rx(self, theta_rad: float) -> np.ndarray:
        """Return the N-element receive steering vector."""

        return self._steering(self.N, theta_rad)

    def steering_derivative_tx(self, theta_rad: float) -> np.ndarray:
        """Return the analytic derivative of the transmit steering vector."""

        return self._steering_derivative(self.M, theta_rad)

    def steering_derivative_rx(self, theta_rad: float) -> np.ndarray:
        """Return the analytic derivative of the receive steering vector."""

        return self._steering_derivative(self.N, theta_rad)

    def get_channel(self, k: int) -> np.ndarray:
        if not 0 <= k < self.K:
            raise IndexError(f"user index {k} is outside [0, {self.K})")
        return self.H[k].copy()

    def get_csi(self) -> np.ndarray:
        return self.H.copy()

    def set_csi(self, channels: np.ndarray) -> None:
        channels = np.asarray(channels, dtype=complex)
        if channels.shape != (self.K, self.M):
            raise ValueError(f"channels must have shape {(self.K, self.M)}")
        if not np.all(np.isfinite(channels)):
            raise ValueError("channels must be finite")
        self.H = channels.copy()

    def regenerate_channels(self, seed: int | None = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.H = self._draw_synthetic_channels()

    def compute_sinr(self, k: int, W: np.ndarray) -> float:
        W = np.asarray(W, dtype=complex)
        if W.shape != (self.M, self.K):
            raise ValueError(f"W must have shape {(self.M, self.K)}")
        return stable_sinr(k, self.get_channel(k), W, self.sigma_c2)

    def compute_sinr_vector(self, W: np.ndarray) -> np.ndarray:
        return np.asarray([self.compute_sinr(k, W) for k in range(self.K)])

    def compute_total_power(self, W: np.ndarray) -> float:
        W = np.asarray(W, dtype=complex)
        if W.shape != (self.M, self.K):
            raise ValueError(f"W must have shape {(self.M, self.K)}")
        return stable_squared_norm(W)


__all__ = ["ISACSystemModel", "PaperParameterProvenance", "dbm_to_watt"]
