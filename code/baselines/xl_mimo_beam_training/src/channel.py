"""Near-field channel model for XL-MIMO systems.

Implements a local scalar spherical-wave/free-space surrogate for an
extremely large-scale MIMO array.  It evaluates the element-to-user distance
directly; it is neither a Fresnel-series expansion nor a Kirchhoff diffraction
integral.

Reference:
    Section III-A of the paper describes the near-field channel model.
    The spherical wavefront model accounts for distance-dependent phase
    variations across the array aperture.

    For antenna n at position d_n and user at distance r, angle theta:
        h_n = (alpha / r_n) * exp(-j * 2*pi / lambda * r_n)

    where r_n = sqrt(r^2 + d_n^2 - 2*r*d_n*sin(theta)) is the distance
    from antenna n to the user (spherical wave model).
"""

from typing import Optional, Tuple

import numpy as np


class NearFieldChannel:
    """Near-field channel model with spherical wavefront propagation.

    Models the channel between a uniform linear array (ULA) and a single-antenna
    user in the radiative near-field region where the planar-wave assumption
    is invalid.

    Args:
        num_antennas: Number of antennas in the ULA (N_t).
        wavelength: Carrier wavelength lambda (meters).
        antenna_spacing: Inter-element spacing (meters). Default: lambda/2.
        bandwidth: System bandwidth (Hz). Default: 1e8 (100 MHz).
        rng: Optional NumPy generator for all stochastic paths and noise.
    """

    def __init__(
        self,
        num_antennas: int = 256,
        wavelength: float = 0.01,  # 30 GHz -> lambda = 1 cm
        antenna_spacing: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        if not isinstance(num_antennas, (int, np.integer)) or num_antennas < 1:
            raise ValueError("num_antennas must be a positive integer")
        if not np.isfinite(wavelength) or wavelength <= 0:
            raise ValueError("wavelength must be positive and finite")
        if antenna_spacing is None:
            antenna_spacing = wavelength / 2.0
        if not np.isfinite(antenna_spacing) or antenna_spacing <= 0:
            raise ValueError("antenna_spacing must be positive and finite")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        self.num_antennas = num_antennas
        self.wavelength = wavelength
        self.antenna_spacing = antenna_spacing
        self.rng = np.random.default_rng() if rng is None else rng

        # Antenna positions (centered ULA)
        self.positions = (
            np.arange(num_antennas) - (num_antennas - 1) / 2.0
        ) * self.antenna_spacing

    def generate_channel(
        self,
        distance: float,
        angle: float,
        path_loss_dB: float = 0.0,
        num_paths: int = 1,
        angle_spread: float = 0.0,
    ) -> np.ndarray:
        """Generate a near-field channel vector.

        Args:
            distance: User distance from the array center (meters).
            angle: User angle of departure in radians (broadside = 0).
            path_loss_dB: Additional path loss in dB.
            num_paths: Number of multipath components.
            angle_spread: Angular spread for multipath (radians).

        Returns:
            Channel vector h of shape (num_antennas,) as complex numpy array.

        Note:
            For single-path (line-of-sight), uses exact spherical wave model.
            For multi-path, adds scattered components with random angles
            within the angular spread.
        """
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("distance must be positive and finite")
        if not np.isfinite(angle):
            raise ValueError("angle must be finite")
        if not np.isfinite(path_loss_dB) or path_loss_dB < 0:
            raise ValueError("path_loss_dB must be non-negative and finite")
        if not isinstance(num_paths, (int, np.integer)) or num_paths < 1:
            raise ValueError("num_paths must be a positive integer")
        if not np.isfinite(angle_spread) or angle_spread < 0:
            raise ValueError("angle_spread must be non-negative and finite")

        # Free-space amplitude is lambda/(4*pi*r_n) in each element.  The
        # component helper supplies the 1/r_n factor, while path_loss_dB is an
        # additional attenuation (hence the negative sign).
        alpha = self.wavelength / (4 * np.pi)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            attenuation = float(np.power(10.0, -path_loss_dB / 20.0))
        if not np.isfinite(attenuation) or attenuation <= 0:
            raise ValueError("path_loss_dB produces an unrepresentable attenuation")
        alpha *= attenuation

        h = np.zeros(self.num_antennas, dtype=np.complex128)

        # Line-of-sight (dominant path)
        h += self._spherical_wave_component(distance, angle, alpha)

        # Additional scattered paths
        for _ in range(1, num_paths):
            scattered_angle = angle + self.rng.uniform(
                -angle_spread, angle_spread
            )
            scattered_distance = distance * self.rng.uniform(0.95, 1.05)
            scattered_alpha = alpha * self.rng.rayleigh(0.3)
            scattered_phase = self.rng.uniform(0, 2 * np.pi)
            h += self._spherical_wave_component(
                scattered_distance, scattered_angle, scattered_alpha
            ) * np.exp(1j * scattered_phase)

        return h

    def _spherical_wave_component(
        self, distance: float, angle: float, amplitude: float
    ) -> np.ndarray:
        """Compute a single spherical wave channel component.

        Args:
            distance: Propagation distance (meters).
            angle: Departure angle (radians).
            amplitude: Path amplitude.

        Returns:
            Complex channel vector for this path component.
        """
        # Distance from each antenna to the user
        r_n = np.hypot(
            distance - self.positions * np.sin(angle),
            self.positions * np.cos(angle),
        )
        if np.any(r_n <= 0):
            raise ValueError("channel geometry places a user on an antenna element")
        # Spherical wave model: amplitude decay + phase rotation
        phase_cycles = np.remainder(r_n, self.wavelength) / self.wavelength
        h_n = (amplitude / r_n) * np.exp(-1j * 2 * np.pi * phase_cycles)
        if not np.all(np.isfinite(h_n)):
            raise ValueError("channel geometry is outside the representable domain")
        return h_n

    def generate_channel_batch(
        self,
        num_samples: int,
        distance_range: Tuple[float, float] = (10.0, 100.0),
        angle_range: Tuple[float, float] = (-np.pi / 3, np.pi / 3),
        num_paths: int = 3,
        angle_spread: float = 0.05,
    ) -> np.ndarray:
        """Generate a batch of near-field channel realizations.

        Args:
            num_samples: Number of channel samples to generate.
            distance_range: (min, max) user distances in meters.
            angle_range: (min, max) user angles in radians.
            num_paths: Number of multipath components per sample.
            angle_spread: Angular spread for scattered paths (radians).

        Returns:
            Channel matrix of shape (num_samples, num_antennas) as complex array.
        """
        if not isinstance(num_samples, (int, np.integer)) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        if (
            len(distance_range) != 2
            or not np.all(np.isfinite(distance_range))
            or not 0 < distance_range[0] < distance_range[1]
        ):
            raise ValueError("distance_range must be finite, positive, and increasing")
        if (
            len(angle_range) != 2
            or not np.all(np.isfinite(angle_range))
            or not angle_range[0] < angle_range[1]
        ):
            raise ValueError("angle_range must be finite and increasing")
        channels = np.zeros((num_samples, self.num_antennas), dtype=np.complex128)
        for i in range(num_samples):
            distance = self.rng.uniform(*distance_range)
            angle = self.rng.uniform(*angle_range)
            channels[i] = self.generate_channel(
                distance, angle, num_paths=num_paths, angle_spread=angle_spread
            )
        return channels

    def estimate_channel(
        self,
        h_true: np.ndarray,
        snr_dB: float = 10.0,
        pilot_length: Optional[int] = None,
    ) -> np.ndarray:
        """Generate an additive-noise channel-estimate surrogate.

        This is ``h_est = h_true + n/sqrt(pilot_length)`` at a requested
        per-pilot SNR relative to the input channel's mean element power. It
        is not an MMSE estimator and does not explicitly model pilot symbols.

        Args:
            h_true: True channel vector of shape (num_antennas,).
            snr_dB: Signal-to-noise ratio in dB for estimation.
            pilot_length: Number of pilot symbols. If None, equals num_antennas.

        Returns:
            Estimated channel vector of shape (num_antennas,).
        """
        if pilot_length is None:
            pilot_length = self.num_antennas
        h_true = np.asarray(h_true, dtype=complex)
        if h_true.shape != (self.num_antennas,) or not np.all(np.isfinite(h_true)):
            raise ValueError(f"h_true must be finite with shape ({self.num_antennas},)")
        if not np.isfinite(snr_dB):
            raise ValueError("snr_dB must be finite")
        if not isinstance(pilot_length, (int, np.integer)) or pilot_length < 1:
            raise ValueError("pilot_length must be a positive integer")

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            magnitudes = np.abs(h_true)
        signal_scale = float(np.max(magnitudes))
        if not np.isfinite(signal_scale):
            raise ValueError("h_true magnitude must be representable in binary64")
        if signal_scale <= 0:
            raise ValueError("h_true must have positive energy")
        normalized = h_true / signal_scale
        normalized_power = float(np.mean(np.abs(normalized) ** 2))
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            snr_linear = float(np.power(10.0, snr_dB / 10.0))
        if not np.isfinite(snr_linear) or snr_linear <= 0:
            raise ValueError("snr_dB must map to a finite positive linear SNR")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            normalized_noise_std = float(
                np.sqrt(normalized_power / snr_linear / pilot_length / 2.0)
            )
        if not np.isfinite(normalized_noise_std) or normalized_noise_std <= 0:
            raise ValueError("requested estimation-noise scale is not representable")

        # Simple LS estimate: h_est = h_true + noise
        normalized_noise = normalized_noise_std * (
            self.rng.standard_normal(self.num_antennas)
            + 1j * self.rng.standard_normal(self.num_antennas)
        )
        with np.errstate(over="ignore", invalid="ignore"):
            noise = signal_scale * normalized_noise
            h_est = h_true + noise
        if not np.all(np.isfinite(h_est)):
            raise ValueError("requested SNR produces a non-finite channel estimate")

        return h_est
