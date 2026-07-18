"""
Utility functions for near-field beam training.

Core utilities for the educational implementation:
- trans_vrf: Phase-to-complex conversion (Eq. in paper)
- rate_func: Spectral efficiency loss function (Eq. 14 in paper)
- load_channel_data: Data loading for .mat files

Reference inspiration: J. Nie, Y. Cui et al., IEEE TMC, 2024.
"""

import logging
import math
from fractions import Fraction
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

Nt = 256  # Default number of transmit antennas
_BINARY64_SUBNORMAL_EXPONENT = -1074
_BINARY64_PRODUCT_EXPONENT = 2 * _BINARY64_SUBNORMAL_EXPONENT


def _aligned_binary64_integer(value: float) -> int:
    """Express one finite binary64 value in units of ``2**-1074``."""
    numerator, denominator = float(value).as_integer_ratio()
    denominator_exponent = denominator.bit_length() - 1
    return numerator << (-_BINARY64_SUBNORMAL_EXPONENT - denominator_exponent)


def _scaled_integer(value: int, exponent: int) -> float:
    """Convert ``value * 2**exponent`` without overflowing too early."""
    if value == 0:
        return 0.0
    magnitude = abs(value)
    shift = max(0, magnitude.bit_length() - 1022)
    leading = float(magnitude >> shift)
    try:
        converted = math.ldexp(leading, exponent + shift)
    except OverflowError:
        converted = math.inf
    return -converted if value < 0 else converted


def _exact_coherent_product(
    channel: complex,
    beamformer: complex,
) -> tuple[int, int]:
    """Return ``conj(channel) * beamformer`` as exact binary integers.

    Each binary64 component is an integer multiple of ``2**-1074``.  The
    product components are therefore exact integer multiples of ``2**-2148``.
    Keeping that wider representation is essential when a rotated product is
    individually outside binary64 but later cancels with another antenna term.
    """

    channel_real = _aligned_binary64_integer(channel.real)
    channel_imag = _aligned_binary64_integer(channel.imag)
    beam_real = _aligned_binary64_integer(beamformer.real)
    beam_imag = _aligned_binary64_integer(beamformer.imag)
    return (
        channel_real * beam_real + channel_imag * beam_imag,
        channel_real * beam_imag - channel_imag * beam_real,
    )


def _exact_coherent_summary(
    channel_row: np.ndarray,
    beamformer_row: np.ndarray,
) -> tuple[float, int, int, complex, tuple[tuple[int, int], ...], np.ndarray]:
    """Return an exact coherent product-sum and rounded fast-path terms."""

    exact_terms = tuple(
        _exact_coherent_product(channel, beamformer)
        for channel, beamformer in zip(
            channel_row,
            beamformer_row,
            strict=True,
        )
    )
    real_integer = sum(term[0] for term in exact_terms)
    imag_integer = sum(term[1] for term in exact_terms)
    if real_integer == 0 and imag_integer == 0:
        return (
            -math.inf,
            0,
            0,
            0.0j,
            exact_terms,
            np.zeros(len(exact_terms), dtype=np.complex128),
        )

    real_value = _scaled_integer(real_integer, _BINARY64_PRODUCT_EXPONENT)
    imag_value = _scaled_integer(imag_integer, _BINARY64_PRODUCT_EXPONENT)
    magnitude = math.hypot(real_value, imag_value)
    if math.isfinite(magnitude) and magnitude > 0.0:
        log_magnitude = math.log(magnitude)
    else:
        squared_integer = real_integer**2 + imag_integer**2
        log_magnitude = (
            0.5 * math.log(squared_integer)
            + _BINARY64_PRODUCT_EXPONENT * math.log(2.0)
        )

    rounded_terms = np.asarray(
        [
            complex(
                _scaled_integer(term_real, _BINARY64_PRODUCT_EXPONENT),
                _scaled_integer(term_imag, _BINARY64_PRODUCT_EXPONENT),
            )
            for term_real, term_imag in exact_terms
        ],
        dtype=np.complex128,
    )

    return (
        log_magnitude,
        real_integer,
        imag_integer,
        complex(real_value, imag_value),
        exact_terms,
        rounded_terms,
    )


def _scaled_exact_integer_ratio(
    numerator: int,
    denominator: int,
    *scales: float,
) -> float:
    """Apply binary64 scales before rounding an exact rational once."""
    if numerator == 0 or any(scale == 0.0 for scale in scales):
        return 0.0
    try:
        scaled = Fraction(numerator, denominator)
        for scale in scales:
            scaled *= Fraction.from_float(scale)
        return float(scaled)
    except OverflowError:
        negative = numerator < 0
        for scale in scales:
            negative = negative != (scale < 0.0)
        sign = -1.0 if negative else 1.0
        return math.copysign(math.inf, sign)


def _check_target_gradient_range(
    gradient: np.ndarray,
    target_dtype: torch.dtype,
    label: str,
) -> None:
    """Reject overflow or nonzero underflow before autograd casts a gradient."""
    component_dtype = (
        np.float32
        if target_dtype in (torch.complex64, torch.float32)
        else np.float64
    )
    components = (gradient.real, gradient.imag) if np.iscomplexobj(gradient) else (
        gradient,
    )
    for component in components:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            rounded = component.astype(component_dtype)
        if np.any(~np.isfinite(component)) or np.any(~np.isfinite(rounded)):
            raise FloatingPointError(
                f"the final {label} gradient is outside the {target_dtype} range"
            )
        if np.any((component != 0.0) & (rounded == 0.0)):
            raise FloatingPointError(
                f"the final {label} gradient would underflow in {target_dtype}"
            )


def _softplus_and_sigmoid(value: float) -> tuple[float, float]:
    """Evaluate softplus and sigmoid without overflow."""
    if value >= 0.0:
        negative_exponential = math.exp(-value)
        return (
            value + math.log1p(negative_exponential),
            1.0 / (1.0 + negative_exponential),
        )
    positive_exponential = math.exp(value)
    return math.log1p(positive_exponential), positive_exponential / (
        1.0 + positive_exponential
    )


class _ExactCoherentRate(torch.autograd.Function):
    """Exact coherent rate with scale-safe final input gradients."""

    @staticmethod
    def forward(  # type: ignore[no-untyped-def]
        ctx,
        channels: torch.Tensor,
        phase_coordinates: torch.Tensor,
        snr_values: torch.Tensor,
        num_antennas: int,
    ) -> torch.Tensor:
        reduced = torch.remainder(phase_coordinates.to(torch.float64), 2.0)
        phase = reduced * math.pi
        beamformers = torch.complex(torch.cos(phase), torch.sin(phase))
        channel_rows = channels.detach().to(torch.complex128).cpu().numpy()
        beamformer_rows = beamformers.detach().cpu().numpy()
        summaries = [
            _exact_coherent_summary(channel_row, beamformer_row)
            for channel_row, beamformer_row in zip(
                channel_rows,
                beamformer_rows,
                strict=True,
            )
        ]
        snr_rows = snr_values.detach().to(torch.float64).cpu().numpy().reshape(-1)

        channel_gradients = np.empty(channel_rows.shape, dtype=np.complex128)
        phase_gradients = np.empty(channel_rows.shape, dtype=np.float64)
        exact_channel_gradients: dict[
            tuple[int, int], tuple[int, int, int, float]
        ] = {}
        exact_phase_gradients: dict[
            tuple[int, int], tuple[int, int, float]
        ] = {}
        rates = np.empty(len(channel_rows), dtype=np.float64)
        for row_index, (beamformer_row, summary, snr_value) in enumerate(
            zip(beamformer_rows, summaries, snr_rows, strict=True)
        ):
            (
                log_magnitude,
                real_sum,
                imag_sum,
                coherent,
                exact_terms,
                rounded_terms,
            ) = summary
            if (real_sum == 0 and imag_sum == 0) or snr_value == 0.0:
                rates[row_index] = 0.0
                channel_gradients[row_index] = 0.0j
                phase_gradients[row_index] = 0.0
                continue

            log_received_snr = (
                math.log(float(snr_value))
                - math.log(num_antennas)
                + 2.0 * log_magnitude
            )
            softplus, sigmoid = _softplus_and_sigmoid(log_received_snr)
            rates[row_index] = softplus / math.log(2.0)
            log_magnitude_gradient_scale = 2.0 * sigmoid / math.log(2.0)

            if np.isfinite(coherent):
                with np.errstate(
                    over="ignore",
                    divide="ignore",
                    invalid="ignore",
                ):
                    direct_channel = (
                        log_magnitude_gradient_scale * beamformer_row / coherent
                    )
                    direct_phase = (
                        -math.pi
                        * log_magnitude_gradient_scale
                        * np.imag(rounded_terms / coherent)
                    )
            else:
                direct_channel = np.full(
                    beamformer_row.shape,
                    np.nan + 1j * np.nan,
                    dtype=np.complex128,
                )
                direct_phase = np.full(
                    beamformer_row.shape,
                    np.nan,
                    dtype=np.float64,
                )
            finite_channel = np.isfinite(direct_channel)
            finite_phase = np.isfinite(direct_phase)
            channel_gradients[row_index, finite_channel] = direct_channel[
                finite_channel
            ]
            phase_gradients[row_index, finite_phase] = direct_phase[finite_phase]
            if np.all(finite_channel) and np.all(finite_phase):
                continue

            denominator = real_sum**2 + imag_sum**2
            for index, ((term_real, term_imag), beamformer) in enumerate(
                zip(exact_terms, beamformer_row, strict=True)
            ):
                # Align the beam coefficient with the 2**-2148 product units
                # used by ``real_sum`` and ``imag_sum`` before forming the exact
                # complex quotient for the channel derivative.
                beam_real = _aligned_binary64_integer(beamformer.real) << 1074
                beam_imag = _aligned_binary64_integer(beamformer.imag) << 1074
                if not finite_channel[index]:
                    channel_gradients[row_index, index] = 0.0j
                    exact_channel_gradients[(row_index, index)] = (
                        beam_real * real_sum + beam_imag * imag_sum,
                        beam_imag * real_sum - beam_real * imag_sum,
                        denominator,
                        log_magnitude_gradient_scale,
                    )
                if not finite_phase[index]:
                    phase_gradients[row_index, index] = 0.0
                    exact_phase_gradients[(row_index, index)] = (
                        term_imag * real_sum - term_real * imag_sum,
                        denominator,
                        -math.pi * log_magnitude_gradient_scale,
                    )

        channel_gradient_tensor = torch.tensor(
            channel_gradients,
            dtype=torch.complex128,
            device=channels.device,
        )
        phase_gradient_tensor = torch.tensor(
            phase_gradients,
            dtype=torch.float64,
            device=phase_coordinates.device,
        )
        ctx.exact_channel_gradients = exact_channel_gradients
        ctx.exact_phase_gradients = exact_phase_gradients
        ctx.channel_dtype = channels.dtype
        ctx.phase_dtype = phase_coordinates.dtype
        ctx.save_for_backward(channel_gradient_tensor, phase_gradient_tensor)
        return torch.tensor(
            rates,
            dtype=torch.float64,
            device=channels.device,
        ).unsqueeze(1)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[no-untyped-def]
        channel_gradient, phase_gradient = ctx.saved_tensors
        output_scales = (
            grad_output.detach().to(torch.float64).cpu().numpy().reshape(-1)
        )
        if not np.all(np.isfinite(output_scales)):
            raise FloatingPointError("the upstream rate gradient must be finite")

        channel_local = channel_gradient.detach().cpu().numpy()
        phase_local = phase_gradient.detach().cpu().numpy()
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            channel_final = channel_local * output_scales[:, None]
            phase_final = phase_local * output_scales[:, None]

        channel_underflow = (
            (channel_local.real != 0.0)
            & (output_scales[:, None] != 0.0)
            & (channel_final.real == 0.0)
        ) | (
            (channel_local.imag != 0.0)
            & (output_scales[:, None] != 0.0)
            & (channel_final.imag == 0.0)
        )
        phase_underflow = (
            (phase_local != 0.0)
            & (output_scales[:, None] != 0.0)
            & (phase_final == 0.0)
        )

        for (row, index), (
            real_numerator,
            imag_numerator,
            denominator,
            local_scale,
        ) in ctx.exact_channel_gradients.items():
            channel_final[row, index] = complex(
                _scaled_exact_integer_ratio(
                    real_numerator,
                    denominator,
                    local_scale,
                    float(output_scales[row]),
                ),
                _scaled_exact_integer_ratio(
                    imag_numerator,
                    denominator,
                    local_scale,
                    float(output_scales[row]),
                ),
            )
            if output_scales[row] != 0.0:
                if (
                    real_numerator != 0
                    and local_scale != 0.0
                    and channel_final[row, index].real == 0.0
                ):
                    channel_underflow[row, index] = True
                if (
                    imag_numerator != 0
                    and local_scale != 0.0
                    and channel_final[row, index].imag == 0.0
                ):
                    channel_underflow[row, index] = True

        for (row, index), (
            numerator,
            denominator,
            local_scale,
        ) in ctx.exact_phase_gradients.items():
            phase_final[row, index] = _scaled_exact_integer_ratio(
                numerator,
                denominator,
                local_scale,
                float(output_scales[row]),
            )
            if (
                numerator != 0
                and local_scale != 0.0
                and output_scales[row] != 0.0
                and phase_final[row, index] == 0.0
            ):
                phase_underflow[row, index] = True

        channel_result = None
        if ctx.needs_input_grad[0]:
            if np.any(channel_underflow) or not np.all(np.isfinite(channel_final)):
                raise FloatingPointError(
                    "the final channel gradient is outside the binary64 range"
                )
            _check_target_gradient_range(
                channel_final,
                ctx.channel_dtype,
                "channel",
            )
            channel_result = torch.tensor(
                channel_final,
                dtype=ctx.channel_dtype,
                device=channel_gradient.device,
            )

        phase_result = None
        if ctx.needs_input_grad[1]:
            if np.any(phase_underflow) or not np.all(np.isfinite(phase_final)):
                raise FloatingPointError(
                    "the final phase gradient is outside the binary64 range"
                )
            _check_target_gradient_range(
                phase_final,
                ctx.phase_dtype,
                "phase",
            )
            phase_result = torch.tensor(
                phase_final,
                dtype=ctx.phase_dtype,
                device=phase_gradient.device,
            )
        return (
            channel_result,
            phase_result,
            None,
            None,
        )


def trans_vrf(temp: torch.Tensor) -> torch.Tensor:
    """Convert phase values to complex unit-modulus coefficients.

    Maps output values from the CNN (in [-1, 1]) to complex exponentials
    representing phase-only beamforming vectors for analog beamforming.

    Args:
        temp: Finite real phase coordinates. CNN outputs lie in [-1, 1];
            external values are allowed and wrap with period 2 because the
            coordinates are multiplied by pi inside a complex exponential.

    Returns:
        Complex beamforming vectors of shape (batch, Nt) with |v| = 1.

    Note:
        This implements the periodic phase mapping
        v_n = exp(j * pi * temp_n).
        For analog beamforming, each antenna applies a phase shift only,
        hence |v_n| = 1 for all n.

    """
    if not isinstance(temp, torch.Tensor) or not torch.is_floating_point(temp):
        raise TypeError("phase coordinates must be a real floating-point tensor")
    if not torch.all(torch.isfinite(temp)):
        raise ValueError("phase coordinates must be finite")
    reduced = torch.remainder(temp, 2.0)
    v_real = torch.cos(reduced * math.pi)
    v_imag = torch.sin(reduced * math.pi)
    vrf = torch.complex(v_real, v_imag)
    return vrf


def rate_func(
    h: torch.Tensor,
    v: torch.Tensor,
    snr_input: torch.Tensor,
    num_antennas: int = Nt,
) -> torch.Tensor:
    """Compute negative spectral efficiency (loss function).

    Calculates the achievable rate under the given beamforming vector and
    channel, then returns its NEGATIVE for use as a loss function to minimize.

    Args:
        h: True channel vectors of shape (batch, Nt), complex-valued.
        v: Phase values from CNN of shape (batch, Nt), real-valued in [-1, 1].
        snr_input: SNR values of shape (batch, 1), linear scale (not dB).
        num_antennas: Number of transmit antennas N_t.

    Returns:
        Negative spectral efficiency of shape (batch, 1).
        Minimizing this loss maximizes the achievable rate.

    Note:
        The spectral efficiency is computed as:
            R = log2(1 + (SNR/N_t) * |h^H v|^2)

    """
    if h.ndim != 2 or v.ndim != 2 or h.shape != v.shape:
        raise ValueError("h and v must have equal shape (batch, num_antennas)")
    if not torch.is_complex(h):
        raise TypeError("h must be a complex tensor")
    if h.dtype not in (torch.complex64, torch.complex128):
        raise TypeError("h must use complex64 or complex128")
    if not torch.is_floating_point(v):
        raise TypeError("v must be a real floating-point phase tensor")
    if v.dtype not in (torch.float32, torch.float64):
        raise TypeError("v must use float32 or float64")
    if not isinstance(num_antennas, int) or num_antennas != h.shape[1]:
        raise ValueError("num_antennas must equal the channel's antenna dimension")
    if snr_input.ndim != 2 or snr_input.shape != (h.shape[0], 1):
        raise ValueError("snr_input must have shape (batch, 1)")
    if not torch.is_floating_point(snr_input) or snr_input.dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise TypeError("snr_input must use float32 or float64")
    if h.device != v.device or h.device != snr_input.device:
        raise ValueError("h, v, and snr_input must be on the same device")
    if not torch.all(torch.isfinite(h)) or not torch.all(torch.isfinite(v)):
        raise ValueError("h and v must be finite")
    if not torch.all(torch.isfinite(snr_input)) or torch.any(snr_input < 0):
        raise ValueError("snr_input must be finite and non-negative")
    if snr_input.requires_grad:
        raise ValueError("snr_input is a fixed condition and cannot require gradients")
    if h.device.type == "mps":
        raise ValueError(
            "rate_func requires a float64/complex128-capable device; "
            "Apple MPS does not currently provide that numerical path"
        )

    # The exact binary64 superaccumulator preserves representable tails across
    # cancellation.  Its custom backward combines the final rate derivative
    # with exact integer ratios before binary64 rounding, so individually
    # overflowing log-magnitude derivatives can still yield finite gradients.
    # It also checks the original input dtypes before autograd can silently
    # overflow or underflow a complex64/float32 gradient cast.
    rate = _ExactCoherentRate.apply(
        h,
        v,
        snr_input,
        num_antennas,
    )

    return -rate


def load_channel_data(
    data_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load channel data from .mat files.

    Loads perfect CSI (pcsi.mat) and estimated CSI (ecsi.mat) from the
    specified directory.

    Args:
        data_path: Path to directory containing pcsi.mat and ecsi.mat.

    Returns:
        Tuple of (H, H_est) where:
        - H: Perfect CSI matrix of shape (num_samples, Nt), complex.
        - H_est: Estimated CSI matrix of shape (num_samples, Nt), complex.
    Raises:
        FileNotFoundError: If either expected file is absent.
        KeyError: If a MAT file does not contain the expected variable.
        ValueError: If the two loaded arrays have inconsistent shapes.
    """
    data_dir = Path(data_path)
    pcsi_path = data_dir / "pcsi.mat"
    ecsi_path = data_dir / "ecsi.mat"

    if not pcsi_path.exists() or not ecsi_path.exists():
        raise FileNotFoundError(
            f"expected both pcsi.mat and ecsi.mat in {data_path}; "
            "omit data_path to request synthetic data explicitly"
        )

    import scipy.io as sio

    logger.info("Loading channel data from %s...", data_path)
    h = np.asarray(sio.loadmat(str(pcsi_path))["pcsi"])
    h_est = np.asarray(sio.loadmat(str(ecsi_path))["ecsi"])
    if h.ndim != 2 or h_est.ndim != 2 or h.shape != h_est.shape:
        raise ValueError("pcsi and ecsi must be two-dimensional arrays of equal shape")
    logger.info("Loaded CSI: perfect shape=%s, estimated shape=%s", h.shape, h_est.shape)
    return h, h_est


def prepare_input_features(h_est: np.ndarray) -> np.ndarray:
    """Convert complex estimated CSI to CNN input format.

    Stacks real and imaginary parts along a new dimension to form
    the CNN input tensor.

    Args:
        h_est: Estimated CSI of shape (num_samples, Nt), complex.

    Returns:
        Input features of shape (num_samples, 1, 2, Nt), float32.
        Channel 0 = real part, Channel 1 = imaginary part.
    """
    h_est = np.asarray(h_est)
    if h_est.ndim != 2 or h_est.size == 0:
        raise ValueError("h_est must be a non-empty two-dimensional array")
    if not np.iscomplexobj(h_est):
        raise TypeError("h_est must contain complex CSI samples")
    if not np.all(np.isfinite(h_est)):
        raise ValueError("h_est must contain only finite values")
    real_part = np.real(h_est)
    imag_part = np.imag(h_est)
    float32_info = np.finfo(np.float32)
    smallest_float32 = float(np.nextafter(np.float32(0), np.float32(1)))
    for component in (real_part, imag_part):
        magnitude = np.abs(component)
        if np.any(magnitude > float32_info.max):
            raise ValueError("h_est exceeds the finite float32 feature domain")
        if np.any((component != 0) & (magnitude < smallest_float32)):
            raise ValueError("h_est has nonzero values below the float32 feature domain")
    # Stack along axis=1 (new axis after batch): (N, 2, Nt) -> add channel dim (N, 1, 2, Nt)
    features = np.stack([real_part, imag_part], axis=1)  # (N, 2, Nt)
    features = np.expand_dims(features, axis=1)  # (N, 1, 2, Nt)
    with np.errstate(over="raise", invalid="raise"):
        features = features.astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("h_est cannot be represented as finite float32 features")
    return features


def generate_synthetic_data(
    num_samples: int = 5000,
    num_antennas: int = 256,
    noise_std: float = 0.1,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic channel data for testing.

    Creates random near-field-like channel realizations with noisy estimates.

    Args:
        num_samples: Number of channel samples.
        num_antennas: Number of antennas N_t.
        noise_std: Standard deviation of estimation noise.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (H, H_est):
        - H: Perfect CSI of shape (num_samples, num_antennas), complex.
        - H_est: Noisy estimated CSI of shape (num_samples, num_antennas), complex.
    """
    if not isinstance(num_samples, (int, np.integer)) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer")
    if not isinstance(num_antennas, (int, np.integer)) or num_antennas < 1:
        raise ValueError("num_antennas must be a positive integer")
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be non-negative and finite")
    rng = np.random.default_rng(seed)

    # Generate random channels with near-field-like structure
    # Spherical wave: phase varies nonlinearly across array
    H = np.zeros((num_samples, num_antennas), dtype=np.complex128)
    positions = (
        np.arange(num_antennas) - (num_antennas - 1) / 2.0
    ) / num_antennas  # normalized positions

    for i in range(num_samples):
        # Random user parameters
        distance = rng.uniform(10, 100)
        angle = rng.uniform(-np.pi / 3, np.pi / 3)

        # Spherical wave model
        r_n = np.sqrt(
            distance**2 + positions**2 - 2 * distance * positions * np.sin(angle)
        )
        wavelength = 0.01
        h_i = np.exp(-1j * 2 * np.pi / wavelength * r_n)
        h_i /= np.linalg.norm(h_i)  # normalize
        H[i] = h_i

    # Add noise for channel estimate
    noise = (noise_std / np.sqrt(2)) * (
        rng.standard_normal((num_samples, num_antennas))
        + 1j * rng.standard_normal((num_samples, num_antennas))
    )
    H_est = H + noise

    return H, H_est


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
) -> None:
    """Save model checkpoint.

    Args:
        model: PyTorch model.
        optimizer: Optimizer instance.
        epoch: Current epoch number.
        loss: Current loss value.
        filepath: Path to save the checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath} (epoch {epoch}, loss {loss:.6f})")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    filepath: str,
    device: str = "cpu",
) -> Tuple[int, float]:
    """Load model checkpoint.

    Args:
        model: PyTorch model to load weights into.
        optimizer: Optimizer to load state into (optional).
        filepath: Path to the checkpoint file.
        device: Device to map the checkpoint to.

    Returns:
        Tuple of (epoch, loss) from the checkpoint.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))
    logger.info(f"Checkpoint loaded from {filepath} (epoch {epoch}, loss {loss:.6f})")
    return epoch, loss
