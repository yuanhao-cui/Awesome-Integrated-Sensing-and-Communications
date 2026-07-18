"""Scale-safe scalar primitives for the resource-allocation surrogate."""

from __future__ import annotations

import math

import numpy as np


_LOG_2 = math.log(2.0)


def _positive_product_ratio(
    numerators: tuple[float, ...],
    denominators: tuple[float, ...],
) -> float:
    """Round a positive product ratio after exponent-safe evaluation."""

    if any(value == 0.0 for value in numerators):
        return 0.0
    mantissa = 1.0
    exponent = 0
    for value in numerators:
        fraction, power = math.frexp(value)
        mantissa *= fraction
        exponent += power
    for value in denominators:
        fraction, power = math.frexp(value)
        mantissa /= fraction
        exponent -= power
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError:
        return float("inf")


def stable_shannon_rates(
    power: np.ndarray,
    bandwidth: np.ndarray,
    gain: np.ndarray,
    noise_psd: float,
) -> np.ndarray:
    """Return ``B log2(1 + p*gain/(N0*B))`` without product overflow.

    At low SNR the bandwidth cancels from the leading term. Evaluating that
    cancellation analytically is essential when ``N0*B`` overflows or the SNR
    itself underflows even though the final bit rate is a representable
    subnormal binary64 value.
    """

    power = np.asarray(power, dtype=float)
    bandwidth = np.asarray(bandwidth, dtype=float)
    gain = np.asarray(gain, dtype=float)
    if power.shape != bandwidth.shape or power.shape != gain.shape:
        raise ValueError("power, bandwidth, and gain must have the same shape")
    if (
        not np.all(np.isfinite(power))
        or not np.all(np.isfinite(bandwidth))
        or not np.all(np.isfinite(gain))
        or np.any(power < 0.0)
        or np.any(bandwidth <= 0.0)
        or np.any(gain < 0.0)
    ):
        raise ValueError(
            "power and gain must be finite and non-negative; bandwidth must "
            "be finite and positive"
        )
    noise_psd = float(noise_psd)
    if not math.isfinite(noise_psd) or noise_psd <= 0.0:
        raise ValueError("noise_psd must be finite and positive")

    rates = np.zeros(power.shape, dtype=float)
    log_noise = math.log(noise_psd)
    for index in np.ndindex(power.shape):
        p_value = float(power[index])
        gain_value = float(gain[index])
        if p_value == 0.0 or gain_value == 0.0:
            continue
        bandwidth_value = float(bandwidth[index])
        snr = _positive_product_ratio(
            (p_value, gain_value),
            (noise_psd, bandwidth_value),
        )
        if snr <= 1.0:
            # Cancel bandwidth before rounding the low-SNR leading scale.
            # This avoids two distinct failure modes: an exact SNR below the
            # binary64 range and double rounding when an exact subnormal SNR
            # first rounds to the smallest binary64 value.  The correction
            # log1p(s)/s is one when s itself underflows; that is also the
            # correctly rounded factor throughout that unresolvable regime.
            low_snr_correction = (
                1.0 if snr == 0.0 else math.log1p(snr) / snr
            )
            rate = _positive_product_ratio(
                (p_value, gain_value, low_snr_correction),
                (noise_psd, _LOG_2),
            )
        elif math.isfinite(snr):
            natural_spectral_efficiency = math.log1p(snr)
            rate = _positive_product_ratio(
                (bandwidth_value, natural_spectral_efficiency),
                (_LOG_2,),
            )
        else:
            # Only an unrepresentable SNR takes the logarithmic asymptote.
            # Here the logarithm is the final well-conditioned quantity; no
            # near-unity ratio is reconstructed from cancelling large logs.
            log_snr = math.fsum(
                [
                    math.log(p_value),
                    math.log(gain_value),
                    -log_noise,
                    -math.log(bandwidth_value),
                ]
            )
            natural_spectral_efficiency = log_snr + math.log1p(
                math.exp(-log_snr)
            )
            rate = _positive_product_ratio(
                (bandwidth_value, natural_spectral_efficiency),
                (_LOG_2,),
            )
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError(
                "communication rate is outside the finite numerical domain"
            )
        rates[index] = rate
    return rates


__all__ = ["stable_shannon_rates"]
