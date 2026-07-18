"""Public API for the educational capacity-distortion surrogate."""

from .bounds import evaluate_reference_endpoints, evaluate_surrogate_curve
from .optimization import (
    covariance_shaping_surrogate,
    isotropic_covariance,
    make_semiunitary_waveform,
    sample_gaussian_waveform,
    sample_row_semiunitary,
    water_filling_covariance,
)
from .system_model import (
    GaussianISACChannel,
    angle_to_channel,
    angle_to_hfunc,
    compute_bfim,
    compute_crb,
    compute_phi_angle,
    compute_rate,
    compute_rate_per_symbol,
    make_uniform_linear_array,
)

__all__ = [
    "GaussianISACChannel",
    "angle_to_channel",
    "angle_to_hfunc",
    "compute_bfim",
    "compute_crb",
    "compute_phi_angle",
    "compute_rate",
    "compute_rate_per_symbol",
    "covariance_shaping_surrogate",
    "evaluate_reference_endpoints",
    "evaluate_surrogate_curve",
    "isotropic_covariance",
    "make_semiunitary_waveform",
    "make_uniform_linear_array",
    "sample_gaussian_waveform",
    "sample_row_semiunitary",
    "water_filling_covariance",
]
