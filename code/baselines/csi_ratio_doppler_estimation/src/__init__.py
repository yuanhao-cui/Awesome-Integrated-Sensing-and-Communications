"""Educational CSI-ratio Doppler-estimation adapters.

The routines are paper-inspired synthetic baselines, not exact reproductions.

Reference:
    "CSI-Ratio-based Doppler Frequency Estimation in Integrated Sensing
    and Communications" by J. Andrew Zhang, Yuanhao Cui et al.
"""

from .signal_model import csi_signal_model, csi_static_dynamic_model, csi_with_doppler
from .csi_ratio import compute_csi_ratio, compute_csi_ratio_multi
from .circle_fit import (
    fit_circle_iterative_weighted,
    fit_circle_kasa,
    least_squares_circle_fit,
)
from .mobius_estimator import mobius_doppler_estimate

__all__ = [
    "csi_signal_model",
    "csi_static_dynamic_model",
    "csi_with_doppler",
    "compute_csi_ratio",
    "compute_csi_ratio_multi",
    "least_squares_circle_fit",
    "fit_circle_kasa",
    "fit_circle_iterative_weighted",
    "mobius_doppler_estimate",
]
