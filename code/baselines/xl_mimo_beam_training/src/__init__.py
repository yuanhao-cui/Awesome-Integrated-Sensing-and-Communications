"""
XL-MIMO Near-Field Beam Training with Deep Learning.

A PyTorch implementation of near-field beam training for extremely large-scale
MIMO systems using a compact encoder-decoder CNN without skip connections.

Reference:
    J. Nie, Y. Cui et al., "Near-Field Beam Training for Extremely Large-Scale
    MIMO Based on Deep Learning," IEEE Transactions on Mobile Computing, 2025.
    arXiv: https://arxiv.org/abs/2406.03249
"""

__version__ = "1.0.0"

from .model import BeamTrainingNet
from .channel import NearFieldChannel
from .beamforming import BeamformingCodebook
from .trainer import Trainer
from .evaluator import Evaluator
from .utils import trans_vrf, rate_func, load_channel_data

__all__ = [
    "BeamTrainingNet",
    "BeamformingCodebook",
    "Evaluator",
    "NearFieldChannel",
    "Trainer",
    "load_channel_data",
    "rate_func",
    "trans_vrf",
]
