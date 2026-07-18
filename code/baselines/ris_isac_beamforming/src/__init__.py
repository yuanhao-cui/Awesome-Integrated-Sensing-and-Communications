"""Educational, post-validated RIS-ISAC feasibility surrogate.

The cited RIS-ISAC paper motivates the topic; this package does not implement
its CRB model, alternating algorithm, or numerical experiments.
"""

from .system_model import RIS_ISAC_System
from .channel_model import RISChannelModel
from .beamforming import BeamformingOptimizer
from .ris_phase import RISPhaseOptimizer
from .snr_constraint import SNRConstrainedSolver
from .ao_solver import AlternatingOptimizationSolver

__all__ = [
    "RIS_ISAC_System",
    "RISChannelModel",
    "BeamformingOptimizer",
    "RISPhaseOptimizer",
    "SNRConstrainedSolver",
    "AlternatingOptimizationSolver",
]
