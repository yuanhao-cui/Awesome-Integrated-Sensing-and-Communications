"""Validated equation-level reference slice for energy-efficient ISAC."""

from .dinkelbach_solver import (
    DinkelbachIteration,
    DinkelbachResult,
    InfeasibleReferenceProblem,
    SingleUserPowerDinkelbach,
)
from .ee_metrics import (
    compute_crb,
    compute_crb_point_target,
    compute_ee_c,
    compute_ee_s,
    compute_sinr,
    compute_sum_rate,
    compute_total_power,
    point_target_information_terms,
)
from .system_model import ISACSystemModel, PaperParameterProvenance, dbm_to_watt

__all__ = [
    "DinkelbachIteration",
    "DinkelbachResult",
    "ISACSystemModel",
    "InfeasibleReferenceProblem",
    "PaperParameterProvenance",
    "SingleUserPowerDinkelbach",
    "compute_crb",
    "compute_crb_point_target",
    "compute_ee_c",
    "compute_ee_s",
    "compute_sinr",
    "compute_sum_rate",
    "compute_total_power",
    "dbm_to_watt",
    "point_target_information_terms",
]

__version__ = "2.0.0"
