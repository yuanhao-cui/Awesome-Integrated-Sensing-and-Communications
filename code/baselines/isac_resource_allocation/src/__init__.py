"""Educational ISAC resource-allocation surrogates.

The modules expose local synthetic detection, localization, and tracking
proxies. They are not a complete reimplementation of the reference paper.

Reference: "Sensing as a Service in 6G Perceptive Networks: A Unified Framework for ISAC Resource Allocation"
Authors: Fuwang Dong, Fan Liu, Yuanhao Cui, Wei Wang, Kaifeng Han, Zhiqin Wang
IEEE Transactions on Wireless Communications, 2023
"""

from .system_model import ISACSystem
from .detection_qos import DetectionQoS
from .localization_qos import LocalizationQoS
from .tracking_qos import TrackingQoS
from .comm_rate import CommunicationRate
from .ao_solver import AOSolver, AOResult
from .fairness import FairnessMetrics, FairnessType

__version__ = "1.0.0"
__all__ = [
    "ISACSystem",
    "DetectionQoS",
    "LocalizationQoS",
    "TrackingQoS",
    "CommunicationRate",
    "AOSolver",
    "AOResult",
    "FairnessMetrics",
    "FairnessType"
]
