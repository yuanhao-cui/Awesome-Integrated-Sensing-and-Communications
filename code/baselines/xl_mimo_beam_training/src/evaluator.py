"""
Evaluation metrics and testing pipeline for beam training.

Provides comprehensive evaluation including spectral efficiency,
beamforming gain, normalized MSE, and visualization.

The metrics are repository diagnostics, not a paper-parity certificate.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .model import BeamTrainingNet
from .utils import (
    load_checkpoint,
    prepare_input_features,
    rate_func,
    trans_vrf,
)

logger = logging.getLogger(__name__)


def _channel_matrix(values: np.ndarray, num_antennas: int, name: str) -> np.ndarray:
    """Return a batch-first channel matrix without collapsing batch size one."""
    matrix = np.asarray(values)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != num_antennas:
        raise ValueError(f"{name} must have shape (samples, {num_antennas})")
    return matrix


class Evaluator:
    """Evaluation pipeline for the beam training model.

    Computes multiple metrics across different SNR regimes:
    - Spectral efficiency (achievable rate in bps/Hz)
    - Beamforming gain |h^H v|^2
    - Phase-aligned MSE against the normalized full-digital channel direction
    - Rate vs SNR curves

    Args:
        model: Trained BeamTrainingNet model.
        device: Device for inference.
        num_antennas: Number of transmit antennas.
    """

    def __init__(
        self,
        model: BeamTrainingNet,
        device: str = "cpu",
        num_antennas: int = 256,
    ):
        self.model = model.to(device)
        self.device = device
        self.num_antennas = num_antennas

    def evaluate_rate_vs_snr(
        self,
        H: np.ndarray,
        H_est: np.ndarray,
        snr_range: Optional[List[int]] = None,
    ) -> Tuple[List[float], List[float]]:
        """Evaluate spectral efficiency across SNR values.

        Args:
            H: Perfect CSI of shape (num_samples, Nt), complex.
            H_est: Estimated CSI of shape (num_samples, Nt), complex.
            snr_range: List of SNR values in dB. Default: [-20, -15, ..., 20].

        Returns:
            Tuple of (snr_dB_list, rate_list) for plotting.
        """
        if snr_range is None:
            snr_range = list(range(-20, 21, 5))

        H_true = _channel_matrix(H, self.num_antennas, "H")
        H_est = _channel_matrix(H_est, self.num_antennas, "H_est")
        H_input = prepare_input_features(H_est)

        H_input_t = torch.tensor(H_input, dtype=torch.float32).to(self.device)
        H_true_t = torch.tensor(H_true, dtype=torch.complex64).to(self.device)

        self.model.eval()
        rates = []

        with torch.no_grad():
            outputs = self.model(H_input_t)  # (N, Nt) phase values

            for snr_dB in snr_range:
                snr_linear = 10 ** (snr_dB / 10.0)
                snr_tensor = torch.full(
                    (H_true_t.shape[0], 1), snr_linear, dtype=torch.float32
                ).to(self.device)

                loss = rate_func(
                    H_true_t,
                    outputs,
                    snr_tensor,
                    num_antennas=self.num_antennas,
                )
                avg_rate = -torch.mean(loss).item()
                rates.append(avg_rate)
                logger.info(f"SNR: {snr_dB} dB, Rate: {avg_rate:.4f} bps/Hz")

        return snr_range, rates

    def compute_beamforming_gain(
        self,
        H: np.ndarray,
        H_est: np.ndarray,
    ) -> np.ndarray:
        """Compute beamforming gain |h^H v|^2 for each sample.

        Args:
            H: Perfect CSI of shape (num_samples, Nt), complex.
            H_est: Estimated CSI of shape (num_samples, Nt), complex.

        Returns:
            Array of beamforming gains of shape (num_samples,).
        """
        H_true = _channel_matrix(H, self.num_antennas, "H")
        H_est = _channel_matrix(H_est, self.num_antennas, "H_est")
        H_input = prepare_input_features(H_est)
        H_input_t = torch.tensor(H_input, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            phases = self.model(H_input_t)  # (N, Nt)
            v = trans_vrf(phases).cpu().numpy()  # (N, Nt) complex

        gains = np.abs(np.sum(np.conj(H_true) * v, axis=1)) ** 2
        return gains

    def compute_normalized_mse(
        self,
        H: np.ndarray,
        H_est: np.ndarray,
    ) -> float:
        """Compute phase-aligned MSE against the full-digital channel direction.

        The reference ``h / ||h||`` is the unconstrained full-digital
        channel direction. It is not a feasible constant-modulus analog
        beamformer when channel magnitudes differ.

        Args:
            H: Perfect CSI of shape (num_samples, Nt), complex.
            H_est: Estimated CSI of shape (num_samples, Nt), complex.

        Returns:
            Average normalized MSE.
        """
        H_true = _channel_matrix(H, self.num_antennas, "H")
        H_est = _channel_matrix(H_est, self.num_antennas, "H_est")
        H_input = prepare_input_features(H_est)
        H_input_t = torch.tensor(H_input, dtype=torch.float32).to(self.device)

        # Unit-norm full-digital channel direction.
        norms = np.linalg.norm(H_true, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("H must not contain zero channel vectors")
        v_reference = H_true / norms  # (N, Nt)

        self.model.eval()
        with torch.no_grad():
            phases = self.model(H_input_t)
            v_pred = trans_vrf(phases).cpu().numpy()  # (N, Nt)

        v_pred /= np.linalg.norm(v_pred, axis=1, keepdims=True)

        # A beamforming vector is unchanged by a common phase rotation.  Align
        # that phase per sample before comparing the two unit-norm vectors.
        phase_inner = np.sum(np.conj(v_pred) * v_reference, axis=1, keepdims=True)
        phase = np.ones_like(phase_inner)
        nonzero = np.abs(phase_inner) > 1e-12
        phase[nonzero] = phase_inner[nonzero] / np.abs(phase_inner[nonzero])
        v_pred_aligned = v_pred * phase

        squared_error = np.sum(np.abs(v_pred_aligned - v_reference) ** 2, axis=1)
        mse = np.mean(squared_error)
        return float(mse)

    def evaluate_all_metrics(
        self,
        H: np.ndarray,
        H_est: np.ndarray,
        snr_range: Optional[List[int]] = None,
    ) -> Dict:
        """Run all evaluation metrics.

        Args:
            H: Perfect CSI.
            H_est: Estimated CSI.
            snr_range: SNR values in dB.

        Returns:
            Dictionary with all computed metrics.
        """
        snr_list, rate_list = self.evaluate_rate_vs_snr(H, H_est, snr_range)
        gains = self.compute_beamforming_gain(H, H_est)
        nmse = self.compute_normalized_mse(H, H_est)

        metrics = {
            "snr_dB": snr_list,
            "spectral_efficiency": rate_list,
            "avg_beamforming_gain_dB": float(10 * np.log10(np.mean(gains))),
            "beamforming_gains": gains,
            "normalized_mse": nmse,
        }

        logger.info(
            f"Evaluation results: "
            f"avg_gain={metrics['avg_beamforming_gain_dB']:.2f} dB, "
            f"NMSE={nmse:.6f}"
        )
        return metrics

    @staticmethod
    def plot_rate_vs_snr(
        snr_range: List[float],
        rates: List[float],
        save_path: Optional[str] = None,
        title: str = "Near-Field Beam Training Performance",
    ) -> None:
        """Plot spectral efficiency vs SNR curve.

        Args:
            snr_range: SNR values in dB.
            rates: Achievable rates in bps/Hz.
            save_path: If provided, save figure to this path.
            title: Plot title.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(snr_range, rates, marker="o", linewidth=2, markersize=6)
        plt.xlabel("SNR (dB)", fontsize=12)
        plt.ylabel("Spectral Efficiency (bps/Hz)", fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        num_antennas: int = 256,
    ) -> "Evaluator":
        """Create an Evaluator from a saved checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint.
            device: Device for inference.
            num_antennas: Number of antennas.

        Returns:
            Initialized Evaluator with loaded model.
        """
        model = BeamTrainingNet(antenna_count=num_antennas)
        load_checkpoint(model, None, checkpoint_path, device)
        model.eval()
        return cls(model, device, num_antennas)
