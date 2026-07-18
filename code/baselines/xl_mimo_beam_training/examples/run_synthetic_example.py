#!/usr/bin/env python3
"""
Run a synthetic educational example for near-field beam training.

No paper dataset or numerical oracle is bundled, so the generated curves are
repository diagnostics rather than paper-result reproductions.

Usage:
    python run_synthetic_example.py [--epochs 200] [--samples 5000] [--device cpu]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt

from src.trainer import Trainer
from src.evaluator import Evaluator
from src.utils import generate_synthetic_data, load_channel_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an educational near-field beam-training example"
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Path to .mat data directory (uses synthetic if not provided)",
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--samples", type=int, default=5000,
        help="Number of synthetic samples (if no .mat data)",
    )
    parser.add_argument(
        "--eval_samples", type=int, default=1000,
        help="Number of synthetic evaluation samples",
    )
    parser.add_argument(
        "--batch_size", type=int, default=100,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to train on",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    torch.manual_seed(args.seed)

    # Configuration
    config = {
        "num_antennas": 256,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "lr_factor": 0.2,
        "lr_patience": 20,
        "min_lr": 5e-5,
        "val_split": 0.1,
        "num_synthetic_samples": args.samples,
        "in_channels": 1,
        "out_channels": 1,
        "init_features": 8,
        "seed": args.seed,
        "log_interval": 10,
        "checkpoint_dir": str(output_dir / "checkpoints"),
    }

    # ── Step 1: Train the model ──
    logger.info("=" * 60)
    logger.info("Training CNN-based beam training model")
    logger.info("=" * 60)

    trainer = Trainer(config, device=args.device)
    trainer.setup_model()
    trainer.load_data(data_path=args.data_path)
    history = trainer.train()

    # ── Step 2: Evaluate ──
    logger.info("=" * 60)
    logger.info("Evaluating trained model")
    logger.info("=" * 60)

    # Load or generate test data
    if args.data_path:
        H, H_est = load_channel_data(args.data_path)
    else:
        H, H_est = generate_synthetic_data(
            num_samples=args.eval_samples,
            num_antennas=256,
            seed=args.seed + 100,
        )

    evaluator = Evaluator(trainer.model, device=args.device)
    metrics = evaluator.evaluate_all_metrics(H, H_est)

    # ── Step 3: Plot and save results ──
    logger.info("=" * 60)
    logger.info("Generating plots")
    logger.info("=" * 60)

    # Rate vs SNR
    evaluator.plot_rate_vs_snr(
        metrics["snr_dB"],
        metrics["spectral_efficiency"],
        save_path=str(output_dir / "rate_vs_snr.png"),
    )

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (-Rate)")
    ax1.set_title("Training Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(metrics["snr_dB"], metrics["spectral_efficiency"], marker="o")
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("Spectral Efficiency (bps/Hz)")
    ax2.set_title("Rate vs SNR")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_dir / "full_results.png"), dpi=150)
    plt.close(fig)
    logger.info(f"Results saved to {output_dir}/")

    # Print summary
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model parameters: {trainer.model.count_parameters():,}")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"Avg beamforming gain: {metrics['avg_beamforming_gain_dB']:.2f} dB")
    logger.info(f"Normalized MSE: {metrics['normalized_mse']:.6f}")
    logger.info("Rate (bps/Hz) at each SNR:")
    for snr, rate in zip(metrics["snr_dB"], metrics["spectral_efficiency"]):
        logger.info(f"  SNR = {snr:3d} dB: {rate:.4f}")


if __name__ == "__main__":
    main()
