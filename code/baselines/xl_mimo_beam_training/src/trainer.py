"""
Training pipeline for the CNN-based beam training model.

Handles data loading, training loop, validation, checkpointing,
and learning rate scheduling.

Reference:
    Paper-inspired training procedure for the educational surrogate.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from .model import BeamTrainingNet
from .utils import (
    generate_synthetic_data,
    load_channel_data,
    load_checkpoint,
    prepare_input_features,
    rate_func,
    save_checkpoint,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Training pipeline for the beam training CNN.

    Manages the complete training workflow including:
    - Data loading (from .mat files or synthetic)
    - Model construction and optimization
    - Training loop with validation
    - Learning rate scheduling (ReduceLROnPlateau)
    - Checkpoint saving/loading

    Args:
        config: Configuration dictionary with training hyperparameters.
            Expected keys: batch_size, num_epochs, learning_rate, etc.
        device: Device to train on ('cpu', 'cuda', 'mps').
    """

    def __init__(self, config: Dict, device: str = "cpu"):
        self.config = config
        self.device = device
        self.model: Optional[BeamTrainingNet] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

    def setup_model(self) -> BeamTrainingNet:
        """Initialize the model, optimizer, and LR scheduler.

        Returns:
            Initialized BeamTrainingNet model.
        """
        seed = self.config.get("seed", 42)
        if not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.model = BeamTrainingNet(
                in_channels=self.config.get("in_channels", 1),
                out_channels=self.config.get("out_channels", 1),
                init_features=self.config.get("init_features", 8),
                antenna_count=self.config.get("num_antennas", 256),
            ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 0.001),
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.get("lr_factor", 0.2),
            patience=self.config.get("lr_patience", 20),
            min_lr=self.config.get("min_lr", 5e-5),
        )

        logger.info(
            f"Model initialized with {self.model.count_parameters():,} parameters"
        )
        return self.model

    def load_data(
        self,
        data_path: Optional[str] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """Load and prepare training/validation data.

        Loads MAT files when ``data_path`` is given. Synthetic data is used
        only when ``data_path`` is omitted, so corrupt or mistyped user paths
        cannot be silently replaced by a different experiment.

        Args:
            data_path: Path to directory with pcsi.mat and ecsi.mat.
                If None, generates synthetic data.

        Returns:
            Tuple of (train_loader, val_loader).
        """
        batch_size = self.config.get("batch_size", 100)
        val_split = self.config.get("val_split", 0.1)
        num_antennas = self.config.get("num_antennas", 256)
        seed = self.config.get("seed", 42)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not np.isfinite(val_split) or not 0 < val_split < 1:
            raise ValueError("val_split must lie strictly between zero and one")
        if not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")

        if data_path is None:
            num_samples = self.config.get("num_synthetic_samples", 5000)
            logger.info(f"Generating synthetic data ({num_samples} samples)...")
            H, H_est = generate_synthetic_data(
                num_samples=num_samples,
                num_antennas=num_antennas,
                seed=seed,
            )
        else:
            H, H_est = load_channel_data(data_path)

        # Prepare input features: (N, 1, 2, Nt)
        H_input = prepare_input_features(H_est)
        H_true = np.asarray(H)
        if H_true.ndim == 1:
            H_true = H_true.reshape(1, -1)
        if H_true.ndim != 2 or H_true.shape[1] != num_antennas:
            raise ValueError(
                f"H must have shape (samples, {num_antennas})"
            )

        # Generate random SNR values per sample (as in original)
        num_samples = H_true.shape[0]
        if num_samples < 2:
            raise ValueError("training/validation split requires at least two samples")
        rng = np.random.default_rng(seed)
        snr_values = np.power(
            10.0,
            rng.integers(-20, 20, size=(num_samples, 1)).astype(np.float32) / 10.0,
        )

        # Convert to tensors
        H_input_t = torch.tensor(H_input, dtype=torch.float32)
        H_true_t = torch.tensor(H_true, dtype=torch.complex64)
        snr_t = torch.tensor(snr_values, dtype=torch.float32)

        # Create dataset
        dataset = TensorDataset(H_input_t, H_true_t, snr_t)
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_data, val_data = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(int(seed)),
        )

        loader_generator = torch.Generator().manual_seed(int(seed))
        self.train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            generator=loader_generator,
        )
        self.val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False
        )

        logger.info(
            f"Data loaded: {train_size} training, {val_size} validation samples"
        )
        return self.train_loader, self.val_loader

    def train(self, num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """Run the training loop.

        Args:
            num_epochs: Number of training epochs. Overrides config if provided.

        Returns:
            Dictionary with 'train_loss' and 'val_loss' lists per epoch.
        """
        if self.model is None:
            self.setup_model()
        if self.train_loader is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        num_epochs = num_epochs or self.config.get("num_epochs", 200)
        checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            for inputs, targets, snr_values in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                snr_values = snr_values.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = rate_func(
                    targets,
                    outputs,
                    snr_values,
                    num_antennas=targets.shape[-1],
                )
                loss = torch.mean(loss)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()

            avg_train_loss = running_loss / len(self.train_loader)
            history["train_loss"].append(avg_train_loss)

            # Validation phase
            val_loss = self._validate()
            history["val_loss"].append(val_loss)

            # LR scheduling
            self.scheduler.step(val_loss)

            # Logging
            current_lr = self.optimizer.param_groups[0]["lr"]
            if (epoch + 1) % self.config.get("log_interval", 10) == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {current_lr:.6f}"
                )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch + 1,
                    val_loss,
                    str(checkpoint_dir / "best_model.pth"),
                )

        # Save final checkpoint
        save_checkpoint(
            self.model,
            self.optimizer,
            num_epochs,
            history["val_loss"][-1],
            str(checkpoint_dir / "final_model.pth"),
        )

        logger.info(f"Training complete. Best validation loss: {best_val_loss:.4f}")
        return history

    def _validate(self) -> float:
        """Run validation and return average loss.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, snr_values in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                snr_values = snr_values.to(self.device)

                outputs = self.model(inputs)
                loss = rate_func(
                    targets,
                    outputs,
                    snr_values,
                    num_antennas=targets.shape[-1],
                )
                loss = torch.mean(loss)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def load_pretrained(self, checkpoint_path: str) -> Tuple[int, float]:
        """Load a pretrained model checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            Tuple of (epoch, loss) from the checkpoint.
        """
        if self.model is None:
            self.setup_model()
        return load_checkpoint(self.model, self.optimizer, checkpoint_path, self.device)
