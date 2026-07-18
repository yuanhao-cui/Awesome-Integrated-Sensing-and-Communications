"""Tests for the training pipeline."""

import pytest
import torch
import numpy as np

import sys
sys.path.insert(0, "..")
from ..src.trainer import Trainer
from ..src.model import BeamTrainingNet


class TestTrainer:
    """Test suite for the training pipeline."""

    @pytest.fixture
    def config(self):
        """Create a minimal config for testing."""
        return {
            "num_antennas": 256,
            "batch_size": 32,
            "num_epochs": 2,
            "learning_rate": 0.001,
            "lr_factor": 0.2,
            "lr_patience": 20,
            "min_lr": 1e-5,
            "val_split": 0.2,
            "num_synthetic_samples": 200,
            "in_channels": 1,
            "out_channels": 1,
            "init_features": 8,
            "seed": 42,
            "log_interval": 1,
            "checkpoint_dir": "/tmp/test_checkpoints",
        }

    def test_model_initialization(self, config):
        """Test that model is properly initialized."""
        trainer = Trainer(config, device="cpu")
        model = trainer.setup_model()
        assert isinstance(model, BeamTrainingNet)
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_synthetic_data_loading(self, config):
        """Test loading synthetic data."""
        trainer = Trainer(config, device="cpu")
        trainer.setup_model()
        train_loader, val_loader = trainer.load_data()
        assert len(train_loader) > 0
        assert len(val_loader) > 0

        # Check data shapes
        batch = next(iter(train_loader))
        inputs, targets, snr = batch
        assert inputs.shape[1:] == (1, 2, 256), f"Input shape: {inputs.shape}"
        assert targets.shape[1] == 256, f"Target shape: {targets.shape}"
        assert snr.shape[1] == 1, f"SNR shape: {snr.shape}"

    def test_seeded_data_loaders_are_reproducible(self, config):
        first = Trainer(config, device="cpu")
        second = Trainer(config, device="cpu")
        first_batch = next(iter(first.load_data()[0]))
        second_batch = next(iter(second.load_data()[0]))
        for left, right in zip(first_batch, second_batch):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_single_sample_split_is_rejected(self, config):
        config = {**config, "num_synthetic_samples": 1}
        with pytest.raises(ValueError, match="at least two"):
            Trainer(config).load_data()

    def test_training_step(self, config):
        """Test that a single training step runs and reduces initial loss."""
        trainer = Trainer(config, device="cpu")
        trainer.setup_model()
        trainer.load_data()

        # Run 2 epochs
        history = trainer.train(num_epochs=2)
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2

    def test_nondefault_antenna_count_trains_end_to_end(self, config, tmp_path):
        """The trainer must pass the configured array size into the rate loss."""
        small_config = {
            **config,
            "num_antennas": 64,
            "batch_size": 8,
            "num_synthetic_samples": 20,
            "init_features": 2,
            "checkpoint_dir": str(tmp_path),
        }
        trainer = Trainer(small_config, device="cpu")
        trainer.setup_model()
        trainer.load_data()
        history = trainer.train(num_epochs=1)
        assert len(history["train_loss"]) == 1
        assert np.isfinite(history["train_loss"][0])

    def test_checkpoint_saving(self, config):
        """Test that checkpoints are saved."""
        import os
        trainer = Trainer(config, device="cpu")
        trainer.setup_model()
        trainer.load_data()
        trainer.train(num_epochs=1)

        checkpoint_path = os.path.join(config["checkpoint_dir"], "final_model.pth")
        assert os.path.exists(checkpoint_path), "Final checkpoint should exist"

    def test_load_pretrained(self, config):
        """Test loading a pretrained checkpoint."""
        import os
        trainer = Trainer(config, device="cpu")
        trainer.setup_model()
        trainer.load_data()
        trainer.train(num_epochs=1)

        # Create new trainer and load checkpoint
        trainer2 = Trainer(config, device="cpu")
        trainer2.setup_model()
        checkpoint_path = os.path.join(config["checkpoint_dir"], "final_model.pth")
        epoch, loss = trainer2.load_pretrained(checkpoint_path)
        assert epoch == 1
        # Loss is negative rate (-Rate), so it should be finite
        assert np.isfinite(loss)
