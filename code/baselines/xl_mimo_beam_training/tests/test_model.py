"""Tests for the BeamTrainingNet CNN model."""

import pytest
import torch

import sys
sys.path.insert(0, "..")
from ..src.model import BeamTrainingNet


class TestBeamTrainingNet:
    """Test suite for the beam training CNN model."""

    @pytest.fixture
    def model(self):
        """Create a standard model for testing."""
        return BeamTrainingNet(in_channels=1, out_channels=1, init_features=8, antenna_count=256)

    def test_model_forward_shape(self, model):
        """Test that forward pass produces correct output shape.

        Input: (batch, 1, 2, 256) -> Output: (batch, 256)
        """
        batch_size = 8
        x = torch.randn(batch_size, 1, 2, 256)
        output = model(x)
        assert output.shape == (batch_size, 256), (
            f"Expected shape ({batch_size}, 256), got {output.shape}"
        )

    def test_model_output_range(self, model):
        """Test that model output is in [-1, 1] (tanh activation)."""
        x = torch.randn(16, 1, 2, 256)
        output = model(x)
        assert torch.all(output >= -1.0), f"Output below -1: min={output.min()}"
        assert torch.all(output <= 1.0), f"Output above 1: max={output.max()}"

    def test_model_batch_independence(self, model):
        """Test that different batch sizes work correctly."""
        for batch_size in [1, 4, 16, 64]:
            x = torch.randn(batch_size, 1, 2, 256)
            output = model(x)
            assert output.shape == (batch_size, 256)

    def test_model_gradient_flow(self, model):
        """Test that gradients flow through the model."""
        x = torch.randn(4, 1, 2, 256, requires_grad=False)
        output = model(x)
        loss = output.mean()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_model_parameter_count(self, model):
        """Test that model has a reasonable number of parameters."""
        num_params = model.count_parameters()
        assert num_params > 0, "Model has no parameters"
        assert num_params < 1e7, f"Model too large: {num_params:,} params"
        # Expected ~few hundred K params for this architecture
        assert num_params > 1e4, f"Model too small: {num_params:,} params"

    def test_model_eval_mode(self, model):
        """Test that model works correctly in eval mode."""
        model.eval()
        x = torch.randn(4, 1, 2, 256)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (4, 256)
        assert torch.all(output >= -1.0) and torch.all(output <= 1.0)

    def test_model_deterministic_in_eval(self, model):
        """Test that model produces same output in eval mode."""
        model.eval()
        x = torch.randn(4, 1, 2, 256)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_model_custom_antenna_count(self):
        """Test model with different antenna counts."""
        for Nt in [64, 128, 512]:
            model = BeamTrainingNet(antenna_count=Nt)
            x = torch.randn(4, 1, 2, Nt)
            output = model(x)
            assert output.shape == (4, Nt), f"Failed for Nt={Nt}"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"out_channels": 2},
            {"antenna_count": 63},
            {"antenna_count": 0},
            {"init_features": 0},
        ],
    )
    def test_unsupported_model_shapes_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            BeamTrainingNet(**kwargs)

    def test_forward_rejects_wrong_antenna_axis(self, model):
        with pytest.raises(ValueError, match="shape"):
            model(torch.randn(2, 1, 2, 128))
