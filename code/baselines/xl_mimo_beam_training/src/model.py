"""
CNN-based near-field beam training model.

Implements a compact convolutional encoder-decoder that maps estimated CSI
to phase-only beamforming vectors for XL-MIMO systems.

Paper-inspired repository architecture:
    Input: Real and imaginary parts of estimated CSI (batch, 1, 2, Nt)
    Output: Phase values in [-1, 1] mapped to unit-modulus coefficients

The encoder progressively downsamples along the antenna dimension while expanding
feature channels, then the decoder upsamples back to the original resolution.
A final linear layer + tanh produces the beamforming phases.

Reference:
    The cited work motivates the CSI-to-phase mapping; this compact UNet-like
    network is not asserted to be an exact architecture transcription.
"""

from collections import OrderedDict

import torch
import torch.nn as nn


class BeamTrainingNet(nn.Module):
    """Compact CNN encoder-decoder for synthetic near-field beam training.

    Takes estimated CSI (real + imaginary concatenated along dim=1) and outputs
    phase values for constructing per-element unit-modulus coefficients.

    Args:
        in_channels: Number of input channels (default: 1 for complex CSI
            stored as real+imag stacked along spatial dim).
        out_channels: Number of output channels from the decoder (default: 1).
        init_features: Base number of feature maps, doubled at each encoder level.
        antenna_count: Number of antennas N_t (default: 256).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_features: int = 8,
        antenna_count: int = 256,
    ):
        super().__init__()
        if not isinstance(in_channels, int) or in_channels < 1:
            raise ValueError("in_channels must be a positive integer")
        if out_channels != 1:
            raise ValueError("out_channels must be 1 for the fixed phase-output head")
        if not isinstance(init_features, int) or init_features < 1:
            raise ValueError("init_features must be a positive integer")
        if not isinstance(antenna_count, int) or antenna_count < 4 or antenna_count % 4:
            raise ValueError("antenna_count must be a positive multiple of 4")
        self.antenna_count = antenna_count
        self.in_channels = in_channels
        features = init_features

        # Encoder path
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.encoder3 = self._block(features * 2, features * 2, name="enc3")

        # Decoder path
        self.upconv2 = nn.ConvTranspose2d(
            features * 2, features * 2, kernel_size=(1, 2), stride=(1, 2)
        )
        self.decoder2 = self._block(features * 2, features, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features, features, kernel_size=(1, 2), stride=(1, 2)
        )
        self.decoder1 = self._block(features, out_channels, name="dec1")

        # Output head: flatten spatial features → phase values
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(antenna_count * 2, antenna_count)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, 2, Nt) containing real and
                imaginary parts of estimated CSI.

        Returns:
            Phase values of shape (batch, Nt) in [-1, 1]. These are multiplied
            by pi in trans_vrf to obtain the actual phases for beamforming.
        """
        expected = (self.in_channels, 2, self.antenna_count)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"x must have shape (batch, {expected[0]}, 2, {expected[2]})")
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        dec2 = self.upconv2(enc3)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(dec1)
        dec1 = self.flatten(dec1)
        dec1 = self.linear(dec1)
        dec1 = self.tanh(dec1)
        return dec1

    @staticmethod
    def _block(in_channels: int, features: int, name: str) -> nn.Sequential:
        """Create a convolutional block with two Conv2d + BN + ReLU layers.

        Args:
            in_channels: Number of input channels.
            features: Number of output feature maps.
            name: Prefix for layer names (for OrderedDict keys).

        Returns:
            Sequential block with conv-norm-relu × 2.
        """
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=2,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=2,
                            padding=0,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
