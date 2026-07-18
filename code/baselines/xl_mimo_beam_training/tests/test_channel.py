"""Tests for the near-field channel model."""

import pytest
import numpy as np

import sys
sys.path.insert(0, "..")
from ..src.channel import NearFieldChannel


class TestNearFieldChannel:
    """Test suite for the near-field channel model."""

    @pytest.fixture
    def channel(self):
        """Create a standard channel model for testing."""
        return NearFieldChannel(
            num_antennas=256, wavelength=0.01, antenna_spacing=0.005
        )

    def test_channel_dimensions(self, channel):
        """Test that generated channel has correct dimensions."""
        h = channel.generate_channel(distance=50.0, angle=0.0)
        assert h.shape == (256,), f"Expected shape (256,), got {h.shape}"
        assert h.dtype == np.complex128

    def test_channel_not_all_zeros(self, channel):
        """Test that channel is not trivially zero."""
        h = channel.generate_channel(distance=50.0, angle=0.0)
        assert np.any(np.abs(h) > 0), "Channel should not be all zeros"

    def test_channel_batch_dimensions(self, channel):
        """Test batch channel generation dimensions."""
        num_samples = 100
        H = channel.generate_channel_batch(num_samples)
        assert H.shape == (num_samples, 256)
        assert H.dtype == np.complex128

    def test_channel_estimation_noise(self, channel):
        """Test that channel estimation adds noise."""
        h_true = channel.generate_channel(distance=50.0, angle=0.0)
        h_est = channel.estimate_channel(h_true, snr_dB=10.0)
        assert h_est.shape == h_true.shape
        # Estimate should differ from true (but be correlated)
        assert not np.allclose(h_true, h_est), "Estimate should have noise"

    def test_additional_path_loss_changes_amplitude(self, channel):
        """An additional 20 dB path loss reduces amplitude by ten."""
        h_0db = channel.generate_channel(
            distance=50.0, angle=0.0, path_loss_dB=0.0
        )
        h_20db = channel.generate_channel(
            distance=50.0, angle=0.0, path_loss_dB=20.0
        )
        np.testing.assert_allclose(
            np.linalg.norm(h_20db) / np.linalg.norm(h_0db),
            0.1,
            rtol=1e-12,
        )

    def test_free_space_amplitude_decreases_with_distance(self, channel):
        """The channel model preserves rather than normalizes path loss away."""
        h_near = channel.generate_channel(distance=50.0, angle=0.0)
        h_far = channel.generate_channel(distance=100.0, angle=0.0)
        assert np.linalg.norm(h_far) < np.linalg.norm(h_near)

    def test_channel_multipath(self, channel):
        """Test multi-path channel generation."""
        h_los = channel.generate_channel(distance=50.0, angle=0.0, num_paths=1)
        h_mp = channel.generate_channel(distance=50.0, angle=0.0, num_paths=5)
        assert h_los.shape == h_mp.shape
        # Multi-path should differ from single-path
        assert not np.allclose(h_los, h_mp)

    def test_different_distances(self, channel):
        """Test channel generation at different distances."""
        h_near = channel.generate_channel(distance=10.0, angle=0.0)
        h_far = channel.generate_channel(distance=200.0, angle=0.0)
        assert h_near.shape == h_far.shape
        # Near-field effect: phase profile should differ
        assert not np.allclose(h_near, h_far)

    def test_seeded_channel_randomness_is_reproducible(self):
        first = NearFieldChannel(num_antennas=16, rng=np.random.default_rng(7))
        second = NearFieldChannel(num_antennas=16, rng=np.random.default_rng(7))
        np.testing.assert_allclose(
            first.generate_channel_batch(4),
            second.generate_channel_batch(4),
            rtol=1e-14,
            atol=0.0,
        )
        h = first.generate_channel(20.0, 0.1)
        h_copy = second.generate_channel(20.0, 0.1)
        np.testing.assert_allclose(
            first.estimate_channel(h, 5.0),
            second.estimate_channel(h_copy, 5.0),
            rtol=1e-14,
            atol=0.0,
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_antennas": 0},
            {"wavelength": 0.0},
            {"antenna_spacing": 0.0},
        ],
    )
    def test_invalid_channel_configuration_is_rejected(self, kwargs):
        with pytest.raises(ValueError):
            NearFieldChannel(**kwargs)

    def test_estimate_channel_validates_inputs(self, channel):
        with pytest.raises(ValueError, match="shape"):
            channel.estimate_channel(np.ones(8), 10.0)
        with pytest.raises(ValueError, match="pilot_length"):
            channel.estimate_channel(np.ones(256), 10.0, pilot_length=0)

    def test_estimation_noise_tracks_channel_power_and_requested_snr(self):
        channel = NearFieldChannel(
            num_antennas=4096,
            rng=np.random.default_rng(11),
        )
        h_true = channel.generate_channel(30.0, 0.1)
        h_est = channel.estimate_channel(h_true, snr_dB=10.0, pilot_length=1)
        empirical_snr = np.mean(np.abs(h_true) ** 2) / np.mean(
            np.abs(h_est - h_true) ** 2
        )
        assert empirical_snr == pytest.approx(10.0, rel=0.06)

    def test_extreme_finite_distance_is_scale_safe(self):
        channel = NearFieldChannel(num_antennas=8, wavelength=0.01)
        h = channel.generate_channel(distance=1e308, angle=0.2)
        assert np.all(np.isfinite(h))
        assert np.all(np.abs(h) > 0)

    @pytest.mark.parametrize("snr_db", [-1e308, 1e308])
    def test_unrepresentable_linear_snr_is_rejected(self, channel, snr_db):
        with pytest.raises(ValueError, match="linear SNR"):
            channel.estimate_channel(np.ones(256, dtype=complex), snr_db)
