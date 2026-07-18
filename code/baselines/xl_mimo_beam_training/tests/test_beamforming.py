"""Tests for beamforming codebook and precoding."""

import pytest
import numpy as np

import sys
sys.path.insert(0, "..")
from ..src.beamforming import BeamformingCodebook
from ..src.channel import NearFieldChannel


class TestBeamformingCodebook:
    """Test suite for beamforming codebook."""

    @pytest.fixture
    def codebook(self):
        """Create a standard codebook for testing."""
        return BeamformingCodebook(
            num_antennas=256, wavelength=0.01, antenna_spacing=0.005
        )

    def test_dft_codebook_dimensions(self, codebook):
        """Test DFT codebook has correct shape."""
        W = codebook.generate_dft_codebook()
        assert W.shape == (256, 256), f"Expected (256, 256), got {W.shape}"

    def test_dft_codebook_orthogonality(self, codebook):
        """Test that DFT codebook columns are orthogonal."""
        W = codebook.generate_dft_codebook()
        # W^H @ W should be identity
        gram = W.conj().T @ W
        identity = np.eye(256)
        assert np.allclose(gram, identity, atol=1e-10), (
            "DFT codebook columns should be orthogonal"
        )

    def test_beamforming_norm(self, codebook):
        """Test that each beamforming vector has unit norm."""
        W = codebook.generate_dft_codebook()
        norms = np.linalg.norm(W, axis=0)
        assert np.allclose(norms, 1.0, atol=1e-10), (
            f"All beamformers should have norm 1, got {norms[:5]}"
        )

    def test_polar_codebook_dimensions(self, codebook):
        """Test polar codebook dimensions."""
        distances = np.array([10.0, 50.0, 100.0])
        angles = np.array([-0.5, 0.0, 0.5])
        num_beams = len(distances) * len(angles)

        W, d_idx, a_idx = codebook.generate_polar_codebook(
            num_beams, distances, angles
        )
        assert W.shape == (256, num_beams)
        assert len(d_idx) == num_beams
        assert len(a_idx) == num_beams

    def test_polar_codebook_norm(self, codebook):
        """Test that polar codebook beams have unit norm."""
        distances = np.array([20.0, 50.0])
        angles = np.array([0.0, 0.3])
        W, _, _ = codebook.generate_polar_codebook(
            len(distances) * len(angles), distances, angles
        )
        norms = np.linalg.norm(W, axis=0)
        assert np.allclose(norms, 1.0, atol=1e-10)

    def test_polar_codebook_extreme_distance_is_finite(self, codebook):
        W, _, _ = codebook.generate_polar_codebook(
            1,
            np.array([1e308]),
            np.array([0.2]),
        )
        assert np.all(np.isfinite(W))
        np.testing.assert_allclose(np.linalg.norm(W[:, 0]), 1.0, rtol=1e-14)

    def test_polar_codebook_rejects_mismatched_beam_count(self, codebook):
        """The declared beam count cannot be silently ignored."""
        with pytest.raises(ValueError, match="grid product"):
            codebook.generate_polar_codebook(
                999,
                np.array([20.0]),
                np.array([0.2]),
            )

    def test_polar_beam_matches_channel_phase_convention(self):
        """A focal beam coherently combines the matching spherical channel."""
        num_antennas = 32
        distance = 20.0
        angle = 0.2
        channel = NearFieldChannel(num_antennas=num_antennas)
        codebook = BeamformingCodebook(num_antennas=num_antennas)
        W, _, _ = codebook.generate_polar_codebook(
            1,
            np.array([distance]),
            np.array([angle]),
        )
        h = channel.generate_channel(distance, angle)
        h_phase = h / np.abs(h) / np.sqrt(num_antennas)
        gain = codebook.compute_beamforming_gain(h_phase, W[:, 0])
        np.testing.assert_allclose(gain, 1.0, rtol=1e-12)

    def test_beamforming_gain(self, codebook):
        """Test beamforming gain computation."""
        # Simple case: aligned h and v should give high gain
        h = np.ones(256, dtype=np.complex128) / 16.0  # unit norm
        v = np.ones(256, dtype=np.complex128) / 16.0
        gain = BeamformingCodebook.compute_beamforming_gain(h, v)
        assert gain > 0.9, f"Aligned beamformer should have high gain, got {gain}"

    def test_normalize_beamformer(self, codebook):
        """Test beamformer normalization."""
        v = np.array([3 + 4j, 1 - 2j, 0.5 + 0.5j])
        v_norm = BeamformingCodebook.normalize_beamformer(v)
        norm = np.linalg.norm(v_norm)
        assert np.isclose(norm, 1.0, atol=1e-10), f"Expected norm 1, got {norm}"

        with pytest.raises(ValueError, match="zero"):
            BeamformingCodebook.normalize_beamformer(np.zeros(4, dtype=complex))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_antennas": 0},
            {"wavelength": 0.0},
            {"antenna_spacing": 0.0},
        ],
    )
    def test_invalid_codebook_configuration_is_rejected(self, kwargs):
        with pytest.raises(ValueError):
            BeamformingCodebook(**kwargs)
