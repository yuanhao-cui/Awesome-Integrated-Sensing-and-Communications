"""End-to-end integration tests with synthetic data."""

import itertools

import pytest
import torch
import numpy as np

import sys
sys.path.insert(0, "..")
from ..src.model import BeamTrainingNet
from ..src.utils import (
    _exact_coherent_summary,
    generate_synthetic_data,
    load_channel_data,
    prepare_input_features,
    rate_func,
    trans_vrf,
)
from ..src.evaluator import Evaluator


class TestEndToEnd:
    """End-to-end tests verifying the complete pipeline."""

    def test_trans_vrf_unit_norm(self):
        """Test that trans_vrf produces unit-norm complex vectors."""
        phases = torch.randn(8, 256) * 0.5  # random phases in reasonable range
        v = trans_vrf(phases)
        assert v.shape == (8, 256)
        assert v.dtype == torch.complex64
        # |v| should be 1 for every element
        magnitudes = torch.abs(v)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6), (
            f"trans_vrf should produce unit-norm values, got range "
            f"[{magnitudes.min():.6f}, {magnitudes.max():.6f}]"
        )

    def test_trans_vrf_external_coordinates_wrap_with_period_two(self):
        coordinates = torch.tensor([[-2.25, -0.5, 0.75, 3.0]])
        torch.testing.assert_close(
            trans_vrf(coordinates),
            trans_vrf(coordinates + 2.0),
        )
        with pytest.raises(ValueError, match="finite"):
            trans_vrf(torch.tensor([[float("nan")]]))
        with pytest.raises(TypeError, match="real floating-point"):
            trans_vrf(torch.tensor([[1 + 0j]]))
        extreme = trans_vrf(torch.tensor([[1e308]], dtype=torch.float64))
        assert torch.all(torch.isfinite(extreme))
        torch.testing.assert_close(torch.abs(extreme), torch.ones_like(extreme.real))

    def test_rate_func_positive(self):
        """Test that spectral efficiency is positive."""
        h = torch.randn(8, 256, dtype=torch.complex64)
        h = h / torch.abs(h).mean()  # normalize
        v = torch.randn(8, 256) * 0.5  # phase values in [-1, 1]
        snr = torch.ones(8, 1) * 10.0  # SNR = 10

        neg_rate = rate_func(h, v, snr)
        assert neg_rate.shape == (8, 1)
        rate = -neg_rate
        assert torch.all(rate > 0), "Spectral efficiency should be positive"

    def test_rate_func_increases_with_snr(self):
        """Test that rate increases with SNR."""
        h = torch.randn(16, 256, dtype=torch.complex64)
        h = h / torch.abs(h).mean()
        v = torch.randn(16, 256) * 0.3

        snr_low = torch.ones(16, 1) * 1.0   # SNR = 1 (0 dB)
        snr_high = torch.ones(16, 1) * 100.0  # SNR = 100 (20 dB)

        rate_low = -torch.mean(rate_func(h, v, snr_low)).item()
        rate_high = -torch.mean(rate_func(h, v, snr_high)).item()

        assert rate_high > rate_low, (
            f"Rate should increase with SNR: {rate_low:.4f} -> {rate_high:.4f}"
        )

    def test_rate_func_uses_hermitian_inner_product(self):
        """The loss uses h^H v, not the physically different h^T v."""
        h = torch.tensor([[1.0 + 0.0j, 0.0 + 1.0j]])
        phases = torch.tensor([[0.0, 0.5]])
        snr = torch.tensor([[1.0]])
        rate = -rate_func(h, phases, snr, num_antennas=2)
        expected = torch.tensor([[np.log2(3.0)]], dtype=rate.dtype)
        torch.testing.assert_close(rate, expected)

    def test_rate_func_preserves_tiny_positive_rate_and_gradient(self):
        h = torch.full((1, 256), 1e-6 + 0j, dtype=torch.complex64)
        phases = torch.zeros((1, 256), dtype=torch.float32)
        snr = torch.ones((1, 1), dtype=torch.float32)
        rate = -rate_func(h, phases, snr, num_antennas=256)
        expected = np.log1p(2.56e-10) / np.log(2.0)
        assert rate.item() == pytest.approx(expected, rel=1e-6)
        assert rate.item() > 0.0
        gradient_phases = torch.linspace(-0.2, 0.2, 256).reshape(1, -1)
        gradient_phases.requires_grad_()
        gradient_rate = -rate_func(h, gradient_phases, snr, num_antennas=256)
        gradient_rate.sum().backward()
        assert gradient_phases.grad is not None
        assert torch.all(torch.isfinite(gradient_phases.grad))
        assert torch.count_nonzero(gradient_phases.grad) > 0

    @pytest.mark.parametrize(
        "h, phases, snr, antennas",
        [
            (
                torch.ones((1, 4), dtype=torch.complex64),
                torch.zeros((1, 3)),
                torch.ones((1, 1)),
                4,
            ),
            (
                torch.ones((1, 4), dtype=torch.complex64),
                torch.zeros((1, 4)),
                -torch.ones((1, 1)),
                4,
            ),
            (
                torch.ones((1, 4), dtype=torch.complex64),
                torch.zeros((1, 4)),
                torch.ones((1, 1)),
                8,
            ),
        ],
    )
    def test_rate_func_rejects_inconsistent_inputs(self, h, phases, snr, antennas):
        with pytest.raises(ValueError):
            rate_func(h, phases, snr, num_antennas=antennas)

    def test_rate_normalization_for_nondefault_antenna_count(self):
        h = torch.ones((1, 64), dtype=torch.complex64)
        phases = torch.zeros((1, 64))
        snr = torch.ones((1, 1))
        rate = -rate_func(h, phases, snr, num_antennas=64)
        assert rate.item() == pytest.approx(np.log2(65.0), rel=1e-6)

    def test_rate_preserves_complex128_cancellation_residual(self):
        h = torch.tensor(
            [[1.0 + 0.0j, -1.0 + 1.0e-10 + 0.0j]],
            dtype=torch.complex128,
        )
        rate = -rate_func(
            h,
            torch.zeros((1, 2), dtype=torch.float32),
            torch.ones((1, 1), dtype=torch.float32),
            num_antennas=2,
        )
        expected = np.log1p(0.5e-20) / np.log(2.0)
        assert rate.item() == pytest.approx(expected, rel=2e-7, abs=0.0)

    def test_rate_preserves_complex64_subnormal_power(self):
        h = torch.tensor([[1.0e-30 + 0.0j]], dtype=torch.complex64)
        component = float(h.real.item())
        rate = -rate_func(h, torch.zeros((1, 1)), torch.ones((1, 1)), 1)
        expected = np.log1p(component**2) / np.log(2.0)
        assert rate.item() == pytest.approx(expected, rel=1e-13, abs=0.0)

    def test_rate_avoids_complex64_large_channel_overflow(self):
        h = torch.tensor([[1.0e20 + 0.0j]], dtype=torch.complex64)
        component = float(h.real.item())
        rate = -rate_func(h, torch.zeros((1, 1)), torch.ones((1, 1)), 1)
        expected = np.log1p(component**2) / np.log(2.0)
        assert torch.isfinite(rate).all()
        assert rate.item() == pytest.approx(expected, rel=1e-14)

    @pytest.mark.parametrize(
        "channel",
        [
            [1e200 + 0j, -1e200 + 0j, 1.0 + 0j],
            [1e200 + 0j, 1.0 + 0j, -1e200 + 0j],
            [1.0 + 0j, 1e200 + 0j, -1e200 + 0j],
        ],
    )
    def test_rate_preserves_tail_after_huge_cancellation_in_any_order(self, channel):
        h = torch.tensor([channel], dtype=torch.complex128)
        rate = -rate_func(h, torch.zeros((1, 3)), torch.ones((1, 1)), 3)
        assert rate.item() == pytest.approx(np.log2(4.0 / 3.0), rel=2e-14)

    def test_rate_combines_tiny_tail_and_large_snr_in_log_domain(self):
        h = torch.tensor(
            [[1e150 + 0j, -1e150 + 0j, 1e-100 + 0j]],
            dtype=torch.complex128,
        )
        snr = torch.tensor([[1e200]], dtype=torch.float64)
        rate = -rate_func(h, torch.zeros((1, 3)), snr, 3)
        assert rate.item() == pytest.approx(np.log2(4.0 / 3.0), rel=2e-14)

    @pytest.mark.parametrize(
        "channel",
        [
            [1e280 + 0j, -1e280 + 0j, 1e-60 + 0j],
            [1e280 + 0j, 1e-60 + 0j, -1e280 + 0j],
            [1e-60 + 0j, 1e280 + 0j, -1e280 + 0j],
        ],
    )
    def test_rate_preserves_tail_below_global_normalization_range(self, channel):
        h = torch.tensor([channel], dtype=torch.complex128)
        snr = torch.tensor([[1e100]], dtype=torch.float64)
        rate = -rate_func(h, torch.zeros((1, 3)), snr, 3)
        expected = np.log1p(1e-20 / 3.0) / np.log(2.0)
        assert rate.item() == pytest.approx(expected, rel=2e-13, abs=0.0)

    @pytest.mark.parametrize(
        "channel",
        list(
            itertools.permutations(
                [1e280 + 0j, -1e280 + 0j, 1e-60 + 0j]
            )
        ),
    )
    def test_extreme_exact_cancellation_has_finite_phase_gradients(self, channel):
        h = torch.tensor([channel], dtype=torch.complex128)
        phases = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
        snr = torch.tensor([[1e100]], dtype=torch.float64)
        rate = -rate_func(h, phases, snr, 3)
        rate.backward()
        assert torch.all(torch.isfinite(phases.grad))
        torch.testing.assert_close(phases.grad, torch.zeros_like(phases.grad))

    def test_rate_phase_gradient_matches_finite_difference_in_ordinary_domain(self):
        h = torch.tensor(
            [[1.0 + 2.0j, 0.5 - 0.3j, -0.2 + 0.7j]],
            dtype=torch.complex128,
        )
        phases = torch.tensor(
            [[0.2, -0.4, 0.1]], dtype=torch.float64, requires_grad=True
        )
        snr = torch.tensor([[3.0]], dtype=torch.float64)
        assert torch.autograd.gradcheck(
            lambda coordinates: rate_func(h, coordinates, snr, 3),
            (phases,),
            eps=1e-6,
            atol=1e-7,
            rtol=1e-6,
        )

    def test_final_rate_gradient_remains_finite_after_stronger_cancellation(self):
        h = torch.tensor(
            [[1e280 + 1e280j, -1e280 - 1e280j, 1e-300 + 0j]],
            dtype=torch.complex128,
        )
        phases = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
        snr = torch.tensor([[1e300]], dtype=torch.float64)
        rate = -rate_func(h, phases, snr, 3)
        rate.backward()
        expected = 2.0 * np.pi / (3.0 * np.log(2.0)) * 1e280
        assert rate.item() == pytest.approx(
            np.log1p(1e-300 / 3.0) / np.log(2.0),
            rel=3e-13,
            abs=0.0,
        )
        assert torch.all(torch.isfinite(phases.grad))
        np.testing.assert_allclose(
            phases.grad.detach().numpy()[0],
            [expected, -expected, 0.0],
            rtol=3e-13,
            atol=0.0,
        )

    def test_unrepresentable_final_phase_gradient_is_explicit(self):
        h = torch.tensor(
            [[1e308 + 1e308j, -1e308 - 1e308j, 1e-300 + 0j]],
            dtype=torch.complex128,
        )
        phases = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
        snr = torch.tensor([[1e300]], dtype=torch.float64)
        rate = -rate_func(h, phases, snr, 3)
        with pytest.raises(FloatingPointError, match="phase gradient"):
            rate.backward()

    def test_upstream_scaling_precedes_final_gradient_range_check(self):
        h = torch.tensor(
            [[1e308 + 1e308j, -1e308 - 1e308j, 1e-300 + 0j]],
            dtype=torch.complex128,
        )
        phases = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
        snr = torch.tensor([[1e300]], dtype=torch.float64)
        scaled_rate = 0.1 * (-rate_func(h, phases, snr, 3))
        scaled_rate.backward()

        expected_magnitude = (
            0.1 * 2.0 * np.pi / (3.0 * np.log(2.0)) * 1e308
        )
        assert torch.all(torch.isfinite(phases.grad))
        np.testing.assert_allclose(
            phases.grad.detach().numpy()[0],
            [expected_magnitude, -expected_magnitude, 0.0],
            rtol=3e-13,
            atol=0.0,
        )

    def test_complex64_channel_gradient_overflow_is_explicit(self):
        h = torch.tensor(
            [[1e-40 + 0j]],
            dtype=torch.complex64,
            requires_grad=True,
        )
        phases = torch.zeros((1, 1), dtype=torch.float64)
        snr = torch.tensor([[1e308]], dtype=torch.float64)
        rate = -rate_func(h, phases, snr, 1)
        with pytest.raises(FloatingPointError, match="channel gradient"):
            rate.backward()

    def test_float32_phase_gradient_overflow_is_explicit(self):
        h = torch.tensor(
            [[1e100 + 1e100j, -1e100 - 1e100j, 1e-100 + 0j]],
            dtype=torch.complex128,
        )
        phases = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
        snr = torch.tensor([[1e100]], dtype=torch.float64)
        rate = -rate_func(h, phases, snr, 3)
        with pytest.raises(FloatingPointError, match="phase gradient"):
            rate.backward()

    def test_roundable_float32_subnormal_gradient_is_preserved(self):
        h = torch.tensor(
            [[1.0 + 0.0j, 0.0 + 1.0j]],
            dtype=torch.complex128,
        )
        phases = torch.zeros((1, 2), dtype=torch.float32, requires_grad=True)
        snr = torch.tensor([[3.0]], dtype=torch.float64)
        smallest = float(np.finfo(np.float32).smallest_subnormal)
        local_magnitude = 0.75 * np.pi / np.log(2.0)
        upstream_scale = 0.75 * smallest / local_magnitude

        scaled_rate = upstream_scale * (-rate_func(h, phases, snr, 2))
        scaled_rate.backward()

        expected = torch.tensor(
            [[-smallest, smallest]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(phases.grad, expected, rtol=0.0, atol=0.0)

    def test_roundable_float32_upper_edge_gradient_is_preserved(self):
        h = torch.tensor(
            [[1.0 + 0.0j, 0.0 + 1.0j]],
            dtype=torch.complex128,
        )
        phases = torch.zeros((1, 2), dtype=torch.float32, requires_grad=True)
        snr = torch.tensor([[3.0]], dtype=torch.float64)
        upstream_scale = 1.0010453452093313e38

        scaled_rate = upstream_scale * (-rate_func(h, phases, snr, 2))
        scaled_rate.backward()

        maximum = float(np.finfo(np.float32).max)
        expected = torch.tensor([[-maximum, maximum]], dtype=torch.float32)
        torch.testing.assert_close(phases.grad, expected, rtol=0.0, atol=0.0)

    def test_exact_products_cancel_before_binary64_rounding(self):
        """Individually overflowing rotated products may have an exact zero sum."""

        amplitude = 1.7e308
        h = torch.tensor(
            [
                [
                    complex(amplitude, amplitude),
                    complex(-amplitude, -amplitude),
                ]
            ],
            dtype=torch.complex128,
            requires_grad=True,
        )
        phases = torch.tensor(
            [[-0.25, -0.25]],
            dtype=torch.float64,
            requires_grad=True,
        )
        snr = torch.ones((1, 1), dtype=torch.float64)

        exact_summary = _exact_coherent_summary(
            h.detach().numpy()[0],
            trans_vrf(phases.detach()).numpy()[0],
        )
        assert exact_summary[1:3] == (0, 0)

        rate = -rate_func(h, phases, snr, num_antennas=2)
        torch.testing.assert_close(rate, torch.zeros_like(rate), rtol=0.0, atol=0.0)
        rate.sum().backward()
        torch.testing.assert_close(h.grad, torch.zeros_like(h), rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            phases.grad,
            torch.zeros_like(phases),
            rtol=0.0,
            atol=0.0,
        )

    def test_rate_gradcheck_covers_channel_and_phase_inputs(self):
        h = torch.tensor(
            [[1.0 + 2.0j, 0.5 - 0.3j, -0.2 + 0.7j]],
            dtype=torch.complex128,
            requires_grad=True,
        )
        phases = torch.tensor(
            [[0.2, -0.4, 0.1]], dtype=torch.float64, requires_grad=True
        )
        snr = torch.tensor([[3.0]], dtype=torch.float64)
        assert torch.autograd.gradcheck(
            lambda channel, coordinates: rate_func(
                channel, coordinates, snr, 3
            ),
            (h, phases),
            eps=1e-6,
            atol=1e-7,
            rtol=1e-6,
        )

    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        H, H_est = generate_synthetic_data(
            num_samples=100, num_antennas=256, seed=42
        )
        assert H.shape == (100, 256)
        assert H_est.shape == (100, 256)
        assert H.dtype == np.complex128
        assert H_est.dtype == np.complex128

    def test_prepare_input_features(self):
        """Test input feature preparation."""
        h_est = np.random.randn(50, 256) + 1j * np.random.randn(50, 256)
        features = prepare_input_features(h_est)
        assert features.shape == (50, 1, 2, 256)
        assert features.dtype == np.float32

    @pytest.mark.parametrize(
        "invalid",
        [
            np.ones(4, dtype=complex),
            np.ones((2, 4), dtype=float),
            np.array([[1.0 + 0.0j, np.nan + 0.0j]]),
        ],
    )
    def test_prepare_input_features_rejects_invalid_csi(self, invalid):
        with pytest.raises((TypeError, ValueError)):
            prepare_input_features(invalid)

    @pytest.mark.parametrize("magnitude", [1e100, 1e-100])
    def test_prepare_input_features_rejects_float32_range_loss(self, magnitude):
        with pytest.raises(ValueError, match="float32 feature domain"):
            prepare_input_features(np.array([[magnitude + 0.0j]]))

    def test_explicit_missing_data_path_raises(self, tmp_path):
        """A mistyped real-data path must not silently select synthetic data."""
        with pytest.raises(FileNotFoundError):
            load_channel_data(str(tmp_path))

    def test_evaluator_batch_one_and_global_phase_invariance(self):
        """Batch size one is preserved and beam NMSE ignores common phase."""

        class FixedPhaseModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("phases", torch.tensor([0.0, 0.5]))

            def forward(self, inputs):
                return self.phases.unsqueeze(0).expand(inputs.shape[0], -1)

        model = FixedPhaseModel()
        evaluator = Evaluator(model, num_antennas=2)
        beam = np.array([[1.0, 1.0j]])
        h = np.exp(1j * 0.73) * beam

        gains = evaluator.compute_beamforming_gain(h, h)
        assert gains.shape == (1,)
        np.testing.assert_allclose(gains[0], 4.0, rtol=1e-6)
        np.testing.assert_allclose(
            evaluator.compute_normalized_mse(h, h),
            0.0,
            atol=1e-12,
        )

    def test_end_to_end_training(self):
        """Test: Train 2 epochs on synthetic data, verify loss decreases."""
        # Generate data
        H, H_est = generate_synthetic_data(
            num_samples=200, num_antennas=256, seed=42
        )
        H_input = prepare_input_features(H_est)
        H_true = np.squeeze(H)
        snr = np.power(
            10.0, np.random.randint(-10, 10, size=(200, 1)).astype(np.float32) / 10.0
        )

        # Convert to tensors
        H_input_t = torch.tensor(H_input, dtype=torch.float32)
        H_true_t = torch.tensor(H_true, dtype=torch.complex64)
        snr_t = torch.tensor(snr, dtype=torch.float32)

        # Model
        model = BeamTrainingNet(antenna_count=256)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train 2 epochs
        losses = []
        for epoch in range(2):
            model.train()
            epoch_loss = 0.0
            batch_size = 50
            num_batches = len(H_input_t) // batch_size

            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                inputs = H_input_t[start:end]
                targets = H_true_t[start:end]
                snr_batch = snr_t[start:end]

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = rate_func(targets, outputs, snr_batch)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

        # Verify loss decreased (or at least didn't increase significantly)
        assert losses[-1] <= losses[0] * 1.1, (
            f"Loss should not increase significantly: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_model_inference_pipeline(self):
        """Test the complete inference pipeline."""
        # Generate data
        H, H_est = generate_synthetic_data(num_samples=20, num_antennas=256, seed=42)

        # Create model and run inference
        model = BeamTrainingNet(antenna_count=256)
        model.eval()

        H_input = prepare_input_features(H_est)
        H_input_t = torch.tensor(H_input, dtype=torch.float32)

        with torch.no_grad():
            phases = model(H_input_t)
            v = trans_vrf(phases)

        assert phases.shape == (20, 256)
        assert v.shape == (20, 256)
        assert torch.all(phases >= -1) and torch.all(phases <= 1)
        assert torch.allclose(torch.abs(v), torch.ones_like(torch.abs(v)), atol=1e-6)
