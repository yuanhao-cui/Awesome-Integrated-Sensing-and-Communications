# XL-MIMO Near-Field Beam Training with Deep Learning

> Educational CNN pipeline mapping synthetic near-field CSI to constant-modulus beamforming phases.
>
> 📄 **Reference**: Jiali Nie, Yuanhao Cui, Zhaohui Yang, Weijie Yuan, and Xiaojun Jing, ["Near-Field Beam Training for Extremely Large-Scale MIMO Based on Deep Learning"](https://doi.org/10.1109/TMC.2024.3462960), *IEEE Transactions on Mobile Computing*, vol. 24, no. 1, pp. 352–362, 2025.
>
> **Evidence level**: educational surrogate. Training data and plots are synthetic; there is no checked-in paper-dataset or numerical parity artifact.

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg?logo=python)](https://www.python.org/)
[![PyTorch 2.5.1](https://img.shields.io/badge/pytorch-2.5.1-ee4c2c?logo=pytorch)](https://pytorch.org/)
[Tests](./tests/)
[![arXiv](https://img.shields.io/badge/arXiv-2406.03249-b31b1b?logo=arxiv)](https://arxiv.org/abs/2406.03249)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/license-CC%20BY--SA%204.0-lightgrey.svg)](../../../LICENSE)

---

## 🎯 What This Implements

An extremely large array may serve users inside its Rayleigh distance, where a
**near-field** (Fresnel) spherical-wave model can be required.  In that regime,
a focusing vector depends on both angle and distance.  A far-field DFT codebook
does not model that range dependence, while a dense two-dimensional search can
be expensive.

This baseline implements a **deep learning-based beam training** example that maps estimated CSI to analog beamforming phases. `trans_vrf` produces per-element unit-modulus coefficients, so the vector norm is `sqrt(N_t)`; `rate_func` applies the corresponding `1/N_t` normalization.  The CNN emits normalized phase coordinates in `[-1, 1]`.  The public mapping also accepts any finite real coordinate and wraps it with period 2 under `exp(jπv)`; it does not silently interpret the coordinate as an unbounded physical angle.

`rate_func` promotes inputs to complex128, converts each channel and phase
component to its exact binary integer representation, and performs both the
complex products and their coherent accumulation before rounding to binary64.
The final `log1p` is evaluated in the log domain. This keeps finite rates for
checked complex64 inputs from `1e-30` through `1e20`, preserves a `1e-60`
residual between opposing `1e280` terms, and makes the checked result independent
of antenna ordering. It also certifies exact cancellation when rotated products
from finite `1.7e308 + j1.7e308` channel components are individually outside
binary64 but their coherent sum is zero. Its custom backward combines the final
rate scale with exact ratios
before rounding; a regression retains finite, analytic phase gradients when a
naive log-magnitude backward would overflow. The upstream loss scale is applied
before this final range decision, so a locally oversized derivative may still
produce a valid final gradient. If that final channel or phase gradient is
outside binary64 or the original `complex64`/`float32` input dtype, backward
propagation raises an explicit range error before an autograd cast can silently
overflow or underflow it. SNR is a fixed condition and cannot request gradients. The exact
reference loss is deterministic but CPU-synchronized and deliberately favors
auditability over training throughput. It requires a float64/complex128-capable
CPU or CUDA device; Apple MPS is rejected explicitly.

CNN input features are float32 by design. `prepare_input_features` rejects
finite complex values that would overflow to infinity or underflow from a
nonzero component to zero during that conversion.

The local training loss is rate-driven rather than phase-MSE-driven. This is an illustrative software pipeline, not evidence of proximity to a theoretical maximum.

### Implemented components

- **CNN beam mapper**: Maps estimated CSI → analog beamforming phases end-to-end
- **Rate-driven loss function**: Directly optimizes spectral efficiency `R = log₂(1 + SNR/N_t · |h^H v|²)`
- **Near-field aware**: Designed for spherical wave propagation in XL-MIMO at mmWave/THz
- **Deterministic synthetic path**: Owned random generators cover channel, estimate, split, shuffle, and SNR sampling

---

## Checked scope

The network is a compact encoder-decoder **without skip connections**. The repository intentionally does not present hand-authored training curves, beam patterns, or method-comparison plots as evidence. Tests cover shapes, gradients, low-SNR numerical stability, channel conventions, seeded data flow, and a short synthetic training path.

---

## 🚀 Quick Start

```bash
# From the repository root, use the complete hashed lock.
uv lock --check
uv sync --locked --only-group ci
.venv/bin/python -W error -m pytest code/baselines/xl_mimo_beam_training/tests -v
.venv/bin/python code/baselines/xl_mimo_beam_training/examples/run_synthetic_example.py --samples 128 --epochs 2 --device cpu

```

### Quick Inference Example

```python
import numpy as np
import torch
from src.model import BeamTrainingNet
from src.channel import NearFieldChannel
from src.utils import prepare_input_features, rate_func

# Load model
model = BeamTrainingNet(antenna_count=256)
# model.load_state_dict(torch.load("checkpoints/best_model.pth"))

# Generate a near-field channel
channel = NearFieldChannel(num_antennas=256, wavelength=0.01)
h = channel.generate_channel(distance=30.0, angle=0.15)

# Prepare input: complex CSI → (batch, 1, 2, N_t)
rng = np.random.default_rng(42)
noise = 0.01 * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
h_est = h + noise
x = torch.tensor(prepare_input_features(h_est[None, :]), dtype=torch.float32)
h_true = torch.tensor(h[None, :], dtype=torch.complex64)

# Predict beamforming phases
model.eval()
with torch.no_grad():
    phases = model(x)  # (1, 256) normalized phase values

# Compute spectral efficiency at 10 dB (linear SNR = 10)
snr = torch.tensor([[10.0]], dtype=torch.float32)
rate = -rate_func(h_true, phases, snr, num_antennas=256)
print(f"Spectral efficiency at 10 dB SNR: {rate.item():.2f} bps/Hz")
```

### Using Real Measurement Data

Place `pcsi.mat` and `ecsi.mat` in the `data/` directory:

```bash
python examples/run_synthetic_example.py --data_path data --epochs 200 --device cuda
```

---

## 📖 Mathematical Background

### Near-Field Channel Model

In the near-field (Fresnel) region, the channel between antenna `n` and a single-antenna user follows the **spherical wave model**:

$$h_n = \frac{\alpha}{r_n} \exp\left(-j \frac{2\pi}{\lambda} r_n\right)$$

where `r_n` is the distance from antenna `n` to the user:

$$r_n = \sqrt{r^2 + d_n^2 - 2 r d_n \sin\theta}$$

Here `r` is the user distance, `d_n` is the position of antenna `n`, `θ` is the angle of arrival, `λ` is the wavelength, and `α` is the path gain. This differs from the far-field model where `r_n ≈ r - d_n sinθ` (planar wave approximation).

### Spectral Efficiency

The achievable rate with the repository's analog beamforming vector **v** (satisfying `|v_n| = 1`, hence `||v||₂ = √N_t`) is:

$$R = \log_2\left(1 + \frac{\rho}{N_t} |\mathbf{h}^H \mathbf{v}|^2\right)$$

where `ρ` is the transmit SNR. Within the phase-only constraint,
`v_phase = h / |h|` is the continuous-phase matched analog oracle under this
module's `hᴴv` convention. It is not full-digital MRT; the unconstrained
unit-norm MRT direction is `h / ||h||₂`.

### Rate-Driven Loss Function

Instead of minimizing MSE between predicted and optimal beamformers, we directly maximize spectral efficiency by minimizing its negative:

$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \log_2\left(1 + \frac{\rho}{N_t} |\mathbf{h}_i^H \mathbf{v}_i|^2\right)$$

where `B` is the batch size and `v_i = trans_vrf(f_θ(h_{est,i}))` is the CNN-predicted beamformer. This end-to-end optimization naturally respects hardware constraints.

### trans_vrf: Phase to Beamforming Vector

The CNN outputs `N_t` real-valued phases `φ_n ∈ [-1, 1]`, which are scaled to `[−π, π]` and converted to a per-element unit-modulus vector:

$$v_n = \exp(j \pi \cdot \phi_n)$$

This enforces the constant-modulus constraint required by phase-only analog beamforming architectures.

---

## 🏗️ Project Structure

```
xl_mimo_beam_training/
├── src/                              # Core implementation
│   ├── __init__.py                  # Package exports
│   ├── model.py                     # Compact convolutional encoder-decoder
│   ├── channel.py                   # NearFieldChannel (spherical wave model)
│   ├── beamforming.py               # Beamforming codebook & precoding
│   ├── trainer.py                   # Training pipeline with checkpointing
│   ├── evaluator.py                 # Metrics & visualization
│   └── utils.py                     # trans_vrf, rate_func, data generation
├── tests/                            # Unit and semantic tests
│   ├── test_model.py                # Architecture & forward pass tests
│   ├── test_channel.py              # Channel model validation tests
│   ├── test_beamforming.py          # Beamforming & codebook tests
│   ├── test_trainer.py              # Training pipeline tests
│   └── test_end_to_end.py           # Full pipeline integration tests
├── examples/
│   └── run_synthetic_example.py     # Train/evaluate synthetic or supplied data
├── configs/
│   └── default.yaml                 # Hyperparameters (Nt, epochs, LR, etc.)
├── data/                             # Data directory (.mat files or synthetic)
│   └── README.md                    # Data preparation instructions
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package installer
└── README.md                         # ← You are here
```

---

## 📚 References

```bibtex
@article{nie2025near,
  title     = {Near-Field Beam Training for Extremely Large-Scale {MIMO} Based on Deep Learning},
  author    = {Nie, Jiali and Cui, Yuanhao and Yang, Zhaohui and Yuan, Weijie and Jing, Xiaojun},
  journal   = {IEEE Transactions on Mobile Computing},
  volume    = {24},
  number    = {1},
  pages     = {352--362},
  year      = {2025},
  doi       = {10.1109/TMC.2024.3462960}
}
```

### Related Work

```bibtex
@article{zhang2026integrated,
  title   = {Integrated Sensing and Communications Over the Years: An Evolution Perspective},
  author  = {Zhang, Di and Cui, Yuanhao and Cao, Xiaowen and Su, Nanchi and
             Gong, Yi and Liu, Fan and Yuan, Weijie and Jing, Xiaojun and
             Zhang, J. Andrew and Xu, Jie and Masouros, Christos and
             Niyato, Dusit and Di Renzo, Marco},
  journal = {IEEE Communications Surveys \& Tutorials},
  volume  = {28},
  pages   = {5014--5048},
  year    = {2026},
  doi     = {10.1109/COMST.2026.3655674}
}
```

---

<p align="center">
  Part of <a href="https://github.com/yuanhao-cui/Awesome-Integrated-Sensing-and-Communications">Awesome-Integrated-Sensing-and-Communications</a>
</p>
