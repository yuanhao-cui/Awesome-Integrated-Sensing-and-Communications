# OFDM Ambiguity Function Analysis for ISAC

> Educational ambiguity-function calculations for parameterized OFDM and LFM waveforms.
>
> 📄 **Background**: Classic radar/communication theory — the ambiguity function is the foundational tool for waveform analysis
> **Evidence level**: educational reference. All figures and numerical comparisons are repository-generated and configuration-dependent.

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg?logo=python)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](./test_ofdm_ambiguity.py)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/license-CC%20BY--SA%204.0-lightgrey.svg)](../../../LICENSE)

---

## 🎯 What This Implements

In radar and ISAC systems, the **ambiguity function** (AF) is the fundamental tool for analyzing waveform resolution. It measures the matched filter output as a function of target delay τ (range) and Doppler frequency ν (velocity), revealing how well a waveform can distinguish closely-spaced targets and suppress false alarms from sidelobes.

This baseline computes and visualizes ambiguity functions for parameterized **OFDM waveforms**. OFDM is used in 5G NR and is frequently studied for ISAC; the exact ambiguity surface depends on symbols, windowing, cyclic prefix, sampling, and normalization.

For comparison, the code also evaluates one rectangularly weighted **LFM (linear frequency modulated) chirp**. LFM commonly exhibits range-Doppler coupling; it is a reference waveform here, not a universal optimum or a communication/sensing performance bound.

The direct LFM generator validates `pulse_width * fs` before integer
conversion and caps allocations at 1,000,000 samples by default. Callers may
set a smaller positive `max_samples`; requests outside the declared finite or
allocation domain fail before an array is created.

---

## 📊 Results

### OFDM Ambiguity Function — 3D Surface

The 3D surface reveals the central peak at the origin (zero delay, zero Doppler) and the sidelobe structure. OFDM's periodic subcarrier structure produces ridges along the delay axis.

![OFDM Ambiguity Function 3D](figures/ofdm_ambiguity_3d.png)

### OFDM Ambiguity Function — Contour Plot

Contour plot with dB-level annotations shows a chosen half-power width and sidelobe pattern. The −3 dB contour is one measurement convention, not a universal resolution limit.

![OFDM Ambiguity Contour](figures/ofdm_ambiguity_contour.png)

### LFM (Chirp) Ambiguity Function — Contour Plot

For the checked-in rectangular weighting and discrete grid, the LFM cut shows the familiar sinc-like sidelobe behavior. Values near −13.2 dB describe the first sidelobe of that specific rectangular-window cut, not a universal LFM constant.

![LFM Ambiguity Contour](figures/lfm_ambiguity_contour.png)

### Resolution Comparison

The conventional reciprocal-bandwidth range cell scales as ΔR = c/(2B), while a rectangular-spectrum full half-power width uses a different constant. Doppler-bin spacing scales as Δν = 1/T_c.

![Resolution Comparison](figures/resolution_comparison.png)

---

## 🚀 Quick Start

```bash
# From the repository root, use the complete hashed lock.
uv lock --check
uv sync --locked --only-group ci
.venv/bin/python -W error -m pytest code/baselines/ofdm_ambiguity_function/test_ofdm_ambiguity.py -v
.venv/bin/python code/baselines/ofdm_ambiguity_function/generate_figures.py --output figures
```

### Generate a Single Ambiguity Function

```python
from ofdm_ambiguity import (
    generate_ofdm_signal,
    compute_ambiguity_function,
    generate_lfm_signal,
    plot_ambiguity_contour,
    compute_papr
)

# Generate OFDM signal (64 subcarriers, QPSK, 16-sample CP)
ofdm_signal = generate_ofdm_signal(n_subcarriers=64, cp_len=16)

# Compute ambiguity function
import numpy as np
tau_range = np.linspace(-40, 40, 81)
nu_range = (np.arange(81) - 40) / 81.0  # avoid duplicate Nyquist endpoints
af = compute_ambiguity_function(ofdm_signal, tau_range, nu_range)

# Plot contour
plot_ambiguity_contour(af, tau_range, nu_range, title="OFDM AF")

# Compare PAPR: OFDM vs LFM
lfm_signal = generate_lfm_signal(bandwidth=20e6, pulse_width=10e-6)
print(f"OFDM PAPR: {10*np.log10(compute_papr(ofdm_signal)):.1f} dB")
print(f"LFM  PAPR: {10*np.log10(compute_papr(lfm_signal)):.1f} dB")
# The LFM value is 0 dB; the random OFDM value depends on its symbols.
```

---

## 📖 Mathematical Background

### Ambiguity Function Definition

The unnormalized ambiguity function characterizes the matched-filter response to a signal with delay τ and Doppler shift ν:

$$\chi(\tau, \nu) = \int s(t)\, s^*(t - \tau)\, e^{j 2\pi \nu t}\, dt$$

`compute_ambiguity_function` returns normalized ambiguity power
$A(\tau,\nu)=|\chi(\tau,\nu)|^2/E^2$, so $A(0,0)=1$. For the unnormalized complex function:

Key properties:
- **Peak at origin**: $|\chi(0, 0)| = E$ (signal energy)
- **Symmetry**: $|\chi(\tau, \nu)| = |\chi(-\tau, -\nu)|$
- **Volume invariance**: $\iint |\chi(\tau, \nu)|^2\, d\tau\, d\nu = E^2$ (total energy preserved)

### OFDM Signal Model

An OFDM signal with N subcarriers is:

$$s(t) = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} X[k]\, e^{j 2\pi k \Delta f\, t}$$

where $X[k]$ are QAM-modulated symbols with unit average power, and $\Delta f = 1/T_s$ is the subcarrier spacing. In discrete time, a cyclic prefix copies the last $N_{CP}$ useful samples:

$$x_{CP}[n] = x[n \bmod N],\qquad n=-N_{CP},\ldots,N-1.$$

### OFDM ambiguity surfaces

There is no symbol-independent Dirichlet–sinc formula for the finite,
cyclic-prefix OFDM realization used here.  The bundled implementation therefore
computes the discrete ambiguity surface directly from the generated samples.
Any expectation over random symbols would additionally need a precisely stated
symbol ensemble, window, occupied-tone set, normalization, and sampling model.

### LFM (Chirp) Comparison

The LFM signal has constant amplitude and quadratic phase:

$$s(t) = e^{j\pi K t^2}, \quad 0 \le t \le T$$

where $K = B/T$ is the chirp rate. Its ambiguity function has a diagonal range-Doppler coupling ridge. Sidelobe levels depend on weighting, sampling, and the selected cut; a rectangular continuous-time cut has a first sinc sidelobe near −13.2 dB.

### PAPR: The ISAC Trade-off

The Peak-to-Average Power Ratio quantifies the power amplifier burden:

$$\text{PAPR} = \frac{\max |s(t)|^2}{\mathbb{E}[|s(t)|^2]}$$

| Waveform | PAPR | Impact |
|----------|------|--------|
| OFDM (64 subcarriers) | Symbol dependent | May require power back-off for a given realization |
| LFM (chirp) | 0 dB | Constant envelope, full PA utilization |

The actual PAPR distribution depends on modulation, occupied tones, oversampling, and signal processing; the bundled function reports one discrete-time realization.

### Resolution Formulas

$$\Delta R = \frac{c}{2B} \qquad \text{(range resolution)}$$

$$\Delta \nu = \frac{1}{T_c} \qquad \text{(Doppler resolution)}$$

where $c$ is the speed of light, $B$ is the bandwidth, and $T_c$ is the coherent processing interval.

---

## 🏗️ Project Structure

```
ofdm_ambiguity_function/
├── ofdm_ambiguity.py          # Core implementation (OFDM/LFM generation, AF computation, plotting)
├── test_ofdm_ambiguity.py     # Test suite
├── generate_figures.py        # Figure generation script
├── README.md                  # ← You are here
│
└── figures/                   # Generated figures
    ├── ofdm_ambiguity_3d.png         # 3D surface of OFDM AF
    ├── ofdm_ambiguity_contour.png    # OFDM AF contour plot
    ├── lfm_ambiguity_contour.png     # LFM AF contour (comparison)
    └── resolution_comparison.png     # Range/Doppler resolution curves
```

### Core Functions

| Function | Description |
|----------|-------------|
| `generate_ofdm_signal()` | Generate OFDM waveform with QAM modulation and cyclic prefix |
| `generate_lfm_signal()` | Generate LFM (chirp) radar signal |
| `compute_ambiguity_function()` | Compute normalized ambiguity power $\lvert\chi\rvert^2/E^2$ on integer-delay and Hz-Doppler axes |
| `compute_ambiguity_function_ofdm()` | Convenience wrapper around the direct AF computation |
| `plot_ambiguity_3d()` | 3D surface visualization |
| `plot_ambiguity_contour()` | Contour plot with dB-level annotations |
| `compute_range_resolution()` | Theoretical range resolution $\Delta R = c/(2B)$ |
| `compute_doppler_resolution()` | Theoretical Doppler resolution $\Delta \nu = 1/T_c$ |
| `compute_papr()` | Peak-to-Average Power Ratio |

---

## 🔬 ISAC Design Implications

### Waveform Selection Guide

| Aspect | OFDM | LFM |
|--------|------|-----|
| Communication support | Native (QAM on subcarriers) | Requires modulation overlay |
| Sensing sidelobes | Symbol/window dependent | Window/chirp dependent |
| PAPR | Modulation dependent | Constant-envelope for this ideal chirp |
| Doppler tolerance | Periodic ambiguity ridges | Continuous coupling ridge |
| MIMO integration | Depends on the chosen precoder and waveform design | Depends on the chosen transmit architecture |
| Standard support | 5G NR, IEEE 802.11 | Radar-specific |

### Key Trade-offs

1. **Bandwidth**: A larger occupied bandwidth narrows reciprocal-bandwidth delay scales; the communication impact depends on whether spectrum is shared or partitioned.
2. **CP length**: A longer CP improves tolerance to channel delay spread but adds overhead; whether CP samples aid sensing depends on the receiver design.
3. **Subcarrier spacing**: Wider spacing shortens an OFDM symbol and changes Doppler sensitivity, while spectral efficiency also depends on CP and scheduling overhead.
4. **Modulation order**: Higher order carries more bits per occupied symbol at a required error rate; it does not by itself guarantee a higher PAPR realization.

---

## 📚 References

```bibtex
@book{richards2005fundamentals,
  title     = {Fundamentals of Radar Signal Processing},
  author    = {Richards, Mark A.},
  year      = {2005},
  publisher = {McGraw-Hill},
  isbn      = {0071444742}
}
```

```bibtex
@book{levanon2004radar,
  title     = {Radar Signals},
  author    = {Levanon, Nadav and Mozeson, Eli},
  year      = {2004},
  publisher = {Wiley},
  doi       = {10.1002/0471663085}
}
```

```bibtex
@article{liu2022isac,
  title   = {Integrated Sensing and Communications: Toward Dual-Functional
             Wireless Networks for 6G and Beyond},
  author  = {Liu, Fan and Cui, Yuanhao and Masouros, Christos and Xu, Jie and
             Han, Tony Xiao and Eldar, Yonina C. and Buzzi, Stefano},
  journal = {IEEE Journal on Selected Areas in Communications},
  volume  = {40},
  number  = {6},
  pages   = {1728--1767},
  year    = {2022},
  doi     = {10.1109/JSAC.2022.3156632}
}
```

```bibtex
@article{sturm2011waveform,
  title   = {Waveform Design and Signal Processing Aspects for Fusion of
             Wireless Communications and Radar Sensing},
  author  = {Sturm, Christian and Wiesbeck, Werner},
  journal = {Proceedings of the IEEE},
  volume  = {99},
  number  = {7},
  pages   = {1236--1259},
  year    = {2011},
  doi     = {10.1109/JPROC.2011.2131110}
}
```

---

<p align="center">
  Part of <a href="https://github.com/yuanhao-cui/Awesome-Integrated-Sensing-and-Communications">Awesome-Integrated-Sensing-and-Communications</a>
</p>
