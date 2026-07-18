# ISAC Resource Allocation Framework

> Educational resource-allocation surrogate covering detection, localization, and tracking proxy metrics.
>
> 📄 **Reference**: Fuwang Dong, Fan Liu, Yuanhao Cui, Wei Wang, Kaifeng Han, and Zhiqin Wang, ["Sensing as a Service in 6G Perceptive Networks: A Unified Framework for ISAC Resource Allocation"](https://doi.org/10.1109/TWC.2022.3219463), *IEEE Transactions on Wireless Communications*, vol. 22, no. 5, pp. 3522–3536, 2023. ([manuscript](https://arxiv.org/abs/2202.09969))
>
> **Evidence level**: educational surrogate. The local objectives and optimizer are documented approximations; generated plots are not paper reproductions.

![Python](https://img.shields.io/badge/python-3.10--3.12-blue)
![SciPy](https://img.shields.io/badge/optimization-SciPy-brightgreen)
![NumPy](https://img.shields.io/badge/numerics-NumPy%20%2B%20SciPy-orange)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/license-CC%20BY--SA%204.0-lightgrey.svg)](../../../LICENSE)

---

## 🎯 What This Implements

In 6G perceptive networks, the same infrastructure serves both communication users and radar sensing targets. The fundamental challenge is **how to allocate limited power and bandwidth** across these competing objectives while guaranteeing a minimum communication rate for each user.

This baseline implements a **local synthetic resource-allocation surrogate** with three sensing Quality-of-Service (QoS) proxies:

- **Detection QoS**: Uses a declared two-degree-of-freedom scaled-central-chi-square model. Its exact probability is `Pfa**(1 / (1 + SNR))`; the implementation evaluates the equivalent log-domain expression so that extremely small false-alarm probabilities remain representable.

- **Localization QoS**: Evaluates declared local range/angle information-bound proxies. Range information depends on allocated bandwidth; angle information uses an explicit fixed noise-equivalent bandwidth. Reference scales convert the two quantities into a dimensionless combined score.

- **Tracking QoS**: Evaluates a same-epoch constant-velocity covariance prediction and range/angle measurement update. The Jacobian is evaluated at the predicted state, measurement covariance is shared with the localization proxy, and scalar objectives use only the position-covariance block. It is not a complete tracker or the paper's joint temporal resource optimizer.

The educational alternating surrogate first solves a rate-feasibility problem when uniform bandwidth is infeasible. It then alternates objective-consistent power and bandwidth updates. Some linear power subproblems have exact solutions; the remaining subproblems use constrained SciPy SLSQP. Every accepted iterate is feasible and non-decreasing in the declared objective. A failed numerical step retains the previous feasible iterate and prevents a convergence claim. The returned point is not certified globally optimal.

## 📊 Results

### Detection Probability & Localization CRB vs. Rate Threshold

The local heuristic can be swept over a per-user rate threshold. Any resulting trend is configuration-specific; monotonic degradation is not guaranteed by this non-convex solver.

Use the scripts under `examples/` to compute a trace from the local source modules; no pre-rendered hand-authored tradeoff curve is presented as evidence.

### Power & Bandwidth Allocation Breakdown

How total resources split across 3 sensing targets, 3 communication users, and 1 ISAC joint user:

Inspect the returned `p` and `b` arrays together with `result.diagnostics` for the actual computed allocation.

### Tracking PCRB Convergence

`TrackingQoS.compute_pcrb` performs one prediction/update covariance-bound step. Repeated behavior exists only when the caller explicitly passes each posterior bound into the next step, as `simulate_tracking` does. Reported scalar values are traces of the 2x2 Cartesian position blocks; position and velocity variances are never added. No monotonic-convergence or steady-state guarantee is claimed.

## 🚀 Quick Start

```bash
# From the repository root, use the complete hashed lock.
uv lock --check
uv sync --locked --only-group ci
.venv/bin/python -W error -m pytest code/baselines/isac_resource_allocation/tests -v

```

### Using the API

```python
import numpy as np
from src import ISACSystem, AOSolver

# Create an ISAC system: 32 Tx/Rx antennas, 3 targets, 3 comm users, 1 ISAC user
system = ISACSystem(Nt=32, Nr=32, Q=3, K=3, L=1, fc=30e9, P_total=40.0, B_total=100e6)

# Solve for detection QoS with max-min fairness
solver = AOSolver(system, qos_type='detection', fairness='maxmin')
result = solver.solve(Gamma_c=1e6)  # 1 Mbit/s minimum rate per user

print(f"Strict local convergence: {result.converged}")
print(f"Certified objective history: {result.objective_history}")
print(f"Detection probabilities: {result.detection_probs}")
print(f"Communication rates:     {result.comm_rates}")
```

### Run the repository example

This command executes the checked implementation and verifies its returned budgets and minimum rates. It prints a local synthetic diagnostic; it is not a paper result.

```bash
.venv/bin/python code/baselines/isac_resource_allocation/examples/run_local_example.py
```

## 📖 Mathematical Background

### System Model

The ISAC system serves M = Q + K + L objects over shared power P_total and bandwidth B_total:

| Symbol | Meaning |
|--------|---------|
| Q | Number of sensing targets |
| K | Number of communication-only users |
| L | Number of joint ISAC users |
| N_t, N_r | Transmit / receive antennas |

**Local synthetic path-loss proxy** (not the paper's Eq. 1):

$$\alpha_q = 10^{-(32.4 + 20\log_{10}(d_q) + 20\log_{10}(f_c))/10}$$

**Sensing SNR** for target q:

$$\text{SNR}_q = \frac{p_q \cdot \beta_q \cdot \sigma_q}{N_0 \cdot b_q}$$

where p_q is power, b_q is bandwidth, β_q is channel gain, σ_q is RCS.

**Communication rate** for user k:

$$R_k = b_k \log_2\!\left(1 + \frac{p_k \cdot \beta_k}{N_0 \cdot b_k}\right)$$

For finite SNR at or below one, the implementation evaluates the equivalent
bandwidth-cancelled low-SNR scale and the correction
$\log(1+\mathrm{SNR})/\mathrm{SNR}$. This prevents a rounded subnormal SNR
from being multiplied back by an enormous bandwidth and silently losing a
representable final rate.

The cited paper's Eq. 9 is a sum-rate expression. This repository instead
imposes the stronger local variant `R_k >= Gamma_c` separately on every
communication and ISAC user; it should not be read as the paper's exact
optimization problem.

### Detection QoS normalization

Under Neyman-Pearson detection with false alarm probability P_fa:

$$\eta=F^{-1}_{\chi^2_2}(1-P_{fa}),\qquad
P_{D,q}=1-F_{\chi^2_2}\!\left(\frac{\eta}{1+\mathrm{SNR}_q}\right)
=P_{fa}^{1/(1+\mathrm{SNR}_q)}.$$

This is a scaled **central** chi-square model; it is not a non-central-chi-square detector. The threshold is evaluated with the inverse survival function, and the probability with `exp(log(Pfa)/(1+SNR))`, avoiding the cancellation in `1 - Pfa` for very small `Pfa`.

### Dimensionally normalized localization proxy

The repository declares the following local information model; it is not identified as a paper equation or as the Fisher information of an unspecified likelihood. Let $g_q=\beta_q\sigma_q$, let $B_{\theta,\mathrm{ref}}$ be a fixed angle-noise bandwidth, and let $d_\lambda$ denote spacing in wavelengths:

$$J_{d,q}=\frac{8\pi^2 p_q g_q b_q}{N_0c^2},\qquad
J_{\theta,q}=\frac{p_qg_q}{N_0B_{\theta,\mathrm{ref}}}
\frac{N_t(N_t^2-1)\pi^2\cos^2\theta_qd_\lambda^2}{6}.$$

The corresponding variance-bound proxies are $J_{d,q}^{-1}$ in square metres and $J_{\theta,q}^{-1}$ in square radians, with zero information mapped to positive infinity. Because $B_{\theta,\mathrm{ref}}$ is fixed, changing allocated bandwidth $b_q$ does not change angle information.

Adding inverse bounds directly would mix physical units. The optimized localization quantity is instead the dimensionless score

$$\rho_q=w_d d_{\mathrm{ref}}^2J_{d,q}
+w_\theta\theta_{\mathrm{ref}}^2J_{\theta,q}.$$

Defaults are $d_{\mathrm{ref}}=1$ m, $\theta_{\mathrm{ref}}=10^{-3}$ rad, $B_{\theta,\mathrm{ref}}=10$ MHz, and $d_\lambda=0.5$. These are declared modeling choices, not inferred paper parameters.

### Local tracking covariance recursion

For each target, state and covariance are predicted to the same epoch,

$$x_k^-=Fx_{k-1},\qquad P_k^-=FP_{k-1}F^T+Q.$$

The range/azimuth Jacobian $H_k$ is evaluated at $x_k^-$. With measurement covariance $R_k$ obtained from the same range/angle proxy above, the implementation uses a solve-based Kalman gain and the Joseph form

$$K_k=P_k^-H_k^T(H_kP_k^-H_k^T+R_k)^{-1},$$

$$P_k^+=(I-K_kH_k)P_k^-(I-K_kH_k)^T+K_kR_kK_k^T.$$

The process covariance is the interval-constant-acceleration form $\sigma_a^2GG^T$. The scalar tracking objective uses $\operatorname{tr}(P_{k,\mathrm{pos}}^+)$ only.

### AO-Inspired Educational Surrogate

```
1. Test whether uniform bandwidth supports all per-user rate floors.
2. If not, solve a Phase-I bandwidth problem that minimizes required user power.
3. Construct a feasible power allocation at the selected bandwidth.
4. REPEAT:
   a. Solve or numerically improve the declared power objective at fixed b.
   b. Numerically improve the same declared objective at fixed p.
   c. Accept only feasible, non-decreasing iterates and retain the best one.
5. Declare convergence only after both subproblems succeed and both the
   objective and allocations satisfy their tolerances.
```

The per-user rate constraint is inverted analytically at fixed bandwidth. Detection max-min and both localization power aggregations use exact monotone/linear updates; other power cases and all bandwidth updates are local constrained numerical searches. Solver diagnostics distinguish a successful update from a retained feasible start. Neither finite termination nor `converged=True` is an optimality certificate.

## 🏗️ Project Structure

```
isac_resource_allocation/
├── configs/
│   └── default.yaml          # Default system & solver parameters
├── src/
│   ├── __init__.py            # Package exports
│   ├── system_model.py        # Local synthetic channels, SNR, and budgets
│   ├── detection_qos.py       # Scaled-central-chi-square probability
│   ├── localization_qos.py    # Dimensionless local information-bound proxy
│   ├── tracking_qos.py        # Same-epoch position-covariance recursion
│   ├── comm_rate.py           # Shannon-form rate computation
│   ├── ao_solver.py           # AO-inspired constrained surrogate
│   └── fairness.py            # Standalone fairness diagnostics; solver uses maxmin/sum
├── tests/
│   ├── test_detection.py      # Detection QoS unit tests
│   ├── test_localization.py   # Localization CRB tests
│   ├── test_tracking.py       # Tracking PCRB tests
│   ├── test_ao_solver.py      # Feasibility, monotonicity & failure tests
│   └── test_integration.py    # End-to-end integration tests
├── examples/
│   └── run_local_example.py   # Deterministic solver diagnostic
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 📚 References

```bibtex
@article{dong2023sensing,
  title   = {Sensing as a Service in 6G Perceptive Networks: A Unified Framework for ISAC Resource Allocation},
  author  = {Dong, Fuwang and Liu, Fan and Cui, Yuanhao and Wang, Wei and Han, Kaifeng and Wang, Zhiqin},
  journal = {IEEE Transactions on Wireless Communications},
  volume  = {22},
  number  = {5},
  pages   = {3522--3536},
  year    = {2023},
  doi     = {10.1109/TWC.2022.3219463}
}
```

### Key Dependencies

- **NumPy** / **SciPy** — Numerical computation and statistical distributions
- **PyYAML** — Configuration parsing used by the integration checks
- **pytest** — Executable verification suite
