# RIS-assisted ISAC SNR-feasibility reference

This directory contains a seeded, post-validated **educational surrogate** for
a narrowband RIS-assisted ISAC feasibility problem. It is motivated by:

R. Liu, M. Li, Q. Liu, and A. L. Swindlehurst, “SNR/CRB-Constrained Joint
Beamforming and Reflection Designs for RIS-ISAC Systems,” *IEEE Transactions on
Wireless Communications*, vol. 23, no. 7, pp. 7456–7470, 2024,
[doi:10.1109/TWC.2023.3341429](https://doi.org/10.1109/TWC.2023.3341429)
([manuscript](https://arxiv.org/abs/2301.11134)).

## Evidence boundary

This code does **not** reproduce the paper's Algorithm 1, two-angle CRB with
nuisance target coefficients, Monte Carlo experiment, or Figures 2–6. It must
not be used as evidence for the paper's reported rate gains. The repository's
former scalar CRB proxy and pre-rendered comparison figures were removed
because they could not support those claims.

The authors publish MATLAB scripts for the paper at
[RangLiu0706/SNR-CRB-constrained-beamforming-for-RIS-ISAC](https://github.com/RangLiu0706/SNR-CRB-constrained-beamforming-for-RIS-ISAC).
That repository states a personal/non-commercial-use condition and does not
provide a standard open-source license, so its code is linked for provenance
and is not copied here. Its stochastic scripts do not fix a random seed or
publish the raw curve data needed for bit-exact comparison.

## Declared local model

For user `k`, the implementation uses one column-channel convention throughout:

```text
h_k = G[k, :] diag(theta) H_BR + h_d[k, :]
SINR_k = |h_k^H w_k|^2 /
         (sum_{j != k} |h_k^H w_j|^2 + sigma^2)
```

For sensing, it declares the local scalar channel

```text
h_s = a_bs + a_ris^T diag(theta) H_BR
SNR_s = h_s^H W W^H h_s / sigma^2
      = sum_k |h_s^H w_k|^2 / sigma^2.
```

The columns of `W` multiply mutually independent, zero-mean, unit-variance
data symbols. Consequently, their received powers add; the code never replaces
this expectation by the coherent quantity `|h_s^H sum_k(w_k)|^2`. For the
explicit regression `M=1`, `K=2`, `h_s=1`, and unit noise, both `W=[1,-1]`
and `W=[1,1]` therefore have sensing SNR 2.

These channel normalizations and synthetic geometry are repository-local
assumptions; they are not the paper's distance-dependent simulation model.
The default values in `configs/default.yaml` are therefore local test values,
not a transcription of the paper's simulation table.

The fixed-phase beamforming step is a minimum-power SOCP with per-user SINR,
sensing-SNR, and total-power constraints. The independent-stream sensing
superlevel set is nonconvex, so the SOCP uses a documented sufficient
condition. For `z=h_s^H W` and a fixed-phase, communication-feasible reference
`z_0=h_s^H W_0`, the exact identity
`||z||_2^2 - (2 Re{z_0^H z} - ||z_0||_2^2) = ||z-z_0||_2^2 >= 0`
gives a global affine lower bound on the covariance power. Requiring that
lower bound to exceed `gamma_s sigma^2` is convex and conservative, contains
no coherent sum of independent streams, and cannot certify a physically
infeasible sensing tuple. The communication minimum-power reference is scaled,
only if needed, by the smallest common factor that meets the full covariance
SNR target and power budget. The affine subproblem therefore has a known
feasible point; otherwise the phase proposal is rejected.

For fixed `W`, one RIS phase cannot generally align the sensing projections of
all independent streams at once. The implementation therefore makes no
joint global-optimality claim. Holding all other phases fixed reduces one
coordinate to `2 Re{conj(theta_l) C_l}`; the implementation exactly accumulates
the cross-stream coefficient `C_l` and selects its phase. Every coordinate is
then post-checked against the physical streamwise SNR and rolled back if it
decreases. A second safeguard accepts an updated phase vector only if the
re-solved feasible transmit power does not increase.

The implementation certifies, on the returned physical vectors:

- all RIS entries have unit modulus;
- total transmit power does not exceed the declared watt budget;
- every user meets its declared linear SINR threshold;
- sensing SNR meets its declared linear threshold; and
- accepted transmit powers are nonincreasing within a declared `1e-8`
  relative numerical acceptance tolerance.

Solver failure or constraint violation is raised; no scientific constraint is
silently dropped.

### Numeric-domain contract

Rates are evaluated as `log1p(SINR) / log(2)`, so the declared regression
`H=1e-10`, `W=1`, and unit noise retains
`1.4426950408889633e-20` bit/s/Hz rather than rounding to zero.

Communication and sensing metrics do not square a floating-point projection
before forming a ratio. Every finite direct and RIS-reflected complex product
is accumulated as an exact signed binary integer/exponent pair. The sensing
numerator then adds the nonnegative power of each data stream in a common
scale; streams are not coherently collapsed. Only the final SINR, SNR, or rate
is rounded to binary64. The certificate checks both equal desired and
interference projections of `1e200` (whose correct SINR is one) and a direct/RIS
path cancellation `1e280 - 1e280 + 1e-60`. Results are invariant to the tested
path ordering. A positive final value outside binary64 raises an explicit range
error instead of silently becoming zero or infinity.

The multi-stream phase update compares physical SNR values with a relative,
scale-free tolerance. Applying the same nonzero complex scale to `a_bs` and
`H_BR` therefore leaves the tested phase decisions invariant and scales sensing
power by the squared magnitude of that common factor. A separate
single-stream-only triangle helper remains as an analytic scale-safety oracle;
it is not used to claim a multi-stream phase optimum. The certificate checks
that helper with `1e200` projection terms even though their individual
floating-point squares would overflow. These exact reference primitives are
deterministic evaluation tools, not differentiable or high-throughput
production kernels.

## Run the executable certificate

From the repository root:

```bash
uv lock --check
uv sync --locked --only-group ci
.venv/bin/python -W error -m code.baselines.ris_isac_beamforming.examples.verify_surrogate --json
.venv/bin/python -W error -m pytest code/baselines/ris_isac_beamforming/tests -q
```

The certificate uses a fixed seed and emits JSON. It checks the two-stream sign
counterexample, non-decreasing phase updates, common-scale covariance, a
single-stream triangle bound, and direct post-evaluation of power, SINR,
sensing SNR, and monotonicity. It also includes an `SINR=1e-20` rate oracle. It
is a certificate for this declared surrogate only, not a paper-result oracle.

Minimal API use:

```python
from code.baselines.ris_isac_beamforming.src import (
    AlternatingOptimizationSolver,
    RIS_ISAC_System,
)

system = RIS_ISAC_System(M=4, K=2, L=30, seed=42)
result = AlternatingOptimizationSolver(
    system,
    problem_type="snr",
    snr_min_dB=5.0,
).solve()
assert result["converged"]
```

`problem_type="crb"` is deliberately rejected because the paper's full FIM is
not implemented.

## Parameter provenance

The paper's published numerical setup is useful as a literature anchor, but
the values below are **not** used by this surrogate: `M=6`, `K=4`, noise power
`-90 dBm`, `L=1024`, BS–RIS distance `50 m`, RIS–target distance `3 m`,
RIS–user distance `8 m`, Rician factor `3 dB`, and self-interference level
`-110 dB`. The manuscript reports stochastic averages and does not disclose a
seed or raw curve points. Consequently, parameter transcription alone would
not establish numerical reproduction.

## Contents

```text
configs/default.yaml       local assumptions with units
examples/verify_surrogate.py
src/channel_model.py       seeded synthetic channels
src/system_model.py        consistent channel and metric convention
src/numerics.py            exact path sums and scale-safe metric evaluation
src/beamforming.py         fixed-phase SOCP and physical post-checks
src/ris_phase.py           monotone rate and streamwise-SNR coordinate search
src/snr_constraint.py      safeguarded feasibility iteration
src/ao_solver.py           narrow public interface
tests/                     unit, counterexample, and certificate tests
```

## License note

This directory follows the parent repository's CC BY-SA 4.0 license. That does
not change the separate terms of the linked author repository.
