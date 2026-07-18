# Energy-efficient ISAC: validated equation-level reference slice

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[Tests](./tests/)
[Reproducibility contract](./reproducibility.yaml)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/license-CC%20BY--SA%204.0-lightgrey.svg)](../../../LICENSE)

Reference: Jiaqi Zou, Songlin Sun, Christos Masouros, Yuanhao Cui,
Ya-Feng Liu, and Derrick Wing Kwan Ng, “Energy-Efficient Beamforming Design
for Integrated Sensing and Communications Systems,” *IEEE Transactions on
Communications*, vol. 72, no. 6, pp. 3766–3782, 2024.
[DOI](https://doi.org/10.1109/TCOMM.2024.3369696) ·
[institutional manuscript](https://discovery.ucl.ac.uk/id/eprint/10188433/) ·
[arXiv preprint](https://arxiv.org/abs/2307.04002)

> **Evidence level: equation-level reference.** This code validates a
> single-user, fixed-beam-direction scalar-power restriction. It does not
> implement the paper's multi-user Algorithm 1, certify a global multi-user
> optimum, or claim parity with a paper figure.

## Why the scope is deliberately narrow

The previous code called proxy SDR/SCA routines “Algorithm 1,” accepted
inaccurate solver states, recovered beams without post-validating the original
constraints, and plotted synthetic curves as reproductions. Those claims were
not scientifically supportable, so the routines and generated figures were
removed.

The public paper reports several simulation parameters, but it does not expose
the channel realizations, communication/sensing noise powers, Monte Carlo
seeds, original solver code, or numerical figure data. Exact paper-curve
replication is therefore not independently testable from public evidence.
This artifact instead makes a smaller claim that has independent numerical
oracles and explicit tolerances.

There are also source-level ambiguities that an implementation must not
hide:

- The displayed point-target CRB in (10) omits the `2L` factor used when the
  same CRB is rearranged in constraint (17).
- The transmit and receive arrays have different declared sizes (`M` and `N`),
  while the compact CRB notation uses a single steering vector.
- The numerical paragraph literally writes `N=20` for the receive array and
  then `N=30` for the frame length. The system model defines frame length as
  `L`, so this artifact records the intended values as `N=20, L=30` and
  discloses the source typo rather than silently changing it.
- The institutional accepted manuscript uses `P0=30 dBm`, whereas the arXiv
  source version uses `P0=33 dBm`. The local certificate declares its selected
  value explicitly and does not treat either version as hidden ground truth.

The implementation evaluates the Fisher information directly from the general
two-array observation model in (9), treats the complex target coefficient as a
nuisance parameter, and uses the `2L` convention of (17). Tests compare this
covariance form against an explicit snapshot-domain Fisher information matrix.

## What is numerically certified

| Claim | Implementation | Independent oracle | Tolerance |
|---|---|---|---|
| SINR and communication EE, (2) and (4) | Stream-excluded, binary-scaled powers | Decimal and analytic counterexamples | `3e-13` relative |
| Radiated power | Normalized binary squared norm | `1e150` amplitude-scale oracle | `2e-15` relative |
| Steering vectors, (6) and (7) | Paper's cosine convention | Central finite difference for derivatives | `1e-8` relative |
| Point-target information from (9)/(17) | General two-array Schur complement | Explicit `L`-snapshot FIM | `5e-12` relative |
| Fixed-direction scalar-power EE optimum | Closed-form inner Dinkelbach step | 500,001-point exhaustive grid | `1.1` grid steps |

The machine-readable contract is [reproducibility.yaml](./reproducibility.yaml).

### Numeric-domain contract

Communication rates are evaluated as `log1p(SINR) / log(2)`. Thus the
regression `H=1e-10`, `W=1`, and `sigma_c2=1` retains the nonzero rate
`1.4426950408889633e-20` bit/s/Hz instead of rounding `1 + SINR` to one.

For every user, interference is accumulated directly over columns `j != k`.
The implementation never computes total received beam power and subtracts the
desired power: that operation loses a `1e-20` interference term beside a unit
desired term. The locked counterexample `h=[1,0]`,
`W=[[1,1e-10],[0,0]]`, and `sigma_c2=1e-30` has SINR
`9.999999999000001e19`; with `H=I2`, its sum rate is
`66.43856189760298` bit/s/Hz, not the former false value
`99.65784284662087`.

Each real and imaginary product in `h_k^H w_j` is first represented exactly as
a signed integer times a power of two. Products are then accumulated with
exact integer exponent buckets, so positive and negative terms cancel before
the residual is converted to a floating-point power. The result is invariant
to antenna ordering and retains the regression
`1e280 - 1e280 + 1e-60 = 1e-60`: its SINR is `1e-120` and its rate is
`1.4426950408889635e-120` bit/s/Hz for unit noise. A single global channel
scale cannot pass this test because it discards the residual more than 324
decades below the large terms.

Received powers remain binary mantissa/exponent pairs until their
dimensionless SINR ratio is formed. This permits finite ratios even when
individual projection powers would overflow or underflow binary64. Radiated
power uses the same scale discipline. Positive SINR, spectral efficiency,
radiated power, or dBm conversion outside the representable output range
raises an explicit `OverflowError` or `FloatingPointError`; exact zero remains
zero. All dBm, wavelength, and antenna-spacing inputs must be finite, with
positive physical values after conversion. These routines are deterministic
binary64 evaluation primitives, not automatic-differentiation operators; the
validated scalar Dinkelbach solver uses its analytic inner solution rather
than differentiating through the accumulator.

The point-target nuisance projection is evaluated after normalizing the beam,
array, and derivative scales. Identifiability thresholds are relative to the
normalized response/derivative geometry, never to an absolute energy floor.
Consequently a common nonzero beam scaling by `s` preserves identifiability
and changes the CRB by exactly `1 / |s|^2`; this is tested from `1e-12` through
`1e12`, including the former false-`inf` case at `s=1e-8`. The physical scale
is restored in the log domain. A mathematically finite CRB above the largest
binary64 value raises `OverflowError`, while a positive CRB below the smallest
binary64 subnormal raises `FloatingPointError`. Neither representability case
is reported as physical unidentifiability.

## Run the verification artifact

From the repository root:

```bash
uv lock --check
uv sync --locked --only-group ci
.venv/bin/python -W error -m code.baselines.isac_energy_efficient_beamforming.examples.verify_reference_slice --json
.venv/bin/python -W error -m pytest -q code/baselines/isac_energy_efficient_beamforming/tests
```

The command exits nonzero unless all numeric comparisons pass. Its JSON output
contains the paper-reported parameters, every local synthetic assumption,
reference and oracle values, errors, tolerances, the Dinkelbach residual, and
post-constraint checks. It also emits the `1e-8` CRB scaling counterexample,
the weak-interference cancellation counterexample, all six permutations of
the 340-decade cancellation-tail example, the `SINR=1e-20` rate oracle, and a
`1e150` total-power amplitude-scale oracle. A rendered curve is never used as
an oracle.

## Model boundary

`SingleUserPowerDinkelbach` fixes a unit beam direction and optimizes only its
radiated power. In this restriction, the sum rate is concave in power and each
Dinkelbach inner problem has the exact solution

```text
p*(lambda) = clip(epsilon / (lambda ln 2) - sigma_c^2 / |h^H v|^2,
                  p_min, P_max).
```

`p_min` is obtained exactly from any declared SINR and CRB constraints. The
returned beam is rejected if the Dinkelbach residual, power budget, SINR, or
CRB post-check fails. Multi-user beam-direction optimization remains out of
scope rather than being represented by an unverified surrogate.

## Parameter provenance

The source paragraph is interpreted, using the paper's own symbol definitions,
as `N=20`, `L=30`; it also gives `P_max=30 dBm`, `epsilon=0.35`, and
`theta=90°`. The source typo and `P0` version difference are disclosed above.
All channel and noise settings in the local certificate are explicitly marked
synthetic. The checked artifact uses smaller arrays for fast deterministic CI;
changing them creates a new experiment, not a paper-figure reproduction.

## Layout

```text
isac_energy_efficient_beamforming/
├── configs/default.yaml
├── examples/verify_reference_slice.py
├── reproducibility.yaml
├── src/
│   ├── dinkelbach_solver.py
│   ├── ee_metrics.py
│   ├── numerics.py
│   ├── quadratic_transform.py
│   └── system_model.py
└── tests/
    ├── test_dinkelbach.py
    ├── test_ee_metrics.py
    ├── test_quadratic.py
    ├── test_reproducibility.py
    └── test_system_model.py
```

## Citation and license

```bibtex
@article{zou2024energy,
  author  = {Zou, Jiaqi and Sun, Songlin and Masouros, Christos and
             Cui, Yuanhao and Liu, Ya-Feng and Ng, Derrick Wing Kwan},
  title   = {Energy-Efficient Beamforming Design for Integrated Sensing and
             Communications Systems},
  journal = {IEEE Transactions on Communications},
  volume  = {72},
  number  = {6},
  pages   = {3766--3782},
  year    = {2024},
  doi     = {10.1109/TCOMM.2024.3369696}
}
```

The implementation is distributed under the repository's
[CC BY-SA 4.0 license](../../../LICENSE).
