# Capacity–distortion educational surrogate

This directory contains a deliberately narrow numerical model for learning and
software regression. It is associated with, but is not a full implementation
of, the following publication:

Yifeng Xiong, Fan Liu, Yuanhao Cui, Weijie Yuan, Tony Xiao Han, and Giuseppe
Caire, “On the Fundamental Tradeoff of Integrated Sensing and Communications
Under Gaussian Channels,” *IEEE Transactions on Information Theory*, vol. 69,
no. 9, pp. 5723–5751, 2023.
([DOI](https://doi.org/10.1109/TIT.2023.3284449),
[author manuscript](https://arxiv.org/abs/2204.06938))

Evidence level: **educational surrogate**. `paper_figure_parity` is false.
There is no digitized paper-curve oracle in this directory, and no generated
number or curve is presented as a match to a published figure.

## Scope

The code implements these precisely stated local quantities:

- Gaussian MIMO mutual information
  `log det(I + Hc Rx Hc^H / sigma_c2)` in nats per channel use;
- a generic information map
  `J = (T / sigma_s2) Phi(Rx) + Jp` and `CRB = tr(J^-1)`;
- a known-unit-gain, single-path ULA angle-information helper, with its analytic
  steering derivative checked against a central finite difference;
- exact communication water filling under `trace(Rx) = P_T M`;
- an explicitly local covariance-shaping objective
  `-(1-alpha) log det(Rx) - alpha log det(I + Hc Rx Hc^H/sigma_c2)`;
- seeded complex-Gaussian and row-semiunitary waveform generators.

The covariance-shaping objective is solved from its scalar KKT conditions in
the eigenbasis of `Hc^H Hc`. It substitutes `-log det(Rx)` for a physical
sensing-error objective and therefore has no general CRB-region interpretation.

The default `Phi(Rx) = Rx` is also only a generic matrix example. A physical
application must supply its own information map and justify its parameter,
nuisance-variable, prior, and noise conventions.

Communication routines normalize channel amplitudes as
`Hc / sqrt(sigma_c2)` before forming any energy quantity. This ordering avoids
silent underflow from separately evaluating `Hc^H Hc` when both the raw channel
and noise variance are extremely small. For example,
`Hc = diag(2e-200, 1e-200)` and `sigma_c2 = 1e-300` retain the representable
normalized gains `4e-100` and `1e-100`. If a normalized channel coefficient or
energy exceeds binary64 range, the API raises `ValueError` rather than
returning a misleading finite result. Normalized energies below binary64's
smallest subnormal value cannot contribute a nonzero binary64 rate.

Water filling selects active modes using ratio comparisons and products; it
does not form `1 / gain` while deciding the active set. Consequently,
representable subnormal gains remain valid—for example,
`Hc = diag(2e-155, 1e-155)` with unit noise gives gains `4e-310` and
`1e-310`, covariance `diag(2, 0)`, and a positive rate near `8e-310`.
The total trace budget `power_per_tx * M` must itself fit binary64 and is
validated before multiplication. The covariance-shaping KKT solver uses
exponent-scaled ratios, so it does not reject a finite solution merely because
the intermediate `budget * gain` or quadratic discriminant would overflow. A
CRB whose reciprocal eigenvalue or summed
trace exceeds binary64 is reported as positive infinity, rather than raising a
floating-point warning or returning a wrapped finite value.

Finite-waveform rate is evaluated from the singular values of
`(Hc / sqrt(sigma_c2)) X / sqrt(T)` instead of first forming `X X^H`; reciprocal
channel/waveform scales such as `1e-200` and `1e200` therefore retain their
finite received product. Its small reference-matrix contraction uses
compensated real/imaginary sums, so an exact `1e280 - 1e280` cancellation does
not erase a representable `1e-60` tail or make the rate depend on column order.
BFIM scaling combines its matrix scale, `T`, and `sigma_s2` in a safe order,
and explicitly rejects a result outside binary64.
Array geometries whose phase derivative cannot be represented are rejected
before a non-finite angle-information value can be returned.

## Paper anchors and local assumptions

The paper’s target-angle numerical configuration states the following common
parameters. They are recorded here for provenance; the certificate does not
instantiate this paper experiment.

| Quantity | Paper value | Local certificate |
|---|---:|---:|
| Transmit antennas | 10 | 2 |
| Sensing receive antennas | 10 | 2 only for the RNG check |
| Communication receive antennas | 1 | 2 |
| Tx/Rx ULA spacing | 0.5 wavelength | diagonal synthetic channel |
| Maximum sensing receive SNR | 20 dB per antenna | not used |
| Maximum communication receive SNR | 33 dB per antenna | not used |
| Angle prior | von Mises, mean 30°, standard deviation 5° | not used |
| Communication bearing / coherent interval | 42° / 3 | not used |
| Reported subspace-overlap coefficient | approximately 0.61 | not used |

The paper’s overlap coefficient is
`h_c^H M_bar h_c / (||h_c||^2 lambda_max(M_bar))`, where `M_bar` is a
prior-averaged sensing-information matrix. It is not the magnitude of a plain
steering-vector inner product. Earlier code used the latter expression and
incorrectly labeled it as approximately 0.61; that case-study path has been
removed. The paper also treats an unknown complex target response and prior
averaging, whereas `compute_phi_angle` explicitly assumes a known unit gain at
a fixed angle. These models must not be conflated.

The fixed certificate uses a two-mode diagonal communication channel, total
covariance trace 2, seed `20260717`, and only the values emitted in its JSON.

## Numerical certificate

From the repository root, run:

```bash
python -m code.baselines.isac_capacity_distortion.examples.verify_surrogate --json
```

The process exits nonzero if any check fails. Its independent checks are:

1. water filling against a 500,001-point simplex grid;
2. the covariance surrogate against a separate 500,001-point simplex grid and
   its KKT stationarity residual;
3. closed-form diagonal BFIM, CRB, weak-information CRB, and SISO rate values;
4. an extreme-scale water-filling/rate case with raw channel amplitudes around
   `1e-200` and noise variance `1e-300`;
5. a subnormal-gain water-filling/rate case at channel scale `1e-155`;
6. scale-safe BFIM checks at information scales `1e308` and `1e-308`;
7. a reciprocal-scale finite-waveform rate oracle using `X=1e200` and
   `Hc=1e-200`;
8. three permutations of a cross-scale cancellation with exact received
   amplitude `1e-60` and rate `log1p(1e-120)`;
9. explicit rejection or positive-infinity behavior for unrepresentable BFIM,
   CRB, array-geometry, and total-power outputs;
10. the ULA angle derivative against a central finite difference;
11. bitwise repeatability for two generators initialized with the declared
   seed;
12. trace, positive-semidefinite, and rate-ordering postconditions.

This certificate validates only the stated local equations. It is not evidence
for any published curve, asymptotic result, signaling theorem, or general
capacity–distortion boundary.

## Quick start

Python 3.10–3.12 is supported by the repository gate.

```bash
uv lock --check
uv sync --locked --only-group ci
.venv/bin/python -m code.baselines.isac_capacity_distortion.examples.demo
.venv/bin/python -W error -m code.baselines.isac_capacity_distortion.examples.verify_surrogate --json
.venv/bin/python -W error -m pytest code/baselines/isac_capacity_distortion/tests
```

The demo uses a fixed synthetic channel and writes no files. Stochastic APIs
require a `numpy.random.Generator`; pass a seeded generator whenever exact
repeatability matters.

## Layout

```text
isac_capacity_distortion/
├── README.md
├── reproducibility.yaml
├── requirements.txt
├── examples/
│   ├── demo.py
│   └── verify_surrogate.py
├── src/
│   ├── bounds.py          # local endpoint/curve evaluation; legacy filename
│   ├── optimization.py    # water filling, KKT solver, waveform generators
│   └── system_model.py    # validated rate, BFIM, CRB, and ULA primitives
└── tests/
```

## Known boundaries

- No paper figure has a numeric oracle here.
- No unknown-gain/nuisance-parameter Bayesian angle FIM is implemented.
- No expectation over a von Mises angle prior or fading ensemble is evaluated.
- No finite-blocklength coding theorem or signaling-distribution capacity is
  implemented; covariance rate is evaluated for Gaussian inputs.
- Direct dependency versions are pinned, but bitwise floating-point identity is
  only expected within the same declared software environment and platform.
- Noise-normalized coefficients and energies must fit binary64; inputs above
  that domain are rejected explicitly, while contributions below its smallest
  subnormal value are numerically zero.
- The total covariance trace must fit binary64. CRB values beyond binary64 are
  represented by positive infinity.
- Observation BFIM scales and angle-derivative geometry must produce finite
  binary64 outputs; the APIs reject larger domains explicitly.

## License

This code is distributed under the repository’s
[CC BY-SA 4.0 license](../../../LICENSE).
