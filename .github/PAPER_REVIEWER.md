# Scientific Code Reviewer Gate

Use this checklist to review ISAC code, manifests, certificates, and numerical
claims. Review what the contribution actually claims; do not silently upgrade a
teaching model into a paper reproduction.

## 1. Establish the evidence level

Every baseline must declare exactly one machine-readable level:

| Manifest value | Review boundary |
|---|---|
| `exact-reproduction` | Original algorithm, data or simulator, parameters, deterministic command, and machine-checked paper-value or figure comparison |
| `equation-level` | Explicitly bounded analytical slice with an independent oracle and no full-paper-parity claim |
| `research-reference` | Cited method or algorithmic structure with targeted tests but no complete original-condition comparison |
| `educational-surrogate` | Material simplification, substituted objective, synthetic-only path, fallback solver, or omitted paper component |

Synthetic data and local proxy equations are allowed only when they are
prominently labeled, justified, deterministic where applicable, and tested.
They can never support an unqualified paper-reproduction claim.

## 2. Check claim-to-evidence alignment

- [ ] README, repository index, `reproducibility.yaml`, executable certificate,
      code comments, filenames, and figure labels state the same evidence level.
- [ ] Exact reproduction identifies the paper version, original artifact/data
      and checksum, parameter mapping, reference figure/table/value, extraction
      method, deterministic environment and command, tolerances, and comparison.
- [ ] Equation-level and research-reference entries state the precise equations,
      algorithms, and omitted components; they explicitly deny unsupported
      full-paper or figure parity.
- [ ] Educational surrogates disclose synthetic data, proxy objectives,
      simplified physics, fallback solvers, and unavailable inputs.
- [ ] Terms such as “optimal,” “reproduced,” “matches,” “state of the art,” and
      performance gains are limited to what a direct oracle proves.

## 3. Review mathematics and implementation

- [ ] Equations use declared units, array shapes, index ranges, signs, complex
      conjugation, normalization, logarithm base, one-way/two-way conventions,
      SNR reference, and dB-to-linear conversion.
- [ ] Paper notation is mapped to code parameters without inventing missing
      values; local assumptions are separated from paper-reported parameters.
- [ ] Dimensional analysis is valid, and unlike physical quantities are not
      combined without explicit normalization.
- [ ] Solvers validate feasibility, budgets, constraints, termination, failure
      status, and returned postconditions; local convergence is not optimality.
- [ ] Finite accepted inputs do not silently overflow, underflow, create NaN, or
      lose meaningful cancellation; unsupported numeric domains fail explicitly.
- [ ] Randomness is injected through an explicit seed or generator, and repeated
      runs meet the declared determinism boundary.

## 4. Require independent executable evidence

- [ ] `reproducibility.yaml` records model, assumptions, parameters, oracle,
      expected values, numeric tolerances, command, checks, and limitations.
- [ ] The command emits one JSON certificate whose baseline, evidence level,
      paper-parity flag, checks, and pass/fail status match the manifest.
- [ ] At least one independent analytic, dense-grid, finite-difference,
      conservation, dimensional, or original-artifact oracle checks each strong
      numerical claim. Reusing the implementation under test is not independent.
- [ ] Tests cover nominal behavior, boundaries, invalid and infeasible inputs,
      extreme representable scales, failure accounting, and changed behavior.
- [ ] Trial rejection, solver failure, NaN/Inf, and Monte Carlo sample counts are
      reported; conditioning on successful trials is not hidden.

## 5. Review figures and reported outputs

Figures are presentation artifacts, never evidence by themselves. A plausible
trend, a round number, overlapping curves, or visual similarity can motivate a
bug investigation but cannot approve or reject a method without an executable
oracle. Review labels, units, uncertainty, sample counts, failure fractions,
provenance, and whether the plotted file was regenerated from the reviewed code.

Reject unsupported figure parity, hard-coded “paper-like” curves, hidden failed
trials, or captions that omit the synthetic/local boundary. Approve only when
the declared evidence level is accurate and every claim remains within its
machine-checked boundary.
