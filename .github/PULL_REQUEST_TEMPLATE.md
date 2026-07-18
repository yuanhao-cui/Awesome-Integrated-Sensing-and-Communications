## Description

Describe the change, its evidence boundary, and any user-visible or scientific claim it affects.

## Type

- [ ] Publication or standards record
- [ ] Baseline or numerical certificate
- [ ] Bug fix
- [ ] Dataset, tool, or benchmark update
- [ ] Documentation or repository infrastructure

## Scientific-content integrity

- [ ] Metadata, author order, venue, year, DOI, correction/retraction status, and standards maturity were checked against primary records.
- [ ] DOI/title/URL conflicts and duplicate records were checked repository-wide.
- [ ] Strong claims have direct evidence and a precise comparison boundary, or were removed.
- [ ] Internal links/fragments resolve and external links were audited; any exception is exact and documented.
- [ ] Data, code, and third-party license provenance is explicit.

## Baseline evidence, when applicable

- [ ] The README declares one evidence level: `exact-reproduction`, `equation-level`, `research-reference`, or `educational-surrogate`.
- [ ] Scope, equations, units, conventions, assumptions, substitutions, synthetic paths, and omitted components are explicit.
- [ ] `reproducibility.yaml` records provenance, parameters, oracle, expected values, tolerances, command, checks, and limitations.
- [ ] The deterministic command emits a matching passing JSON certificate; figure appearance alone is not used as evidence.
- [ ] Changed behavior has targeted tests for nominal, boundary, invalid, infeasible, numeric-range, and failure paths.
- [ ] Seeds and rejected/failed trials are accounted for; solver convergence is not described as optimality.
- [ ] Any exact-reproduction claim includes original data or simulator identity/checksum and a machine-checked paper-value comparison.

## Software and repository integrity

- [ ] The full strict test suite, CFF/YAML validation, lint, link inventory, and relevant live-link audit pass.
- [ ] Aggregate single-process repository coverage remains at least 70%; no per-baseline 80% threshold is claimed.
- [ ] Dependencies remain compatible with the supported Python matrix and follow the root pin/constraint policy.
- [ ] Generated artifacts were regenerated from the reviewed code and contain no stale or misleading outputs.
