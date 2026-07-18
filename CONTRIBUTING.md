# Contributing

This repository is maintained as academic infrastructure. Every contribution must be traceable to primary evidence and must preserve the distinction between a citation, an implementation, a numerical reproduction, and a standard.

## Editorial standard

The repository adopts its own curation policy informed by the publicly stated [IEEE Communications Surveys & Tutorials policies and guidelines](https://www.comsoc.org/publications/journals/ieee-comst/policies-guidelines) and IEEE's [peer-review ethics guidance](https://journals.ieeeauthorcenter.ieee.org/submit-your-article-for-peer-review/ethics-in-peer-review/). This is not an IEEE review, certification, or endorsement.

Reviewers assess:

- timeliness and relevance to ISAC;
- technical correctness and whether conclusions stay within the cited model or experiment;
- completeness and balance of references for survey-style additions;
- originality relative to already indexed work, without unsupported priority claims;
- source quality, metadata consistency, and disclosure of corrections or retractions;
- reproducibility evidence for code and numerical claims.

## Adding or updating a publication

### Required record

Provide all fields available in the version of record:

1. exact title, preserving mathematical symbols and subtitle;
2. all authors in publication order;
3. venue, volume, issue, pages or article number, and publication year;
4. canonical DOI URL;
5. arXiv identifier only when useful for an openly accessible manuscript or when no version of record is known;
6. category and a neutral one-sentence scope description;
7. date and source used to verify the metadata.

Use publisher/DOI metadata first, then an official institutional record. Use Crossref as a metadata cross-check. Google Scholar, search-result pages, citation aggregators, and generated text are discovery aids, not final evidence.

### Identifier and title consistency

Before submitting:

- normalize case, punctuation, whitespace, and Unicode when checking titles;
- confirm that one DOI or publisher identifier maps to exactly one work;
- confirm that one normalized title does not point to conflicting DOI, venue, or year records;
- reuse the same canonical DOI URL wherever the work appears;
- remove duplicate preprint rows after a version of record is indexed, unless the relationship is explicitly useful;
- reject an entry if a URL resolves to a different title or if the conflict cannot be resolved from primary records.

### Strong claims

Terms such as “first,” “latest,” “state of the art,” “optimal,” “real-time,” “deployed,” “standards-compliant,” citation counts, and performance gains require a direct primary source and the precise comparison boundary. Prefer a neutral description when priority or completeness cannot be proven.

Do not compare numbers across papers without aligning signal bandwidth, carrier frequency, array geometry, power and noise definitions, channel/target model, dataset split, metrics, and uncertainty.

### Corrections, expressions of concern, and retractions

Check the publisher record and Crossmark or equivalent status before adding or materially updating an item. If a work has a correction, link and describe it. If it is retracted, remove it from ordinary recommendation tables and document the reason in the audit report. Open an issue when status is ambiguous.

### Topic classification

Place an item under its principal technical contribution. Cross-list only when each category adds clear navigational value. Similar terminology does not establish equivalence: for example, RIS, metasurface, near-field, optical, cell-free, and multimodal systems carry different models and evidence.

## Standards and pre-standardization

Standards entries must link to the official organization page and record:

- organization, exact document or project identifier, title, release/version, and verification date;
- maturity: published standard, normative specification, recommendation, study report, work item, draft project, or pre-standardization report;
- which statement is normative and which is informative.

Never turn a study report or draft into a published standard. Recheck status immediately before merging because release and project status can change.

## Datasets and tools

Dataset additions must link to the official project or archival record, identify modalities and task scope, and state a version for any scale claim. Record licensing, privacy/consent, calibration, synchronization, split, and citation constraints where applicable.

Tool additions must use the official project page. Do not infer bandwidth, antenna count, protocol, chipset, driver, or operating-system compatibility from a product name; cite the version-specific source or omit the field. External licenses remain controlling.

## Code evidence levels

Every baseline README and repository index must use one of these levels:

| Level | Manifest value | Minimum evidence |
|---|---|---|
| Exact reproduction | `exact-reproduction` | Original algorithm, data or simulator, parameters, deterministic command, and machine-checked comparison to specified paper values or figures with declared absolute/relative tolerances |
| Equation-level reference | `equation-level` | An explicitly bounded analytical slice, deterministic command, independent numeric or analytic oracles, declared tolerances, and an explicit no-full-paper-parity boundary |
| Research reference | `research-reference` | A cited method or algorithmic structure with targeted correctness tests, but no complete accepted comparison under the paper's original conditions |
| Educational surrogate | `educational-surrogate` | Any material simplification, substitute objective, synthetic-only data path, fallback solver, or omitted paper component |

A file named “reproduce,” a similar-looking plot, or passing unit tests is not sufficient for exact reproduction. Repository-generated plots must be labeled as examples until the comparison artifact passes review.

### Baseline submission requirements

A new or modified baseline must include:

- a README with the system model, equations, assumptions, provenance, evidence level, and limitations;
- pinned or constrained dependencies compatible with the repository's supported Python matrix;
- targeted tests for dimensions, domains, invariants, edge cases, infeasible inputs, convergence or termination behavior, and deterministic seeds where supported;
- an executable example using documented parameters;
- source citations with correct title, full author order, venue, year, DOI, and code-paper relationship;
- license language consistent with the repository root and notices for third-party material.

The aggregate single-process coverage hard gate is 70%. This is a repository-level minimum, not an assertion that every baseline has 70% or 80% coverage. New or changed code must add tests that exercise the changed behavior; coverage cannot substitute for mathematical correctness.

### Reproducible dependency environment

The `ci`, `integrity`, and `cff-validation` groups in `pyproject.toml` declare
the exact direct dependencies. `uv.lock` is the checked, universal resolution for CPython
3.10–3.12 on Ubuntu x86_64 and macOS arm64; it fixes every transitive version,
platform marker, distribution URL, and SHA-256 hash. The root requirements
files mirror the direct pins for baseline constraints and Dependabot discovery,
but they are not standalone environment locks. `cffconvert` 2.0.0 requires
JSON Schema 3.x, whereas repository metadata is validated with the pinned
Draft 2020-12 implementation in JSON Schema 4.x. These environments are
therefore explicit conflicting lock groups and must never be installed
together. The only unavoidable source
build in the current environment is also constrained to the exact setuptools
version recorded in the lock.

Use the uv version pinned in the workflows and create the complete development
environment with:

```console
uv lock --check
uv sync --locked --only-group ci
.venv/bin/python scripts/check_dependency_lock.py
uv pip check --python .venv/bin/python
UV_PROJECT_ENVIRONMENT=.venv-cff uv sync --locked --only-group cff-validation
.venv-cff/bin/cffconvert --validate
```

When a dependency is deliberately updated, change the matching pin in
`pyproject.toml` and the appropriate root requirements file, regenerate
`uv.lock`, and review the complete lock diff. Never substitute a runtime
`pip freeze` for the input lock or weaken a platform incompatibility by leaving
a version open; use an explicit environment marker when a fork is necessary.

### Upgrading to exact reproduction

An upgrade request must add a machine-readable manifest containing:

- paper DOI and exact version;
- original or archived data/simulator identity and checksum;
- parameter mapping from paper notation to configuration;
- deterministic command and environment;
- reference table/figure/value identifiers;
- extracted reference values with extraction method;
- comparison output, tolerances, and pass/fail result.

If proprietary data, unavailable hardware, or undocumented parameters prevent the comparison, retain the equation-level-reference, research-reference, or educational-surrogate label as warranted and state the blocker.

## Benchmark results

The benchmark directory currently has no verified leaderboard. A future submission must define task, dataset/simulator version, split, metrics, units, uncertainty, baselines, environment, command, result artifact, commit provenance, and independent rerun tolerance before any ranking is published.

## Pull-request evidence checklist

- [ ] Exact metadata was verified from the version of record or other primary source.
- [ ] DOI/title/URL conflicts and duplicates were checked repository-wide.
- [ ] Corrections and retraction status were checked.
- [ ] Strong claims have direct evidence and explicit scope, or were removed.
- [ ] Standards maturity and verification date are explicit.
- [ ] Internal links and fragments resolve; external links were audited.
- [ ] Code tests pass and changed behavior has targeted coverage.
- [ ] Evidence level is accurate; numerical reproduction has a comparison artifact.
- [ ] CFF, YAML, Markdown tables, and citation data validate.
- [ ] External licenses, data terms, and third-party notices are respected.

Maintainers may decline technically related work when evidence is incomplete, metadata is unresolved, the scope duplicates a stronger indexed record, or long-term link/reproduction maintenance is impractical.

## Conduct and disclosure

Follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Disclose conflicts of interest when proposing or reviewing your own work, collaborators' work, or competing implementations. Critique evidence and methods, not people.
