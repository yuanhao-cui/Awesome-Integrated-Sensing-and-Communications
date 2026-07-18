# Benchmark Status and Submission Contract

There is currently no verified cross-method leaderboard in this directory. The repository does not provide a benchmark runner or a published results file, so placeholder scores and instructions for nonexistent commands have been removed.

## Minimum protocol for a future benchmark

A benchmark result is eligible for review only when the contribution records:

1. the exact task, dataset or simulator version, preprocessing, split, and exclusion rules;
2. all channel, waveform, array, power, noise, target, and hardware assumptions;
3. metric definitions, units, aggregation, confidence intervals, and failure handling;
4. dependency lockfile or pinned environment, deterministic seeds where supported, and the exact command;
5. a machine-readable result artifact and provenance linking it to a commit;
6. comparison methods run under the same protocol, with original-source citations;
7. independent rerun evidence within stated numerical tolerances.

## Metric discipline

BER, communication rate, detection probability, localization error, CRB, energy efficiency, latency, and Pareto summaries describe different objectives. A single “Pareto score” is not meaningful until its normalization, integration domain, dominance rule, and uncertainty treatment are fixed.

## Publication rule

No row should be called “state of the art,” “best,” or “reproduced” solely because a script completed. Numerical claims require a checked result artifact and comparison to the cited reference under aligned assumptions.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the repository-wide evidence policy.
