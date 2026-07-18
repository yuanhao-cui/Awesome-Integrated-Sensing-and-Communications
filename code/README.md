# ISAC Reference Implementations

The code is organized as tested research and educational baselines. Passing unit tests establishes the tested software properties; it does not by itself establish numerical reproduction of a paper.

## Evidence levels

| Level | Manifest value | Required evidence |
|---|---|---|
| Exact reproduction | `exact-reproduction` | Original algorithm, data or simulator, parameters, deterministic command, and a machine-checked comparison to specified paper values or figures with declared tolerances |
| Equation-level reference | `equation-level` | Explicitly bounded analytical slice, deterministic command, independent numeric or analytic oracles, declared tolerances, and no full-paper parity claim |
| Research reference | `research-reference` | Implements a cited method or algorithmic structure and has targeted tests, but lacks a complete paper-value comparison under the original conditions |
| Educational surrogate | `educational-surrogate` | Simplifies, substitutes, or synthesizes part of the cited model; useful for study or testing but not a paper reproduction |

Until a baseline supplies all exact-reproduction evidence, its figures must be described as repository-generated examples rather than reproduced paper figures.

## Current classification

| Baseline | Evidence level | Boundary |
|---|---|---|
| [CSI-ratio Doppler estimation](baselines/csi_ratio_doppler_estimation/) | Educational surrogate | Synthetic estimators; no disclosed paper subset or hardware-data replay |
| [Capacity–distortion](baselines/isac_capacity_distortion/) | Educational surrogate | Uses tractable numerical objectives and examples; not the paper's complete numerical pipeline |
| [Energy-efficient beamforming](baselines/isac_energy_efficient_beamforming/) | Equation-level reference | Single-user fixed-direction slice checked against independent grid, FIM, and derivative oracles; no figure parity |
| [ISAC resource allocation](baselines/isac_resource_allocation/) | Educational surrogate | Proxy allocation/QoS solver and synthetic scenarios; no accepted paper-value comparison |
| [OFDM ambiguity function](baselines/ofdm_ambiguity_function/) | Educational surrogate | Standalone waveform illustration, not a paper reproduction |
| [RIS-ISAC beamforming](baselines/ris_isac_beamforming/) | Educational surrogate | Local SNR-feasibility model; no implementation of the paper's CRB, algorithm, or figures |
| [XL-MIMO beam training](baselines/xl_mimo_beam_training/) | Educational surrogate | Synthetic default data and simplified evaluation; no original-data replay |

## Test gate

Run the repository's documented test command from the repository root. The aggregate single-process coverage hard gate is 70%. This is a minimum CI threshold, not a claim that every baseline has 70% or 80% coverage. New or changed code must add targeted tests for its mathematical invariants, edge cases, and failure modes.

## Adding or upgrading a baseline

Follow [CONTRIBUTING.md](../CONTRIBUTING.md). A request to upgrade an evidence level must include the provenance, deterministic command, reference values, tolerances, and generated comparison artifact.
