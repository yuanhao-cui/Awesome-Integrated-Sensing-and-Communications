# Awesome Integrated Sensing and Communications

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Paper: IEEE COMST 2026](https://img.shields.io/badge/Paper-IEEE%20COMST%202026-blue?logo=ieee)](https://doi.org/10.1109/COMST.2026.3655674)
[![arXiv: 2504.06830](https://img.shields.io/badge/arXiv-2504.06830-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.06830)
[![CI](https://img.shields.io/github/actions/workflow/status/yuanhao-cui/Awesome-Integrated-Sensing-and-Communications/test.yml?label=tests)](https://github.com/yuanhao-cui/Awesome-Integrated-Sensing-and-Communications/actions)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

A curated research index and a set of tested reference implementations for integrated sensing and communications (ISAC), accompanying:

**[Integrated Sensing and Communications Over the Years: An Evolution Perspective](https://doi.org/10.1109/COMST.2026.3655674)**

Di Zhang; Yuanhao Cui; Xiaowen Cao; Nanchi Su; Yi Gong; Fan Liu; Weijie Yuan; Xiaojun Jing; J. Andrew Zhang; Jie Xu; Christos Masouros; Dusit Niyato; Marco Di Renzo.

IEEE Communications Surveys & Tutorials, vol. 28, pp. 5014–5048, 2026.

## Evidence and scope

- The bibliography is curated and non-exhaustive; it is not a citation ranking.
- Publication metadata and standards status were checked through 2026-07-18 using DOI/publisher records and official standards-body pages.
- A DOI-linked version of record is preferred over duplicate preprint and search-result links.
- “First,” “latest,” “state of the art,” performance, deployment, and standards-compliance claims are omitted unless a primary source directly supports them.
- Code tests establish only the properties they exercise. They do not, by themselves, establish numerical reproduction of a paper.
- See the [audit report](AUDIT_REPORT.md) for corrections, evidence boundaries, and gate definitions.

## Research map

| Area | Curated file | Scope |
|---|---|---|
| Surveys and tutorials | [paper/surveys.md](paper/surveys.md) | General surveys, signal-processing overviews, limits, and mobile networks |
| Theory and limits | [paper/theory.md](paper/theory.md) | Information, estimation, detection, and beamforming tradeoffs |
| Waveforms and signals | [paper/waveform.md](paper/waveform.md) | OTFS, single-carrier, FMCW, index modulation, and beamforming |
| Antennas and surfaces | [paper/antenna.md](paper/antenna.md) | Near-field/XL-MIMO arrays, RIS, and metasurfaces |
| Optical ISAC | [paper/optical.md](paper/optical.md) | Fibre, optical wireless, photonic, and illumination systems |
| Networked ISAC | [paper/network.md](paper/network.md) | Multistatic, cooperative, distributed-MIMO, cell-free, UAV, and satellite systems |
| AI and machine learning | [paper/ai_ml.md](paper/ai_ml.md) | Model-driven learning, deep learning, multimodal, and task-oriented systems |
| Security and privacy | [paper/security.md](paper/security.md) | Explicit threat models, eavesdropping, artificial noise, and adversarial surfaces |
| Standards | [paper/standardization.md](paper/standardization.md) | Official 3GPP, IEEE, ITU-R, and ETSI status records |
| Applications | [paper/application.md](paper/application.md) | Aerial, space, optical, and fibre contexts |

## Selected source-verified publications

| Publication | Bibliographic record | Why it is indexed |
|---|---|---|
| [Integrated Sensing and Communications: Toward Dual-Functional Wireless Networks for 6G and Beyond](https://doi.org/10.1109/JSAC.2022.3156632) | IEEE JSAC, vol. 40, no. 6, pp. 1728–1767, 2022 | General ISAC survey and tutorial |
| [An Overview of Signal Processing Techniques for Joint Communication and Radar Sensing](https://doi.org/10.1109/JSTSP.2021.3113120) | IEEE JSTSP, vol. 15, no. 6, pp. 1295–1315, 2021 | Signal-processing overview |
| [Seventy Years of Radar and Communications: The Road from Separation to Integration](https://doi.org/10.1109/MSP.2023.3272881) | IEEE Signal Processing Magazine, vol. 40, no. 5, pp. 106–121, 2023 | Historical synthesis |
| [A Survey on Fundamental Limits of Integrated Sensing and Communication](https://doi.org/10.1109/COMST.2022.3149272) | IEEE COMST, vol. 24, no. 2, pp. 994–1034, 2022 | Fundamental-limit survey |
| [On the Fundamental Tradeoff of Integrated Sensing and Communications Under Gaussian Channels](https://doi.org/10.1109/TIT.2023.3284449) | IEEE Transactions on Information Theory, vol. 69, no. 9, pp. 5723–5751, 2023 | Gaussian-channel capacity–distortion analysis |
| [Cramér-Rao Bound Optimization for Joint Radar-Communication Beamforming](https://doi.org/10.1109/TSP.2021.3135692) | IEEE Transactions on Signal Processing, vol. 70, pp. 240–253, 2022 | CRB-oriented beamforming |
| [Toward Seamless Sensing Coverage for Cellular Multi-Static Integrated Sensing and Communication](https://doi.org/10.1109/TWC.2023.3325849) | IEEE Transactions on Wireless Communications, vol. 23, no. 6, pp. 5363–5376, 2024 | Cellular multistatic sensing |
| [Toward 6G Networks: A Survey on Integrated Sensing and Communication in Cell-Free Massive MIMO](https://doi.org/10.1109/JIOT.2026.3693228) | IEEE Internet of Things Journal, vol. 13, no. 14, pp. 30028–30052, 2026 | 2026 survey of cell-free massive-MIMO ISAC |

Full author lists and topic boundaries are in the category files.

## Standards snapshot

Standards change independently of this repository. Status below was checked on 2026-07-18:

- [IEEE 802.11bf-2025](https://standards.ieee.org/ieee/802.11bf/11574/) is a published active WLAN sensing amendment.
- [3GPP TS 22.137: Integrated Sensing and Communication](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=4198) is a Release-19 Stage-1 ISAC specification under change control.
- [Recommendation ITU-R M.2160-0](https://www.itu.int/rec/R-REC-M.2160-0-202311-I/en) is the IMT-2030 framework recommendation and includes integration of sensing and communication among its usage scenarios and capabilities.
- [ETSI Industry Specification Group on ISAC](https://www.etsi.org/technical-groups/isac/) performs pre-standardization work; its group reports are not an air-interface standard.
- Release-20 studies and draft items must not be described as a completed 6G ISAC standard. See the [standards page](paper/standardization.md) for official status links and document types.

## Code: evidence-graded baselines

The repository uses four evidence levels:

1. **Exact reproduction** — original algorithm, data or simulator, parameters, deterministic command, and machine-checked comparison to specified paper values or figures with declared tolerances.
2. **Equation-level reference** — an explicitly bounded analytical slice with independent numerical or analytic oracles and declared tolerances, but no full paper-result parity.
3. **Research reference** — a cited method or algorithmic structure with targeted tests, but without the complete original-condition numerical comparison.
4. **Educational surrogate** — a simplified, substituted, or synthetic model intended for study or software testing.

Their machine-readable manifest values are `exact-reproduction`, `equation-level`,
`research-reference`, and `educational-surrogate`.

| Baseline | Current evidence level | Evidence boundary |
|---|---|---|
| [CSI-ratio Doppler estimation](code/baselines/csi_ratio_doppler_estimation/) | Educational surrogate | Synthetic estimators; no disclosed paper subset or hardware-data replay |
| [Capacity–distortion](code/baselines/isac_capacity_distortion/) | Educational surrogate | Tractable numerical objectives and examples; not the complete paper pipeline |
| [Energy-efficient beamforming](code/baselines/isac_energy_efficient_beamforming/) | Equation-level reference | Single-user fixed-direction slice checked against grid, explicit FIM, and finite-difference oracles; no figure parity |
| [ISAC resource allocation](code/baselines/isac_resource_allocation/) | Educational surrogate | Proxy allocation/QoS solver and synthetic scenarios; no accepted paper-value comparison |
| [OFDM ambiguity function](code/baselines/ofdm_ambiguity_function/) | Educational surrogate | Standalone waveform illustration |
| [RIS-ISAC beamforming](code/baselines/ris_isac_beamforming/) | Educational surrogate | Local SNR-feasibility model; the paper's CRB/algorithm/figures are not implemented |
| [XL-MIMO beam training](code/baselines/xl_mimo_beam_training/) | Educational surrogate | Synthetic default data and simplified evaluation |

The aggregate single-process coverage hard gate is 70%. This is a repository-level minimum, not a claim that every baseline has 70% or 80% coverage. New or changed code must add targeted tests. See [code/README.md](code/README.md) for the executable evidence contract.

## Datasets, tools, and benchmarks

- [Datasets](datasets/README.md) records official sources and clearly separates RF, multimodal, radar, and robotic data.
- [Tools](tools/README.md) links only to identifiable official project pages and avoids unverified hardware capability tables.
- [Benchmark status](benchmark/README.md) states that no verified cross-method leaderboard currently exists and defines the minimum protocol for a future one.

External projects retain their own licenses and usage terms. Inclusion is not an endorsement or a compatibility guarantee.

## Two strict gates

Every accepted revision must satisfy two independent gates:

| Gate | Required checks |
|---|---|
| Scientific-content integrity | Canonical identifiers; exact title/author/venue/year metadata; one identifier–one work consistency; primary-source standards status; explicit evidence boundaries; no unsupported priority, performance, deployment, or reproduction claims |
| Software and repository integrity | Passing tests; aggregate single-process coverage at or above 70%; valid CFF/YAML; no broken internal links or fragments; external-link audit with documented exceptions; deterministic reproduction checks where reproduction is claimed |

Passing one gate does not compensate for failing the other.

## Citation

~~~bibtex
@article{zhang2026integrated,
  author  = {Zhang, Di and Cui, Yuanhao and Cao, Xiaowen and Su, Nanchi and Gong, Yi
             and Liu, Fan and Yuan, Weijie and Jing, Xiaojun and Zhang, J. Andrew
             and Xu, Jie and Masouros, Christos and Niyato, Dusit and Di Renzo, Marco},
  title   = {Integrated Sensing and Communications Over the Years: An Evolution Perspective},
  journal = {IEEE Communications Surveys \& Tutorials},
  volume  = {28},
  pages   = {5014--5048},
  year    = {2026},
  doi     = {10.1109/COMST.2026.3655674}
}
~~~

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

## Contributing and license

Contributions must follow [CONTRIBUTING.md](CONTRIBUTING.md), including the primary-source metadata, conflict-resolution, retraction/correction, standards-status, and code-evidence rules.

Repository-authored material is licensed under [CC BY-SA 4.0](LICENSE). External publications, datasets, code, and tools remain governed by their respective rights holders and licenses.
