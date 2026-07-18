# Repository Audit Report

**Audit date and evidence cutoff:** 2026-07-18

**Repository:** [yuanhao-cui/Awesome-Integrated-Sensing-and-Communications](https://github.com/yuanhao-cui/Awesome-Integrated-Sensing-and-Communications)

**Scope:** all tracked documentation and links, publication metadata, standards status, citation metadata, datasets/tools, baseline documentation, Python implementations, tests, and repository gates.

## Executive finding

The initial repository was not a defensible academic bibliography or reproduction suite. It reused IEEE document URLs for unrelated titles, sent many entries to Google Scholar searches, contained incorrect author/venue/year/DOI records, presented study or draft activity as standards completion, overstated dataset attributes, and treated passing software tests or similar-looking plots as paper reproduction.

The content has therefore been reduced to a smaller, auditable index rather than preserving unverified breadth:

- 59 publication rows across nine topical files, representing 49 unique DOI-linked works; cross-listed works use the same DOI and metadata.
- 12 standards/pre-standardization rows linked to official IEEE, 3GPP, ITU-R,
  or ETSI records and labeled by maturity.
- no Google Scholar discovery links or reused IEEE document-number placeholders in the curated root/topic files;
- explicit evidence levels for every local baseline;
- corrected CFF and BibTeX data for the associated IEEE COMST survey;
- conservative dataset and tool directories with source and licensing boundaries;
- no claim that a cross-method leaderboard exists.

The lists are deliberately non-exhaustive. Accuracy and traceability take precedence over item count.

## Audit method

### Publication and factual checks

1. Normalize titles, DOI strings, author names, and URLs.
2. Check one identifier–one work and one normalized title–one version-of-record consistency.
3. Verify exact title, complete author order, venue, volume/issue/pages, year, and DOI against publisher deposits or Crossref.
4. Prefer canonical DOI links; retain arXiv only as a clearly labeled manuscript or when no version of record is known.
5. Remove unsupported priority, completeness, citation-count, deployment, performance, and “state of the art” claims.
6. Require contributors to check publisher correction/retraction notices when
   adding or materially revising a record; this audit does not claim a complete
   publisher-notice screening of every retained work.
7. Apply a 2026-07-18 cutoff and include recent work only when a formal publication record existed by that date.

The curation policy is informed by the public [IEEE Communications Surveys & Tutorials policies and guidelines](https://www.comsoc.org/publications/journals/ieee-comst/policies-guidelines) and IEEE [peer-review ethics guidance](https://journals.ieeeauthorcenter.ieee.org/submit-your-article-for-peer-review/ethics-in-peer-review/). This is a repository policy, not IEEE review or endorsement.

### Code and reproduction checks

The audit separates:

- **exact reproduction:** original method/data/parameters plus deterministic execution and a machine-checked paper-value comparison with declared tolerances;
- **equation-level reference:** an explicitly bounded analytical slice with independent numerical or analytic oracles and declared tolerances, without full-paper parity;
- **research reference:** a cited algorithmic structure with targeted tests but no accepted complete paper-result comparison;
- **educational surrogate:** a materially simplified, substituted, fallback, or synthetic-only path.

Unit tests, test count, coverage, and generated plots are insufficient on their own to establish exact reproduction.

### Link checks

Checks cover Markdown destinations, images, relative paths, fragments, canonical repository casing, DOI/title correspondence, and external reachability. Redirects and bot-blocked primary sources are recorded separately from genuine 404/invalid destinations.

The initial independent scans reported:

- offline Markdown scan: 27 files, 455 link occurrences, 332 unique destinations; the known internal placeholder was tools/README.md's PyRadar link;
- external scan: 372 occurrences, 289 unique destinations, 354 successful responses including 68 redirects, and 18 errors;
- content-correspondence review: at least 33 distinct titles had definite wrong IEEE article-number links, 40 distinct Google Scholar search URLs failed the primary-source rule, and eight cited DOI strings were unregistered.

The obsolete link_check_report.md was removed so that this report is the single audit record.

The final DOI audit found 109 DOI occurrences and 54 unique DOI records.
Crossref registered all 54. Every DOI had title evidence, with no unregistered
DOI, identifier–title conflict, or title mismatch after identifier-only labels
were excluded. Across 69 complete structured citation records (59 catalogue
rows, nine BibTeX entries, and one CFF preferred citation), title, full
publication-order authors, venue, and year matched 69/69. All optional fields
present in Crossref also matched: volume 68/68, issue 61/61, pages 65/65, and
article number 3/3. The only normalized-name exceptions were publication
bylines that retain initials used by the publications; every structured field
uses the complete publisher-deposited display name and publication order.

All 54 unique DOI-linked works are now frozen in the version-controlled
[authority metadata snapshot](audit/authority-metadata.json), with a separate
[JSON Schema](audit/authority-metadata.schema.json). The compact Crossref
snapshot records canonical DOI, exact title, complete deposited author order,
venue, year, volume, and, where deposited, issue, pages, or article number. Its
catalogue section covers the 49 works underlying all 59 topical rows, with
required-field coverage of 49/49 for DOI, title, authors, venue, year, and
volume; optional-field coverage is 46/49 issues, 47/49 page fields, and 2/49
article numbers. A separate auxiliary-reference section covers the remaining
five works cited by the code baselines: four journal articles and one
monograph. Each record retains its exact Crossref API source URL, publisher,
source type, deposit-update timestamp, and the 2026-07-18 retrieval date.

Standards evidence is deliberately not represented as Crossref data. The same
snapshot contains a separate 12-record official-source section with identifier,
official title, status, maturity, official URL, evidence boundary, and
2026-07-18 as-of date. The offline
[authority verifier](scripts/verify_authority_metadata.py) parses every
catalogue and standardization table field, all nine DOI-bearing BibTeX entries,
and the CFF preferred citation; compares them with the reviewed snapshot;
executes the declared JSON Schema Draft 2020-12; validates authority URL
shape/domain; and emits a SHA-256-bound JSON artifact in Gate 2. Adversarial
tests independently prove that changing one DOI's title, author list, venue, or
year—or adding an unreviewed manifest field—is rejected. This deterministic
gate does not claim to re-query Crossref or standards sites during CI;
refreshes are explicit review events, while the independent Lychee job remains
the live URL reachability gate.

The deterministic Markdown audit scanned 28 files and 248 references (119
unique external URLs), with zero malformed destinations, missing local targets,
placeholder destinations, or missing fragments. The independent Lychee 0.24.2
audit checked 385 destinations (277 unique): 381 succeeded, zero failed, and
four exact URLs used documented narrow exclusions for a bot-blocked or
non-HTML authority endpoint. Redirect counts are intentionally omitted because
they vary with live server behavior.

## High-impact factual corrections

### Associated survey and citation

The authoritative record is:

[Integrated Sensing and Communications Over the Years: An Evolution Perspective](https://doi.org/10.1109/COMST.2026.3655674), Di Zhang; Yuanhao Cui; Xiaowen Cao; Nanchi Su; Yi Gong; Fan Liu; Weijie Yuan; Xiaojun Jing; J. Andrew Zhang; Jie Xu; Christos Masouros; Dusit Niyato; Marco Di Renzo, IEEE Communications Surveys & Tutorials, vol. 28, pp. 5014–5048, 2026.

The root BibTeX and `preferred-citation` in CITATION.cff now include the DOI,
volume, pages, complete publication-order author list, and Yuanhao Cui's
publisher-deposited ORCID 0000-0001-6323-8559. The CFF top-level software
author is separately limited to Yuanhao Cui, the only author evidenced by the
repository's commit history; survey coauthors are not misattributed as software
authors. The invalid year-only CFF `date-released` field was removed.

### Representative publication corrections

| Work | Corrected record |
|---|---|
| Seventy Years of Radar and Communications | Fan Liu; Le Zheng; Yuanhao Cui; Christos Masouros; Athina P. Petropulu; Hugh Griffiths; Yonina C. Eldar; IEEE Signal Processing Magazine 40(5):106–121 (2023); [DOI](https://doi.org/10.1109/MSP.2023.3272881) |
| Gaussian-channel ISAC tradeoff | Yifeng Xiong; Fan Liu; Yuanhao Cui; Weijie Yuan; Tony Xiao Han; Giuseppe Caire; IEEE Transactions on Information Theory 69(9):5723–5751 (2023); [DOI](https://doi.org/10.1109/TIT.2023.3284449) |
| RIS SNR/CRB design | Rang Liu; Ming Li; Qian Liu; A. Lee Swindlehurst; IEEE Transactions on Wireless Communications 23(7):7456–7470 (2024); [DOI](https://doi.org/10.1109/TWC.2023.3341429) |
| XL-MIMO beam training | Jiali Nie; Yuanhao Cui; Zhaohui Yang; Weijie Yuan; Xiaojun Jing; IEEE Transactions on Mobile Computing 24(1):352–362 (2025); [DOI](https://doi.org/10.1109/TMC.2024.3462960) |

The category files contain full publication-order author lists rather than ambiguous “et al.” metadata.

### Standards status

| Item | Correct status at 2026-07-18 | Primary record |
|---|---|---|
| IEEE 802.11bf | IEEE 802.11bf-2025 is a published active WLAN sensing amendment; approved 2025-05-28, published 2025-09-26 | [IEEE SA](https://standards.ieee.org/ieee/802.11bf/11574/) |
| 3GPP TS 22.137 | Release-19 Stage-1 ISAC specification, under change control; portal version 19.1.0 at verification | [3GPP portal](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=4198) |
| 3GPP Release 20 | Contains studies and draft work; this audit does not describe it as a completed 6G ISAC air-interface standard | [3GPP status report](https://www.3gpp.org/dynareport?code=status-report.htm) |
| ITU-R M.2160-0 | In-force IMT-2030 framework recommendation; integration of sensing and communication is among its usage scenarios/capabilities | [ITU](https://www.itu.int/rec/R-REC-M.2160-0-202311-I/en) |
| ETSI ISG ISAC | Pre-standardization group; its GR documents are reports, not a deployed air-interface standard | [ETSI](https://www.etsi.org/technical-groups/isac/) |
| IEEE P802.15.4ab | Active PAR at verification; not presented as a published standard | [IEEE SA](https://standards.ieee.org/ieee/802.15.4ab/10694/) |

Unsupported claims about WRC spectrum being specifically allocated to ISAC, fixed 3GPP KPI/use-case counts, and completed Release-20 standardization were removed.

### Dataset corrections

- [XRF55](https://aiotgroup.github.io/XRF55/) is multimodal, not WiFi-only: its source reports 42.9K RF samples, 55 classes, 39 subjects, four scenes, and WiFi/RFID/mmWave/Kinect modalities.
- [MM-Fi](https://ntu-aiot-lab.github.io/mm-fi) reports more than 320K synchronized frames, five modalities, 40 subjects, and 25 action categories; the former 27-action value was removed.
- [Widar 3.0](https://tns.thss.tsinghua.edu.cn/widar3.0/) reports 16 users, 15 gestures, 15 locations, five orientations, and three environments; the former 16-gesture value was corrected.
- Hardware/tool bandwidth, subcarrier, antenna, and protocol tables were removed where no version-specific official evidence was attached.

## Baseline metadata and evidence audit

| Baseline | Correct cited record | Evidence level and boundary |
|---|---|---|
| CSI-ratio Doppler estimation | Xinyu Li; J. Andrew Zhang; Kai Wu; Yuanhao Cui; Xiaojun Jing, IEEE Sensors Journal 22(21):20886–20895 (2022), [DOI](https://doi.org/10.1109/JSEN.2022.3208272) | Educational surrogate; synthetic estimators, without the paper's disclosed hardware subset or curve data |
| Capacity–distortion | Yifeng Xiong; Fan Liu; Yuanhao Cui; Weijie Yuan; Tony Xiao Han; Giuseppe Caire, IEEE TIT 69(9):5723–5751 (2023), [DOI](https://doi.org/10.1109/TIT.2023.3284449) | Educational surrogate; repository objectives/examples do not constitute the paper's complete capacity–distortion numerical pipeline |
| Energy-efficient beamforming | Jiaqi Zou; Songlin Sun; Christos Masouros; Yuanhao Cui; Ya-Feng Liu; Derrick Wing Kwan Ng, IEEE TCOM 72(6):3766–3782 (2024), [DOI](https://doi.org/10.1109/TCOMM.2024.3369696) | Equation-level reference; single-user fixed-direction slice checked against dense-grid, explicit-FIM, and finite-difference oracles, without figure parity |
| ISAC resource allocation | Fuwang Dong; Fan Liu; Yuanhao Cui; Wei Wang; Kaifeng Han; Zhiqin Wang, IEEE TWC 22(5):3522–3536 (2023), [DOI](https://doi.org/10.1109/TWC.2022.3219463) | Educational surrogate; proxy allocation/QoS solver and synthetic scenarios, without an accepted paper-value comparison |
| OFDM ambiguity function | Educational waveform-analysis baseline rather than a single-paper reproduction | Educational surrogate; sidelobe and resolution statements are parameter- and window-dependent |
| RIS-ISAC beamforming | Rang Liu; Ming Li; Qian Liu; A. Lee Swindlehurst, IEEE TWC 23(7):7456–7470 (2024), [DOI](https://doi.org/10.1109/TWC.2023.3341429) | Educational surrogate; local SNR-feasibility certificate only, with the paper's CRB/algorithm/figures explicitly out of scope |
| XL-MIMO beam training | Jiali Nie; Yuanhao Cui; Zhaohui Yang; Weijie Yuan; Xiaojun Jing, IEEE TMC 24(1):352–362 (2025), [DOI](https://doi.org/10.1109/TMC.2024.3462960) | Educational surrogate; synthetic default data and no original-data result replay |

No local baseline currently satisfies the repository's exact-reproduction definition. Generated figures are repository outputs unless and until an explicit numerical comparison artifact passes.

### Public artifact and result-parity boundary

| Baseline | Public artifact reviewed | Consequence for this repository |
|---|---|---|
| CSI-ratio | The paper identifies a Widar 2.0 hardware setup and the project publishes a sample archive, but the exact paper subset, raw trajectories, and curve values are not disclosed | Synthetic estimator invariants can be tested; hardware-result or figure parity cannot be certified |
| Capacity–distortion | No author code, original channel realizations, seed, or raw curve values were located | Paper parameters are recorded for provenance, but local proxy objectives are not paper-result oracles |
| Energy-efficient beamforming | No public author implementation, channel realizations, noise values, seeds, or curve data were located | The retained artifact is limited to a single-user equation-level slice with independent grid/FIM/finite-difference oracles |
| Resource allocation | The [author MATLAB repository](https://github.com/FuwangDong/2023-TWC-Sensing-AS-Service-Resource-Allocation) was reviewed at commit `c91892260325dd40f911d760db548f68d4d8b614`; it requires CVX/SeDuMi, discloses no fixed seed, and has no standard license file | The code is linked, not copied; the local Python model remains an independently tested educational surrogate |
| RIS beamforming | The [author MATLAB repository](https://github.com/RangLiu0706/SNR-CRB-constrained-beamforming-for-RIS-ISAC) was reviewed at commit `85cfeb2e3112013dc391d04b7f2102e28175ee0c`; it states a personal/non-commercial condition and provides no fixed Monte Carlo seed or raw curve values | The former scalar CRB proxy and similarity plots were removed; only the declared local SNR-feasibility certificate remains |
| XL-MIMO beam training | The [first-author repository](https://github.com/fly-winder/near-field-beamforming-using-deeplearning) was reviewed at commit `c3f5b66afa82b56b09e9a783a5f1615a59b8c2fd`; it has no standard license, omits the `pcsi`/`ecsi` inputs, and its script hyperparameters differ from Table III | Author weights/CSV files are not treated as a turnkey figure oracle; the local default path is explicitly synthetic |
| OFDM ambiguity | This is a survey-inspired waveform utility, not a reproduction package for one paper | Analytic signal-processing invariants are the only accepted numerical evidence |

The energy manuscript also contains a disclosed symbol typo (`N=30` where the
defined frame-length symbol is `L`) and a `P0` difference between the accepted
institutional manuscript and the arXiv source. The baseline records both rather
than silently selecting a version.

The aggregate single-process coverage hard gate is 70%. The final locked
Python 3.12 run measured 85.05% statement coverage (3,739 of 4,396 statements)
with 664/664 tests passing. Locked Python 3.10 and 3.11 environments
independently passed the same 664-test suite. This replaces the unsupported
claim that every baseline had at least 80% coverage; new or modified code must
still add targeted tests rather than relying on the aggregate percentage.

The seven machine-readable baseline certificates contain 99 declared
checks and passed on all three Python versions. A separate read-only numerical
red team evaluated 3,000 actual-binary resource-allocation inputs against
500-digit oracles (worst accepted error: 2 ULP), permutation and cancellation
tails, exact integer coherent sums, analytic gradients, and IEEE 754 binary32
round-to-nearest-even boundaries at the subnormal and finite upper edges.
These checks establish the declared equation/invariant slices; they do not
upgrade any educational surrogate to a paper reproduction.

## Recent formally published additions

To bring the index through the cutoff without speculative “latest” labeling, the audit added selected 2026 version-of-record publications, including:

- [Toward 6G Networks: A Survey on Integrated Sensing and Communication in Cell-Free Massive MIMO](https://doi.org/10.1109/JIOT.2026.3693228);
- [Large AI Model for Multimodal Integrated Sensing and Communication](https://doi.org/10.1109/MNET.2026.3661589);
- [Task-Oriented Integrated Sensing and Communication for Multidevice Cooperative Motion Recognition](https://doi.org/10.1109/TMC.2026.3664359);
- [Robust Design of Integrated Sensing and Communication in LEO Satellite Systems](https://doi.org/10.1109/JIOT.2026.3687912);
- [Integrated sensing and communication for optical transmission networks](https://doi.org/10.1364/JOCN.584918);
- [Integrated sensing and communication system exceeding a 200 km repeaterless fiber link](https://doi.org/10.1364/PRJ.585704).

Inclusion means that a formal record existed by 2026-07-18, not that the work is historically first or technically superior.

## Two strict gates

| Gate | Pass criteria | Audited-tree result |
|---|---|---|
| Gate 1 — software and simulation integrity | Exact direct dependencies; Python 3.10–3.12 CI matrix; warning-free compilation and tests; valid CFF/YAML; seven machine-readable JSON certificates; aggregate single-process coverage at least 70%; numerical paper-value comparison whenever exact reproduction is claimed | **PASS locally.** No-cache locked Python 3.10.20 and locked Python 3.11.15/3.12.13 environments each passed 664/664 strict tests and all seven certificates (99/99 checks). The covered Python 3.12 run measured 85.05% (3,739/4,396 statements). Dependency audit, install consistency, CFF 1.2/YAML validation, Ruff, YAML lint, compilation, and whitespace checks all passed; compilation and tests were warning-free. The final uv refresh emitted one non-fatal normalization warning for an upstream package's legacy version specifier, without lock drift or install inconsistency. The workflow repeats the covered run on Python 3.10, 3.11, and 3.12 and separately performs no-cache macOS arm64 installs for all three versions. |
| Gate 2 — scholarly content and link integrity | Canonical identifiers; exact metadata; no title/identifier collisions; official standards records and maturity; evidence-bounded descriptions; no unsupported reproduction claims; no malformed or broken tracked links; narrowly documented exact endpoint exclusions | **PASS locally.** The reviewed snapshot covers 54/54 DOI authorities, all 69 structured citation records, and 12 official standards records; its SHA-256 is `3e34b11ebaa31778093d6e76f259e497afd75f2c7c732fe284d46d29f14028a0`. The deterministic audit found zero failures across 28 Markdown files, 248 references, and 119 unique external URLs. Independent Lychee checked 385 destinations (277 unique): 381 succeeded, zero failed, and four exact documented exclusions were applied. |

Passing one gate does not compensate for failing the other. Repository policy
requires both workflows to pass before merge; each uploads machine-readable
evidence, and neither uses a fail-open command. Classic protection for `main`
was enabled and independently read back through the GitHub API on 2026-07-18.
It requires the eight exact Gate 1/Gate 2 job names from GitHub Actions
(`app_id` 15368), strict branch updating, a pull request, resolved
conversations, and linear history. The rule applies to administrators and
disables force pushes and branch deletion. Because the repository had one
collaborator at the audit date, its solo-maintainer review policy sets required
approvals to zero and does not require approval of the latest reviewable push;
the maintainer can merge only after the eight automated checks and the other
protection conditions pass. These automated gates do not constitute
independent human peer review.

## Remaining limits

- External sites can change or rate-limit automated checks after the audit date.
- Standards status can change after 2026-07-18 and must be rechecked before downstream reliance.
- A curated list cannot prove literature completeness; it documents its search and evidence boundary instead.
- Software tests cannot validate an unstated physical model, unavailable original data, or unpublished experimental conditions.
- External publications, datasets, and tools retain their own licenses and terms; repository inclusion does not relicense or endorse them.
