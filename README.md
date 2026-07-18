# ⚡ Awesome Integrated Sensing and Communications (ISAC)

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20COMST%202026-blue?logo=ieee)](https://doi.org/10.1109/COMST.2026.3655674)
[![arXiv](https://img.shields.io/badge/arXiv-2504.06830-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.06830)
[![Featured](https://img.shields.io/badge/Featured-53-orange)](#-featured-papers)
[![Datasets](https://img.shields.io/badge/Datasets-11-purple)](#-datasets--benchmarks)
[![Tools](https://img.shields.io/badge/Tools-10-green)](#-open-source-and-research-tools)
[![Baselines](https://img.shields.io/badge/Baselines-7-2ea44f)](#-reproducible-baselines--evidence-graded)
[![Stars](https://img.shields.io/github/stars/yuanhao-cui/Awesome-Integrated-Sensing-and-Communications?style=social)](https://github.com/yuanhao-cui/Awesome-Integrated-Sensing-and-Communications)
[![Tests](https://img.shields.io/github/actions/workflow/status/yuanhao-cui/Awesome-Integrated-Sensing-and-Communications/test.yml?label=CI)](https://github.com/yuanhao-cui/Awesome-Integrated-Sensing-and-Communications/actions)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

> 📡 A curated list of **Integrated Sensing and Communications** resources,
> accompanying our survey: **“Integrated Sensing and Communications Over the
> Years: An Evolution Perspective”** (IEEE COMST, 2026).

The homepage retains the original visual roadmap, timelines, featured-paper
tables, tools, datasets, code, and benchmark sections. Incorrect identifiers,
metadata, standards status, dataset statistics, hardware specifications, and
reproduction claims have been corrected against primary or publisher records.

---

## 📝 Citation

If you find this repository useful, please cite our survey:

```bibtex
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
```

**Authors in publication order**: Di Zhang · **Yuanhao Cui** · Xiaowen Cao ·
Nanchi Su · Yi Gong · Fan Liu · Weijie Yuan · Xiaojun Jing · J. Andrew Zhang ·
Jie Xu · Christos Masouros · Dusit Niyato · Marco Di Renzo

Machine-readable metadata is available in [CITATION.cff](CITATION.cff).

---

## 📑 Table of Contents

- [📝 Citation](#-citation)
- [🧬 The Evolution of ISAC](#-the-evolution-of-isac)
- [📅 Evolution Timelines](#-evolution-timelines-by-subfield)
- [⭐ Featured Papers](#-featured-papers)
- [📚 All Papers by Topic](#-all-papers-by-topic)
- [📋 Standards Snapshot](#-standards-snapshot)
- [🧰 Open-Source and Research Tools](#-open-source-and-research-tools)
- [🔗 Related Projects](#-related-projects)
- [📊 Datasets & Benchmarks](#-datasets--benchmarks)
- [💻 Reproducible Baselines](#-reproducible-baselines--evidence-graded)
- [🏆 Leaderboard](#-leaderboard)
- [✅ Two Strict Gates](#-two-strict-gates)
- [🤝 Contributing](#-contributing)

---

## 🔎 Curation and Evidence Notes

- The bibliography is curated and non-exhaustive; inclusion is not a citation
  ranking, endorsement, or claim of historical priority.
- Publication metadata was checked through **2026-07-18** using DOI, publisher,
  Crossref, or official preprint records. DOI-linked versions of record are
  preferred over search-result and duplicate preprint links.
- Standards status was checked on the same date using official 3GPP, IEEE,
  ITU-R, and ETSI records. A study, report, draft, specification, and published
  standard are not interchangeable maturity levels.
- “First,” “latest,” “state of the art,” performance, deployment, and
  standards-compliance claims require a primary source and an explicit
  comparison boundary.
- Passing code tests establishes only the properties exercised by those tests;
  it does not by itself establish numerical reproduction of a paper.
- The [audit report](AUDIT_REPORT.md) records corrections, remaining limits,
  and the two independent acceptance gates.

---

## 🧬 The Evolution of ISAC

Our survey organizes ISAC research along **five evolutionary axes**.

> **Reading note.** The arrows below are conceptual coverage maps, not claims of
> technical priority, linear replacement, or universal performance improvement.
> Dates in the detailed timelines are publication or standards-record dates for
> representative, source-verified entries—not dates of first invention.

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                         🧬 THE EVOLUTION OF ISAC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📡 SPECTRUM / PHYSICAL DOMAIN
   RF waveforms ━━━━━ arrays · RIS · near field ━━━━━ optical fibre · wireless · illumination

🌐 NETWORK
   Link / cell design ━━━━━ multistatic · multi-BS ━━━━━ D-MIMO · cell-free · aerial · space

🧠 SENSING
   Signal-level / single-modal ━━━━━ learning-assisted ━━━━━ multimodal · task-oriented

🔒 SECURITY
   Communication confidentiality ━━━━━ sensing protection · privacy ━━━━━ network · surface threats

📋 STANDARDIZATION
   Research ━━━━━ pre-standardization ━━━━━ study / report / draft ━━━━━ specification / standard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              Representative verified records through 2026—not origin dates
```

| Axis | Evolution lens | Source-verified topics |
|---|---|---|
| 📡 **Spectrum / physical domain** | RF-domain designs → optical-domain systems | OTFS, single-carrier, FMCW, arrays, RIS, near field, optical fibre, optical wireless, illumination |
| 🌐 **Network** | Link/cell designs → cooperative and distributed networks | Cellular multistatic sensing, multi-BS cooperation, D-MIMO, cell-free MIMO, UAV, satellite |
| 🧠 **Sensing** | Signal-level or single-modal processing → multimodal and task-oriented systems | Deep learning, model-driven learning, multimodal integration, cooperative motion recognition |
| 🔒 **Security** | Communication confidentiality → joint sensing, privacy, and adversarial-surface models | Malicious targets, target eavesdropping, artificial noise, sensing eavesdroppers, unauthorized RIS |
| 📋 **Standardization** | Research and pre-standardization → differentiated reports, drafts, specifications, and standards | 3GPP Release 19/20 records, IEEE 802.11bf, IEEE P802.15.4ab, ITU-R M.2160-0, ETSI ISG ISAC |

---

## 📅 Evolution Timelines by Subfield

### 📡 Axis 1: RF → Optical ISAC

```text
2020 ─── OTFS and cyclic-prefixed single-carrier designs
  │      Index-modulated dual-function signaling
  │
2021 ─── Signal-processing overview and adaptive automotive waveforms
  │      FMCW/index-modulated joint radar-communications
  │
2022 ─── CRB-oriented multi-antenna waveform and beamforming design
  │
2023 ─── Optical-fibre sensing and data transmission
  │      RIS-aided joint beamforming and reflection design
  │
2024 ─── Near-field ISAC and optical-ISAC architecture records
  │      Optical-wireless experiment and RIS SNR/CRB design
  │
2025 ─── Learning-based near-field beam training for XL-MIMO
  │      Space-time-coding metasurface architecture
  │
2026 ─── Rotatable-array beamforming and optical transmission networks
         Long-reach fibre and laser-headlamp ISAC records
```

Representative records: [waveform and signal design](paper/waveform.md),
[antennas, arrays, and surfaces](paper/antenna.md), and
[optical ISAC](paper/optical.md). Performance values are not compared across
incompatible carrier, bandwidth, aperture, power, channel, detector, or metric
definitions.

### 🌐 Axis 2: Single-Cell → Multi-Cell and Distributed Networks

```text
2023 ─── UAV-enabled joint maneuver and beamforming design
  │
2024 ─── Cellular multistatic sensing coverage
  │      Multi-BS cooperative sensing and stochastic-geometry analysis
  │
2025 ─── Distributed-MIMO communication, localization, and sensing
  │
2026 ─── Cell-free massive-MIMO ISAC survey
         UAV-swarm sensing/communication/control and LEO-satellite design
```

Representative records: [networked and cooperative ISAC](paper/network.md) and
[aerial and space applications](paper/application.md). Results remain conditional
on geometry, synchronization, fronthaul, interference, blockage, channel
knowledge, and the evidence level of each cited study.

### 🧠 Axis 3: Single-Modal → Multi-Modal and Task-Oriented Sensing

```text
2024 ─── Survey of intelligent multimodal sensing-communication integration
  │      Deep-learning uplink design and model-driven passive ISAC
  │
2025 ─── Deep-learning-based near-field beam training
  │
2026 ─── Large-model architecture for multimodal ISAC
         Task-oriented multidevice cooperative motion recognition
```

Representative records: [AI and machine learning for ISAC](paper/ai_ml.md).
The [dataset catalogue](datasets/README.md) separately records WiFi-CSI,
multimodal RF/vision, and automotive/robotic radar resources, but does not assign
unsupported field-origin dates. Model conclusions remain conditional on the
reported data, split, baseline, channel, sensing condition, compute budget, and
random seeds.

### 🔒 Axis 4: Security & Privacy

```text
2021 ─── Malicious-target model combining radar, communications, and jamming
  │
2022 ─── Target-eavesdropping model exploiting interference for resilience
  │
2024 ─── Artificial-noise design protecting the sensing functionality
  │      Cell-free information/sensing eavesdroppers and privacy-risk survey
  │
2026 ─── Adversarial or unauthorized-RIS threat model
```

Representative records: [security, privacy, and resilience](paper/security.md).
Each result applies only to its explicit protected assets, adversary model, trust
boundary, channel assumptions, and evaluated attacks; inclusion is not a
universal security or privacy guarantee.

### 📋 Axis 5: Standardization

```text
2021 ─── IEEE P802.15.4ab PAR approved
  │      Active draft project at the 2026-07-18 snapshot
  │
2023 ─── ITU-R M.2160-0 approved and in force as the IMT-2030 framework
  │      ETSI launched ISG ISAC for pre-standardization work
  │
2025 ─── ETSI GR ISC 001 published with 18 advanced use cases
  │      IEEE 802.11bf-2025 approved and published as an active standard
  │
2026 ─── ETSI GR ISC 003 and GR ISC 004 published
  │      3GPP Release-20 architecture and NR studies under change control
  │      3GPP TS 23.137 remains a 0.x Stage-2 draft
  │
As of ─── 3GPP TS 22.137 (Release 19, Stage 1) is under change control
2026-07-18    Study reports and drafts are not described as completed
              or deployed Release-20 air-interface standards
```

See the [official-record standards snapshot](paper/standardization.md) for
document type, version, maturity, verification date, and source links. Standards
status can change and must be rechecked before downstream reliance.

---

## ⭐ Featured Papers

The original **six-section layout and all 44 entries are retained**. This
53-entry view adds nine DOI-linked records: four established surveys spanning
fundamental limits, mobile networks, signal design, and cell-free networks,
plus five selected recent papers. Twenty-seven entries are also present in the
machine-frozen topical catalogue; 24 additional formal records were restored
after title-level Crossref/DOI verification, one formal early-access record is
kept outside the volume-complete catalogue, and one item is explicitly
identified as a preprint.

Within each section, a version of record is preferred to a preprint; ordering
then considers direct ISAC relevance, breadth or technical contribution, venue
standing in the corresponding subfield, publication completeness, and recency
as a tie-breaker. A targeted review of the maintainer's recent publications
used these same criteria: no author receives a dedicated section or special
styling, and overlapping, preprint-only, editorial, or out-of-scope work is not
promoted solely because of authorship. Dynamic citation counts are not used.
The sections cover different research questions, so position is not a universal
cross-topic quality ranking.

### 🔥 Landmark Surveys

| Paper | Authors | Venue | Year |
|---|---|---|---|
| [Integrated Sensing and Communications Over the Years: An Evolution Perspective](https://doi.org/10.1109/COMST.2026.3655674) | Di Zhang; Yuanhao Cui; Xiaowen Cao; Nanchi Su; Yi Gong; Fan Liu; Weijie Yuan; Xiaojun Jing; J. Andrew Zhang; Jie Xu; Christos Masouros; Dusit Niyato; Marco Di Renzo | IEEE COMST | 2026 |
| [A Survey on Fundamental Limits of Integrated Sensing and Communication](https://doi.org/10.1109/COMST.2022.3149272) | An Liu; Zhe Huang; Min Li; Yubo Wan; Wenrui Li; Tony Xiao Han; Chenchen Liu; Rui Du; Danny Kai Pin Tan; Jianmin Lu; Yuan Shen; Fabiola Colone; Kevin Chetty | IEEE COMST | 2022 |
| [Sensing With Communication Signals: From Information Theory to Signal Processing](https://doi.org/10.1109/JSAC.2025.3614025) | Fan Liu; Ya-Feng Liu; Yuanhao Cui; Christos Masouros; Jie Xu; Tony Xiao Han; Stefano Buzzi; Yonina C. Eldar; Shi Jin | IEEE JSAC | 2026 |
| [Integrated Sensing and Communications: Toward Dual-Functional Wireless Networks for 6G and Beyond](https://doi.org/10.1109/JSAC.2022.3156632) | Fan Liu; Yuanhao Cui; Christos Masouros; Jie Xu; Tony Xiao Han; Yonina C. Eldar; Stefano Buzzi | IEEE JSAC | 2022 |
| [Enabling Joint Communication and Radar Sensing in Mobile Networks—A Survey](https://doi.org/10.1109/COMST.2021.3122519) | J. Andrew Zhang; Md. Lushanur Rahman; Kai Wu; Xiaojing Huang; Y. Jay Guo; Shanzhi Chen; Jinhong Yuan | IEEE COMST | 2022 |
| [A Survey on Wi-Fi Sensing Generalizability: Taxonomy, Techniques, Datasets, and Future Research Prospects](https://doi.org/10.1109/COMST.2026.3670854) | Fei Wang; Tingting Zhang; Wei Xi; Han Ding; Ge Wang; Di Zhang; Yuanhao Cui; Fan Liu; Jinsong Han; Jie Xu; Tony Xiao Han | IEEE COMST | 2026 |
| [Seventy Years of Radar and Communications: The road from separation to integration](https://doi.org/10.1109/MSP.2023.3272881) | Fan Liu; Le Zheng; Yuanhao Cui; Christos Masouros; Athina P. Petropulu; Hugh Griffiths; Yonina C. Eldar | IEEE SPM | 2023 |
| [An Overview of Signal Processing Techniques for Joint Communication and Radar Sensing](https://doi.org/10.1109/JSTSP.2021.3113120) | J. Andrew Zhang; Fan Liu; Christos Masouros; Robert W. Heath; Zhiyong Feng; Le Zheng; Athina Petropulu | IEEE JSTSP | 2021 |
| [Joint Radar and Communication Design: Applications, State-of-the-Art, and the Road Ahead](https://doi.org/10.1109/TCOMM.2020.2973976) | Fan Liu; Christos Masouros; Athina P. Petropulu; Hugh Griffiths; Lajos Hanzo | IEEE TCOM | 2020 |
| [Integrated Sensing and Communication Signals Toward 5G-A and 6G: A Survey](https://doi.org/10.1109/JIOT.2023.3235618) | Zhiqing Wei; Hanyang Qu; Yuan Wang; Xin Yuan; Huici Wu; Ying Du; Kaifeng Han; Ning Zhang; Zhiyong Feng | IEEE IoT-J | 2023 |
| [Toward 6G Networks: A Survey on Integrated Sensing and Communication in Cell-Free Massive MIMO](https://doi.org/10.1109/JIOT.2026.3693228) | Manzoor Ahmed; Ali Arshad Nasir; Mudassir Masood; Kamran Ali Memon; Khurram Karim Qureshi; Feroz Khan; Touseef Hussain; Wali Ullah Khan; Fang Xu; Zhu Han | IEEE IoT-J | 2026 |
| [Integrated Sensing and Communication: Towards Multifunctional Perceptive Network](https://arxiv.org/abs/2510.14358) | Yuanhao Cui; Jiali Nie; Fan Liu; Weijie Yuan; Zhiyong Feng; Xiaojun Jing; Yulin Liu; Jie Xu; Christos Masouros; Shuguang Cui | arXiv preprint | 2025 |

### 📡 RF ISAC — Antenna & Waveform

| Paper | Venue | Year | Key contribution or scope |
|---|---|---|---|
| [Integrated sensing and communication based on space-time-coding metasurfaces](https://doi.org/10.1038/S41467-025-57137-6) | Nature Communications | 2025 | Space-time-coding metasurface architecture |
| [MIMO-OFDM ISAC Waveform Design for Range-Doppler Sidelobe Suppression](https://doi.org/10.1109/TWC.2024.3503605) | IEEE TWC | 2025 | Range-Doppler sidelobe-aware waveform design |
| [On the Effectiveness of OTFS for Joint Radar Parameter Estimation and Communication](https://doi.org/10.1109/TWC.2020.2998583) | IEEE TWC | 2020 | Delay-Doppler-domain parameter estimation and communication |
| [RIS-Aided Integrated Sensing and Communication: Joint Beamforming and Reflection Design](https://doi.org/10.1109/TVT.2023.3248657) | IEEE TVT | 2023 | Joint active beamforming and RIS reflection design |
| [Multi-user ISAC through Stacked Intelligent Metasurfaces: New Algorithms and Experiments](https://doi.org/10.1109/GLOBECOM52923.2024.10901440) | IEEE GLOBECOM | 2024 | Stacked-metasurface multi-user algorithms and experiments |
| [Fixed and Movable Antenna Technology for 6G Integrated Sensing and Communication](https://doi.org/10.16798/j.issn.1003-0530.2024.08.001) | Journal of Signal Processing | 2024 | Fixed, distributed, and movable antenna architectures for ISAC |
| [Sparse MIMO for ISAC: New Opportunities and Challenges](https://doi.org/10.1109/MWC.001.2400201) | IEEE Wireless Communications | 2025 | Sparse-array opportunities and design challenges for ISAC |
| [From OTFS to DD-ISAC: Integrating Sensing and Communications in the Delay Doppler Domain](https://doi.org/10.1109/MWC.018.2300607) | IEEE Wireless Communications | 2024 | Delay-Doppler-domain ISAC overview |
| [Smart Radio Environments Empowered by Reconfigurable Intelligent Surfaces: How It Works, State of Research, and The Road Ahead](https://doi.org/10.1109/JSAC.2020.3007211) | IEEE JSAC | 2020 | Enabling RIS background; not itself an ISAC-specific result |

### 🔦 Optical ISAC

| Paper | Venue | Year | Key contribution or scope |
|---|---|---|---|
| [Integrated sensing and communication in an optical fibre](https://doi.org/10.1038/S41377-022-01067-1) | Light: Science & Applications | 2023 | Optical-fibre sensing and data transmission |
| [Multi-Channel Photonic THz-ISAC System Based on Integrated LFM-QAM Waveform](https://doi.org/10.1109/JLT.2024.3392282) | IEEE Journal of Lightwave Technology | 2024 | Multi-channel photonic THz ISAC with an integrated waveform |
| [Photonic-Based Flexible Integrated Sensing and Communication With Multiple Targets Detection Capability for W-Band Fiber-Wireless Network](https://doi.org/10.1109/TMTT.2024.3355936) | IEEE TMTT | 2024 | W-band fibre-wireless ISAC with multiple-target detection |
| [Photonics-aided integrated sensing and communications in mmW bands based on a DC-offset QPSK-encoded LFMCW](https://doi.org/10.1364/OE.474055) | Optics Express | 2022 | Photonics-aided mmWave ISAC using a coded LFMCW waveform |
| [Optical Integrated Sensing and Communication: Architectures, Potentials and Challenges](https://doi.org/10.1109/IOTM.001.2300196) | IEEE Internet of Things Magazine | 2024 | Optical ISAC architectures, potentials, and challenges |
| [W-Band Photonics-aided ISAC Wireless System Sharing OFDM Signal as Communication and Sensing](https://doi.org/10.1364/OFC.2024.Tu3K.4) | OFC | 2024 | Photonics-aided W-band communication and sensing experiment |

### 🌐 Network Architecture

| Paper | Venue | Year | Key contribution or scope |
|---|---|---|---|
| [Co-Design of Sensing, Communications, and Control for Low-Altitude Wireless Networks](https://doi.org/10.1109/TMC.2025.3581616) | IEEE TMC | 2025 | Joint sensing, communication, and control co-design for low-altitude wireless networks |
| [Precoding for Multi-Cell ISAC: From Coordinated Beamforming to Coordinated Multipoint and Bi-Static Sensing](https://doi.org/10.1109/TWC.2024.3417713) | IEEE TWC | 2024 | Multi-cell coordinated precoding and bi-static sensing |
| [Toward Seamless Sensing Coverage for Cellular Multi-Static Integrated Sensing and Communication](https://doi.org/10.1109/TWC.2023.3325849) | IEEE TWC | 2024 | Cellular multistatic sensing coverage |
| [Joint Maneuver and Beamforming Design for UAV-Enabled Integrated Sensing and Communication](https://doi.org/10.1109/TWC.2022.3211533) | IEEE TWC | 2023 | Coupled UAV trajectory and beamforming design |
| [Cooperative ISAC Networks: Opportunities and Challenges](https://doi.org/10.1109/MWC.008.2400151) | IEEE Wireless Communications | 2025 | Cooperative-network ISAC opportunities and challenges |
| [Simultaneous Sensing Data Acquisition and Sharing in Low-Altitude Wireless Networks: Fundamental Limits and Signaling Design](https://doi.org/10.1109/JSTSP.2026.3696543) | IEEE JSTSP (Early Access) | 2026 | Formal DOI record; fundamental limits and signaling for sensing-data acquisition and sharing |
| [Interference Mitigation for Network-Level ISAC: An Optimization Perspective](https://doi.org/10.1109/MCOM.001.2300674) | IEEE Communications Magazine | 2024 | Network-level interference mitigation |
| [Deep Cooperation in ISAC System: Resource, Node and Infrastructure Perspectives](https://doi.org/10.1109/IOTM.001.2400042) | IEEE Internet of Things Magazine | 2024 | Cooperation across resources, nodes, and infrastructure |
| [UAV Meets Integrated Sensing and Communication: Challenges and Future Directions](https://doi.org/10.1109/MCOM.008.2200510) | IEEE Communications Magazine | 2023 | UAV-enabled ISAC challenges and research directions |
| [Air-Ground Integrated Sensing and Communications: Opportunities and Challenges](https://doi.org/10.1109/MCOM.007.2200459) | IEEE Communications Magazine | 2023 | Air-ground ISAC opportunities and challenges |

### 🧠 AI/ML for ISAC

| Paper | Venue | Year | Key contribution or scope |
|---|---|---|---|
| [Intelligent Multi-Modal Sensing-Communication Integration: Synesthesia of Machines](https://doi.org/10.1109/COMST.2023.3336917) | IEEE COMST | 2024 | Multimodal sensing-communication integration survey |
| [Joint Sensing, Communication, and Computation for Vertical Federated Edge Learning in Edge Perception Networks](https://doi.org/10.1109/TMC.2026.3674960) | IEEE TMC | 2026 | Vertical federated edge learning across sensing, communication, and computation |
| [Toward Ambient Intelligence: Federated Edge Learning With Task-Oriented Sensing, Computation, and Communication Integration](https://doi.org/10.1109/JSTSP.2022.3226836) | IEEE JSTSP | 2023 | Task-oriented federated edge learning across sensing, computation, and communication |
| [ISAC-NET: Model-Driven Deep Learning for Integrated Passive Sensing and Communication](https://doi.org/10.1109/TCOMM.2024.3375818) | IEEE TCOM | 2024 | Model-driven learning for passive sensing and communication |
| [AI-Enhanced Integrated Sensing and Communications: Advancements, Challenges, and Prospects](https://doi.org/10.1109/MCOM.001.2300724) | IEEE Communications Magazine | 2024 | Survey-style perspective on AI-enhanced ISAC |
| [Sensing-Assisted High Reliable Communication: A Transformer-Based Beamforming Approach](https://doi.org/10.1109/JSTSP.2024.3405859) | IEEE JSTSP | 2024 | Transformer-based beamforming using sensing information |
| [Edge Perception: Intelligent Wireless Sensing at Network Edge](https://doi.org/10.1109/MCOM.001.2300660) | IEEE Communications Magazine | 2025 | Edge-oriented wireless sensing framework |
| [AI-Driven Integration of Sensing and Communication in the 6G Era](https://doi.org/10.1109/MNET.2023.3326064) | IEEE Network | 2024 | AI-driven ISAC integration perspective |
| [Deep CLSTM for Predictive Beamforming in Integrated Sensing and Communication-Enabled Vehicular Networks](https://doi.org/10.23919/JCIN.2022.9906941) | JCIN | 2022 | CLSTM-based predictive beamforming for vehicular ISAC |
| [Penetrative AI: Making LLMs Comprehend the Physical World](https://doi.org/10.1145/3638550.3641130) | ACM HotMobile | 2024 | LLM-oriented physical sensing research adjacent to ISAC |

### 🔒 Security

| Paper | Venue | Year | Key contribution or scope |
|---|---|---|---|
| [Multi-Antenna Signal Masking and Round-Trip Transmission for Privacy-Preserving Wireless Sensing](https://doi.org/10.1109/TIFS.2024.3414185) | IEEE TIFS | 2024 | Multi-antenna signal masking for sensing privacy |
| [Secure Radar-Communication Systems With Malicious Targets: Integrating Radar, Communications and Jamming Functionalities](https://doi.org/10.1109/TWC.2020.3023164) | IEEE TWC | 2021 | Malicious-target and joint-jamming threat model |
| [Secure Dual-Functional Radar-Communication Transmission: Exploiting Interference for Resilience Against Target Eavesdropping](https://doi.org/10.1109/TWC.2022.3156893) | IEEE TWC | 2022 | Target-eavesdropping and interference design |
| [Securing the Sensing Functionality in ISAC Networks: An Artificial Noise Design](https://doi.org/10.1109/TVT.2024.3422036) | IEEE TVT | 2024 | Artificial-noise design for sensing security |
| [Privacy and Security in Ubiquitous Integrated Sensing and Communication: Threats, Challenges and Future Directions](https://doi.org/10.1109/IOTM.001.2300180) | IEEE Internet of Things Magazine | 2024 | Privacy and security threat survey |
| [PriSense: Privacy-Preserving Wireless Sensing for Vital Signs Monitoring](https://doi.org/10.1109/LWC.2024.3434470) | IEEE WCL | 2024 | Privacy-preserving vital-sign sensing |

---

## 📚 All Papers by Topic

The machine-frozen topical catalogue contains **66 publication rows covering 54
unique DOI-linked works**. A work may appear in more than one topic, so row counts
must not be added and presented as a unique-paper count. Standards are counted
separately because they are not research papers.

| Category | File | Curated rows | Description |
|---|---|---:|---|
| 📖 Surveys & Tutorials | [paper/surveys.md](paper/surveys.md) | 10 | General surveys, historical synthesis, limits, signals, and mobile networks |
| 📐 Theory & Bounds | [paper/theory.md](paper/theory.md) | 5 | Information, estimation, detection, CRB, and tradeoffs |
| 📡 Waveform Design | [paper/waveform.md](paper/waveform.md) | 8 | OTFS, single-carrier, FMCW, index modulation, and beamforming |
| 📡 Antenna Technology | [paper/antenna.md](paper/antenna.md) | 6 | Near-field/XL-MIMO arrays, RIS, and metasurfaces |
| 🔦 Optical ISAC | [paper/optical.md](paper/optical.md) | 6 | Optical fibre, optical wireless, photonic, and illumination systems |
| 🌐 Network Architecture | [paper/network.md](paper/network.md) | 8 | Multistatic, multi-BS, D-MIMO, cell-free, UAV, and satellite systems |
| 🧠 AI/ML for ISAC | [paper/ai_ml.md](paper/ai_ml.md) | 9 | Deep learning, model-driven learning, multimodal, and task-oriented systems |
| 🔒 Security & Privacy | [paper/security.md](paper/security.md) | 6 | Threat models, eavesdropping, artificial noise, privacy, and adversarial surfaces |
| 🏗️ Applications | [paper/application.md](paper/application.md) | 8 | Aerial, space, optical, and fibre contexts |
| 📋 Standardization | [paper/standardization.md](paper/standardization.md) | 12 official records | 3GPP, IEEE, ITU-R, and ETSI status records |

---

## 📋 Standards Snapshot

Status below was checked against official records on **2026-07-18**.

| Item | Correct status at verification | Primary record |
|---|---|---|
| IEEE 802.11bf | IEEE 802.11bf-2025 is a published active WLAN sensing amendment; approved 2025-05-28 and published 2025-09-26 | [IEEE SA](https://standards.ieee.org/ieee/802.11bf/11574/) |
| 3GPP TS 22.137 | Release-19 Stage-1 ISAC specification under change control; portal version 19.1.0 at verification | [3GPP portal](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=4198) |
| 3GPP Release 20 | Architecture and NR studies plus draft Stage-2 work; not described here as a completed air-interface standard | [3GPP status report](https://www.3gpp.org/dynareport?code=status-report.htm) |
| ITU-R M.2160-0 | In-force IMT-2030 framework recommendation; integration of sensing and communication is among its usage scenarios and capabilities | [ITU-R](https://www.itu.int/rec/R-REC-M.2160-0-202311-I/en) |
| ETSI ISG ISAC | Pre-standardization group; its GR documents are reports rather than a deployed air-interface standard | [ETSI](https://www.etsi.org/technical-groups/isac/) |
| IEEE P802.15.4ab | Active draft project/PAR at verification; not a published standard | [IEEE SA](https://standards.ieee.org/ieee/802.15.4ab/10694/) |

See [paper/standardization.md](paper/standardization.md) for all 12 records,
document versions, maturity labels, dates, and evidence boundaries.

---

## 🧰 Open-Source and Research Tools

> Official project pages were checked on 2026-07-18. Hardware, firmware, driver,
> operating-system, protocol, bandwidth, and antenna compatibility can vary by
> release; verify the exact project documentation before purchase or deployment.

### WiFi channel and beamforming measurements

| Tool | Primary purpose | Official project |
|---|---|---|
| PicoScenes | WiFi channel-state and physical-layer measurement platform | [PicoScenes](https://ps.zpj.io/) |
| Nexmon CSI | CSI extraction for supported Broadcom WiFi chipsets | [Nexmon CSI](https://github.com/seemoo-lab/nexmon_csi) |
| Linux 802.11n CSI Tool | CSI measurement with supported Intel 5300 hardware and drivers | [Intel CSI Tool](https://dhalperi.github.io/linux-80211n-csitool/) |
| Atheros CSI Tool | CSI extraction for supported Atheros hardware | [Atheros CSI Tool](https://github.com/xieyaxiongfly/Atheros-CSI-Tool) |
| ZTE WiFi Sensing | WiFi-sensing software and examples for hardware documented by the project | [ZTE WiFi Sensing](https://github.com/WiFiZTE2025/ZTE_WiFi_Sensing) |
| Wi-ESP | WiFi measurement tooling for devices documented by the project | [Wi-ESP](https://github.com/wrlab/Wi-ESP) |
| BFM-Tool | WiFi beamforming-feedback collection and analysis | [BFM-Tool](https://github.com/Enze-Yi/BFM-tool) |

### Simulation and numerical computing

| Project | Access model | Primary purpose | Official project |
|---|---|---|---|
| SciPy | Open source | Numerical and signal-processing routines for Python | [SciPy](https://scipy.org/) |
| MATLAB Radar Toolbox | Commercial | Radar modeling, simulation, and signal processing | [Radar Toolbox](https://www.mathworks.com/products/radar.html) |
| Wireless InSite | Commercial | Radio-propagation and channel modeling | [Wireless InSite](https://www.remcom.com/wireless-insite-propagation-software) |

> See [tools/README.md](tools/README.md) for the complete directory and its
> verification boundary. Inclusion is not a performance endorsement, security
> audit, or compatibility guarantee. External tools retain their own licenses
> and usage terms.

---

## 🔗 Related Projects

| Project | Relationship to this index | Link |
|---|---|---|
| SDP — Sensing Data Protocol | Protocol-level abstraction and benchmark repository for scalable wireless sensing | [Repository](https://github.com/yuanhao-cui/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing) |
| Must-Reading-on-ISAC | Related collection of ISAC papers, research resources, and open-source code | [Repository](https://github.com/yuanhao-cui/Must-Reading-on-ISAC) |
| CRB-ISAC beamforming code | External companion implementation retained for navigation; not one of the seven locally certified baselines | [Repository](https://github.com/yuanhao-cui/crb-isac-tap-2022) |
| VFEEL code | External companion implementation retained for navigation; not one of the seven locally certified baselines | [Repository](https://github.com/yuanhao-cui/VFEEL-Joint-Sensing-Communication-and-Computation-for-Vertical-Federated-Edge-Learning) |

> External companion repositories are not covered by this repository's baseline
> manifests, test counts, evidence levels, or license. Assess their code, data,
> results, and terms independently.

---

## 📊 Datasets & Benchmarks

### RF and multimodal human sensing

| Dataset | Modalities or data scope | Source-supported scale recorded here | Task scope | Official source |
|---|---|---|---|---|
| XRF55 | WiFi, RFID, mmWave, and Kinect | 42.9K synchronized RF samples; 55 classes; 39 subjects; four scenes | Multimodal human sensing | [XRF55](https://aiotgroup.github.io/XRF55/) |
| Widar 3.0 | WiFi CSI | 16 users; 15 gestures; 15 locations; five orientations; three environments | Gesture recognition | [Widar 3.0](https://tns.thss.tsinghua.edu.cn/widar3.0/) |
| MM-Fi | Five synchronized modalities | More than 320K synchronized frames; 40 subjects; 25 action categories | Multimodal human sensing | [MM-Fi](https://ntu-aiot-lab.github.io/mm-fi) |
| SignFi | WiFi CSI | — | Sign-language recognition | [SignFi](https://github.com/yongsen/SignFi) |
| NTU-Fi | WiFi CSI data and benchmark code | — | Human-activity recognition | [NTU-Fi benchmark](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark) |
| WiAR | WiFi-based sensing data and resources | — | Activity recognition | [WiAR](https://github.com/linteresa/WiAR) |
| OPERAnet | Radio-frequency and vision-based sensors | — | Multimodal activity recognition | [OPERAnet](https://springernature.figshare.com/collections/A_Comprehensive_Multimodal_Activity_Recognition_Dataset_Acquired_from_Radio_Frequency_and_Vision-Based_Sensors/5551209) |

### Automotive and robotic radar datasets

| Dataset | Modalities or data scope | Task scope | Official source |
|---|---|---|---|
| RadarScenes | Automotive radar sequences | Object detection and tracking | [RadarScenes](https://radar-scenes.com/) |
| Oxford Radar RobotCar | Radar extension to the Oxford RobotCar dataset | Autonomous-driving and robotic radar research | [Oxford Radar RobotCar](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/) |
| RADIATE | Radar, LiDAR, and camera | Adverse-weather road perception | [RADIATE](https://pro.hw.ac.uk/radiate/) |
| nuScenes | Radar, LiDAR, cameras, and other vehicle sensors | Multisensor autonomous-driving perception | [nuScenes](https://www.nuscenes.org/) |

> A dash means that this index does not make a version-specific scale claim;
> consult the official source. Before comparing methods, record the exact release,
> license and consent conditions, preprocessing, calibration, synchronization,
> split, seeds, metrics, and evaluation code. See
> [datasets/README.md](datasets/README.md) and the
> [benchmark protocol](benchmark/README.md).

---

## 💻 Reproducible Baselines — Evidence-Graded

These are runnable research and educational reference packages, not complete
end-user applications. Completing a documented command and receiving a passing
certificate establishes only the declared equations, numerical invariants, and
software behavior. It does **not** by itself establish reproduction of a paper's
figures, experiments, hardware results, or complete algorithm.

| Evidence level | Manifest value | Required interpretation |
|---|---|---|
| Exact reproduction | `exact-reproduction` | Original algorithm, data or simulator, parameters, deterministic command, and machine-checked comparison to specified paper values or figures with declared tolerances |
| Equation-level reference | `equation-level` | Explicitly bounded analytical slice with independent numerical or analytic oracles and declared tolerances; no full-paper parity |
| Research reference | `research-reference` | Cited method or algorithmic structure with targeted tests, but no complete original-condition numerical comparison |
| Educational surrogate | `educational-surrogate` | Simplified, substituted, fallback, or synthetic model for study and software testing |

| # | Baseline | Publication or literature anchor | Current evidence level | Evidence boundary | Documentation |
|---:|---|---|---|---|---|
| 1 | CSI-ratio Doppler estimation | Li et al., *IEEE Sensors Journal*, 2022 ([DOI](https://doi.org/10.1109/JSEN.2022.3208272)) | Educational surrogate | Synthetic estimators; no disclosed paper subset or hardware-data replay | [README](code/baselines/csi_ratio_doppler_estimation/) · [Manifest](code/baselines/csi_ratio_doppler_estimation/reproducibility.yaml) |
| 2 | Capacity–distortion | Xiong et al., *IEEE Transactions on Information Theory*, 2023 ([DOI](https://doi.org/10.1109/TIT.2023.3284449)) | Educational surrogate | Tractable numerical objectives and examples; not the paper's complete numerical pipeline | [README](code/baselines/isac_capacity_distortion/) · [Manifest](code/baselines/isac_capacity_distortion/reproducibility.yaml) |
| 3 | Energy-efficient beamforming | Zou et al., *IEEE Transactions on Communications*, 2024 ([DOI](https://doi.org/10.1109/TCOMM.2024.3369696)) | Equation-level reference | Single-user fixed-direction slice checked against grid, explicit-FIM, and finite-difference oracles; no figure parity | [README](code/baselines/isac_energy_efficient_beamforming/) · [Manifest](code/baselines/isac_energy_efficient_beamforming/reproducibility.yaml) |
| 4 | ISAC resource allocation | Dong et al., *IEEE Transactions on Wireless Communications*, 2023 ([DOI](https://doi.org/10.1109/TWC.2022.3219463)) | Educational surrogate | Proxy allocation/QoS solver and synthetic scenarios; no accepted paper-value comparison | [README](code/baselines/isac_resource_allocation/) · [Manifest](code/baselines/isac_resource_allocation/reproducibility.yaml) |
| 5 | OFDM ambiguity function | Survey-inspired waveform-analysis utility | Educational surrogate | Standalone waveform illustration; resolution and sidelobes depend on parameters and windowing | [README](code/baselines/ofdm_ambiguity_function/) · [Manifest](code/baselines/ofdm_ambiguity_function/reproducibility.yaml) |
| 6 | RIS-ISAC beamforming | R. Liu et al., *IEEE Transactions on Wireless Communications*, 2024 ([DOI](https://doi.org/10.1109/TWC.2023.3341429)) | Educational surrogate | Local SNR-feasibility model; the paper's CRB, full algorithm, and figures are out of scope | [README](code/baselines/ris_isac_beamforming/) · [Manifest](code/baselines/ris_isac_beamforming/reproducibility.yaml) |
| 7 | XL-MIMO beam training | Nie et al., *IEEE Transactions on Mobile Computing*, 2025 ([DOI](https://doi.org/10.1109/TMC.2024.3462960)) | Educational surrogate | Synthetic default data and simplified evaluation; no original-data result replay | [README](code/baselines/xl_mimo_beam_training/) · [Manifest](code/baselines/xl_mimo_beam_training/reproducibility.yaml) |

No local baseline currently qualifies as an exact reproduction, and every
manifest declares `paper_figure_parity: false`. Repository-generated figures are
examples unless a paper-value comparison artifact with declared tolerances is
accepted.

> **Audited snapshot, 2026-07-18:** the homepage-preservation revision passed
> 666/666 strict tests in the locked Python 3.12 environment; all seven
> executable certificates passed their 99/99 declared checks; and the covered
> run measured 85.05% aggregate statement coverage. The protected workflow
> repeats the current tree on Python 3.10, 3.11, and 3.12 before merge. These
> are repository-level software results, not seven paper-reproduction
> certificates. See [code/README.md](code/README.md) and the
> [audit report](AUDIT_REPORT.md).

---

## 🏆 Leaderboard

There is currently **no verified cross-method leaderboard** in this repository.
No benchmark runner or published ranking artifact is presented as if it existed.

A future result is eligible for review only when it records:

| Requirement | Minimum evidence |
|---|---|
| Task and data | Exact task, dataset or simulator version, preprocessing, split, and exclusions |
| Physical model | Channel, waveform, array, power, noise, target, and hardware assumptions |
| Metrics | Definitions, units, aggregation, uncertainty or confidence intervals, and failure handling |
| Environment | Pinned dependencies, deterministic seeds where supported, and exact command |
| Results | Machine-readable artifact tied to a repository commit |
| Comparisons | Competing methods evaluated under the same protocol with primary-source citations |
| Independent verification | A separate rerun within declared numerical tolerances |

BER, communication rate, detection probability, localization error, CRB,
energy efficiency, latency, and Pareto summaries measure different objectives;
they must not be collapsed into one ranking without a declared normalization and
comparison protocol. See [benchmark/README.md](benchmark/README.md).

---

## ✅ Two Strict Gates

Every accepted revision must pass both independent gates; passing one cannot
compensate for failing the other.

| Gate | Required checks |
|---|---|
| **Gate 1 — Software and simulation integrity** | Exact direct dependencies; Python 3.10–3.12 locked matrix; no-cache macOS arm64 installs; warning-free tests and compilation; valid CFF/YAML; aggregate coverage ≥70%; seven machine-readable certificates; deterministic comparison whenever reproduction is claimed |
| **Gate 2 — Scholarly content and link integrity** | Canonical identifiers; correct title/author/venue/year metadata; one identifier–one work consistency; official standards status; explicit evidence boundaries; no unsupported priority, performance, deployment, or reproduction claims; deterministic internal-link scan and independent live-link audit |

The protected `main` branch requires all eight exact GitHub Actions jobs, strict
branch updating, resolved conversations, and linear history. Administrators are
also subject to the rule; force pushes and branch deletion are disabled. The
solo-maintainer policy requires no second-person approval, but it does not bypass
either automated gate. See [AUDIT_REPORT.md](AUDIT_REPORT.md) for exact results
and limits.

---

## 🤝 Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before
opening a pull request. It explains:

- 📖 how to add or correct publications using canonical identifiers and primary evidence;
- 📋 how to record standards and pre-standardization work at the correct maturity;
- 📊 how to add datasets and tools with source, version, license, and compatibility boundaries;
- 💻 how to contribute or upgrade an evidence-graded baseline;
- 🏆 what evidence is required before submitting benchmark results;
- ✅ the scholarly-content and software-integrity checklist used during review.

Strong claims such as “first,” “state of the art,” “optimal,” “deployed,” or
“reproduced” require direct evidence and an explicit comparison boundary.

---

## 📜 License

Repository-authored material is licensed under the
[Creative Commons Attribution-ShareAlike 4.0 International License](LICENSE).
Linked publications, datasets, tools, and external repositories retain their own
licenses and terms; inclusion here does not relicense or endorse them.

---

## 🙏 Acknowledgements

Inspired by
[awesome-wireless-sensing-generalization](https://github.com/airslab2020/awesome-wireless-sensing-generalization),
[Must-Reading-on-ISAC](https://github.com/yuanhao-cui/Must-Reading-on-ISAC),
and the broader wireless-sensing and awesome-lists communities.
