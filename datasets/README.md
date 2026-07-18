# Public Datasets Relevant to ISAC

> Curated starting points, not a claim that every dataset contains a complete communication-and-sensing link. Source pages and the three explicitly reported scale summaries below were checked on 2026-07-18.

## RF and multimodal human sensing

| Dataset | Source-supported content | Official project or repository |
|---|---|---|
| XRF55 | 42.9K synchronized RF samples; 55 classes; 39 subjects; four scenes; WiFi, RFID, mmWave, and Kinect modalities | [XRF55 project](https://aiotgroup.github.io/XRF55/) |
| MM-Fi | More than 320K synchronized frames; five modalities; 40 subjects; 25 action categories | [MM-Fi project](https://ntu-aiot-lab.github.io/mm-fi) |
| Widar 3.0 | WiFi CSI gesture data from 16 users, 15 gestures, 15 locations, five orientations, and three environments | [Widar 3.0 project](https://tns.thss.tsinghua.edu.cn/widar3.0/) |
| SignFi | WiFi CSI sign-language recognition data and code | [SignFi repository](https://github.com/yongsen/SignFi) |
| NTU-Fi | WiFi CSI data and benchmark code for human-activity recognition | [WiFi CSI sensing benchmark repository](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark) |
| WiAR | WiFi-based activity-recognition dataset and resources | [WiAR repository](https://github.com/linteresa/WiAR) |
| OPERAnet | Multimodal activity-recognition collection using radio-frequency and vision-based sensors | [OPERAnet collection](https://springernature.figshare.com/collections/A_Comprehensive_Multimodal_Activity_Recognition_Dataset_Acquired_from_Radio_Frequency_and_Vision-Based_Sensors/5551209) |

## Automotive and robotic radar datasets

| Dataset | Modalities or task scope | Official project |
|---|---|---|
| RadarScenes | Automotive radar object-detection and tracking sequences | [RadarScenes](https://radar-scenes.com/) |
| Oxford Radar RobotCar | Radar extension to the Oxford RobotCar autonomous-driving dataset | [Oxford Radar RobotCar](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/) |
| RADIATE | Radar, LiDAR, and camera data for adverse-weather road perception | [RADIATE](https://pro.hw.ac.uk/radiate/) |
| nuScenes | Multisensor autonomous-driving dataset including radar, LiDAR, and cameras | [nuScenes](https://www.nuscenes.org/) |

## Use and licensing boundary

- Read each source's license, terms of use, consent or privacy conditions, and citation instructions before downloading or redistributing data.
- Verify units, coordinate frames, synchronization, calibration, train/test splits, and subject or scene independence before comparing methods.
- A result is reproducible only when the exact dataset release, preprocessing, split, random seeds, metrics, and evaluation code are recorded.
- Dataset statistics can change across versions; the project source is authoritative if it differs from this snapshot.

Contributions should follow the [curation and evidence policy](../CONTRIBUTING.md).
