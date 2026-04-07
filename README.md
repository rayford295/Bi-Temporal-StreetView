# Bi-Temporal-StreetView-Damage

**Hyperlocal disaster damage assessment using bi-temporal street-view imagery and pre-trained vision models**

[![CEUS](https://img.shields.io/badge/Journal-CEUS-blue.svg)](https://doi.org/10.1016/j.compenvurbsys.2025.102335)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.compenvurbsys.2025.102335-blue.svg)](https://doi.org/10.1016/j.compenvurbsys.2025.102335)
[![arXiv](https://img.shields.io/badge/arXiv-2504.09066-b31b1b.svg)](https://arxiv.org/abs/2504.09066)
[![Dataset](https://img.shields.io/badge/Dataset-Figshare-blue)](https://doi.org/10.6084/m9.figshare.28801208.v2)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/Rayford295/BiTemporal-StreetView-Damage)

---

## Overview

This repository presents a **bi-temporal street-view image analysis framework** for hyperlocal disaster damage assessment. By fusing pre- and post-disaster imagery through a **dual-channel architecture** with Swin Transformer and ConvNeXt backbones, the framework improves both classification accuracy and spatial interpretability of damage detection.

**Key results:**
- 2,249 labeled street-view image pairs with severity annotations
- Accuracy: 66.14% (post-only) → **77.11%** (bi-temporal fusion)
- Grad-CAM visualization confirms improved attention focus with pre-disaster context

---

## Methodology

<p align="center">
  <img src="https://github.com/rayford295/Bi-Temporal-StreetView/blob/main/images/dual_channel.drawio%20(2).png" width="700"/>
</p>

**Pipeline:** paired image normalization → feature extraction (Swin Transformer / ConvNeXt) → dual-channel fusion → severity classification (*mild / moderate / severe*) → Grad-CAM interpretation.

---

## Study Area

Focused on **Horseshoe Beach, Florida**, severely impacted by **Hurricane Milton (2024)**.

| Study Area | Damage Heatmap |
|:---:|:---:|
| <img src="https://github.com/rayford295/Bi-Temporal-StreetView/blob/main/images/study_area_disaster%20damage_made.png" width="320"/> | <img src="https://github.com/rayford295/Bi-Temporal-StreetView/blob/main/images/heatmap%20all.drawio.png" width="320"/> |

---

## Dataset

Pre- and post-disaster street-view image pairs with georeferenced annotations and severity labels.

- **Figshare:** [10.6084/m9.figshare.28801208.v2](https://doi.org/10.6084/m9.figshare.28801208.v2)
- **Hugging Face:** [Rayford295/BiTemporal-StreetView-Damage](https://huggingface.co/datasets/Rayford295/BiTemporal-StreetView-Damage)

---

## Recognition

Presented at **AAG Annual Meeting 2025** — GISS Specialty Group Paper Competition, **Honorable Mention**
Session: 360, Level 3, Huntington Place — March 25, 2025, 10:10–11:30 AM

> This work forms the foundational framework later extended into [DamageArbiter](https://github.com/rayford295/DamageArbiter) (AAG 2026).

---

## Citation

```bibtex
@article{YANG2025102335,
  title   = {Hyperlocal disaster damage assessment using bi-temporal street-view imagery
             and pre-trained vision models},
  journal = {Computers, Environment and Urban Systems},
  volume  = {121},
  pages   = {102335},
  year    = {2025},
  doi     = {10.1016/j.compenvurbsys.2025.102335},
  author  = {Yifan Yang and Lei Zou and Bing Zhou and Daoyang Li and
             Binbin Lin and Joynal Abedin and Mingzheng Yang}
}
```

---

## Contact

**Yifan Yang** — Department of Geography, Texas A&M University
[yyang295@tamu.edu](mailto:yyang295@tamu.edu) · [rayford295.github.io](https://rayford295.github.io)
