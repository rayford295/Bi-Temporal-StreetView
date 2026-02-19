# 🛰️ BiTemporal-StreetView-Damage

**Hyperlocal disaster damage assessment using bi-temporal street-view imagery and pre-trained vision models**

<p align="center">
  <img src="https://github.com/rayford295/BiTemporal-StreetView-Damage/blob/main/images/0204-06.png" alt="Study Area Map" width="600"/>
</p>

---

## 📘 Overview

This repository presents a **bi-temporal street-view image analysis framework** for **hyperlocal disaster damage assessment**.  
By integrating **pre- and post-disaster imagery** through **pre-trained vision and vision–language models**, this approach improves both classification accuracy and interpretability of damage detection.

### 🔍 Key Contributions

- ✅ **Dual-channel architecture** for pre– and post-disaster fusion.  
- 📸 **2,249 labeled street-view image pairs**, annotated with detailed impact levels.  
- 📈 **Performance Gain**: Accuracy increased from 66.14% (post-only) → **77.11% (bi-temporal)**.  
- 🔥 **Grad-CAM visualization** demonstrates improved attention focus using pre-disaster inputs.  
- 🏙️ Supports **fine-grained and rapid damage mapping** for climate-resilient urban planning.

---

## 🧩 Methodology

<p align="center">
  <img src="https://github.com/rayford295/Bi-Temporal-StreetView/blob/main/images/dual_channel.drawio%20(2).png" alt="Dual-Channel Architecture" width="700"/>
</p>

<p align="center"><i>Figure 1: Dual-channel architecture for bi-temporal disaster damage assessment.</i></p>

### Model Pipeline
1. **Pre-processing:** Normalize paired street-view images (pre-/post-disaster).  
2. **Feature Extraction:** Use pre-trained Swin Transformer & ConvNeXt backbones.  
3. **Dual-Channel Fusion:** Fuse embeddings via a feature-fusion head for comparative reasoning.  
4. **Classification:** Predict severity levels (mild, moderate, severe).  
5. **Visualization:** Apply Grad-CAM to interpret key spatial attention areas.

---

## 🌍 Study Area
<p align="center">
  <img src="https://github.com/rayford295/Bi-Temporal-StreetView/blob/main/images/study_area_disaster%20damage_made.png" alt="Study Area Map" width="700"/>
</p>

The study focuses on **Horseshoe Beach, Florida**, which was severely impacted by **Hurricane Milton (2024)**.  
Bi-temporal street-view imagery was collected to model the extent and types of disaster damage across different locations.

<p align="center">
  <img src="https://github.com/rayford295/Bi-Temporal-StreetView/blob/main/images/heatmap%20all.drawio.png" alt="Damage Distribution Heatmap" width="700"/>
</p>

<p align="center"><i>Figure 2: Heatmap visualization of disaster severity distribution across Horseshoe Beach, Florida.</i></p>

---

## 📂 Dataset

You can access the **bi-temporal street-view disaster dataset** via the DOI below:

> 📁 **Yang, Yifan (2025)**.  
> *Perceiving Multidimensional Disaster Damages from Street–View Images Using Visual–Language Models*.  
> figshare. Dataset. [https://doi.org/10.6084/m9.figshare.28801208.v2](https://doi.org/10.6084/m9.figshare.28801208.v2)

or

The primary hosting platform is **Hugging Face Datasets**, which provides a version-controlled repository for convenient access, inspection, and integration with machine learning workflows:

🔗 https://huggingface.co/datasets/Rayford295/BiTemporal-StreetView-Damage


**Dataset Contents:**
- Paired pre-/post-disaster street-view images  
- Location and damage-type annotations  
- Severity labels: *Mild, Moderate, Severe*  
- Sample imagery from **Horseshoe Beach, FL** (Hurricane Milton, 2024)

---
## 🏛️ Conference Presentation (AAG 2025)

This work was **accepted for presentation** at the **2025 Annual Meeting of the American Association of Geographers (AAG 2025)**.
- **Session:** GISS Specialty Group — **Paper Competition II**  
- **Official AAG Link:** https://aag.secure-platform.com/aag2025/organizations/main/gallery/rounds/131/details/83465

**Presentation Schedule:**
- **Date:** 3/25/2025
- **Time:** 10:10 AM - 11:30 AM 
- **Location:** 360, Level 3, Huntington Place

**Paper Title:**  
*Hyperlocal Disaster Damage Assessment Using Bi-Temporal Street-View Imagery and Pre-Trained Image Processing Models*

This presentation represents an **early-stage foundation** of the bi-temporal disaster perception framework, which later evolved into the **DamageArbiter** disagreement-driven arbitration framework presented at AAG 2026.


---
## 🧠 Paper Reference

[![CEUS](https://img.shields.io/badge/Journal-CEUS-blue.svg)](https://doi.org/10.1016/j.compenvurbsys.2025.102335)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.compenvurbsys.2025.102335-blue.svg)](https://doi.org/10.1016/j.compenvurbsys.2025.102335)
[![arXiv](https://img.shields.io/badge/arXiv-2504.09066-b31b1b.svg)](https://arXiv.org/abs/2504.09066)

If you use this repository, please cite **both** the *Computers, Environment and Urban Systems* article and the *arXiv* preprint.

<details>
<summary><b>📖 APA Citation (click to expand)</b></summary>

Yang, Y., Zou, L., Zhou, B., Li, D., Lin, B., Abedin, J., & Yang, M. (2025). Hyperlocal disaster damage assessment using bi-temporal street-view imagery and pre-trained vision models. Computers, Environment and Urban Systems, 121, 102335.

</details>

<details>
<summary><b>🧾 BibTeX (click to expand)</b></summary>

```bibtex
@article{YANG2025102335,
title = {Hyperlocal disaster damage assessment using bi-temporal street-view imagery and pre-trained vision models},
journal = {Computers, Environment and Urban Systems},
volume = {121},
pages = {102335},
year = {2025},
issn = {0198-9715},
doi = {https://doi.org/10.1016/j.compenvurbsys.2025.102335},
url = {https://www.sciencedirect.com/science/article/pii/S0198971525000882},
author = {Yifan Yang and Lei Zou and Bing Zhou and Daoyang Li and Binbin Lin and Joynal Abedin and Mingzheng Yang},
keywords = {Disaster resilience, Street-view imagery, Dual-channel neural network, Pre-trained vision model, Damage estimation}
}
