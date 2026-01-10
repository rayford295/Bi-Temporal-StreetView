# 🛰️ BiTemporal-StreetView-Damage

Hyperlocal disaster damage assessment using bi-temporal street-view imagery and pre-trained vision models.

<p align="center">
  <img src="https://github.com/rayford295/BiTemporal-StreetView-Damage/blob/main/images/0204-06.png" alt="Study Area Map" width="600"/>
</p>


---

## 📌 Introduction

This repository presents a novel framework for **bi-temporal street-view image analysis**, aimed at advancing hyperlocal disaster damage assessment. We integrate **pre- and post-disaster imagery** using **pre-trained vision and vision-language models** to classify and localize disaster impact more accurately.

### 🔍 Key Contributions

- ✅ **Dual-channel model** for fusing pre- and post-disaster street-view images.
- 📸 **2,249 labeled street-view image pairs**, annotated with fine-grained disaster impact.
- 📈 **Performance**: Accuracy improved from 66.14% (post-only) to 77.11% (bi-temporal).
- 🔥 **Grad-CAM visualization** confirms the added value of pre-disaster imagery for model focus.
- 🏙️ Enables **rapid and fine-grained damage mapping**, supporting climate-resilient urban planning.

<p align="center">
  <img src="https://raw.githubusercontent.com/rayford295/BiTemporal-StreetView-Damage/main/images/dual_channel.drawio%20(2).png" alt="Dual-Channel Architecture" width="600"/>
</p>

<p align="center"><i>Figure: Dual-channel architecture for bi-temporal disaster damage assessment.</i></p>
---

## 📂 Dataset

You can access the **bi-temporal street-view disaster dataset** from the following DOI:

> 📁 Yang, Yifan (2025).  
> *Perceiving Multidimensional Disaster Damages from Street–View Images Using Visual–Language Models*.  
> figshare. Dataset. https://doi.org/10.6084/m9.figshare.28801208.v2

The dataset includes:
- Pre- and post-disaster images
- Location and damage type annotations
- Severity scores (mild, moderate, severe)
- Sample image regions from Horseshoe Beach, Florida, after Hurricane Milton

---
## 🧠 Paper Reference
### 📚 Citation

[![CEUS](https://img.shields.io/badge/Journal-CEUS-blue.svg)](https://doi.org/10.1016/j.compenvurbsys.2025.102335)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.compenvurbsys.2025.102335-blue.svg)](https://doi.org/10.1016/j.compenvurbsys.2025.102335)
[![arXiv](https://img.shields.io/badge/arXiv-2504.09066-b31b1b.svg)](https://arXiv.org/abs/2504.09066)

If you use this repository, please cite **both** the CEUS article and the arXiv preprint.

---

<details>
<summary><b>📖 APA Citation (click to expand)</b></summary>

Yang, Y., Zou, L., Zhou, B., Li, D., Lin, B., Abedin, J., & Yang, M. (2025). *Hyperlocal disaster damage assessment using bi-temporal street-view imagery and pre-trained vision models*. *Computers, Environment and Urban Systems, 121*, 102335. https://doi.org/10.1016/j.compenvurbsys.2025.102335

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
keywords = {Disaster resilience, Street-view imagery, Dual-channel neural network, Pre-trained vision model, Damage estimation},
abstract = {Street-view images offer unique advantages for disaster damage estimation as they capture impacts from a visual perspective and provide detailed, on-the-ground insights. Despite several investigations attempting to analyze street-view images for damage estimation, they mainly focus on using post-disaster images. The potential of time-series street-view images remains underexplored. Pre-disaster images provide valuable benchmarks for accurate damage estimations at building and street levels. These images could also aid annotators in objectively labeling post-disaster impacts, improving the reliability of labeled data sets for model training, and potentially enhancing the model performance in damage evaluation. The goal of this study is to estimate hyperlocal, on-the-ground disaster damages using bi-temporal street-view images and advanced pre-trained vision models. Street-view images before and after 2024 Hurricane Milton in Horseshoe Beach, Florida, were collected for experiments. The objectives are: (1) to assess the performance gains of incorporating pre-disaster street-view images as a no-damage category in fine-tuning pre-trained models, including Swin Transformer and ConvNeXt, for damage level classification; (2) to design and evaluate a dual-channel algorithm that reads pair-wise pre- and post-disaster street-view images for hyperlocal damage assessment. The results indicate that incorporating pre-disaster street-view images and employing a dual-channel processing framework can significantly enhance damage assessment accuracy. The accuracy improves from 66.14 % with the Swin Transformer baseline to 77.11 % with the dual-channel Feature-Fusion ConvNeXt model. Gradient-weighted Class Activation Mapping (Grad-CAM) shows that incorporating pre-disaster images improves the pre-trained vision model's capacity to focus on major changes between pre- and post-disaster images, thus enhancing model performance. This research evaluates the technical solutions and challenges of assessing disaster damages at hyperlocal spatial resolutions with bi-temporal street-view images, providing valuable insights to support effective decision-making in disaster management and resilience planning.}
}


## 🗂 Repository Structure

```bash
BiTemporal-StreetView-Damage/
│
├── codes/                          # Model training and evaluation scripts
├── images/                         # Project figures
│   ├── study_area_disaster_damage_made.png
│   ├── architect1.drawio (1).png
│   ├── design experiment.drawio (1).png
│   ├── dual_channel.drawio (2).png
│   ├── 0204-06.png
│   ├── readme.txt
├── LICENSE
├── README.md
└── .gitignore


