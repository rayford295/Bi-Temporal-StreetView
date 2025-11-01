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
- Severity scores (minor, moderate, severe)
- Sample image regions from Horseshoe Beach, Florida, after Hurricane Milton

---
## 🧠 Paper Reference
### 📚 Citation

[![CEUS](https://img.shields.io/badge/Journal-CEUS-blue.svg)](https://doi.org/10.1016/j.compenvurbsys.2025.102372)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.compenvurbsys.2025.102372-blue.svg)](https://doi.org/10.1016/j.compenvurbsys.2025.102372)
[![arXiv](https://img.shields.io/badge/arXiv-2504.09066-b31b1b.svg)](https://arxiv.org/abs/2504.09066)

If you use this repository, please cite **both** the CEUS article and the arXiv preprint.

---

<details>
<summary><b>📖 APA Citation (click to expand)</b></summary>

Yang, Y., Zou, L., Zhou, B., Li, D., Lin, B., Abedin, J., & Yang, M. (2025). *Hyperlocal disaster damage assessment using bi-temporal street-view imagery and pre-trained vision models*. *Computers, Environment and Urban Systems, 112*, 102372. https://doi.org/10.1016/j.compenvurbsys.2025.102372

</details>

<details>
<summary><b>🧾 BibTeX (click to expand)</b></summary>

```bibtex
@article{yang2025hyperlocal,
  title        = {Hyperlocal disaster damage assessment using bi-temporal street-view imagery and pre-trained vision models},
  author       = {Yang, Yifan and Zou, Lei and Zhou, Bing and Li, Daoyang and Lin, Binbin and Abedin, Joynal and Yang, Mingzheng},
  journal      = {Computers, Environment and Urban Systems},
  volume       = {112},
  pages        = {102372},
  year         = {2025},
  doi          = {10.1016/j.compenvurbsys.2025.102372},
  publisher    = {Elsevier},
  url          = {https://doi.org/10.1016/j.compenvurbsys.2025.102372}
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


