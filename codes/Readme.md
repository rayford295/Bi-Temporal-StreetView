# 🧠 Code Overview

This folder contains the core implementation and experiment scripts for the **Bi-Temporal Street-View Disaster Damage Assessment** project.  
Each file supports a distinct component of the model training, evaluation, and interpretability workflow.

## 📂 File Descriptions

### `swin_transformer.py`
Implements the **Dual-Swin Bi-Temporal Classifier** for disaster damage assessment.  
This script supports feature extraction and fine-tuning using paired pre- and post-disaster images.

### `convnext_split_experiment.py`
Contains the **ConvNeXt Tiny** model training and evaluation pipeline.  
Includes multi-class classification, accuracy computation, and Grad-CAM visualization integration.

### `Grad-CAM Visualization.py`
Provides **multi-backbone Grad-CAM visualization** for interpretability analysis.  
It can be used to compare spatial attention across different model backbones (e.g., Swin, ConvNeXt).

### `generate_summary__GPT_4o_mini.ipynb`
A notebook for **summarizing and documenting experimental results** using GPT-4o-mini.  
Includes automated text summarization and report generation for paper preparation.

---

## 📘 Usage Notes

1. Make sure to install all dependencies listed in the root `requirements.txt`. Using Google Colab would be the best option, as all the code for this project can be run on Colab. 
2. Adjust data paths (`train_split.csv`, `test_split.csv`, and `images/`) as needed.  
3. Training scripts can be executed individually to reproduce reported results.

---

## 📩 Contact

If you find any missing or incomplete code, please feel free to contact:

**Yifan Yang**  
Texas A&M University – GEAR Lab  
📧 [yyf990925@gmail.com](mailto:yyf990925@gmail.com)

