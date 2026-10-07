# 🧠 Brain Tumor Detection & Classification using Hybrid CNN–ViT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

> **A two-stage, explainable, reproducible medical AI pipeline for brain tumor diagnosis from MRI images**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [System Architecture](#-system-architecture)
- [Model Design](#️-model-design)
- [Dataset](#️-dataset)
- [Results](#-results)
- [Explainability](#-explainability)
- [Installation & Usage](#-installation--usage)
- [Project Structure](#-project-structure)
- [Limitations](#️-limitations)
- [Key Contributions](#-key-contributions)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Overview

This project implements a **two-stage brain tumor diagnosis system** that mimics real clinical workflow:

```
Screening → Diagnosis → Explanation
```

### **Stage 1: Binary Classification**
Detects whether a tumor is present (Tumor / No Tumor)

### **Stage 2: Multiclass Classification**
If tumor is detected, classifies it into:
- 🔴 **Glioma**
- 🟡 **Meningioma**
- 🔵 **Pituitary**
- ⚪ **Normal**

### **Key Features**
✅ Hybrid CNN + Vision Transformer (ViT) architecture  
✅ Grad-CAM based explainability for tumor localization  
✅ Trained on **~20,000 MRI images** from multiple public datasets  
✅ Fully reproducible training & evaluation pipeline  

---

## 💡 Problem Statement

### Challenges with Existing Systems:
❌ Use only single-stage classification (binary OR multiclass)  
❌ Trained on limited, single-source datasets  
❌ Lack interpretability and explainability  
❌ Poor reproducibility  

### Our Solution:
✅ Multi-stage diagnostic pipeline  
✅ Heterogeneous MRI dataset integration  
✅ Attention-based visual explanations  
✅ Complete reproducible codebase  

---

## 🧩 System Architecture

```
┌─────────────┐
│  MRI Image  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│  Binary Classification      │
│  (Tumor / No Tumor)         │
└──────┬──────────────────────┘
       │
       │ if Tumor detected
       ▼
┌─────────────────────────────┐
│  Multiclass Classification  │
│  (Glioma/Meningioma/        │
│   Pituitary/Normal)         │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Grad-CAM Visualization     │
│  (Tumor Localization)       │
└─────────────────────────────┘
```

---

## 🏗️ Model Design

### Architecture Components

#### **1. Binary Classification Model**
- **Input**: 224×224 MRI images
- **Architecture**: Hybrid CNN + Vision Transformer
- **Output**: 2 classes (Tumor / No Tumor)

#### **2. Multiclass Classification Model**
- **Input**: 224×224 MRI images
- **Architecture**: Hybrid CNN + Vision Transformer
- **Output**: 4 classes (Glioma / Meningioma / Pituitary / Normal)

#### **3. Vision Transformer Components**
- **PatchExtractor**: Divides images into 16×16 patches
- **PatchEncoder**: Adds positional embeddings
- **Multi-head Self-Attention**: Captures global dependencies
- **Feed-Forward Transformer Blocks**: Deep feature learning

---

## 🗂️ Dataset

### Training Data
- **Total Images**: ~20,000 MRI scans
- **Sources**: Multiple public MRI datasets
- **Processing**: Converted .mat tumor datasets into image format

### Dataset Structure

#### Binary Classification
```
binary/
├── train/
│   ├── no_tumor/
│   └── tumor/
├── val/
│   ├── no_tumor/
│   └── tumor/
└── test/
    ├── no_tumor/
    └── tumor/
```

#### Multiclass Classification
```
multiclass/
├── train/
│   ├── Glioma/
│   ├── Meningioma/
│   ├── Normal/
│   └── Pituitary/
├── val/
│   ├── Glioma/
│   ├── Meningioma/
│   ├── Normal/
│   └── Pituitary/
└── test/
    ├── Glioma/
    ├── Meningioma/
    ├── Normal/
    └── Pituitary/
```

---

## 📊 Results

### 🔹 Binary Classification Model

| Metric | Score |
|--------|-------|
| **Accuracy** | 93% |
| **ROC-AUC** | 0.987 |
| **Tumor Recall** | 92% |

**Confusion Matrix:**
```
                Predicted
              No Tumor  Tumor
Actual No      196       4
       Tumor    97     1093
```

---

### 🔹 Multiclass Classification Model

| Metric | Score |
|--------|-------|
| **Accuracy** | 94% |
| **Macro F1-Score** | 0.94 |

**Confusion Matrix:**
```
              Glioma  Meningioma  Normal  Pituitary
Glioma          163      17         2        6
Meningioma       10     172         5        0
Normal            1       4       186        0
Pituitary         1       0         0      187
```

---

### 🔹 End-to-End Pipeline Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 85% |
| **Macro F1-Score** | 0.85 |

**Confusion Matrix:**
```
              Glioma  Meningioma  Normal  Pituitary
Glioma          150      14        18        6
Meningioma        9     121        57        0
Normal            1       1       189        0
Pituitary         1       0         7      180
```

> **Note**: The drop in pipeline accuracy is due to error propagation from the binary stage. This is expected in cascaded medical decision systems and can be optimized by tuning the binary classification threshold.

---

## 🔍 Explainability

### Grad-CAM Visualization

The system generates **attention-based heatmaps** that:
- ✅ Highlight regions influencing predictions
- ✅ Visualize potential tumor locations
- ✅ Improve model trustworthiness
- ✅ Provide qualitative validation

#### Example Output:
```
Original Image → Grad-CAM Heatmap → Overlay
```

⚠️ **Important**: This is localization, not segmentation. The heatmaps show areas of attention, not precise tumor boundaries.

---

## 🚀 Installation & Usage

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.x
```

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run Inference Pipeline
```bash
python test_pipeline.py
```

### 3️⃣ Generate Grad-CAM Visualizations
```bash
python gradcam_visualization.py
```

---

## 📁 Project Structure

```
brain-tumor-ai/
│
├── dataset_builder/          # Dataset preparation scripts
├── models/                   # Saved model weights
├── training/                 # Training scripts
├── evaluation/               # Evaluation metrics & plots
├── explainability/           # Grad-CAM implementation
│   └── gradcam.py
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── LICENSE                   # MIT License
```

---

## ⚠️ Limitations

This project has several important limitations:

- 🔸 Uses **public datasets only** (not clinically validated)
- 🔸 **Class imbalance** present in training data
- 🔸 Grad-CAM provides **localization, not segmentation**
- 🔸 **Not clinically validated** or approved for medical diagnosis
- 🔸 Requires further testing on diverse patient populations

> **⚕️ Medical Disclaimer**: This project is intended for **research and educational purposes only**. It should not be used for actual medical diagnosis without proper clinical validation and regulatory approval.

---

## 🌟 Key Contributions

1. **Two-Stage Diagnostic Pipeline** – Mimics clinical screening workflow
2. **Hybrid CNN–ViT Architecture** – Combines local and global feature learning
3. **Multi-Dataset Integration** – Trained on ~20,000 diverse MRI images
4. **Explainable AI** – Grad-CAM visualization for interpretability
5. **Reproducible Evaluation** – Complete metrics and confusion matrices
6. **Open Source** – Fully documented and shareable codebase

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{brain-tumor-cnn-vit,
  title={Hybrid CNN–Vision Transformer based Brain Tumor Detection and Classification with Explainable AI},
  author={Varun P},
  year={2025},
  publisher={GitHub},
  url={https://github.com/Varun-ai07/brain-tumor-ai}
}
```

---

## 👨‍💻 Author

Project: Brain Tumor Detection & Classification using Hybrid CNN–ViT

**VARUN P**  
AI Architect & Quantum Computing Researcher  

[![Portfolio](https://img.shields.io/badge/Portfolio--brightgreen?style=flat&logo=googlechrome)](https://varun-ai07.github.io)
[![Email](https://img.shields.io/badge/Email--red?style=flat&logo=gmail)](mailto:jp.vxrun@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn--blue?style=flat&logo=linkedin)](https://linkedin.com/in/jp-varun)
[![GitHub](https://img.shields.io/badge/GitHub--black?style=flat&logo=github)](https://github.com/Varun-ai07)


**SURYA PRAKASH V**
Full Stack Developer & AI Engineer

[![Email](https://img.shields.io/badge/Email--red?style=flat&logo=gmail)](mailto:v.surya.prakash.2210@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn--blue?style=flat&logo=linkedin)](https://linkedin.com/in/v-suryaprakash)
[![GitHub](https://img.shields.io/badge/GitHub--black?style=flat&logo=github)](https://github.com/v-suryaprakash)

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

##  Acknowledgments

- Public MRI datasets used for training
- TensorFlow and Keras communities
- Vision Transformer implementation references

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star!**

Made with ❤️ for advancing medical AI research

</div>
