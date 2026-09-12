# 🩺 Melanoma Skin Lesion Analysis

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Medical%20Imaging-green)
![Explainable AI](https://img.shields.io/badge/XAI-Grad--CAM%20%7C%20LIME%20%7C%20SHAP-purple)
![Project](https://img.shields.io/badge/Project-Research-red)

## 📂 Project Overview

This project presents a deep learning framework for **automated melanoma detection from dermoscopic skin-lesion images**.

The study combines image preprocessing, lesion segmentation, deep convolutional neural networks, explainable artificial intelligence, external validation, fairness analysis, and uncertainty evaluation to investigate the reliability and generalization of AI-based melanoma classification.

The project uses **HAM10000** as the primary development dataset and evaluates model generalization using **Derm7pt** and **PAD-UFES-20**.

A lightweight attention-based architecture, **LAMNet**, is also developed and compared with established deep learning architectures including AlexNet, VGG16, VGG19, ResNet50, and EfficientNet-B0.

> **Important:** This is a research-stage system and is not intended for clinical diagnosis or medical decision-making.

---

## 🎯 Project Objectives

* Develop robust preprocessing techniques for dermoscopic images.
* Apply **Weighted Median Filtering** and **Watershed Transformation** for image preprocessing.
* Develop a **U-Net-based lesion segmentation pipeline**.
* Train and compare multiple deep learning classification architectures.
* Develop a lightweight attention-based model named **LAMNet**.
* Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.
* Perform independent external validation across different datasets.
* Analyze model decisions using **Grad-CAM, LIME, SHAP, and attention maps**.
* Investigate model performance across **Fitzpatrick skin-tone groups**.
* Analyze confidence, calibration, and uncertainty.
* Study the limitations of melanoma AI models under dataset shift and class imbalance.

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Scikit-learn
* OpenCV
* Matplotlib
* PIL
* Grad-CAM
* LIME
* SHAP
* U-Net
* Convolutional Neural Networks
* CUDA / NVIDIA GPU
* Kaggle
* Google Colab
* Google Drive
* Jupyter Notebook

---

# 📊 Datasets

The project uses three dermatology datasets.

### 1. HAM10000

The primary dataset used for model development and internal evaluation.

It contains dermoscopic images covering multiple skin-lesion categories. For this study, the task is converted into a binary classification problem:

* **0 — Non-Melanoma**
* **1 — Melanoma**

Official evaluation set:

* **1,502 images**
* **1,335 Non-Melanoma**
* **167 Melanoma**

HAM10000 is used for:

* preprocessing
* segmentation
* model training
* validation
* internal testing
* XAI analysis

---

### 2. Derm7pt

Derm7pt is used for **cross-dataset external evaluation**.

The frozen dataset split contains:

* **1,011 cases**
* **413 training cases**
* **203 validation cases**
* **395 test cases**
* **759 Non-Melanoma**
* **252 Melanoma**

Both clinical and dermoscopic images are available in the dataset.

The models trained using HAM10000 are evaluated on Derm7pt to investigate cross-dataset generalization.

---

### 3. PAD-UFES-20

PAD-UFES-20 is used as an **independent external validation dataset**.

The dataset contains:

* **2,298 images**
* **1,373 patients**
* **1,641 lesions**
* **52 Melanoma**
* **2,246 Non-Melanoma**

The extremely low melanoma prevalence makes this dataset particularly useful for evaluating model behavior under severe class imbalance.

---

# 🔬 Research Workflow

```text
                    Melanoma Skin Lesion Analysis
                              │
                              ▼
                       Dataset Preparation
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
          HAM10000                      External Data
                │                    Derm7pt / PAD-UFES-20
                ▼
        Image Preprocessing
                │
        ┌───────┴────────┐
        ▼                ▼
 Weighted Median     Watershed
    Filtering        Transform
        │                │
        └───────┬────────┘
                ▼
          U-Net Segmentation
                │
                ▼
       Deep Learning Models
                │
     ┌──────────┼──────────┐
     ▼          ▼          ▼
  AlexNet    VGG16/VGG19  ResNet50
                │
                ▼
        EfficientNet-B0
                │
                ▼
             LAMNet
                │
                ▼
        Model Evaluation
                │
       ┌────────┼─────────┐
       ▼        ▼         ▼
   Internal   External    XAI
   Testing    Validation
                │
                ▼
       Fairness & Uncertainty
```

---

# 🧠 Image Preprocessing & Segmentation

The preprocessing pipeline investigates traditional and deep-learning-based techniques for improving lesion representation.

### Weighted Median Filtering

Weighted median filtering is used as a preprocessing technique to reduce image noise while preserving important lesion structures and boundaries.

### Watershed Transformation

Watershed-based processing is investigated for separating lesion regions from surrounding skin structures.

### U-Net Segmentation

A U-Net architecture is used to perform lesion segmentation.

The official HAM10000 test evaluation achieved:

| Metric         |      Score |
| -------------- | ---------: |
| Dice           | **0.9241** |
| IoU            | **0.8727** |
| Pixel Accuracy | **0.9678** |

These results indicate strong overlap between predicted and reference lesion masks.

---

# 🤖 Deep Learning Classification Models

Six classification architectures are evaluated:

### Benchmark Models

* AlexNet
* VGG16
* VGG19
* ResNet50
* EfficientNet-B0

### Proposed Lightweight Architecture

* **LAMNet — Lightweight Attention-based Melanoma Network**

All models perform binary classification:

```text
Input Image
     ↓
Feature Extraction
     ↓
Deep Neural Network
     ↓
Melanoma Probability
     ↓
Melanoma / Non-Melanoma
```

---

# 🧩 LAMNet

LAMNet is a lightweight attention-based convolutional architecture developed as part of this project.

The architecture incorporates:

* Channel Attention
* Spatial Attention
* Residual connections
* Convolutional blocks
* Batch normalization
* Global average pooling
* Dropout

The final architecture contains approximately:

**2.43 million trainable parameters**

This makes LAMNet substantially smaller than several conventional deep CNN architectures evaluated in the study.

The objective is not simply to maximize accuracy, but to investigate the trade-off between:

* model size
* computational efficiency
* melanoma sensitivity
* generalization
* interpretability

---

# 📈 HAM10000 Classification Results

Official test-set results:

| Model           |   Accuracy |  Precision |     Recall |   F1-Score |    ROC-AUC |
| --------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| AlexNet         |     84.55% |     38.60% | **65.87%** |     48.67% |     87.04% |
| VGG16           |     85.62% |     41.52% | **71.86%** |     52.63% |     89.61% |
| VGG19           | **90.35%** | **56.88%** |     54.49% |     55.66% |     87.25% |
| ResNet50        |     88.15% |     47.72% |     68.86% | **56.37%** | **90.90%** |
| EfficientNet-B0 |     89.55% |     53.47% |     46.11% |     49.52% |     88.84% |
| LAMNet          |     77.23% |     28.50% |     69.46% |     40.42% |     81.97% |

### 🏆 Key Findings

* **ResNet50** achieved the highest F1-score and ROC-AUC.
* **VGG19** achieved the highest accuracy and precision.
* **VGG16** achieved the highest melanoma recall.
* **LAMNet** achieved relatively high melanoma recall while using only approximately **2.43M parameters**.
* LAMNet did **not** outperform the strongest benchmark architectures overall.

---

# 🌍 Cross-Dataset Validation

A major focus of the project is evaluating whether models trained on HAM10000 maintain their performance on different datasets.

## Derm7pt

The HAM10000-trained models were evaluated on Derm7pt without retraining.

The results demonstrated a clear change in model behavior across datasets.

LAMNet achieved:

| Metric    |    Derm7pt |
| --------- | ---------: |
| Accuracy  |     60.00% |
| Precision |     35.82% |
| Recall    | **71.29%** |
| F1-Score  | **47.68%** |
| ROC-AUC   |     72.30% |

Among the evaluated models:

* **ResNet50** achieved the highest accuracy and ROC-AUC.
* **LAMNet** achieved the highest recall and F1-score.

This demonstrates that model rankings can change substantially under dataset shift.

---

# 🌎 PAD-UFES-20 External Validation

PAD-UFES-20 presents a more challenging external evaluation because only:

**52 of 2,298 images were melanoma.**

This corresponds to approximately **2.26% melanoma prevalence**.

LAMNet achieved:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 70.10% |
| Precision |  2.26% |
| Recall    | 28.85% |
| F1-Score  |  4.18% |
| ROC-AUC   | 47.80% |

The results demonstrate that high accuracy can be misleading when melanoma is severely underrepresented.

Therefore, the project emphasizes the importance of:

* Recall
* Precision
* F1-score
* ROC-AUC
* Confusion matrices
* Calibration
* External validation

rather than relying on accuracy alone.

---

# 🔍 Explainable AI

The project incorporates multiple explainability techniques to investigate how the models make predictions.

### Grad-CAM

Grad-CAM is used to visualize image regions that contribute to CNN predictions.

The analysis compares the attention patterns of:

* AlexNet
* VGG16
* VGG19
* ResNet50
* EfficientNet-B0

The visual analysis suggests that deeper architectures such as ResNet50 and EfficientNet-B0 can produce more lesion-focused activation patterns in selected examples, while some models show stronger activation around edges or background regions.

These observations are exploratory and should not be interpreted as proof of clinically correct reasoning.

---

### LIME

LIME is used to identify superpixel regions that contribute positively or negatively to individual predictions.

---

### SHAP

SHAP-based analysis provides another perspective on feature contributions and model behavior.

---

### LAMNet Attention Maps

LAMNet's spatial attention maps are visualized across multiple network blocks to examine how learned attention evolves from early to deeper feature representations.

---

# ⚖️ Fairness Analysis

The project also investigates model performance across **Fitzpatrick skin-tone groups** using PAD-UFES-20 metadata.

The exploratory analysis indicates that melanoma recall can vary between Fitzpatrick groups.

For the evaluated ResNet50 model, the observed recall values ranged across the available groups, with a maximum observed recall difference of approximately:

**35.71 percentage points**

This analysis highlights the importance of:

* balanced representation
* subgroup evaluation
* fairness-aware metrics
* diverse dermatological datasets

before considering clinical deployment.

---

# 🎯 Model Calibration & Uncertainty

The project investigates whether model confidence corresponds to actual predictive reliability.

Calibration analysis includes:

* Brier Score
* Expected Calibration Error
* Prediction entropy
* High-confidence errors

Among the HAM10000 models, EfficientNet-B0 showed the lowest Brier score and ECE among the evaluated models.

The analysis demonstrates that a model with high classification performance can still produce overconfident incorrect predictions.

Therefore, confidence estimates should be considered alongside classification metrics.

---

# 💡 Key Insights

* **Segmentation:** U-Net achieved strong lesion segmentation performance with a Dice score of **0.9241**.
* **Classification:** ResNet50 provided the strongest overall HAM10000 F1/AUC combination.
* **Accuracy:** VGG19 achieved the highest HAM10000 accuracy.
* **Sensitivity:** VGG16 achieved the highest HAM10000 melanoma recall.
* **Efficiency:** LAMNet uses approximately **2.43M parameters**, offering a lightweight alternative.
* **Generalization:** Performance changes substantially when models are evaluated on Derm7pt and PAD-UFES-20.
* **Class imbalance:** PAD-UFES-20 demonstrates why accuracy alone is insufficient for melanoma detection.
* **Explainability:** Grad-CAM, LIME, SHAP, and attention maps provide complementary views of model behavior.
* **Fairness:** Performance differences across skin-tone groups require further investigation.
* **Reliability:** Calibration and uncertainty analysis are important for assessing whether predictions can be trusted.
* **Clinical translation:** Further dermatologist comparison, prospective validation, and regulatory evaluation are required before clinical use.

---

# 🛠️ Skills Demonstrated

* Medical Image Analysis
* Computer Vision
* Deep Learning
* Convolutional Neural Networks
* PyTorch
* Image Preprocessing
* Image Segmentation
* U-Net
* Transfer Learning
* Attention Mechanisms
* Model Evaluation
* Cross-Dataset Validation
* Explainable AI
* Grad-CAM
* LIME
* SHAP
* Fairness Analysis
* Model Calibration
* Uncertainty Analysis
* Research Methodology
* Data Analysis
* Python Programming

---

# 📁 Repository Structure

```text
Melanoma-Skin-Lesion-Analysis/
│
├── notebooks/
│   ├── 01_HAM10000_Project_Setup.ipynb
│   ├── 02_Preprocessing_Segmentation.ipynb
│   ├── 03_HAM10000_U-Net_Baseline.ipynb
│   ├── 04_Derm7pt_Classification_Baselines.ipynb
│   ├── 05_LAMNet_HAM10000-Training.ipynb
│   ├── 06_PAD_UFES_20_External_Validation.ipynb
│   └── 07_XAI_SHAP_HAM10000.ipynb
│
├── results/
│   ├── classification_results/
│   ├── segmentation_results/
│   ├── external_validation/
│   ├── xai_results/
│   └── fairness_results/
│
├── figures/
│   ├── segmentation/
│   ├── classification/
│   ├── gradcam/
│   ├── lime/
│   ├── shap/
│   └── attention_maps/
│
├── documentation/
│   └── research_paper/
│
├── README.md
└── .gitignore
```

> Large datasets and trained model checkpoints should not be stored directly in the GitHub repository. Dataset download/access instructions and artifact locations should be documented separately.

---

# 🚀 How to Run

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the project

```bash
cd Melanoma-Skin-Lesion-Analysis
```

### 3. Install the required libraries

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib opencv-python pillow
```

For explainability experiments:

```bash
pip install grad-cam lime shap
```

### 4. Open the notebooks

The notebooks can be opened using:

* Jupyter Notebook
* JupyterLab
* Google Colab
* Kaggle Notebooks
* VS Code

### 5. Follow the research workflow

Start with:

```text
01 → Project Setup
02 → Preprocessing & Segmentation
03 → U-Net Baseline
04 → Classification Baselines
05 → LAMNet Training
06 → External Validation
07 → XAI & SHAP
```

---

# 📌 Project Outcome

This project demonstrates a complete research-oriented workflow for melanoma skin-lesion analysis, combining:

**Preprocessing → Segmentation → Classification → External Validation → Explainability → Fairness → Uncertainty**

The main finding is that strong internal test performance does not necessarily guarantee reliable performance on independent datasets.

The PAD-UFES-20 evaluation particularly demonstrates how severe class imbalance and domain differences can cause accuracy to appear high while melanoma detection remains poor.

The study therefore emphasizes **robust evaluation and reliability rather than accuracy alone**.

---

# 🔮 Future Enhancements

* Train the final proposed architecture using a unified multi-dataset strategy.
* Perform controlled component-wise ablation studies for LAMNet.
* Improve cross-dataset domain adaptation.
* Expand fairness analysis using larger and more balanced skin-tone datasets.
* Perform statistical significance testing between models.
* Improve uncertainty estimation using ensemble or Bayesian approaches.
* Conduct a formal dermatologist reader study.
* Develop a clinical-style inference interface.
* Perform prospective clinical validation.
* Investigate regulatory requirements for future medical-device development.

---



# 👩‍💻 Author

### Akansha Walia

**MCA Student | Data Analytics | Machine Learning | AI Research**

Interested in:

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Computer Vision
* Medical AI
* Explainable AI
* Data Analytics

---

## ⭐ Acknowledgement

This project is developed as an academic and research-oriented investigation into deep learning approaches for melanoma skin-lesion analysis.

The work is intended for **research and educational purposes** and should not be used as a substitute for professional medical diagnosis.


