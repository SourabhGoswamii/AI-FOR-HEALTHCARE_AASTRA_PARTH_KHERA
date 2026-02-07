# AI-for-Healthcare_AASTRA_Parth-Khera  
## MRI-Based Neurological Disorder Detection

![Subject-Level MRI Classification Pipeline](images/pipeline_overview.png)

---

## Overview
This project implements an **end-to-end healthcare AI pipeline** for neurological disorder detection using **T1-weighted brain MRI scans**.  
The system covers **dataset preprocessing**, **binary classification**, and **multi-class classification**, designed for early screening and clinical decision support.

All experiments were conducted on a **Linux GPU server accessed securely via VPN and SSH**.

---

## Dataset
- **MRI:** Brain MRI scans in DICOM (.dcm) format  
- **Labels:** CSV file containing subject-level diagnoses:
  - CN – Cognitively Normal  
  - MCI – Mild Cognitive Impairment  
  - AD – Alzheimer’s Disease  

The CSV file is treated as the **single source of ground truth**.

---

## Task 1 — Dataset Preprocessing

### Objective
Ensure **data integrity, standardization, and reproducibility** prior to model training.

### Pipeline
- Automatic data extraction  
- Recursive DICOM loading  
- Subject-level 3D MRI volume reconstruction  
- Background noise removal and global intensity normalization  
- Central slice extraction (3D → 2D)  
- Label encoding  
- Leakage-free train/test split  

### Compliance
- No data augmentation  
- No label modification  
- No sample addition or removal  
- No class rebalancing  
- No manual or disease-specific region selection  

All preprocessing steps are **uniform, deterministic, and reproducible**.

---

## Task 2 — Binary Classification (CN vs AD)

![Binary Classification Pipeline](images/task2_binary_pipeline.png)
![Grad-CAM Visualization](images/gradcam_binary.png)

### Objective
Detect **Alzheimer’s Disease (AD)** from **Cognitively Normal (CN)** subjects using MRI data.

### Method
- Central MRI slices resized to **128 × 128**
- **2D CNN** with slice-level feature extraction
- Subject-level feature aggregation
- Sigmoid output layer  
- Loss: Binary Crossentropy  
- Optimizer: Adam  
- Early stopping  

### Evaluation
- Accuracy  
- Precision, Recall, F1-score  
- Confusion Matrix  
- ROC–AUC  

### Result
✅ Achieved **>91% test accuracy**, satisfying the task requirement.

---

## Task 3 — Multi-Class Classification (CN vs MCI vs AD)

![Multi-Class ROC Curves](images/task3_multiclass_roc.png)
![Confusion Matrix](images/task3_confusion_matrix.png)

### Objective
Distinguish between **CN**, **MCI**, and **AD** in a clinically realistic setting.

### Method
- Slice-based learning with subject-level aggregation  
- **2D CNN** with softmax output  
- Loss: Categorical Crossentropy  
- Optimizer: Adam  
- Early stopping  

### Evaluation
- Overall and per-class accuracy  
- Precision, Recall, F1-score  
- Confusion matrix  

### Result
✅ Achieved **>55% accuracy**, exceeding the required threshold.

---

## Final Summary
This project delivers a **compliant, reproducible, and clinically relevant MRI-based AI system** capable of:

- Standardized MRI preprocessing  
- Reliable binary Alzheimer’s disease detection  
- Meaningful multi-class neurological classification  

The pipeline is suitable for **real-world screening and decision-support scenarios**.

---
