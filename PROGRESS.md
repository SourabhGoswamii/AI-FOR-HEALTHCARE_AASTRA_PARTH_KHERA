# Project Progress Report  
## AI-for-Healthcare_AASTRA_Parth-Khera

This document records the **development progress** of the MRI-based neurological disorder detection project, covering preprocessing, binary classification, and multi-class classification.

All work was carried out on a **Linux GPU server accessed securely via VPN and SSH**.

---

## Phase 1 — Problem Understanding & Environment Setup

### Completed
- Studied the healthcare problem statement and evaluation criteria
- Identified strict constraints (no data manipulation, no augmentation)
- Set up a secure Linux GPU environment using VPN and SSH

### Outcome
- Clear task separation (Task 1, Task 2, Task 3)
- Controlled and reproducible execution environment

---

## Phase 2 — Task 1: Dataset Preprocessing

### Work Done
- Loaded raw **T1-weighted brain MRI scans** in DICOM format
- Parsed subject-level diagnostic labels from CSV
- Treated CSV as the **single source of ground truth**
- Implemented:
  - Recursive loading of DICOM slices
  - Subject-level 3D MRI reconstruction
  - Background noise removal
  - Global intensity normalization
  - Central slice extraction (3D → 2D)
  - Leakage-free train/test split

### Compliance
- No data augmentation  
- No label modification  
- No sample addition or removal  
- No class rebalancing  
- No manual or disease-specific region selection  

### Reference Image
**Preprocessing outcome visualization**  
![Preprocessing Result](Images/Progress/preprocessing.png)

### Outcome
- Generated a clean and standardized MRI dataset ready for learning tasks

---

## Phase 3 — Task 2: Binary Classification (CN vs AD)

### Work Done
- Designed a **2D CNN-based binary classification model**
- Used central MRI slices resized to **128 × 128**
- Learned discriminative features at slice level
- Training setup:
  - Loss: Binary Crossentropy
  - Optimizer: Adam
  - Early stopping

### Evaluation
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC–AUC

### Results
- Achieved **>91% test accuracy**
- Clear separation between CN and AD classes

### Reference Images
**Binary classification analysis**  
![Binary Classification Result](Images/Progress/binary_result.png)  
![Binary Classification Metrics](Images/Progress/binary_metrics.png)

---

## Phase 4 — Task 3: Multi-Class Classification (CN vs MCI vs AD)

### Work Done
- Extended the model to a **multi-class classification setting**
- Encoded labels:
  - CN → 0  
  - MCI → 1  
  - AD → 2  
- Training setup:
  - Loss: Categorical Crossentropy
  - Optimizer: Adam
  - Early stopping

### Evaluation
- Overall accuracy
- Per-class precision, recall, and F1-score
- Confusion matrix

### Results
- Achieved **>55% accuracy**
- Effective identification of MCI, the most challenging class

### Reference Images
**Multi-class classification analysis**  
![Multi-Class Result](Images/Progress/multiclass_result.png)  
![Multi-Class Confusion Matrix](Images/Progress/multiclass_confusion.png)

---

## Phase 5 — Validation & Review

### Work Done
- Verified consistency between preprocessing and training outputs
- Reviewed model behavior across tasks
- Ensured all steps complied with stated constraints

### Reference Image
**Overall model behavior visualization**  
![Model Review](Images/Progress/model_review.png)

---

## Current Status
✅ Task 1 completed  
✅ Task 2 completed  
✅ Task 3 completed  
✅ Documentation completed  

---

## Final Remarks
This progress report documents the successful completion of all required tasks, resulting in a **compliant, reproducible, and clinically relevant MRI-based AI system** suitable for neurological disorder screening.
