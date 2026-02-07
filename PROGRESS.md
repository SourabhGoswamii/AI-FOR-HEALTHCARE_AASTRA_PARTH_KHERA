# Project Progress Report  
## AI-for-Healthcare_AASTRA_Parth-Khera

This document summarizes the progress of the MRI-based neurological disorder detection project, covering dataset preprocessing, binary classification, and multi-class classification.

All work was executed on a **Linux GPU server accessed securely via VPN and SSH**.

---

## Phase 1 — Setup and Problem Understanding

- Analyzed the problem statement and evaluation requirements  
- Defined strict compliance constraints (no data manipulation, augmentation, or rebalancing)  
- Configured a secure Linux GPU environment for development  

---

## Phase 2 — Task 1: Dataset Preprocessing

**Objective:** Prepare a clean, standardized, and AI-ready MRI dataset.

**Work Completed:**
- Recursive loading of nested DICOM MRI scans  
- Parsing subject-level diagnostic labels from CSV  
- Treating CSV as the single source of ground truth  
- Reconstruction of subject-level 3D MRI volumes  
- Background noise removal and global intensity normalization  
- Central slice extraction (3D → 2D)  
- Leakage-free train/test data splitting  

**Compliance:**
- No data augmentation  
- No label modification  
- No sample addition or removal  
- No class rebalancing  
- No manual or disease-specific region selection  

**Outcome:**  
A consistent and standardized MRI dataset ready for learning tasks.

---

## Phase 3 — Task 2: Binary Classification (CN vs AD)

**Objective:** Distinguish Cognitively Normal (CN) subjects from Alzheimer’s Disease (AD) patients using MRI data.

**Approach:**
- Central MRI slices resized to 128 × 128  
- 2D CNN with sigmoid output  
- Loss: Binary Crossentropy  
- Optimizer: Adam  
- Early stopping to prevent overfitting  

**Evaluation Metrics:**
- Accuracy  
- Precision, Recall, F1-score  
- Confusion Matrix  
- ROC–AUC  

**Results:**
- Achieved **>91% test accuracy**, meeting the task requirement  

---

## Phase 4 — Task 3: Multi-Class Classification (CN vs MCI vs AD)

**Objective:** Classify MRI scans into CN, MCI, and AD in a clinically realistic setting.

**Approach:**
- Slice-based learning with subject-level aggregation  
- 2D CNN with softmax output  
- Loss: Categorical Crossentropy  
- Optimizer: Adam  
- Early stopping  

**Evaluation Metrics:**
- Overall accuracy  
- Per-class precision, recall, and F1-score  
- Confusion matrix  

**Results:**
- Achieved **>55% accuracy**, exceeding the required threshold  

---

## Phase 5 — Validation and Review

- Verified consistency between preprocessing and classification outputs  
- Reviewed model behavior across binary and multi-class tasks  
- Ensured all steps strictly followed compliance constraints  

---

## Current Status

| Task | Status |
|------|--------|
| Task 1 — Dataset Preprocessing | ✅ Completed |
| Task 2 — Binary Classification | ✅ Completed |
| Task 3 — Multi-Class Classification | ✅ Completed |

---

## Final Remarks

This progress report documents the successful completion of all required tasks, resulting in a **compliant, reproducible, and clinically relevant MRI-based AI system** suitable for neurological disorder screening and evaluation.
