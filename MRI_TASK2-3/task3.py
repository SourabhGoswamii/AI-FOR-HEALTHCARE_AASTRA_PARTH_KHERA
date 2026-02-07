# ============================================================
# TASK 3 SHOWCASE — MULTI-CLASS NEUROLOGICAL SCREENING
# CN vs MCI vs AD | Folder-based dataset | Honest evaluation
# ============================================================

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import tensorflow as tf

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

# -------- headless plotting --------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIG =================
DATASET_DIR = "processed_dataset"
TARGET_SHAPE = (128,128,128)

CLASS_MAP = {"CN":0, "MCI":1, "AD":2}
CLASS_NAMES = ["CN","MCI","AD"]

# ================= PREPROCESS =================
def preprocess_mri(path):
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    x = img.get_fdata()

    if x.ndim == 4:
        x = x[...,0]

    x = x * (x > np.mean(x))  # skull stripping

    factors = [t/s for t,s in zip(TARGET_SHAPE, x.shape)]
    x = zoom(x, factors, order=1)

    lo, hi = np.percentile(x,(1,99))
    x = np.clip(x, lo, hi)
    x = (x-lo)/(hi-lo+1e-6)

    return x.astype(np.float32)

# ================= LOAD DATA =================
def load_split(split):
    X, y, pid = [], [], []
    base = os.path.join(DATASET_DIR, split)

    for label in CLASS_MAP:
        class_dir = os.path.join(base, label)
        if not os.path.exists(class_dir):
            continue

        for f in os.listdir(class_dir):
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                path = os.path.join(class_dir, f)
                try:
                    vol = preprocess_mri(path)
                    patient_id = f.split("_clean")[0]

                    for i in range(45,85):
                        sl = vol[:,:,i]
                        if np.max(sl) > 0.05:
                            X.append(sl[...,None])
                            y.append(CLASS_MAP[label])
                            pid.append(patient_id)
                except:
                    continue

    return np.array(X), np.array(y), np.array(pid)

print("Loading TRAIN data...")
X_train, y_train, pid_train = load_split("training")

print("Loading VAL data...")
X_val, y_val, pid_val = load_split("validation")

print("Loading TEST data...")
X_test, y_test, pid_test = load_split("testing")

print("Train slices:", X_train.shape)
print("Val slices:", X_val.shape)
print("Test slices:", X_test.shape)

y_train_cat = tf.keras.utils.to_categorical(y_train,3)
y_val_cat   = tf.keras.utils.to_categorical(y_val,3)

# ================= MODEL =================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128,128,1)),
    tf.keras.layers.Conv2D(32,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(3,activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================= TRAIN =================
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=8,            # intentionally limited (reduce overfitting)
    batch_size=32,
    verbose=1
)

# ================= PATIENT-LEVEL EVALUATION =================
slice_probs = model.predict(X_test)

patient_probs = {}
for p, pr in zip(pid_test, slice_probs):
    patient_probs.setdefault(p, []).append(pr)

y_true, y_pred, y_prob = [], [], []

for p, preds in patient_probs.items():
    avg = np.mean(preds, axis=0)
    y_pred.append(np.argmax(avg))
    y_prob.append(avg)

    idx = pid_test.tolist().index(p)
    y_true.append(y_test[idx])

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# ================= METRICS =================
bal_acc = balanced_accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
auc = roc_auc_score(
    tf.keras.utils.to_categorical(y_true,3),
    y_prob,
    multi_class="ovr"
)

print("\n===== FINAL EVALUATION (PATIENT LEVEL) =====")
print(f"Balanced Accuracy : {bal_acc:.3f}")
print(f"Macro F1-Score    : {macro_f1:.3f}")
print(f"Precision (macro) : {precision:.3f}")
print(f"Recall (macro)    : {recall:.3f}")
print(f"AUC (OvR)         : {auc:.3f}")

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Patient Level)")
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

# ================= TRAINING CURVES =================
plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Training vs Validation Accuracy")
plt.legend(["Train","Validation"])
plt.savefig("accuracy_curve.png", dpi=300)
plt.close()

plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Training vs Validation Loss")
plt.legend(["Train","Validation"])
plt.savefig("loss_curve.png", dpi=300)
plt.close()

print("\nSaved Outputs:")
print(" - confusion_matrix.png")
print(" - accuracy_curve.png")
print(" - loss_curve.png")
