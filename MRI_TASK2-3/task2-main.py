# ============================================================
# TASK 2 FINAL — BINARY MRI CLASSIFICATION (CN vs AD)
# Folder-based | Transfer Learning | Full Evaluation
# ============================================================

import os
import numpy as np
import nibabel as nib
import cv2
from scipy.ndimage import zoom
import tensorflow as tf

from tensorflow.keras import layers, models, applications
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

# ---------- headless plotting ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# GPU CONFIG
# ============================================================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

print("GPUs:", gpus)

# ============================================================
# CONFIG
# ============================================================
DATASET_DIR = "processed_dataset"   # training / validation / testing
IMG_SIZE = 224
NUM_SLICES = 32
EPOCHS = 30
BATCH = 4

CLASS_MAP = {"CN": 0, "AD": 1}
CLASS_NAMES = ["CN", "AD"]

# ============================================================
# MRI PREPROCESS
# ============================================================
def load_subjects(split):
    X, y = [], []

    for cname, label in CLASS_MAP.items():
        folder = os.path.join(DATASET_DIR, split, cname)
        if not os.path.exists(folder):
            continue

        for f in os.listdir(folder):
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                path = os.path.join(folder, f)
                try:
                    img = nib.load(path).get_fdata()

                    # intensity normalization
                    img = (img - np.mean(img)) / (np.std(img) + 1e-8)

                    mid = img.shape[2] // 2
                    slices = []

                    for i in range(mid - NUM_SLICES // 2,
                                   mid + NUM_SLICES // 2):
                        if 0 <= i < img.shape[2]:
                            sl = cv2.resize(img[:, :, i],
                                            (IMG_SIZE, IMG_SIZE))

                            # simple augmentation
                            if split == "training" and np.random.rand() > 0.5:
                                sl = cv2.flip(sl, 1)

                            sl = np.stack([sl] * 3, axis=-1)
                            slices.append(sl)

                    X.append(np.array(slices))
                    y.append(label)

                except:
                    continue

    return np.array(X, dtype=np.float32), np.array(y)

# ============================================================
# LOAD DATA
# ============================================================
print("Loading TRAIN data...")
X_train, y_train = load_subjects("training")

print("Loading VAL data...")
X_val, y_val = load_subjects("validation")

print("Loading TEST data...")
X_test, y_test = load_subjects("testing")

print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)

# ============================================================
# CLASS WEIGHTS (IMBALANCE SAFE)
# ============================================================
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: weights[0], 1: weights[1]}
print("Class weights:", class_weights)

# ============================================================
# MODEL (RESNET50 + SLICE AGGREGATION)
# ============================================================
base = applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

for layer in base.layers[:-20]:
    layer.trainable = False

inp = layers.Input(shape=(NUM_SLICES, IMG_SIZE, IMG_SIZE, 3))

x = layers.TimeDistributed(base)(inp)
x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
x = layers.GlobalAveragePooling1D()(x)

x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)

out = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inp, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# ============================================================
# TRAIN
# ============================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    class_weight=class_weights,
    verbose=1
)

# ============================================================
# EVALUATION
# ============================================================
prob = model.predict(X_test).flatten()

# medically safer threshold
THRESHOLD = 0.4
pred = (prob > THRESHOLD).astype(int)

bal_acc = balanced_accuracy_score(y_test, pred)
auc = roc_auc_score(y_test, prob)
macro_f1 = f1_score(y_test, pred, average="macro")
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)

print("\n===== FINAL EVALUATION (TASK 2) =====")
print(f"Balanced Accuracy : {bal_acc:.3f}")
print(f"AUC              : {auc:.3f}")
print(f"Macro F1         : {macro_f1:.3f}")
print(f"Precision        : {precision:.3f}")
print(f"Recall           : {recall:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=CLASS_NAMES))

# ============================================================
# CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — CN vs AD")
plt.savefig("task2_confusion_matrix.png", dpi=300)
plt.close()

# ============================================================
# ROC CURVE
# ============================================================
fpr, tpr, _ = roc_curve(y_test, prob)
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title("ROC Curve — CN vs AD")
plt.savefig("task2_roc_curve.png", dpi=300)
plt.close()

# ============================================================
# TRAINING CURVES
# ============================================================
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.legend()
plt.title("Accuracy Curve")
plt.savefig("task2_accuracy_curve.png", dpi=300)
plt.close()

plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.legend()
plt.title("Loss Curve")
plt.savefig("task2_loss_curve.png", dpi=300)
plt.close()

print("\nSaved Outputs:")
print(" - task2_confusion_matrix.png")
print(" - task2_roc_curve.png")
print(" - task2_accuracy_curve.png")
print(" - task2_loss_curve.png")


