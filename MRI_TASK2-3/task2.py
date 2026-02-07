# ============================================================
# TASK 2 — Binary Neurological Classification (CN vs AD)
# SUBJECT-LEVEL DEEP LEARNING MODEL (STRICTLY COMPLIANT)
# ============================================================

import os
import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix
)

# ---------------- CONFIG ----------------
DATASET = "/nlsasfs/home/gpucbh/vyakti7/AI_Project/MRI_TASK1/processed_dataset"
IMG_SIZE = 224
NUM_SLICES = 32
EPOCHS = 20
BATCH = 8

# ---------------- MRI LOADER ----------------
def load_subjects(split):
    X, y = [], []
    classes = {"CN": 0, "AD": 1}

    for cname, label in classes.items():
        folder = os.path.join(DATASET, split, cname)
        if not os.path.exists(folder):
            continue

        for f in os.listdir(folder):
            if f.endswith(".nii.gz"):
                img = nib.load(os.path.join(folder, f)).get_fdata()

                # Normalize MRI
                img = (img - np.mean(img)) / (np.std(img) + 1e-8)

                mid = img.shape[2] // 2
                slices = []

                for i in range(mid - NUM_SLICES//2, mid + NUM_SLICES//2):
                    if 0 <= i < img.shape[2]:
                        sl = cv2.resize(img[:, :, i], (IMG_SIZE, IMG_SIZE))
                        sl = np.stack([sl]*3, axis=-1)
                        slices.append(sl)

                X.append(np.array(slices))
                y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)

# ---------------- LOAD DATA ----------------
X_train, y_train = load_subjects("training")
X_val, y_val     = load_subjects("validation")
X_test, y_test   = load_subjects("testing")

# Scale to [0,1]
X_train /= np.max(X_train)
X_val   /= np.max(X_train)
X_test  /= np.max(X_train)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

# ---------------- MODEL ----------------

# CNN backbone (feature extractor)
base_cnn = applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_cnn.trainable = False

# Input: (subjects, slices, H, W, C)
inp = layers.Input(shape=(NUM_SLICES, IMG_SIZE, IMG_SIZE, 3))

# Apply CNN on each slice
x = layers.TimeDistributed(base_cnn)(inp)
x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)

# SUBJECT-LEVEL AGGREGATION (KEY STEP)
x = layers.GlobalAveragePooling1D()(x)

# Classifier
x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inp, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# ---------------- TRAIN ----------------
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH
)

# ---------------- EVALUATION ----------------
print("\nFINAL TEST RESULTS (SUBJECT LEVEL)")

prob = model.predict(X_test).flatten()
pred = (prob > 0.5).astype(int)

print("Balanced Accuracy:",
      balanced_accuracy_score(y_test, pred))

print("AUC:", roc_auc_score(y_test, prob))

print("\nClassification Report:")
print(classification_report(y_test, pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
