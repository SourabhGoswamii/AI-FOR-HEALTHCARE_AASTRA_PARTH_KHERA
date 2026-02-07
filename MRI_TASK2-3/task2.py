# ============================================================
# TASK 2 FINAL — AI NEUROLOGICAL CLASSIFICATION (CN vs AD)
# FULL GPU + BALANCED TRAINING + ALL METRICS + CURVES
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# ================= GPU BOOST =================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

print("GPU Active:", gpus)

# ================= CONFIG =================
DATASET = "/nlsasfs/home/gpucbh/vyakti7/AI_Project/MRI_TASK1/processed_dataset"
IMG_SIZE = 160
NUM_SLICES = 32
EPOCHS = 40
BATCH = 2

# ================= LOAD MRI =================
def load_subjects(split):
    X, y = [], []
    classes = {"CN":0, "AD":1}

    for cname,label in classes.items():
        folder = os.path.join(DATASET, split, cname)
        if not os.path.exists(folder):
            continue

        for f in os.listdir(folder):
            if f.endswith(".nii.gz"):
                img = nib.load(os.path.join(folder,f)).get_fdata()

                img = (img - np.mean(img)) / (np.std(img)+1e-8)

                mid = img.shape[2]//2
                slices=[]

                for i in range(mid-NUM_SLICES//2, mid+NUM_SLICES//2):
                    if 0<=i<img.shape[2]:
                        sl = cv2.resize(img[:,:,i],(IMG_SIZE,IMG_SIZE))
                        sl = np.stack([sl]*3,axis=-1)
                        slices.append(sl)

                X.append(np.array(slices))
                y.append(label)

    return np.array(X,dtype=np.float32), np.array(y)

# ================= LOAD DATA =================
print("\nLoading dataset...")
X_train,y_train = load_subjects("training")
X_val,y_val     = load_subjects("validation")
X_test,y_test   = load_subjects("testing")

print("Train:",X_train.shape," Val:",X_val.shape," Test:",X_test.shape)

print("\nClass distribution:")
print("Train CN:",np.sum(y_train==0))
print("Train AD:",np.sum(y_train==1))

# ================= CLASS WEIGHTS =================
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: weights[0], 1: weights[1]}
print("Class weights:",class_weights)

# ================= MODEL =================
inp = layers.Input(shape=(NUM_SLICES,IMG_SIZE,IMG_SIZE,3))

def small_cnn():
    m = models.Sequential([
        layers.Conv2D(32,3,activation='relu',padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64,3,activation='relu',padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(128,3,activation='relu',padding='same'),
        layers.GlobalAveragePooling2D()
    ])
    return m

cnn = small_cnn()
x = layers.TimeDistributed(cnn)(inp)
x = layers.GlobalAveragePooling1D()(x)

x = layers.Dense(128,activation='relu')(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(1,activation='sigmoid',dtype='float32')(x)

model = models.Model(inp,out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4),
    loss="binary_crossentropy",
    metrics=["accuracy",tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# ================= TRAIN =================
print("\nTraining started...\n")

history = model.fit(
    X_train,y_train,
    validation_data=(X_val,y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    class_weight=class_weights
)

# ================= FINAL EVALUATION =================
print("\n\n================ MODEL PERFORMANCE REPORT ================")

prob = model.predict(X_test).flatten()

THRESH = 0.40
pred = (prob > THRESH).astype(int)

# ===== METRICS =====
bal_acc = balanced_accuracy_score(y_test,pred)
auc = roc_auc_score(y_test,prob)
macro_f1 = f1_score(y_test,pred,average='macro')
precision = precision_score(y_test,pred,zero_division=0)
recall = recall_score(y_test,pred,zero_division=0)

print("\n📊 BASIC METRICS")
print("--------------------------------")
print(f"Balanced Accuracy : {bal_acc:.3f}")
print(f"AUC Score         : {auc:.3f}")
print(f"Macro F1 Score    : {macro_f1:.3f}")
print(f"Precision         : {precision:.3f}")
print(f"Recall            : {recall:.3f}")

print("\n📋 Classification Report")
print("--------------------------------")
print(classification_report(y_test,pred,zero_division=0))

# ===== CONFUSION MATRIX =====
print("\n🧠 Confusion Matrix")
cm = confusion_matrix(y_test,pred)
print(cm)

plt.figure(figsize=(5,4))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
            xticklabels=["CN","AD"],
            yticklabels=["CN","AD"])
plt.title("Confusion Matrix")
plt.show()

# ===== ROC =====
print("\n📈 ROC Curve")
fpr,tpr,_ = roc_curve(y_test,prob)

plt.figure()
plt.plot(fpr,tpr,label=f"AUC={auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve")
plt.legend()
plt.show()

# ===== TRAIN CURVES =====
print("\n📉 Training Curves")

plt.figure()
plt.plot(history.history['accuracy'],label="Train")
plt.plot(history.history['val_accuracy'],label="Val")
plt.title("Accuracy Curve")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'],label="Train")
plt.plot(history.history['val_loss'],label="Val")
plt.title("Loss Curve")
plt.legend()
plt.show()

print("\n================ CHECKLIST ================")
print("✔ Balanced Accuracy")
print("✔ AUC")
print("✔ Macro F1")
print("✔ Precision & Recall")
print("✔ Confusion Matrix")
print("✔ ROC Curve")
print("✔ Training Curve")
print("✔ Validation Curve")
print("ALL REQUIREMENTS DONE")
