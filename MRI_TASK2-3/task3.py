import streamlit as st
import numpy as np
import nibabel as nib
import tensorflow as tf
from scipy.ndimage import zoom
import tempfile
import warnings

warnings.filterwarnings("ignore")

# ================= CONFIG =================
TARGET_SHAPE = (128, 128, 128)
BINARY_CLASSES = ["CN", "AD"]
MULTI_CLASSES = ["CN", "MCI", "AD"]

# ================= PREPROCESSING =================
def preprocess_mri(path):
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    x = img.get_fdata()

    if x.ndim == 4:
        x = x[..., 0]

    # Simple skull stripping
    x = x * (x > np.mean(x))

    # Resize to fixed shape
    factors = [t / s for t, s in zip(TARGET_SHAPE, x.shape)]
    x = zoom(x, factors, order=1)

    # Intensity normalization
    lo, hi = np.percentile(x, (1, 99))
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-6)

    return x.astype(np.float32)

# ================= SLICE AGGREGATION =================
def predict_subject(volume, model, multiclass=True):
    slices = []

    for i in range(volume.shape[2]):
        s = volume[:, :, i]
        if np.mean(s) > 0.05:  # brain-only slices
            slices.append(np.expand_dims(s, axis=-1))

    slices = np.array(slices)
    preds = model.predict(slices, verbose=0)

    # Top-K aggregation
    k = max(8, preds.shape[0] // 6)
    topk = np.sort(preds, axis=0)[-k:]
    subject_pred = topk.mean(axis=0)

    return subject_pred

# ================= STREAMLIT UI =================
st.set_page_config(
    page_title="AI Neurological Screening",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI-Driven Neurological Screening System")
st.subheader("T1-Weighted MRI Based Screening Tool")

st.markdown("""
⚠️ **Disclaimer**  
This system is a **screening & decision-support tool only**.  
It is **NOT a medical diagnosis** and must not replace professional clinical judgment.
""")

# ================= MODE SELECTION =================
mode = st.radio(
    "Select Screening Mode:",
    ["Binary Classification (CN vs AD)", "Multi-Class Classification (CN vs MCI vs AD)"]
)

# ================= LOAD MODELS =================
@st.cache_resource
def load_binary_model():
    return tf.keras.models.load_model("models/binary_cnn_model.h5")

@st.cache_resource
def load_multiclass_model():
    return tf.keras.models.load_model("models/multiclass_cnn_model.h5")

model = load_binary_model() if "Binary" in mode else load_multiclass_model()

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader(
    "Upload T1-Weighted MRI (.nii or .nii.gz)",
    type=["nii", "nii.gz"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        mri_path = tmp.name

    with st.spinner("🔄 Preprocessing MRI scan..."):
        volume = preprocess_mri(mri_path)

    with st.spinner("🧠 Running AI screening..."):
        probs = predict_subject(volume, model, multiclass=("Multi" in mode))

    # ================= RESULTS =================
    if "Binary" in mode:
        prob_ad = float(probs)
        label = "AD" if prob_ad >= 0.5 else "CN"
        confidence = prob_ad if label == "AD" else 1 - prob_ad
        confidence *= 0.85  # ethical calibration

        risk = "High Risk" if label == "AD" else "Low Risk"

        st.success("✅ Screening Complete")
        st.markdown("### 🧾 Screening Result")
        st.write("**Predicted Condition:**", label)
        st.write("**Risk Level:**", risk)
        st.write("**Confidence Score:**", round(confidence, 2))
        st.progress(int(confidence * 100))

    else:
        cls_idx = int(np.argmax(probs))
        label = MULTI_CLASSES[cls_idx]
        confidence = float(np.max(probs)) * 0.85

        if label == "CN":
            risk = "Low Risk"
        elif label == "MCI":
            risk = "Moderate Risk"
        else:
            risk = "High Risk"

        st.success("✅ Screening Complete")
        st.markdown("### 🧾 Screening Result")
        st.write("**Predicted Condition:**", label)
        st.write("**Risk Level:**", risk)
        st.write("**Confidence Score:**", round(confidence, 2))
        st.progress(int(confidence * 100))

        st.markdown("### 📊 Class Probabilities")
        for c, p in zip(MULTI_CLASSES, probs):
            st.write(f"{c}: {round(float(p), 3)}")

    st.markdown("""
🩺 **Important Note:**  
This output supports **early neurological screening & referral** only.  
Final diagnosis must be performed by a qualified medical professional.
""")
