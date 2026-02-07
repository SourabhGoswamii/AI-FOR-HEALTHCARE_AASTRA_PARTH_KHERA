import streamlit as st
import numpy as np
import nibabel as nib
import tensorflow as tf
from scipy.ndimage import zoom
import tempfile, os, warnings

warnings.filterwarnings("ignore")

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="AI Neurological Screening",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
body { background: linear-gradient(to right, #0f2027, #203a43, #2c5364); }
.title { font-size: 40px; font-weight: bold; text-align: center; color: #00ffff; }
.card { background: rgba(255,255,255,0.08); padding: 20px; border-radius: 15px; margin-top: 20px; }
.result { font-size: 22px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =========================================
# SIDEBAR
# =========================================
st.sidebar.title("⚙️ Screening Mode")

task = st.sidebar.radio(
    "Select Task",
    ["Binary (CN vs AD)", "Multi-Class (CN vs MCI vs AD)"]
)

MODEL_DIR = "models"

if "Binary" in task:
    MODEL_NAME = "task2_binary_cnn.h5"
    CLASS_NAMES = ["Cognitively Normal (CN)", "Alzheimer's Disease (AD)"]
else:
    MODEL_NAME = "task3_multiclass_cnn.h5"
    CLASS_NAMES = ["Cognitively Normal (CN)", "MCI", "Alzheimer's Disease (AD)"]

# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model(os.path.join(MODEL_DIR, MODEL_NAME))

# =========================================
# PREPROCESSING
# =========================================
TARGET_SHAPE = (128, 128, 128)

def preprocess_mri(path):
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    x = img.get_fdata()

    if x.ndim == 4:
        x = x[..., 0]

    # Skull stripping (simple threshold)
    x = x * (x > np.mean(x))

    # Resize
    factors = [t / s for t, s in zip(TARGET_SHAPE, x.shape)]
    x = zoom(x, factors, order=1)

    # Intensity normalization
    lo, hi = np.percentile(x, (1, 99))
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)

    return x.astype(np.float32)

# =========================================
# SLICE-WISE PREDICTION
# =========================================
def predict_patient(volume, model):
    slices = []

    for i in range(volume.shape[2]):
        sl = volume[:, :, i]
        if np.max(sl) > 0.05:
            sl = np.expand_dims(sl, axis=(-1, 0))  # (1,128,128,1)
            slices.append(sl)

    if len(slices) == 0:
        return np.zeros(len(CLASS_NAMES))

    slices = np.vstack(slices)
    preds = model.predict(slices, verbose=0)

    return preds.mean(axis=0)

# =========================================
# UI
# =========================================
st.markdown('<div class="title">🧠 AI Neurological Screening</div>', unsafe_allow_html=True)
st.markdown(f"### Mode: **{task}**")

uploaded = st.file_uploader("Upload T1-weighted MRI (.nii / .nii.gz)", type=["nii", "nii.gz"])

if uploaded:
    suffix = ".nii.gz" if uploaded.name.endswith(".gz") else ".nii"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        path = tmp.name

    with st.spinner("Preprocessing MRI..."):
        volume = preprocess_mri(path)

    # Show preview
    mid = volume[:, :, volume.shape[2] // 2]
    st.image(mid.T, caption="Axial MRI Slice", use_column_width=True, clamp=True)

    with st.spinner("Running AI Model..."):
        probs = predict_patient(volume, model)

    idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[idx]
    confidence = float(probs[idx])

    # Risk
    if "Alzheimer" in pred_class:
        risk = "🔴 High Risk"
    elif "MCI" in pred_class:
        risk = "🟡 Moderate Risk"
    else:
        risk = "🟢 Low Risk"

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.success("✅ Screening Completed")
    st.markdown(f"<p class='result'>Prediction: {pred_class}</p>", unsafe_allow_html=True)
    st.write(f"**Confidence:** {confidence:.2f}")
    st.write(f"**Risk Level:** {risk}")
    st.progress(int(confidence * 100))
    st.markdown('</div>', unsafe_allow_html=True)

    os.remove(path)
