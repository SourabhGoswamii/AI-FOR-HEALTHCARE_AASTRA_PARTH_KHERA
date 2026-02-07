import streamlit as st
import numpy as np
import nibabel as nib
import tensorflow as tf
from scipy.ndimage import zoom
import tempfile, os, warnings

warnings.filterwarnings("ignore")

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="AI Neurological Screening",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
body { background: linear-gradient(to right, #0f2027, #203a43, #2c5364); }
.title { font-size: 42px; font-weight: bold; text-align: center; color: #00ffff; }
.subtitle { text-align: center; color: #d0ffff; font-size: 18px; }
.card { background: rgba(255,255,255,0.1); padding: 25px; border-radius: 18px; margin-top: 25px; }
.metric { font-size: 20px; font-weight: bold; }
.footer { text-align:center; font-size:13px; color:#b0c4de; margin-top:30px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 AI Neurological Screening System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">MRI-based Early Neurological Risk Screening</div>', unsafe_allow_html=True)

st.markdown("""
⚠️ **Disclaimer**  
This system is a **screening & decision-support tool only**.  
It does **NOT provide medical diagnosis**.
""")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("⚙️ Screening Options")

task = st.sidebar.radio(
    "Select Screening Mode",
    ["Binary (CN vs AD)", "Multi-Class (CN vs MCI vs AD)"]
)

MODEL_DIR = "models"

if "Binary" in task:
    MODEL_NAME = "task2_binary_cnn.h5"
    CLASS_NAMES = ["Cognitively Normal (CN)", "Alzheimer’s Disease (AD)"]
else:
    MODEL_NAME = "task3_multiclass_cnn.h5"
    CLASS_NAMES = ["Cognitively Normal (CN)", "MCI", "Alzheimer’s Disease (AD)"]

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# ======================================================
# LOAD MODEL (SAFE)
# ======================================================
@st.cache_resource
def load_model_safe(path):
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)

model = load_model_safe(MODEL_PATH)

if model is None:
    st.error("❌ Trained model not found")
    st.info(f"Expected file: `{MODEL_PATH}`")
    st.info("➡️ Train the model using task2.py / task3.py and save it inside `models/` folder.")
    st.stop()

# ======================================================
# PREPROCESSING
# ======================================================
TARGET_SHAPE = (128, 128, 128)

def preprocess_mri(path):
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    x = img.get_fdata()

    if x.ndim == 4:
        x = x[..., 0]

    # Simple skull stripping
    x = x * (x > np.mean(x))

    # Resize
    factors = [t / s for t, s in zip(TARGET_SHAPE, x.shape)]
    x = zoom(x, factors, order=1)

    # Intensity normalization
    lo, hi = np.percentile(x, (1, 99))
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-6)

    return x.astype(np.float32)

# ======================================================
# SUBJECT-LEVEL PREDICTION
# ======================================================
def predict_subject(volume, model):
    slices = []

    for i in range(volume.shape[2]):
        sl = volume[:, :, i]
        if np.mean(sl) > 0.05:
            slices.append(sl[..., None][None, ...])

    slices = np.vstack(slices)
    preds = model.predict(slices, verbose=0)

    k = max(8, preds.shape[0] // 6)
    topk = np.sort(preds, axis=0)[-k:]
    return topk.mean(axis=0)

# ======================================================
# UI
# ======================================================
uploaded = st.file_uploader(
    "📤 Upload T1-Weighted MRI (.nii / .nii.gz)",
    type=["nii", "nii.gz"]
)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        mri_path = tmp.name

    with st.spinner("🔄 Preprocessing MRI scan..."):
        volume = preprocess_mri(mri_path)

    # Preview slice
    mid = volume[:, :, volume.shape[2] // 2]
    st.image(mid.T, caption="Axial MRI Slice Preview", use_column_width=True)

    with st.spinner("🧠 Running AI Screening..."):
        probs = predict_subject(volume, model)

    idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[idx]
    confidence = float(probs[idx])

    # Risk logic
    if "Alzheimer" in pred_class:
        risk = "🔴 High Risk"
    elif "MCI" in pred_class:
        risk = "🟡 Moderate Risk"
    else:
        risk = "🟢 Low Risk"

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.success("✅ Screening Completed")
    st.markdown(f"<div class='metric'>Prediction: {pred_class}</div>", unsafe_allow_html=True)
    st.write(f"**Risk Level:** {risk}")
    st.write(f"**Confidence Score:** {confidence:.2f}")
    st.progress(int(confidence * 100))
    st.markdown('</div>', unsafe_allow_html=True)

    os.remove(mri_path)

st.markdown('<div class="footer">AI-assisted neurological screening • Research use only</div>', unsafe_allow_html=True)
