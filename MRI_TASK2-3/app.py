import streamlit as st
import numpy as np
import nibabel as nib
import tensorflow as tf
from scipy.ndimage import zoom
import tempfile
import warnings
import os

warnings.filterwarnings("ignore")

# =========================================
# PAGE CONFIG & CSS
# =========================================
st.set_page_config(
    page_title="AI Neurological Screening",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
body { background: linear-gradient(to right, #0f2027, #203a43, #2c5364); }
.big-title { font-size: 42px; font-weight: bold; text-align: center; color: #00e6e6; }
.card { background-color: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; margin-top: 20px; box-shadow: 0px 0px 20px rgba(0,255,255,0.2); }
.result-box { font-size: 20px; font-weight: bold; padding: 15px; border-radius: 10px; background: rgba(0,255,200,0.1); }
</style>
""", unsafe_allow_html=True)

# =========================================
# 1. MODEL SELECTION (SIDEBAR)
# =========================================
st.sidebar.title("⚙️ Settings")
task_option = st.sidebar.radio(
    "Select Screening Task:",
    ["Binary (CN vs AD)", "Multi-Class (CN vs MCI vs AD)"]
)

# Define configurations based on selection
MODEL_DIR = "/nlsasfs/home/gpucbh/vyakti7/AI_Project/MRI_TASK2-3/models"

if "Binary" in task_option:
    MODEL_FILENAME = "task2_highest_accuracy_2dcnn.h5"
    CLASS_NAMES = ["Cognitively Normal (CN)", "Alzheimer's Disease (AD)"]
    st.sidebar.info("Loading Binary Model (2 Classes)")
else:
    MODEL_FILENAME = "task3_cn_mci_ad_model.h5"
    CLASS_NAMES = ["Cognitively Normal (CN)", "Mild Cognitive Impairment (MCI)", "Alzheimer's Disease (AD)"]
    st.sidebar.info("Loading Multi-Class Model (3 Classes)")

# =========================================
# 2. MODEL LOADING
# =========================================
@st.cache_resource
def load_selected_model(filename):
    model_path = os.path.join(MODEL_DIR, filename)
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
        
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model based on the user's choice
model = load_selected_model(MODEL_FILENAME)

# =========================================
# 3. PREPROCESSING FUNCTIONS
# =========================================
TARGET_SHAPE = (128, 128, 128)

def preprocess_mri(nifti_path):
    img = nib.load(nifti_path)
    img = nib.as_closest_canonical(img)
    x = img.get_fdata()
    
    # Ensure 3D
    if x.ndim == 4: x = x[..., 0]

    # Skull Strip
    x = x * (x > np.mean(x))

    # Spatial Normalize (Resize)
    factors = [t / s for t, s in zip(TARGET_SHAPE, x.shape)]
    x = zoom(x, factors, order=1)

    # Intensity Normalize
    x = np.nan_to_num(x)
    lo, hi = np.percentile(x, (1, 99))
    x = np.clip(x, lo, hi)
    x = ((x - lo) / (hi - lo + 1e-8)).astype(np.float32)

    return x

def predict_patient(volume, model):
    slices = []
    # Extract all slices
    for i in range(volume.shape[2]):
        slice_ = volume[:, :, i]
        # Skip empty slices (optional optimization)
        if np.max(slice_) > 0:
            slice_ = np.expand_dims(slice_, axis=-1)
            slices.append(slice_)

    if len(slices) == 0: return np.zeros((1, len(CLASS_NAMES)))

    slices = np.array(slices)
    preds = model.predict(slices, verbose=0)
    
    # Average prediction across all slices
    patient_pred = preds.mean(axis=0)
    return patient_pred

# =========================================
# 4. MAIN UI & LOGIC
# =========================================
st.markdown('<p class="big-title">🧠 AI Neurological Screening System</p>', unsafe_allow_html=True)
st.markdown(f"### Current Mode: **{task_option}**")

uploaded_file = st.file_uploader("Upload T1-weighted MRI (.nii or .nii.gz)", type=["nii", "nii.gz"])

if uploaded_file:
    # Fix for file extension error
    suffix = ".nii.gz" if uploaded_file.name.endswith(".gz") else ".nii"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner("🧠 Preprocessing MRI Scan..."):
            volume = preprocess_mri(tmp_path)

        # Preview Middle Slice
        mid_slice = volume[:, :, volume.shape[2]//2]
        st.image(mid_slice.T, caption="MRI Preview (Axial Slice)", use_column_width=True, clamp=True)

        with st.spinner(f"🤖 Analyzing with {MODEL_FILENAME}..."):
            probs = predict_patient(volume, model)

        # Get Prediction
        pred_idx = np.argmax(probs)
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        # Risk Logic
        if "Alzheimer" in pred_class:
            risk = "🔴 High Risk"
        elif "MCI" in pred_class:
            risk = "🟡 Moderate Risk"
        else:
            risk = "🟢 Low Risk"

        # Display Results
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.success("✅ Screening Completed")
        st.markdown(f'<div class="result-box">Prediction: {pred_class}</div>', unsafe_allow_html=True)
        st.write(f"**Confidence Score:** {confidence:.2f}")
        st.write(f"**Risk Level:** {risk}")
        st.progress(int(confidence * 100))
        st.markdown('</div>', unsafe_allow_html=True)

        # Cleanup
        os.remove(tmp_path)

    except Exception as e:
        st.error(f"Error processing MRI: {e}")