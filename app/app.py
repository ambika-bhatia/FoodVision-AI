from PIL import Image
import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Food Vision AI",
    page_icon="🔍",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .stApp {
        background-color: #0a0e14;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    .block-container {
        padding-top: 2.5rem;
        max-width: 700px;
    }

    /* Header */
    .app-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 1px;
        margin-bottom: 2px;
    }
    .app-title span {
        color: #00e5a0;
    }
    .app-subtitle {
        text-align: center;
        color: #6b7684;
        font-size: 0.9rem;
        margin-bottom: 30px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Scan frame container */
    .scan-frame {
        position: relative;
        border: 1px solid #1c2530;
        border-radius: 12px;
        background: #0e131c;
        padding: 30px;
    }
    .scan-frame::before,
    .scan-frame::after,
    .corner-tl, .corner-tr, .corner-bl, .corner-br {
        content: '';
        position: absolute;
        width: 22px;
        height: 22px;
        border-color: #00e5a0;
    }
    .corner-tl { top: -1px; left: -1px; border-top: 3px solid #00e5a0; border-left: 3px solid #00e5a0; border-radius: 8px 0 0 0; }
    .corner-tr { top: -1px; right: -1px; border-top: 3px solid #00e5a0; border-right: 3px solid #00e5a0; border-radius: 0 8px 0 0; }
    .corner-bl { bottom: -1px; left: -1px; border-bottom: 3px solid #00e5a0; border-left: 3px solid #00e5a0; border-radius: 0 0 0 8px; }
    .corner-br { bottom: -1px; right: -1px; border-bottom: 3px solid #00e5a0; border-right: 3px solid #00e5a0; border-radius: 0 0 8px 0; }

    div[data-testid="stFileUploader"] section {
        border: 1px dashed #2a3542;
        border-radius: 10px;
        background-color: #0e131c;
    }
    div[data-testid="stFileUploader"] label {
        color: #8b93a1 !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] span,
    div[data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #b8c0cc !important;
    }
    /* Status line */
    .status-line {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #00e5a0;
        font-size: 0.8rem;
        font-family: monospace;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }
    .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00e5a0;
    }

    /* Result panel */
    .result-panel {
        background: #0e131c;
        border: 1px solid #1c2530;
        border-left: 3px solid #00e5a0;
        border-radius: 8px;
        padding: 20px 24px;
        margin-top: 18px;
    }
    .result-label {
        font-size: 0.75rem;
        color: #6b7684;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .result-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #ffffff;
        text-transform: capitalize;
        margin-bottom: 4px;
    }
    .result-confidence {
        color: #00e5a0;
        font-family: monospace;
        font-size: 0.95rem;
    }

    .other-label {
        font-size: 0.75rem;
        color: #6b7684;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 24px 0 10px 0;
    }
    .other-row {
        display: flex;
        justify-content: space-between;
        color: #8b93a1;
        font-family: monospace;
        font-size: 0.85rem;
        padding: 6px 0;
        border-bottom: 1px solid #1c2530;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="app-title">FOOD<span>VISION</span> AI</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Snap a photo, get an instant identification.</div>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "food_classifier_v2.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

class_names = [
    "apple_pie", "bibimbap", "cannoli", "edamame", "falafel",
    "french_toast", "ice_cream", "ramen", "sushi", "tiramisu"
]

# ---------------- SCAN FRAME ----------------
st.markdown('<div class="scan-frame"><div class="corner-tl"></div><div class="corner-tr"></div><div class="corner-bl"></div><div class="corner-br"></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a food image to scan",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:

    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Running inference..."):
        prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    st.markdown('<div class="status-line"><div class="dot"></div>SCAN COMPLETE</div>', unsafe_allow_html=True)

    st.markdown(f"""
        <div class="result-panel">
            <div class="result-label">Identified as</div>
            <div class="result-value">{predicted_class.replace('_', ' ')}</div>
            <div class="result-confidence">confidence: {confidence:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

    top_indices = np.argsort(prediction[0])[::-1][1:4]
    other_rows = ""
    for idx in top_indices:
        label = class_names[idx].replace('_', ' ')
        score = float(prediction[0][idx]) * 100
        other_rows += f'<div class="other-row"><span>{label}</span><span>{score:.1f}%</span></div>'

    st.markdown(f'<div class="other-label">Other matches</div>{other_rows}', unsafe_allow_html=True)

else:
    st.markdown('<div class="status-line"><div class="dot"></div>AWAITING INPUT</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(
    "<p style='text-align:center; color:#3d4552; font-size:0.75rem; margin-top:30px; font-family:monospace;'>MobileNetV2 · Transfer Learning · TensorFlow</p>",
    unsafe_allow_html=True
)