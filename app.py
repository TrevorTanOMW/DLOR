import streamlit as st
import numpy as np
from PIL import Image
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fish Species Classifier",
    page_icon="🐟",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

/* Remove default Streamlit padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 680px;
}

/* Upload box */
.stFileUploader > div {
    border: 1.5px dashed #d1d5db !important;
    border-radius: 12px !important;
    background: #fafafa !important;
    transition: border-color 0.2s;
}
.stFileUploader > div:hover {
    border-color: #6b7280 !important;
}

/* Result card */
.result-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin-top: 1.2rem;
    text-align: center;
}

.result-card .species {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #0f172a;
    margin: 0;
    letter-spacing: -0.5px;
}

.result-card .confidence {
    font-size: 1rem;
    color: #64748b;
    margin-top: 0.3rem;
    font-weight: 300;
}

.confidence-bar-bg {
    background: #e2e8f0;
    border-radius: 99px;
    height: 8px;
    margin: 1rem auto 0;
    max-width: 320px;
}

.confidence-bar-fill {
    background: #0f172a;
    border-radius: 99px;
    height: 8px;
    transition: width 0.6s ease;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 1.5rem 0;
}

/* Hide Streamlit branding */
#MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Black Sea Sprat",
    "Gilt-Head Bream",
    "Hourse Mackerel",
    "Red Mullet",
    "Red Sea Bream",
    "Sea Bass",
    "Shrimp",
    "Striped Red Mullet",
    "Trout",
]

IMG_SIZE = (224, 224)

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """
    Load TFLite model.
    Swap MODEL_PATH to your actual .tflite file when ready.
    """
    MODEL_PATH = "model.tflite"  # ← replace with your model path

    if not os.path.exists(MODEL_PATH):
        return None  # placeholder mode

    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


def predict(interpreter, img_array):
    """Run inference with TFLite interpreter."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]  # shape: (9,)


def remove_background(image: Image.Image) -> Image.Image:
    from rembg import remove
    result = remove(image.convert("RGB"))
    background = Image.new("RGB", result.size, (255, 255, 255))
    background.paste(result, mask=result.split()[3])
    return background


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Remove background, resize, normalize, add batch dim."""
    img = remove_background(image)
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("## 🐟 Fish Species Classifier")
st.markdown(
    "<p style='color:#64748b; margin-top:-0.5rem; margin-bottom:1.5rem;'>"
    "Upload a fish image to identify its species.</p>",
    unsafe_allow_html=True,
)

interpreter = load_model()

if interpreter is None:
    st.info(
        "**Placeholder mode** — no `model.tflite` found. "
        "Drop your trained model file next to `app.py` and restart.",
        icon="ℹ️",
    )

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded_file:
    image = Image.open(uploaded_file)

    # Show image — constrained width
    st.image(image, use_container_width=True, caption="Uploaded image")

    st.markdown("<hr>", unsafe_allow_html=True)

    if interpreter is None:
        # ── Demo / placeholder prediction ─────────────────────────────────────
        st.markdown("**Prediction (demo — random placeholder)**")
        rng = np.random.default_rng(seed=42)
        probs = rng.dirichlet(np.ones(9))
        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        predicted_class = CLASS_NAMES[top_idx]
        note = " *(placeholder)*"
    else:
        # ── Real model inference ───────────────────────────────────────────────
        with st.spinner("Analyzing..."):
            img_array = preprocess_image(image)
            probs = predict(interpreter, img_array)
        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        predicted_class = CLASS_NAMES[top_idx]
        note = ""

    # Result card
    bar_width = int(top_conf * 100)
    st.markdown(f"""
    <div class="result-card">
        <p class="species">{predicted_class}{note}</p>
        <p class="confidence">{top_conf*100:.1f}% confidence</p>
        <div class="confidence-bar-bg">
            <div class="confidence-bar-fill" style="width:{bar_width}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Subtle footnote
    st.markdown(
        f"<p style='text-align:center; color:#94a3b8; font-size:0.78rem; margin-top:1rem;'>"
        f"Model: ResNet50V2 · 9 classes · 224×224 input</p>",
        unsafe_allow_html=True,
    )
