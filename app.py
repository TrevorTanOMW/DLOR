import streamlit as st
import numpy as np
from PIL import Image
import os
import time

st.set_page_config(
    page_title="AquaSort — Fish Species Classifier",
    page_icon="",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: #0a0a0a;
    color: #e8e8e8;
}

.block-container {
    padding: 3rem 2rem 4rem;
    max-width: 720px;
}

.header {
    border-bottom: 1px solid #222;
    padding-bottom: 2rem;
    margin-bottom: 2rem;
}

.header-label {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: #555;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.header-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #f0f0f0;
    letter-spacing: -1px;
    line-height: 1;
}

.header-title span { color: #00e5a0; }

.header-sub {
    font-size: 0.72rem;
    color: #444;
    margin-top: 0.8rem;
    letter-spacing: 0.05em;
}

.status-bar {
    display: flex;
    gap: 2rem;
    padding: 0.9rem 0;
    border-top: 1px solid #1a1a1a;
    border-bottom: 1px solid #1a1a1a;
    margin-bottom: 2rem;
    flex-wrap: wrap;
}

.status-item { display: flex; flex-direction: column; gap: 0.2rem; }

.status-key {
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    color: #444;
    text-transform: uppercase;
}

.status-val {
    font-size: 0.78rem;
    color: #00e5a0;
    font-weight: 500;
}

.stFileUploader > div {
    border: 1px solid #222 !important;
    border-radius: 4px !important;
    background: #0f0f0f !important;
    transition: border-color 0.2s, background 0.2s;
}

.stFileUploader > div:hover {
    border-color: #00e5a0 !important;
    background: #0a1510 !important;
}

.divider {
    border: none;
    border-top: 1px solid #1a1a1a;
    margin: 1.5rem 0;
}

.result-panel {
    background: #0d0d0d;
    border: 1px solid #1e1e1e;
    border-left: 3px solid #00e5a0;
    border-radius: 4px;
    padding: 1.8rem 2rem;
    margin-top: 1rem;
}

.result-label {
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: #444;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.result-species {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #f0f0f0;
    letter-spacing: -0.5px;
    line-height: 1.1;
}

.result-confidence-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 1.2rem;
}

.result-conf-num {
    font-size: 0.85rem;
    color: #00e5a0;
    font-weight: 500;
    min-width: 55px;
}

.conf-track {
    flex: 1;
    height: 3px;
    background: #1e1e1e;
    border-radius: 2px;
    overflow: hidden;
}

.conf-fill {
    height: 3px;
    background: #00e5a0;
    border-radius: 2px;
}

.meta-row {
    display: flex;
    gap: 2rem;
    margin-top: 1.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid #1a1a1a;
    flex-wrap: wrap;
}

.meta-key {
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    color: #333;
    text-transform: uppercase;
}

.meta-val {
    font-size: 0.72rem;
    color: #555;
    margin-top: 0.15rem;
}

.footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #151515;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.footer-text {
    font-size: 0.6rem;
    color: #2a2a2a;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

div[data-testid="stImage"] img { border-radius: 4px; }
#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #222; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = [
    "Black Sea Sprat", "Gilt-Head Bream", "Hourse Mackerel",
    "Red Mullet", "Red Sea Bream", "Sea Bass",
    "Shrimp", "Striped Red Mullet", "Trout",
]

IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    MODEL_PATH = "model.tflite"
    if not os.path.exists(MODEL_PATH):
        return None
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def predict(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

def remove_background(image):
    from rembg import remove
    result = remove(image.convert("RGB"))
    background = Image.new("RGB", result.size, (255, 255, 255))
    background.paste(result, mask=result.split()[3])
    return background

def preprocess_image(image):
    img = remove_background(image)
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

interpreter = load_model()
status_color = "#00e5a0" if interpreter else "#ff4444"
status_text = "ONLINE" if interpreter else "OFFLINE"

st.markdown(f"""
<div class="header">
    <div class="header-label">Aquatic Species Identification System</div>
    <div class="header-title">Aqua<span>Sort</span></div>
    <div class="header-sub">Automated fish classification for industrial sorting pipelines</div>
</div>

<div class="status-bar">
    <div class="status-item">
        <span class="status-key">System</span>
        <span class="status-val" style="color:{status_color};">{status_text}</span>
    </div>
    <div class="status-item">
        <span class="status-key">Architecture</span>
        <span class="status-val">ResNet50V2</span>
    </div>
    <div class="status-item">
        <span class="status-key">Classes</span>
        <span class="status-val">9 species</span>
    </div>
    <div class="status-item">
        <span class="status-key">Test Accuracy</span>
        <span class="status-val">100.0%</span>
    </div>
    <div class="status-item">
        <span class="status-key">Quantization</span>
        <span class="status-val">INT8</span>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop image here or click to browse — JPG, PNG, WEBP accepted",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="visible",
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if interpreter is None:
        rng = np.random.default_rng(seed=42)
        probs = rng.dirichlet(np.ones(9))
        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        predicted_class = CLASS_NAMES[top_idx]
        inference_ms = 0
        mode_note = " <span style='font-size:0.85rem;color:#444;'>(demo)</span>"
    else:
        with st.spinner("Analyzing..."):
            t0 = time.time()
            img_array = preprocess_image(image)
            probs = predict(interpreter, img_array)
            inference_ms = int((time.time() - t0) * 1000)
        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        predicted_class = CLASS_NAMES[top_idx]
        mode_note = ""

    conf_pct = top_conf * 100

    st.markdown(f"""
    <div class="result-panel">
        <div class="result-label">Identified Species</div>
        <div class="result-species">{predicted_class}{mode_note}</div>
        <div class="result-confidence-row">
            <span class="result-conf-num">{conf_pct:.1f}%</span>
            <div class="conf-track">
                <div class="conf-fill" style="width:{int(conf_pct)}%;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Inference Time", f"{inference_ms} ms")
    with col2:
        st.metric("Input Size", "224×224 px")
    with col3:
        st.metric("Preprocessing", "BG + norm")
    with col4:
        st.metric("Confidence", f"{conf_pct:.2f}%")

st.markdown("""
<div class="footer">
    <span class="footer-text">AquaSort Industrial Classification System</span>
    <span class="footer-text">ResNet50V2 · TFLite INT8 · 9-class fish identifier</span>
</div>
""", unsafe_allow_html=True)
