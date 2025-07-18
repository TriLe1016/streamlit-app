import os
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from main import LitModel

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Paddy Disease Classifier",
    page_icon="🌾",
    layout="centered",
)

# ------------------- CUSTOM CSS -------------------
st.markdown("""
<style>
    body {
        background-color: #f4f6f7;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stFileUploader {
        border: 2px dashed #2e7d32;
        border-radius: 12px;
        padding: 20px;
        background-color: #ffffff;
    }
    .result-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 30px;
        text-align: center;
    }
    .header {
        color: #2e7d32;
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .subheader {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- LOAD MODEL -------------------
ckpt_path = os.path.join(os.path.dirname(__file__), "mobilenet-best.ckpt")
model = LitModel.load_from_checkpoint(ckpt_path, input_shape=(3, 224, 224), num_classes=10)
model.to("cpu")
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class_names = [
    'bacterial_leaf_blight',
    'bacterial_leaf_streak',
    'bacterial_panicle_blight',
    'blast',
    'brown_spot',
    'dead_heart',
    'downy_mildew',
    'hispa',
    'normal',
    'tungro'
]

# ------------------- UI HEADER -------------------
st.markdown('<div class="header">🌾 Paddy Disease Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload an image of a rice leaf to identify possible diseases using AI.</div>', unsafe_allow_html=True)

# ------------------- FILE UPLOAD -------------------
uploaded_file = st.file_uploader("📤 Upload a leaf image (PNG, JPG, JPEG)", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    with st.spinner("🧠 Analyzing with AI model..."):
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f"### 🩺 **Prediction:** `{class_names[pred]}`")
    st.markdown(f"### 🔍 **Confidence:** `{confidence:.2%}`")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- FOOTER -------------------
st.markdown("""
<hr style="margin-top: 50px;"/>
<div style='text-align: center; color: #888; font-size: 14px; margin-top: 20px;'>
    © 2025 - Developed with ❤️ using Streamlit & PyTorch for smart agriculture 🌱
</div>
""", unsafe_allow_html=True)
