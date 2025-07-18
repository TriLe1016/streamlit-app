import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from main import LitModel  # Make sure LitModel is imported properly

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stFileUploader {
        background-color: #ffffff;
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 10px;
    }
    .result-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .header {
        color: #2E7D32;
        font-size: 36px;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        color: #424242;
        font-size: 18px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load trained model
model = LitModel.load_from_checkpoint("D:\code (1)\code\mobilenet-best.ckpt", input_shape=(3, 224, 224), num_classes=10)
model.to("cpu")
model.eval()

# Define transform (same as test_transform used in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define label mapping
class_names = ['bacterial_leaf_blight',
               'bacterial_leaf_streak',
               'bacterial_panicle_blight',
               'blast',
               'brown_spot',
               'dead_heart',
               'downy_mildew',
               'hispa',
               'normal',
               'tungro']

# App title and description
st.markdown('<div class="header">🌾 Paddy Disease Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload an image of a rice leaf to detect potential diseases with our AI model.</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📤 Upload a leaf image (PNG, JPG, JPEG)", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    # Display uploaded image
    col1, col2 = st.columns([1, 1])
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process image with spinner
    with col2:
        with st.spinner("🧠 Analyzing image..."):
            # Preprocess image
            input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred].item()

        # Display results in a styled box
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f"### Predicted Disease: **{class_names[pred]}**")
        st.markdown(f"### Confidence Score: **{confidence:.2%}**")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; color: #757575; margin-top: 40px;'>
    Powered by Streamlit & PyTorch | Developed with ❤️ for agricultural innovation
</div>
""", unsafe_allow_html=True)