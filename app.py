"""Streamlit demo for concrete crack classification."""
from pathlib import Path

import streamlit as st
from PIL import Image

from src.predict import load_predictor, predict_image

st.set_page_config(page_title="Concrete Crack Detection", page_icon="🧱")
st.title("Concrete Crack Detection")
st.write("Upload a concrete surface image to classify it as crack or negative.")

checkpoint = Path("outputs/best_model.pth")
if not checkpoint.exists():
    st.warning("No trained checkpoint found. Train the model first using the README instructions.")
    st.stop()

@st.cache_resource
def get_predictor():
    return load_predictor(str(checkpoint))

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded concrete image", use_container_width=True)
    label, confidence = predict_image(image, get_predictor())
    st.subheader(f"Prediction: {label}")
    st.metric("Model confidence", f"{confidence * 100:.1f}%")
    st.caption("This is an image classifier output, not an engineering assessment or structural safety rating.")
