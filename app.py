import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page config
st.set_page_config(page_title="Soil Detection", layout="centered")

st.title("🌍 Soil Type Detection")
st.write("Upload an image to detect soil type")

# Load model
@st.cache_resource
def load_model():
    return YOLO("model/best.pt")

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Soil"):
        results = model.predict(image, conf=0.4)

        result_img = results[0].plot()
        st.image(result_img, caption="Detection Result", use_container_width=True)
