import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# =============================
# Page config
# =============================
st.set_page_config(page_title="Soil Detection", layout="centered")
st.title("🌍 Soil Type Detection")
st.write("Upload an image to detect the soil type")

# =============================
# Load YOLO ONNX model
# =============================
@st.cache_resource
def load_model():
    return YOLO("best.onnx")

model = load_model()

# =============================
# Upload image
# =============================
uploaded_file = st.file_uploader(
    "Upload Soil Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    if st.button("Detect Soil"):
        results = model(
            image,
            conf=0.4,
            imgsz=640,
            device="cpu"
        )

        r = results[0]

        if len(r.boxes) == 0:
            st.warning("No soil detected")
        else:
            st.image(
                r.plot(),
                caption="✅ Detection Result",
                use_container_width=True
            )

            st.subheader("📊 Detected Soil Types")

            for cls, conf in zip(
                r.boxes.cls.cpu().numpy(),
                r.boxes.conf.cpu().numpy()
            ):
                cls = int(cls)
                st.write(
                    f"• {model.names[cls]} — {conf:.2f}"
                )
