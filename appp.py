import streamlit as st
from ultralytics import YOLO
from PIL import Image

# =============================
# Page config
# =============================
st.set_page_config(page_title="Soil Type Detection", layout="centered")
st.title("🌍 Soil Type Detection")
st.write("Upload an image to detect the soil type")

# =============================
# Soil class names (match your training)
# =============================
CLASS_NAMES = [
    "Alluvial Soil",  # 0
    "Black Soil",     # 1
    "Clay Soil",      # 2
    "Red Soil"        # 3
]

# =============================
# Load YOLO ONNX model
# =============================
@st.cache_resource
def load_model():
    return YOLO("best.onnx")  # your ONNX model path

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
        # Run inference
        results = model(
            image,
            conf=0.4,     # minimum confidence threshold
            imgsz=640,    # input image size for YOLO
            device="cpu"  # or "cuda" if GPU available
        )

        r = results[0]

        if len(r.boxes) == 0:
            st.warning("⚠️ No soil detected")
        else:
            # Show detection plot
            st.image(
                r.plot(),
                caption="✅ Detection Result",
                use_container_width=True
            )

            # Show detected soil types
            st.subheader("📊 Detected Soil Types")

            for cls, conf in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                cls = int(cls)
                st.write(f"• {CLASS_NAMES[cls]} — {conf:.2f}")
