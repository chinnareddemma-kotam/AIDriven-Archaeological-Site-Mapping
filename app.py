import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Soil Detection | YOLOv8",
    layout="wide"
)

st.title("🌍 Soil Type Detection Dashboard")
st.markdown("Upload an image to detect **Soil Types** using YOLOv8")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(r"C:\Users\welcome\Desktop\aidriven-archaeological-site-mapping\qwerty\runs\detect\train\weights\best.pt"):
        st.error("❌ Model file best.pt not found")
        st.stop()
    return YOLO(r"C:\Users\welcome\Desktop\aidriven-archaeological-site-mapping\qwerty\runs\detect\train\weights\best.pt")

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.25,
    step=0.05
)

img_size = st.sidebar.selectbox(
    "Image Size",
    [320, 416, 512, 640],
    index=3
)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼 Original Image")
        st.image(image, use_container_width=True)

    # YOLO inference
    with st.spinner("🔍 Detecting soil type..."):
        results = model.predict(
            source=img_np,
            conf=conf_threshold,
            imgsz=img_size,
            device="cpu"
        )[0]

    # Draw results
    annotated_img = results.plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("🧪 Detection Output")
        st.image(annotated_img, use_container_width=True)

    # ---------------- DETECTION DETAILS ----------------
    st.markdown("### 📊 Detection Details")

    if results.boxes is None or len(results.boxes) == 0:
        st.warning("⚠️ No soil detected")
    else:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            st.write(
                f"🟢 **{class_name}** — Confidence: **{conf:.2f}**"
            )

else:
    st.info("👆 Upload an image to start detection")
