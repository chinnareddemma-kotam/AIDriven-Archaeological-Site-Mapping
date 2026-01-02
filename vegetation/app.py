import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Vegetation Segmentation", layout="wide")
st.title("Vegetation Segmentation Dashboard")

# ---------------- LOAD MODEL ----------------
model = YOLO(
    r"C:\Users\welcome\Desktop\vegtation_segmentation\runs\segment\train\weights\best.pt"
)

# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.header("⚙️ Settings")

resize_dim = st.sidebar.selectbox(
    "YOLO Input Size",
    [256, 320, 416, 512, 640],
    index=4
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.05, 0.9, 0.1, 0.05
)

veg_threshold = st.sidebar.slider(
    "Vegetation Dominance Threshold (%)",
    0, 100, 50
)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    # ---------------- READ ORIGINAL IMAGE ----------------
    image_bytes = uploaded_file.read()
    original_img = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    H, W = original_img.shape[:2]

    # ---------------- RESIZE ONLY FOR YOLO ----------------
    yolo_img = cv2.resize(original_img_rgb, (resize_dim, resize_dim))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼 Original Image")
        st.image(original_img_rgb, use_container_width=True)

    # ---------------- YOLO INFERENCE ----------------
    results = model(
        yolo_img,
        conf=confidence,
        classes=[0]  # vegetation class
    )[0]

    if results.masks is None:
        st.warning("⚠️ No vegetation detected")
    else:
        # ---------------- MASK HANDLING (ACCURATE) ----------------
        masks = results.masks.data.cpu().numpy()

        combined_mask = np.zeros((H, W), dtype=np.uint8)

        for mask in masks:
            # Resize mask back to ORIGINAL image size
            mask = cv2.resize(
                mask,
                (W, H),
                interpolation=cv2.INTER_NEAREST
            )

            combined_mask = np.maximum(
                combined_mask,
                (mask > 0.3).astype(np.uint8)
            )

        # ---------------- OVERLAY ----------------
        overlay = original_img_rgb.copy()
        overlay[combined_mask == 1] = (0, 255, 0)

        output = cv2.addWeighted(
            original_img_rgb, 0.7,
            overlay, 0.3, 0
        )

        with col2:
            st.subheader("🌱 Segmentation Output")
            st.image(output, use_container_width=True)

        # ---------------- ACCURATE VEGETATION % ----------------
        veg_pixels = np.count_nonzero(combined_mask)
        total_pixels = H * W
        veg_percent = (veg_pixels / total_pixels) * 100

        st.markdown("### 📊 Accurate Vegetation Analysis")
        st.metric("🌱 Vegetation Coverage", f"{veg_percent:.2f}%")

        if veg_percent >= veg_threshold:
            st.success("✅ Vegetation Dominant Area")
        else:
            st.warning("⚠️ Non-Vegetation Dominant Area")
else:
    st.info("👆 Upload an image to start segmentation")
