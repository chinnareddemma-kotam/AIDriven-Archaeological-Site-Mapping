import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Soil Type Detection",
    page_icon="🌱",
    layout="centered"
)

st.title("🌍 Soil Type Detection")
st.markdown("Upload a soil image to detect the **soil type** using an ONNX model.")

# ---------------- LOAD ONNX MODEL ----------------
@st.cache_resource
def load_model():
    try:
        return ort.InferenceSession(
            "best.onnx",
            providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

session = load_model()

# ---------------- CLASS NAMES ----------------
# UPDATE this according to your dataset
CLASS_NAMES = [
    "Red Soil",
    "Black Soil",
    "Alluvial Soil",
    "Clay Soil"
]

# ---------------- IMAGE PREPROCESS ----------------
def preprocess(image, img_size=640):
    image = image.resize((img_size, img_size))
    image = np.array(image).astype(np.float32)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC → CHW
    image = np.expand_dims(image, axis=0)
    return image

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload Soil Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file and session is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect Soil Type"):
        with st.spinner("Running inference..."):
            input_tensor = preprocess(image)

            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_tensor})

        # ---------------- POSTPROCESS (YOLOv8 ONNX) ----------------
        preds = outputs[0][0]  # shape: (num_boxes, 85)

        conf_threshold = 0.4
        detections = []

        for pred in preds:
            confidence = pred[4]
            if confidence > conf_threshold:
                class_id = np.argmax(pred[5:])
                class_conf = pred[5 + class_id]
                detections.append((class_id, class_conf))

        # ---------------- DISPLAY RESULTS ----------------
        if detections:
            st.subheader("✅ Detected Soil Types")
            for cls_id, conf in detections:
                st.write(f"• **{CLASS_NAMES[cls_id]}** — Confidence: `{conf:.2f}`")
        else:
            st.warning("⚠️ No soil detected in the image.")

else:
    st.info("👆 Upload an image to start detection.")
