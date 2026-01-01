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
    return ort.InferenceSession(
        "best.onnx",
        providers=["CPUExecutionProvider"]
    )

session = load_model()

# ---------------- AUTO DETECT NUMBER OF CLASSES ----------------
input_shape = session.get_outputs()[0].shape
NUM_CLASSES = input_shape[-1] - 4  # YOLOv8 format

# ---------------- CLASS NAMES ----------------
# You can rename these safely
DEFAULT_CLASSES = [
    "Red Soil",
    "Black Soil",
    "Alluvial Soil",
    "Clay Soil"
]

# Extend class list if model has more classes
CLASS_NAMES = DEFAULT_CLASSES + [
    f"Class {i}" for i in range(len(DEFAULT_CLASSES), NUM_CLASSES)
]

# ---------------- IMAGE PREPROCESS ----------------
def preprocess(image, img_size=640):
    image = image.resize((img_size, img_size))
    img = np.array(image).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- POSTPROCESS (YOLOv8 ONNX) ----------------
def postprocess(outputs, img_shape, conf_thres=0.4):
    h, w = img_shape
    detections = []

    preds = outputs[0][0]  # (num_boxes, 4 + num_classes)

    for pred in preds:
        class_scores = pred[4:]
        cls_id = int(np.argmax(class_scores))
        conf = float(class_scores[cls_id])

        if conf > conf_thres:
            x, y, bw, bh = pred[:4]

            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)

            detections.append((x1, y1, x2, y2, cls_id, conf))

    return detections

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload Soil Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect Soil Type"):
        with st.spinner("Running inference..."):
            input_tensor = preprocess(image)
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_tensor})

        img_np = np.array(image)
        detections = postprocess(outputs, img_np.shape[:2])

        if detections:
            for x1, y1, x2, y2, cls_id, conf in detections:
                class_name = (
                    CLASS_NAMES[cls_id]
                    if cls_id < len(CLASS_NAMES)
                    else f"Class {cls_id}"
                )

                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(
                    img_np,
                    label,
                    (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            st.image(img_np, caption="✅ Detection Result", use_container_width=True)

            st.subheader("📊 Detected Soil Types")
            for _, _, _, _, cls_id, conf in detections:
                name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class {cls_id}"
                st.write(f"• **{name}** — `{conf:.2f}`")
        else:
            st.warning("⚠️ No soil detected.")

else:
    st.info("👆 Upload an image to start detection.")
