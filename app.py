import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image

# =============================
# Page config
# =============================
st.set_page_config(page_title="Soil Detection", layout="centered")
st.title("🌍 Soil Type Detection")
st.write("Upload a soil image to detect the soil type")

# =============================
# Correct class names (from training)
# =============================
CLASS_NAMES = [
    "Alluvial Soil",
    "Black Soil",
    "Clay Soil",
    "Red Soil"
]

NUM_CLASSES = len(CLASS_NAMES)

# =============================
# Load ONNX model
# =============================
@st.cache_resource
def load_model():
    return ort.InferenceSession(
        "best.onnx",
        providers=["CPUExecutionProvider"]
    )

session = load_model()
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# =============================
# Preprocess
# =============================
def preprocess(image, img_size=640):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    scale = img_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized

    img = canvas.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)

    return img, scale

# =============================
# Postprocess (YOLOv8 ONNX CORRECT)
# =============================
def postprocess(output, scale, conf_thres=0.4):
    detections = []

    preds = np.squeeze(output)  # (84, 8400)
    preds = preds.T             # (8400, 84)

    for pred in preds:
        x, y, w, h = pred[:4]
        class_scores = pred[4:4 + NUM_CLASSES]

        cls_id = np.argmax(class_scores)
        conf = class_scores[cls_id]

        if conf < conf_thres:
            continue

        x1 = int((x - w / 2) / scale)
        y1 = int((y - h / 2) / scale)
        x2 = int((x + w / 2) / scale)
        y2 = int((y + h / 2) / scale)

        detections.append((x1, y1, x2, y2, cls_id, float(conf)))

    return detections

# =============================
# Draw boxes
# =============================
def draw_boxes(image, detections):
    img = image.copy()
    for x1, y1, x2, y2, cls_id, conf in detections:
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    return img

# =============================
# Streamlit UI
# =============================
uploaded_file = st.file_uploader(
    "Upload Soil Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    if st.button("Detect Soil"):
        img_np, scale = preprocess(image)
        output = session.run([output_name], {input_name: img_np})[0]

        detections = postprocess(output, scale)

        if not detections:
            st.warning("No soil detected")
        else:
            result = draw_boxes(np.array(image), detections)
            st.image(result, caption="✅ Detection Result", use_container_width=True)

            st.subheader("📊 Detected Soil Types")
            for _, _, _, _, cls_id, conf in detections:
                st.write(f"• {CLASS_NAMES[cls_id]} — {conf:.2f}")
