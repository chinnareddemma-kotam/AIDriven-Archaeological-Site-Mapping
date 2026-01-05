import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import build_model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pth")
# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load checkpoint
# ----------------------------
model_path = r"C:\Users\welcome\Desktop\soildetection\Soil_detection\models\best.pth"
ckpt = torch.load(MODEL_PATH, map_location=device)

# ----------------------------
# Class names
# ----------------------------
class_names = ckpt["class_names"]

# ----------------------------
# Build model (🔥 RESNET-50 🔥)
# ----------------------------
model = build_model(
    "resnet50",                # ✅ CORRECT MODEL
    num_classes=len(class_names),
    pretrained=False
).to(device)

# ----------------------------
# Load weights (auto-detect)
# ----------------------------
if "model_state_dict" in ckpt:
    state_dict = ckpt["model_state_dict"]
elif "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
elif "model" in ckpt:
    state_dict = ckpt["model"]
else:
    state_dict = ckpt

model.load_state_dict(state_dict, strict=True)
model.eval()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Soil Type Classification", page_icon="🌱")

st.title("🌱 Soil Type Classification")
st.markdown("""
Welcome! This Website helps you **identify the type of soil** from an image.  
**How to use:**
1. Upload a clear image of soil using the uploader below.
2. The AI model will predict the soil type for you.
3. Supported soil types: **Red Soil, Black Soil, Clay Soil, Alluvial Soil**.
4. See the predicted soil type instantly!
""")

st.write("Upload a soil image and get the predicted soil type")

uploaded_file = st.file_uploader(
    "Upload soil image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        pred_idx = torch.argmax(outputs, dim=1).item()

    st.success(f"🌱 Predicted Soil Type: **{class_names[pred_idx]}**")
