# 🌍 EcoExplore AI – Soil Detection & Vegetation Segmentation

## 📌 Project Overview

**EcoExplore AI** is an intelligent image analysis application designed for **archaeological site mapping and environmental analysis**. The project combines **soil detection** and **vegetation segmentation** to help identify land patterns that may indicate archaeological significance.

The system processes images to distinguish:

* 🟤 **Soil regions** (bare land, excavation-prone areas)
* 🌿 **Vegetation regions** (plants, grass, crops)

This aids researchers, archaeologists, and environmental analysts in **site exploration, land-use assessment, and ecological monitoring**.

---

## 🎯 Key Features

* Comparison of multiple deep learning models (YOLOv8 Nano, Small, Medium)
* Accuracy-based model selection strategy
* Final adoption of **DeepLabV3** for superior semantic segmentation performance
* Automated soil detection using segmentation masks
* Accurate vegetation segmentation at pixel level
* Modular project structure for easy experimentation

---

## 🧠 Technologies Used

* **Language:** Python

* **Model Comparison:** YOLOv8 (Nano, Small, Medium)

* **Final Selected Model:** **DeepLabV3** (Semantic Segmentation)

* **Frameworks & Libraries:**

  * TensorFlow / PyTorch
  * Ultralytics YOLOv8
  * OpenCV
  * NumPy
  * Matplotlib

* **Techniques:**

  * Model accuracy comparison
  * Semantic image segmentation
  * Image preprocessing & augmentation

* **Language:** Python

* **Deep Learning Model:** **DeepLabV3** (Semantic Segmentation)

* **Frameworks & Libraries:**

  * TensorFlow / PyTorch (DeepLabV3 implementation)
  * OpenCV
  * NumPy
  * Matplotlib

* **Techniques:**

  * Semantic image segmentation
  * Image preprocessing & augmentation
  * Color space analysis (RGB / HSV)


---

## ⚙️ How the System Works

1. **Model Evaluation Phase** – YOLOv8 Nano, Small, and Medium models were trained and evaluated
2. **Accuracy Comparison** – Models were compared based on segmentation accuracy and performance
3. **Model Selection** – DeepLabV3 was chosen due to higher accuracy and better pixel-level segmentation
4. **Input Image** – Images are loaded from the assets folder or user input
5. **Preprocessing** – Image resizing, normalization, and noise reduction
6. **DeepLabV3 Segmentation** – Semantic segmentation produces detailed masks
7. **Soil Detection** – Soil regions extracted from segmentation output
8. **Vegetation Segmentation** – Vegetation areas isolated accurately
9. **Visualization** – Outputs displayed for analysis

---

## 🚀 Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/chinnareddemma-kotam/EcoExplore-AI.git
cd EcoExplore-AI
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
python appp.py
```

---

## 📊 Output

* Highlighted soil regions
* Segmented vegetation masks
* Visual comparison between original and processed images

---

## 🌱 Applications

* Archaeological site mapping
* Environmental monitoring
* Precision agriculture
* Land cover analysis
* AI-assisted exploration systems

---

## 🔮 Future Enhancements

* Fine-tuning DeepLabV3 on custom archaeological datasets
* Multi-class land cover classification (soil, vegetation, water, structures)
* NDVI-based vegetation health analysis
* GIS & satellite image integration
* Web-based dashboard (Streamlit/Flask)

---
## Live Demo
https://ecoexplore-vegetationsegmentation-soildetection.streamlit.app/
## 👨‍💻 Author

**Chinna Reddemma Kotam**
AI & Web Development Enthusiast


