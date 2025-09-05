# 🩺 Chest X-Ray Diagnostic Assistant (Testing Build)

An interactive **Streamlit** app that uses a deep-learning model to classify chest X-ray images and show **Grad-CAM** heatmaps as visual explanations.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chest-xray-app-mfawfpdbjakjdr7bsotynh.streamlit.app/)  
**Live demo:** https://chest-xray-app-mfawfpdbjakjdr7bsotynh.streamlit.app/

> **Disclaimer:** For **testing and educational** purposes only (OmniDent.ai task). This is **not** a medical device and **not** a substitute for professional medical advice, diagnosis, or treatment.

---

## ✨ Features
- **Multi-Class Classification:** `Normal`, `COVID`, `Lung Opacity`, `Viral Pneumonia`
- **Visual Explanations:** Grad-CAM heatmaps highlight important regions
- **Side-by-Side View:** Original image vs. annotated heatmap
- **Interactive UI:** Built with Streamlit
- **Cloud-Ready:** Easy deploy to Streamlit Community Cloud

---

## 🧠 Model
- **Architecture:** DenseNet-121  
- **Pre-trained on:** ImageNet  
- **Task:** Multi-class classification (4 classes)

---

## 🏥 Dataset
- **Source:** *COVID-19 Radiography Database* (Kaggle)  
- **Classes:** `Normal`, `COVID`, `Lung_Opacity`, `Viral Pneumonia`  
- **Format:** `.png` images in class folders  
- **Augmentations:** Horizontal flip, rotation, normalization

> Please follow the dataset’s license/usage terms on Kaggle.

---

## 📂 Folder Structure
```
/chest-xray-app/
├─ app.py                       # Streamlit app
├─ classification_model.pth     # Trained PyTorch weights (LFS or runtime download)
├─ requirements.txt             # Python deps
├─ assets/
│   └─ RoC.png                  # Example ROC figure (optional)
└─ README.md
```

---

## ⚙️ Local Setup

### 1) Prerequisites
- Python **3.9+**
- Git & **Git LFS** (`git lfs install`)

### 2) Clone
```bash
git clone https://github.com/azeem-aslam-ch/chest-xray-app
cd chest-xray-app
```

### 3) Install deps
```bash
pip install -r requirements.txt
```

### 4) Model weights
Place `classification_model.pth` in the project root.  
*Alternatively*, set an environment/secret **`MODEL_URL`** that points to a direct download for the weights (see **Cloud Deploy**).

### 5) Run
```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push your repo to **GitHub** (include `app.py`, `requirements.txt`, and use **Git LFS** if you commit the model).
2. On Streamlit Cloud → **New app** → select repo/branch → **Main file = `app.py`**.
3. If you **don’t** commit the model, add a secret for runtime download:
   - App → **Settings → Secrets** → add:
     ```yaml
     MODEL_URL: "https://drive.google.com/uc?id=<FILE_ID>&export=download"
     ```
4. In `app.py`, load and cache once (example):
   ```python
   import os, pathlib, requests, streamlit as st
   WEIGHTS_PATH = pathlib.Path("classification_model.pth")
   url = st.secrets.get("MODEL_URL") or os.getenv("MODEL_URL")
   if url and not WEIGHTS_PATH.exists():
       with requests.get(url, stream=True) as r:
           r.raise_for_status()
           with open(WEIGHTS_PATH, "wb") as f:
               for chunk in r.iter_content(chunk_size=8192):
                   if chunk: f.write(chunk)
   ```
   > For large Drive files that need confirmation tokens, consider `gdown` or hosting on Hugging Face.

---

## 🖱️ How to Use
1. Launch the app (locally or on Streamlit Cloud).  
2. Upload a chest X-ray (`.png/.jpg`).  
3. See class probabilities and the **Grad-CAM** heatmap beside the original image.

---

## 📊 Evaluation (example)
- ROC-AUC per class ≈ **0.99–1.00** on the validation set  
- Include your figures in `assets/` and reference them here:

<p align="center">
  <a href="assets/RoC.png">
    <img src="assets/RoC.png" width="600" alt="ROC curve">
  </a>
</p>

> Validate on a **patient-level** hold-out or external test set. Also check precision-recall, confusion matrix, and calibration.

---

## 🛠️ Tech Stack
- **Python**, **PyTorch**
- **Streamlit**
- **OpenCV**, **Pillow**, **NumPy**
- **Grad-CAM** utilities

---

## 🔒 Config (optional)
`.streamlit/config.toml`
```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = true
```
---

## 📜 License
Choose a license (e.g., **MIT**) and place it in `LICENSE`.

---

## 🙏 Acknowledgements
- COVID-19 Radiography Database (Kaggle)
- PyTorch & Streamlit communities

---

## 📣 Repo Description (short)
Streamlit app for classifying chest X-rays (Normal, COVID, Lung Opacity, Viral Pneumonia) with Grad-CAM explanations. Testing build. One-click deploy to Streamlit Cloud. **Live demo:** https://chest-xray-app-mfawfpdbjakjdr7bsotynh.streamlit.app/
