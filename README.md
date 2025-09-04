# 🩺 Chest X-Ray Diagnostic Assistant (OmniDent.ai — Testing Build)

An interactive **Streamlit** app that uses a deep-learning model to classify chest X-ray images and show a **Grad-CAM** heatmap as a visual explanation.

> **Disclaimer:** This app is for **testing and educational** purposes only (OmniDent.ai task). It is **not** a medical device and is **not** a substitute for professional medical advice, diagnosis, or treatment.

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
├─ app.py                      # Streamlit app
├─ classification_model.pth    # Trained PyTorch weights (use Git LFS or download at runtime)
├─ requirements.txt            # Python deps
├─ README.md                   # This file

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
*Alternatively*, set an environment variable `https://drive.google.com/file/d/1LRP3o3TBXXp-_xIPEzLt8dArt1YCcTRs/view?usp=drive_link` to auto-download at startup (see **Cloud Deploy**).

### 5) Run
```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push your repo to **GitHub** (include `app.py`, `requirements.txt`, and use **Git LFS** for the model if you commit it).
2. Go to Streamlit Cloud → **New app** → select your repo/branch → **Main file = `app.py`**.
3. If you *don’t* commit the model file, set a secret to download it:
   - In your app → **Settings → Secrets** → add:
     ```yaml
     https://drive.google.com/file/d/1LRP3o3TBXXp-_xIPEzLt8dArt1YCcTRs/view?usp=drive_link"
     ```
4. In `app.py`, load `MODEL_URL` and download/cache once.

---

## 🖱️ How to Use
1. Launch the app (locally or Streamlit Cloud).
2. Upload a chest X-ray (`.png/.jpg`).
3. View predicted class probabilities and the **Grad-CAM** heatmap next to the original image.

---

## 📊 Evaluation (example)
- ROC-AUC per class ≈ **0.99–1.00** on the validation set  
- Include your figures in `assets/` and reference them here:

![ROC](<img width="1001" height="855" alt="image" src="https://github.com/user-attachments/assets/79e33bfd-8800-4fd7-ba9c-76b6ee65261d" />
)

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
Streamlit app for classifying chest X-rays (Normal, COVID, Lung Opacity, Viral Pneumonia) with Grad-CAM explanations. OmniDent.ai testing build. One-click deploy to Streamlit Cloud.
