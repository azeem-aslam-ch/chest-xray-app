🩺 Chest X-Ray Diagnostic Assistant (OmniDent.ai — Testing Build)

An interactive Streamlit app that uses a deep-learning model to classify chest X-ray images and show a Grad-CAM heatmap as a visual explanation.

Disclaimer: This app is for testing and educational purposes only (OmniDent.ai task). It is not a medical device and is not a substitute for professional medical advice, diagnosis, or treatment.

✨ Features

Multi-Class Classification: Normal, COVID, Lung Opacity, Viral Pneumonia

Visual Explanations: Grad-CAM heatmaps highlight important regions

Side-by-Side View: Original image vs. annotated heatmap

Interactive UI: Built with Streamlit

Cloud-Ready: Easy deploy to Streamlit Community Cloud

Docker Support: Containerized run on any server

🧠 Model

Architecture: DenseNet-121

Pre-trained on: ImageNet

Task: Multi-class classification (4 classes)

🏥 Dataset

Source: COVID-19 Radiography Database (Kaggle)

Classes: Normal, COVID, Lung_Opacity, Viral Pneumonia

Format: .png images in class folders

Augmentations: Horizontal flip, rotation, normalization

Be sure to follow the dataset’s license/usage terms on Kaggle.

📂 Folder Structure
/chest-xray-app/
├─ app.py                      # Streamlit app
├─ classification_model.pth    # Trained PyTorch weights (use Git LFS or download at runtime)
├─ requirements.txt            # Python deps
├─ README.md                   # This file
├─ src/                        # (optional) helper code
├─ assets/                     # screenshots/figures (e.g., roc.png)
└─ .streamlit/                 # (optional) Streamlit config
   └─ config.toml

⚙️ Local Setup
1) Prerequisites

Python 3.9+

Git & Git LFS (git lfs install)

2) Clone
git clone https://github.com/your-username/chest-xray-app.git
cd chest-xray-app

3) Install deps
pip install -r requirements.txt

4) Model weights

Place classification_model.pth in the project root.
Alternatively, set an environment variable MODEL_URL to auto-download at startup (see Cloud Deploy section).

5) Run
streamlit run app.py

☁️ Deploy on Streamlit Community Cloud

Push your repo to GitHub (include app.py, requirements.txt, and use Git LFS for the model if you commit it).

Go to Streamlit Cloud → New app → select your repo/branch → Main file = app.py.

If you don’t commit the model file, set a secret to download it:

In your app → Settings → Secrets → add:

MODEL_URL: "https://your-storage/classification_model.pth"


In app.py, load MODEL_URL and download/cache once (example code included in the app).

🐳 Docker (Optional)

Dockerfile

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLEXsrfPROTECTION=true
ENV STREAMLIT_SERVER_PORT=8501

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


Build & run

docker build -t chest-xray-app:latest .
docker run -p 8501:8501 chest-xray-app:latest
# open http://localhost:8501


With remote model URL

docker run -p 8501:8501 -e MODEL_URL="https://your-storage/classification_model.pth" chest-xray-app:latest

🖱️ How to Use

Launch the app (locally, Docker, or Streamlit Cloud).

Upload a chest X-ray (.png/.jpg).

View predicted class probabilities and the Grad-CAM heatmap next to the original image.

📊 Evaluation (example)

ROC-AUC per class ≈ 0.99–1.00 on the validation set

Include your figures in assets/ and reference them here:

Validate on a patient-level hold-out or external test set. Also check precision-recall, confusion matrix, and calibration.

🛠️ Tech Stack

Python, PyTorch

Streamlit

OpenCV, Pillow, NumPy

Grad-CAM utilities

🔒 Config (optional)

.streamlit/config.toml

[server]
headless = true
enableCORS = false
enableXsrfProtection = true

🗺️ Roadmap

Add external test evaluation

Add batch upload

Add DICOM support and windowing

Export PDF report

📜 License

Choose a license (e.g., MIT) and place it in LICENSE.

🙏 Acknowledgements

COVID-19 Radiography Database (Kaggle)

PyTorch & Streamlit communities

📣 Citation

If you use this code, please cite this repository and the dataset you trained on.
