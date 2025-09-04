🩺 Chest X-Ray Diagnostic Assistant

An interactive web application built with Streamlit that uses a deep learning model to classify chest X-ray images and provide visual explanations for its predictions.

📌 Features
✅ Multi-Class Classification: Classifies images into Normal, COVID, Lung Opacity, and Viral Pneumonia.

🔍 Visual Explanations: Uses Grad-CAM to generate heatmaps highlighting areas of interest.

📊 Interactive Interface: Built with Streamlit for a user-friendly experience.

🚀 Cloud-Ready: Designed for easy deployment on Streamlit Cloud.

📁 Side-by-Side View: Compares the original X-ray with the annotated heatmap image.

🧠 Model Used
Model Type

Architecture

Pre-trained on

Task

Classifier

DenseNet-121

ImageNet

Multi-Label Classification

🏥 Dataset Overview
Source: COVID-19 Radiography Database on Kaggle.

Classes: Normal, COVID, Lung_Opacity, Viral Pneumonia.

Format: .png images sorted into category folders.

Augmentations: Horizontal Flip, Rotation, Normalization.

⚙️ Setup for Local Development
1. Prerequisites
Python 3.9+

Git & Git LFS (git lfs install)

2. Clone the Repository
git clone [https://github.com/your-username/chest-xray-app.git](https://github.com/your-username/chest-xray-app.git)
cd chest-xray-app

3. Install Dependencies
pip install -r requirements.txt

4. Download Model File
Place the classification_model.pth file in the root directory of the project.

5. Run the App
streamlit run app.py

☁️ Deployment
This application is designed for easy deployment on Streamlit Cloud.

Push the project (including app.py, requirements.txt, and the model file using Git LFS) to a public GitHub repository.

Sign up for a free account on Streamlit Cloud using your GitHub account.

From your Streamlit dashboard, click "New app" and select the repository. Streamlit will handle the rest.

⚠️ Disclaimer
This tool is an academic project and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

📂 Folder Structure
/chest-xray-app/
|
|-- 📄 app.py                  # The main Streamlit application script
|-- 🧠 classification_model.pth    # The pre-trained PyTorch model
|-- 🛒 requirements.txt         # A list of required Python libraries
|-- 📄 README.md                 # This file
