🩺 Chest X-Ray Diagnostic Assistant


An interactive web application built with Streamlit that uses a deep learning model to classify chest X-ray images into four categories: Normal, COVID, Lung Opacity, and Viral Pneumonia. The app also provides visual explanations using Grad-CAM to highlight areas of interest that influenced the model's prediction.



🌟 Key Features
Image Classification: Upload a chest X-ray (.png, .jpg, .jpeg) and get a prediction with confidence scores.

Visual Explanation (Grad-CAM): See a heatmap overlay on the X-ray that shows where the model is "looking" to make its decision.

Side-by-Side Comparison: Compare the original image with the annotated heatmap image.

Interactive Controls:

Adjust the opacity of the heatmap overlay with a slider.

Select which class you want to see the heatmap for via a dropdown menu.

Error Handling: Gracefully handles invalid or corrupted file uploads.

🛠️ Technology Stack
Framework: Streamlit

Deep Learning: PyTorch, Torchvision

Model Visualization: TorchCAM

Image Processing: Albumentations, OpenCV, Pillow (PIL)

Data Handling: Pandas, NumPy

⚙️ Setup and Installation (Running Locally)
To run this application on your local machine, please follow these steps.

1. Prerequisites
Python 3.9+

Git installed.

Git LFS installed for handling the large model file. Run git lfs install once after installation.

2. Clone the Repository
Open your terminal and clone this repository:

git clone [https://github.com/your-username/chest-xray-app.git](https://github.com/your-username/chest-xray-app.git)
cd chest-xray-app

3. Install Dependencies
Install all the required Python libraries using the requirements.txt file:

pip install -r requirements.txt

4. Add the Model File
This project requires the pre-trained model file classification_model.pth.

Download the model file from the source (e.g., Google Drive).

Place it in the root directory of this project folder (/chest-xray-app/).

5. Run the Application
Once the setup is complete, run the following command in your terminal:

streamlit run app.py

Your web browser will open a new tab with the application running locally!

☁️ Deployment
This application is designed to be deployed on Streamlit Cloud.

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
