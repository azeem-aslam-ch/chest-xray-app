import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

# ==============================================================================
# DEPLOYMENT INSTRUCTIONS (for Streamlit Cloud)
# ==============================================================================
# 1. Create a GitHub Repository:
#    - Create a new public repository on GitHub.
#    - Add these three files to it: `app.py`, `requirements.txt`, `classification_model.pth`.
#    - Note: For larger models, you might need to use Git LFS (Large File Storage).
#
# 2. Sign up for Streamlit Cloud:
#    - Go to https://streamlit.io/cloud and sign up using your GitHub account.
#
# 3. Deploy the App:
#    - From your Streamlit Cloud dashboard, click "New app".
#    - Choose the GitHub repository you just created.
#    - The branch should be 'main' and the main file path should be 'app.py'.
#    - Click "Deploy!". Streamlit will handle the rest.
# ==============================================================================


# --- App Configuration ---
st.set_page_config(
    page_title="Chest X-Ray Diagnostic Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- Model and Preprocessing Configuration ---
MODEL_PATH = "classification_model.pth"
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASSES = ['Normal', 'COVID', 'Lung_Opacity', 'Viral Pneumonia']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Caching the Model and CAM Extractor ---
@st.cache_resource
def load_model(model_path):
    """Loads the pre-trained DenseNet-121 model and the CAM extractor."""
    try:
        model = models.densenet121()
        model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        # The target layer for DenseNet-121's last convolutional block
        cam_extractor = GradCAM(model, 'features.denseblock4.denselayer16.conv2')
        return model, cam_extractor
    except FileNotFoundError:
        st.error(f"Model file not found at '{model_path}'. Please make sure it's in the same directory as the app.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None, None

# --- Image Transformation ---
def transform_image(image_bytes):
    """
    Applies the same transformations as the validation set.
    """
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
    try:
        image = Image.open(image_bytes).convert("RGB")
        image_np = np.array(image)
        augmented = transform(image=image_np)
        return augmented['image'], image # Return original PIL image too
    except Exception:
        # Handle invalid image files gracefully
        return None, None

# --- Main App Interface ---
st.title("🩺 Chest X-Ray Diagnostic Assistant")
st.markdown("""
Welcome to the AI-powered diagnostic tool. Upload a chest X-ray to get a classification and see a visual explanation of the result.
**Disclaimer:** This tool is for informational purposes only and not a substitute for professional medical advice.
""")

# --- Sidebar ---
st.sidebar.header("About This App")
st.sidebar.markdown("""
This application uses a **DenseNet-121** model to classify chest X-rays and **Grad-CAM** to highlight areas of interest for the model's prediction.
""")

st.sidebar.header("Visualization Parameters")
opacity = st.sidebar.slider("Heatmap Opacity", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
class_to_visualize = st.sidebar.selectbox("Show Heatmap For:", CLASSES)

# --- File Uploader and Prediction Logic ---
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["png", "jpg", "jpeg"])

model, cam_extractor = load_model(MODEL_PATH)

if model and uploaded_file is not None:
    image_tensor, original_image = transform_image(uploaded_file)
    
    if image_tensor is None:
        st.error("The uploaded file could not be processed. Please upload a valid image file (PNG, JPG, JPEG).")
    else:
        st.header("Analysis Results")
        
        # --- Run Inference ---
        input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        with st.spinner("Analyzing the image..."):
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

            # Create a DataFrame for the results
            results_df = pd.DataFrame({
                'Condition': CLASSES,
                'Confidence': probabilities
            })
            results_df = results_df.sort_values(by='Confidence', ascending=False).reset_index(drop=True)
        
        top_prediction = results_df.iloc[0]
        st.success(f"**Top Prediction:** {top_prediction['Condition']} with a confidence of {top_prediction['Confidence']:.2%}")
        st.write("The model has analyzed the image and provides the following confidence scores:")
        st.dataframe(results_df.style.format({'Confidence': '{:.2%}'}), use_container_width=True)

        st.header("Visual Explanation (Grad-CAM)")
        
        # --- Generate and Display Grad-CAM ---
        with st.spinner("Generating heatmap..."):
            # Get the index of the class to visualize
            class_idx = CLASSES.index(class_to_visualize)
            
            # We need to re-run the forward pass *without* no_grad for the hook to work
            model_output = model(input_tensor)
            activation_map = cam_extractor(class_idx, model_output)[0].cpu()

            # Overlay the heatmap
            result = overlay_mask(original_image, to_pil_image(activation_map, mode='F'), alpha=opacity)
        
        # Display side-by-side comparison
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption='Original Uploaded Image', use_column_width=True)
        with col2:
            st.image(result, caption=f'Annotated Image (Heatmap for "{class_to_visualize}")', use_column_width=True)

elif not model:
    st.warning("Please place the `classification_model.pth` file in the application's root directory.")

