import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Load model once
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Prabhav619/catanddog_image_classifier",  # Replace with your repo
        filename="model.keras",                           # Or model.h5 based on your format
        repo_type="model"
    )
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Image preprocessing
def preprocess_image(image):
    image = image.convert("RGB")  # 💡 Ensure it's 3 channels (RGB)
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Streamlit UI
st.title("🐶🐱 Cat vs Dog Image Classifier")
st.markdown("Upload an image to find out whether it's a **cat or dog**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        input_img = preprocess_image(image)
        prediction = model.predict(input_img)[0][0]

        if prediction > 0.5:
            st.success(f"Prediction: 🐶 Dog ({prediction:.2f})")
        else:
            st.success(f"Prediction: 🐱 Cat ({1 - prediction:.2f})")
