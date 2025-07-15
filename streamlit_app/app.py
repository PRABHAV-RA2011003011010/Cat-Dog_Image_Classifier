import streamlit as st
import requests

st.title("🐶🐱 Cat vs Dog Image Classifier (via FastAPI)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify via API"):
        files = {"file": uploaded_file.getvalue()}
        try:
            response = requests.post("http://fastapi:8000/predict", files=files)
            data = response.json()
            label = "🐶 Dog" if data["result"] == "dog" else "🐱 Cat"
            st.success(f"Prediction: {label} ({data['confidence']:.2f})")
        except Exception as e:
            st.error(f"API call failed: {e}")
