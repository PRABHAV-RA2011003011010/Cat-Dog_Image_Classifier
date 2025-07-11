from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from huggingface_hub import hf_hub_download

app = FastAPI()

model_path = hf_hub_download(
    repo_id="Prabhav619/catanddog_image_classifier",
    filename="model.keras",
    repo_type="model"
)
model = tf.keras.models.load_model(model_path)

def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    input_img = preprocess_image(image_data)
    prediction = model.predict(input_img)[0][0]

    result = "dog" if prediction > 0.5 else "cat"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)
    return {"result": result, "confidence": confidence}
