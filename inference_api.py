from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import uvicorn

# Initialize API 
app = FastAPI(title="Person vs Horse Classifier")

# Load model
MODEL_PATH = "model/person_vs_horse_model.h5"
model = load_model(MODEL_PATH)

# Define image size and preprocessing function
IMG_SIZE = (224, 224)
def preprocess_image(image_bytes):
    """Preprocess the image for prediction."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # format (1, 224, 224, 3)
    return img

# Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = preprocess_image(contents)
    pred = model.predict(img)[0][0]

    if pred > 0.5:
        label = "Is a person"
        confidence = pred  # Probability of being a person
    else:
        label = "Is a horse"
        confidence = 1 - pred  # Probability of being a horse

    return {"prediction": label, "confidence": float(confidence)} 

 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
