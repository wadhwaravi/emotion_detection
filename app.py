from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "emotion_model.keras")
model = load_model(model_path)
labels = ["angry", "happy", "sad", "neutral"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (48,48))
    img = img / 255.0
    img = np.reshape(img, (1,48,48,1))

    prediction = model.predict(img)
    emotion = labels[np.argmax(prediction)]

    return {"emotion": emotion}