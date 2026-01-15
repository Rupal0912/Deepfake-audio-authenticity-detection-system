from fastapi import FastAPI, UploadFile, File, HTTPException
import joblib
import numpy as np
import tempfile
import os
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Import ML pipeline
from ml.preprocess import preprocess_audio
from ml.features import extract_features

app = FastAPI(title="Deepfake Audio Detection API")
# Serve static frontend files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

MODEL_PATH = "models/rf_model.pkl"

# Load model once at startup
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Preprocess + extract features
        audio = preprocess_audio(tmp_path)
        features = extract_features(audio).reshape(1, -1)

        # Predict
        prediction = model.predict(features)[0]

        result = "real" if prediction == 0 else "fake"

        return {"prediction": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
