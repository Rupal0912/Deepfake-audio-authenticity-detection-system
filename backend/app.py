# # backend/app.py
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import joblib, tempfile, os
# from pathlib import Path
# # from utils.feature_extractor import extract_features_from_file
# from backend.utils.feature_extractor import extract_features_from_file
# # add these imports at top of backend/app.py
# from fastapi.staticfiles import StaticFiles
# from pathlib import Path

# # locate frontend folder (adjust path if your repo layout differs)
# FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

# # mount frontend as static files (only if it exists)
# if FRONTEND_DIR.exists():
#     app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
#     print("Serving frontend from", FRONTEND_DIR)
# else:
#     print("Frontend folder not found at", FRONTEND_DIR)

# # MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "xgb_model.pkl"
# MODEL_PATH = Path(__file__).resolve().parent / "model" / "xgb_model.pkl"
# FEATURES_PATH = Path(__file__).resolve().parent / "model" / "feature_columns.json"

# app = FastAPI(title="Audio Authenticity API")
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# # load model
# model = joblib.load(str(MODEL_PATH))
# print("Loaded model from", MODEL_PATH)
# backend/app.py (replace top of file with this block)
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import joblib, os, tempfile

# --- paths ---
BASE_DIR = Path(__file__).resolve().parents[1]      # audio-authenticity folder
BACKEND_DIR = Path(__file__).resolve().parent       # backend folder
MODEL_PATH = BACKEND_DIR / "model" / "xgb_model.pkl"
FEATURES_PATH = BACKEND_DIR / "model" / "feature_columns.json"
FRONTEND_DIR = BASE_DIR / "frontend"

# --- create the FastAPI app first ---
app = FastAPI(title="Audio Authenticity API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- mount frontend static files AFTER app is created ---
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    print("Serving frontend from", FRONTEND_DIR)
else:
    print("Frontend folder not found at", FRONTEND_DIR)

# --- lazy model load (safe) ---
_model = None
def load_model():
    global _model
    if _model is None:
        if MODEL_PATH.exists():
            _model = joblib.load(str(MODEL_PATH))
            print("Loaded model from", MODEL_PATH)
        else:
            print("Model not found at", MODEL_PATH, "- API will return 503 for /predict until model is added.")
            _model = None
    return _model
mdl = load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    mdl = load_model()
    if mdl is None:
        return JSONResponse({"error":"Model not available. Place xgb_model.pkl in backend/model/ and restart."}, status_code=503)

    # save uploaded file
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    try:
        from backend.utils.feature_extractor import extract_features_from_file
        X = extract_features_from_file(tmp_path)
        proba = mdl.predict_proba(X)[0].tolist()
        pred = int(mdl.predict(X)[0])
        return JSONResponse({"prediction": pred, "probability": proba, "labels": mdl.classes_.tolist()})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

