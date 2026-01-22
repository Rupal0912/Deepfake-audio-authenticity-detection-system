# 🎧 Deepfake Audio Authenticity Detection System

A **production-ready ML system** to detect whether an uploaded audio file is **Real Human Speech** or **AI-Generated / Deepfake Audio**.

This project is not a demo script — it is a **fully deployed, containerized, end‑to‑end system** covering:

* Audio preprocessing & feature extraction
* ML model training & version compatibility
* FastAPI backend
* Frontend integration
* Dockerized deployment on Render

🔗 **Live Demo**: [https://deepfake-audio-authenticity-detection.onrender.com](https://deepfake-audio-authenticity-detection.onrender.com)

---

## 🚀 Key Highlights

* 🔍 **ML-based Deepfake Detection** using audio signal features
* 🧠 Model retrained with strict **production version parity** (no sklearn mismatch)
* ⚙️ **FastAPI backend** with file upload support
* 🎨 Simple frontend UI for real-time testing
* 🐳 **Dockerized** for consistent builds
* ☁️ **Deployed on Render** (publicly accessible)

This project intentionally focuses on **engineering correctness**, not just model accuracy.

---

## 🏗️ System Architecture

```
┌──────────────┐
│   Browser    │
│ (Frontend)   │
└──────┬───────┘
       │  Audio Upload (POST)
       ▼
┌──────────────────────────┐
│   FastAPI Application    │
│  (api/app.py)            │
│                          │
│  - /predict endpoint     │
│  - UploadFile handling   │
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐
│ Audio Preprocessing      │
│ (ml/preprocess.py)       │
│                          │
│ - Resampling             │
│ - Mono conversion        │
│ - Silence handling       │
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐
│ Feature Extraction       │
│ (ml/features.py)         │
│                          │
│ - Spectral features      │
│ - MFCC-based stats       │
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐
│ ML Inference              │
│ (Random Forest Model)    │
│ models/rf_model.pkl      │
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐
│ Prediction Response      │
│ { Real / AI-generated }  │
└──────────────────────────┘
```

---

## 🧠 Machine Learning Pipeline

### 1. Data Handling

* Audio files loaded using `librosa`
* Converted to a consistent sample rate
* Normalized and validated

### 2. Feature Extraction

Extracted features include:

* Spectral centroid
* Spectral bandwidth
* Zero-crossing rate
* MFCC statistical aggregates

These features are designed to capture **artifacts common in synthetic audio**.

### 3. Models Trained

* **Scaled Logistic Regression** (baseline)
* **Random Forest Classifier** (final model)

### 4. Final Model

* **Random Forest** selected due to superior validation performance
* Serialized as `rf_model.pkl`
* Retrained under **scikit-learn 1.4.2** to match production runtime

---

## 🧪 Validation Results (Final Training)

```
Accuracy: 98%

Class 0 (Real Audio):
Precision: 0.99 | Recall: 0.98

Class 1 (AI-generated Audio):
Precision: 0.98 | Recall: 0.99
```

⚠️ Note: Metrics are secondary here — **deployment correctness and compatibility** were the main goals.

---

## 🛠️ Tech Stack

**Backend**

* Python 3.10
* FastAPI
* Uvicorn

**ML / Audio**

* scikit-learn 1.4.2
* numpy 1.26.4
* scipy 1.11.4
* librosa
* soundfile

**Frontend**

* HTML
* CSS
* JavaScript (Fetch API)

**DevOps / Deployment**

* Docker
* Render

---

## 📦 Project Structure

```
Deepfake-audio/
├── api/
│   ├── app.py           # FastAPI app
│   ├── inference.py     # Model loading & prediction
│   └── schemas.py
│
├── ml/
│   ├── train.py         # Model training script
│   ├── preprocess.py   # Audio preprocessing
│   └── features.py     # Feature extraction
│
├── models/
│   └── rf_model.pkl     # Trained ML model
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🐳 Docker & Deployment

The application is fully containerized.

Key deployment considerations:

* Exact dependency pinning
* Python version parity (3.10)
* Model retraining to avoid sklearn incompatibility warnings

Deployment is handled automatically by **Render** on push to `main`.

---

## ⚠️ Known Limitations & Future Improvements

* Large model file (~62 MB) — should be moved to object storage or Git LFS
* No authentication (public demo)
* Synchronous inference (can be async/queued)
* Dataset not included in repo

---

## 📌 Why This Project Matters

This project demonstrates:

* Real-world ML deployment challenges
* Version mismatch debugging
* End-to-end ownership (data → model → API → UI → cloud)

It reflects **engineering maturity**, not just ML theory.

---

## 👤 Author

**Rupal**
B.Tech CSE Student

---

## 📄 License

This project is for educational and demonstration purposes.
