# 🎧 Audio Authenticity — Deepfake Audio Detection Tool

## 🔍 Problem Statement
With the rapid rise of AI-generated voices, it is becoming increasingly difficult to verify whether audio evidence is real or fake. Audio deepfakes can be misused in:

- Scams and financial fraud
- Political misinformation
- Fake voice evidence in legal cases
- Impersonation of celebrities & leaders

✅ **This project detects whether an audio sample is Human or AI-Generated.**

---

## ✅ Key Features

| Feature | Description |
|--------|-------------|
| 🎤 Audio Upload & Analysis | Users upload audio files for authenticity check |
| 📊 Probability Score | Model outputs confidence %: Real vs Fake |
| 🧠 ML Model | Extracts MFCC + Spectrogram features for classification |
| ⚙️ Backend API | FastAPI-based prediction service |
| 🌐 Frontend UI | Clean and interactive interface |
| 🐳 Docker Support | Containerized deployment |
| 🔁 CI/CD Pipeline | GitHub Actions automated workflow (DevOps) |

---

## 🧠 Machine Learning Approach

| Step | Technique |
|------|----------|
| Feature Extraction | MFCC, Mel Spectrograms using `librosa` |
| Model | CNN / SVM Binary Classifier |
| Evaluation | Accuracy, ROC-AUC Score |

📌 Dataset:
- ✅ LibriSpeech / VCTK for **real** audio
- ✅ ASVspoof / Coqui TTS generated data for **fake** audio

---

## 🏗 Project Architecture

User → Frontend → REST API → Feature Extraction → Model → Authenticity Result

yaml
Copy code

---

## 📂 Folder Structure

audio-authenticity/
│
├── frontend/ # Web UI
├── backend/ # API + Model Inference
├── ml_model/ # Training Notebook + Data
├── docker-compose.yml # Multi-service orchestration
├── Dockerfile # Project containerization
└── .github/workflows/ # CI/CD pipeline automation

yaml
Copy code

---

## 🧪 How to Run Locally

### ✅ 1️⃣ Create Virtual Environment

cd backend
pip install -r requirements.txt

shell
Copy code

### ✅ 2️⃣ Start FastAPI Backend

uvicorn app:app --reload

yaml
Copy code

✅ Backend API → `http://127.0.0.1:8000/predict`

### ✅ 3️⃣ Start Frontend

Open `frontend/index.html` in browser  
(or serve using Live Server Extension)

---

## 🐳 Docker Deploy (Optional)

docker-compose up --build

yaml
Copy code

---

## 🔁 DevOps Workflow (CI/CD)

GitHub Actions Pipeline automated tasks:

✔ Build Docker Image  
✔ Install Dependencies  
✔ Run Linting & Tests  
✔ Deploy Backend to Cloud (future scope)  

`deploy.yml` included inside `.github/workflows/`

---

## 📈 Future Enhancements

🚀 Browser extension to analyze YouTube/Instagram audio  
🎙️ Live microphone stream verification  
🌍 Multi-language support  
🛡 Blockchain logging for digital evidence integrity  
🔊 Detection of cloned voices of a specific person  

---

## 👩‍💻 Tech Stack

| Category | Tech |
|---------|-----|
| Frontend | HTML, CSS, JavaScript |
| Backend | FastAPI, Python |
| ML / Audio Processing | librosa, scikit-learn / PyTorch |
| Deployment | Docker, GitHub Actions |
| Data Format | .wav, .mp3 |

---

## 👨‍🏫 Academic Use
This project demonstrates skills in:

✅ Machine Learning  
✅ Audio Signal Processing  
✅ Web Development  
✅ DevOps (CI/CD + Docker)

Perfect for: Major Project | Internship | Resume Portfolio ✅

---

## 🤝 Contributors
👤 Your Name — Data Science & Full Stack Development

---

## 📜 License
MIT License — free for academic use

---

## ⭐ Support
If you like this project, please ⭐ the repository!