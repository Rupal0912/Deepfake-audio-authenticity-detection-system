# src/train.py
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from preprocess import preprocess_audio
from features import extract_features

DATA_ROOT = r"D:\3_Datasets\archive (1)\for-norm\for-norm"

def load_split(split):
    X, y = [], []

    for label, cls in enumerate(["real", "fake"]):
        folder = os.path.join(DATA_ROOT, split, cls)
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                path = os.path.join(folder, file)
                audio = preprocess_audio(path)
                feats = extract_features(audio)
                X.append(feats)
                y.append(label)

    return np.array(X), np.array(y)


print("Loading training data...")
X_train, y_train = load_split("training")

print("Loading validation data...")
X_val, y_val = load_split("validation")

# Baseline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

print("Training Scaled Logistic Regression (Pipeline)...")

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000))
])

lr_pipeline.fit(X_train, y_train)

print("Validation Results (Scaled LR):")
print(classification_report(y_val, lr_pipeline.predict(X_val)))

# Main model
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

print("Validation Results (RF):")
print(classification_report(y_val, rf.predict(X_val)))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/rf_model.pkl")

print("Model saved.")
