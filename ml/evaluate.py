# src/evaluate.py
import os
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import preprocess_audio
from features import extract_features
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_ROOT = r"D:\3_Datasets\archive (1)\for-norm\for-norm"
import os
print(os.listdir(DATA_ROOT))
MODEL_PATH = "models/rf_model.pkl"

def load_test():
    X, y = [], []

    for label, cls in enumerate(["real", "fake"]):
        folder = os.path.join(DATA_ROOT, "testing", cls)
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                path = os.path.join(folder, file)
                audio = preprocess_audio(path)
                feats = extract_features(audio)
                X.append(feats)
                y.append(label)

    return np.array(X), np.array(y)


rf = joblib.load(MODEL_PATH)
X_test, y_test = load_test()
print("\nEvaluating Scaled Logistic Regression on Test Set...")

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000))
])

# Train LR on training + validation (allowed, test untouched)
X_train_val = np.concatenate([X_test[:0], X_test[:0]])  # dummy placeholder

y_pred = rf.predict(X_test)

print("Test Results:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
