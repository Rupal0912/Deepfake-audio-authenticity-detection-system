# backend/utils/feature_extractor.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf

# CONFIG (match training)
TARGET_SR = 16000
DURATION = 4.0
TARGET_LEN = int(TARGET_SR * DURATION)
N_MFCC = 20
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torchaudio transforms
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
).to(DEVICE)
to_db = torchaudio.transforms.AmplitudeToDB(stype="power").to(DEVICE)
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=TARGET_SR,
    n_mfcc=N_MFCC,
    melkwargs={"n_fft":N_FFT,"hop_length":HOP_LENGTH,"n_mels":N_MELS}
).to(DEVICE)

FEATURES_PATH = Path(__file__).resolve().parents[1] / "model" / "feature_columns.json"

def extract_features_from_file(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)

    if sr != TARGET_SR:
        y = torchaudio.functional.resample(torch.tensor(y).unsqueeze(0), sr, TARGET_SR).squeeze(0).numpy()
        sr = TARGET_SR

    if len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)))
    else:
        y = y[:TARGET_LEN]

    x = torch.tensor(y).float().unsqueeze(0).to(DEVICE)

    mfcc = mfcc_transform(x).squeeze(0).cpu().numpy()
    mel = mel_spec(x)
    mel_db = to_db(mel + 1e-9).squeeze(0).cpu().numpy()
    rms = float(torch.sqrt(torch.mean(x**2)).cpu().numpy())
    freqs = torch.linspace(0, TARGET_SR/2, N_MELS, device=DEVICE)
    E = mel.squeeze(0).sum(dim=1) + 1e-12
    centroid = float((E * freqs).sum().item() / E.sum().item())
    zcr = float(((y[:-1] * y[1:]) < 0).mean())

    feat = {}
    for i in range(min(N_MFCC, mfcc.shape[0])):
        feat[f"mfcc_{i+1}_mean"] = float(mfcc[i].mean())
        feat[f"mfcc_{i+1}_std"]  = float(mfcc[i].std())

    feat["mel_db_mean"] = float(mel_db.mean())
    feat["mel_db_std"]  = float(mel_db.std())
    feat["rms"] = rms
    feat["centroid"] = centroid
    feat["zcr"] = zcr

    # reorder and fill missing with zeros according to saved columns
    cols = json.loads(FEATURES_PATH.read_text())
    row = {c: float(feat.get(c, 0.0)) for c in cols}
    return pd.DataFrame([row])
