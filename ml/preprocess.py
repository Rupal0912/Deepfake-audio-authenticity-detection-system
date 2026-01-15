# src/preprocess.py
import librosa
import numpy as np

TARGET_SR = 16000
TARGET_DURATION = 2.0  # seconds
TARGET_LENGTH = int(TARGET_SR * TARGET_DURATION)


def preprocess_audio(file_path):
    """
    Loads an audio file and applies:
    - mono conversion
    - resampling to 16kHz
    - trimming/padding to 2 seconds
    - amplitude normalization
    """
    # Load audio
    y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

    # Trim or pad
    if len(y) > TARGET_LENGTH:
        y = y[:TARGET_LENGTH]
    else:
        pad_width = TARGET_LENGTH - len(y)
        y = np.pad(y, (0, pad_width), mode="constant")

    # Normalize amplitude
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    return y
