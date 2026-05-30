import joblib
import librosa
import numpy as np

# Load Model
model = joblib.load(
    "models/emotion_rf_model.pkl"
)

# Audio File Path
file_path = input(
    "Enter audio file path: "
)

# Load Audio
audio, sr = librosa.load(
    file_path,
    duration=3,
    offset=0.5
)

# Extract MFCC
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=40
)

mfcc_mean = np.mean(
    mfcc.T,
    axis=0
)

mfcc_mean = mfcc_mean.reshape(1, -1)

# Predict
prediction = model.predict(
    mfcc_mean
)

print("\nPredicted Emotion:", prediction[0])
