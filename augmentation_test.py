import librosa
import numpy as np

audio, sr = librosa.load(
    "dataset/Actor_01/03-01-03-01-01-01-01.wav"
)

# Noise
noise_audio = audio + 0.005 * np.random.randn(len(audio))

# Pitch Shift
pitch_audio = librosa.effects.pitch_shift(
    audio,
    sr=sr,
    n_steps=2
)

print("Original:", audio.shape)
print("Noise:", noise_audio.shape)
print("Pitch:", pitch_audio.shape)
