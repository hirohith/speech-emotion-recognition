import os
import numpy as np
import librosa

dataset_path = "dataset"

emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful"
}

features = []
labels = []

for actor_folder in os.listdir(dataset_path):

    actor_path = os.path.join(dataset_path, actor_folder)

    if os.path.isdir(actor_path):

        for file in os.listdir(actor_path):

            if file.endswith(".wav"):

                emotion_code = file.split("-")[2]

                if emotion_code in emotion_map:

                    file_path = os.path.join(actor_path, file)

                    audio, sr = librosa.load(
                        file_path,
                        duration=3,
                        offset=0.5
                    )

                    # MFCC
                    mfcc = librosa.feature.mfcc(
                        y=audio,
                        sr=sr,
                        n_mfcc=40
                    )
                    mfcc_mean = np.mean(mfcc.T, axis=0)

                    # Chroma
                    chroma = librosa.feature.chroma_stft(
                        y=audio,
                        sr=sr
                    )
                    chroma_mean = np.mean(chroma.T, axis=0)

                    # Spectral Centroid
                    spectral = librosa.feature.spectral_centroid(
                        y=audio,
                        sr=sr
                    )
                    spectral_mean = np.mean(spectral)

                    # Zero Crossing Rate
                    zcr = librosa.feature.zero_crossing_rate(audio)
                    zcr_mean = np.mean(zcr)

                    # RMS Energy
                    rms = librosa.feature.rms(y=audio)
                    rms_mean = np.mean(rms)

                    feature_vector = np.hstack([
                        mfcc_mean,
                        chroma_mean,
                        spectral_mean,
                        zcr_mean,
                        rms_mean
                    ])

                    features.append(feature_vector)
                    labels.append(
                        emotion_map[emotion_code]
                    )

X = np.array(features)
y = np.array(labels)

print("Feature Shape:", X.shape)
print("Label Shape:", y.shape)
