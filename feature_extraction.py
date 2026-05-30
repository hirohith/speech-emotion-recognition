import os
import numpy as np
import pandas as pd
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

                    # Load audio
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

                    # Mean of MFCC coefficients
                    mfcc_mean = np.mean(
                        mfcc.T,
                        axis=0
                    )

                    features.append(mfcc_mean)
                    labels.append(
                        emotion_map[emotion_code]
                    )

X = np.array(features)
y = np.array(labels)

print("Feature Shape:", X.shape)
print("Label Shape:", y.shape)

print("\nFirst Feature Vector:")
print(X[0])
