import os
import numpy as np
import librosa
import joblib

from sklearn.ensemble import RandomForestClassifier

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

                    mfcc = librosa.feature.mfcc(
                        y=audio,
                        sr=sr,
                        n_mfcc=40
                    )

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

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X, y)

joblib.dump(
    rf,
    "models/emotion_rf_model.pkl"
)

print("Model Saved Successfully")
