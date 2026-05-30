import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

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

                    if mfcc.shape[1] < 173:
                        pad_width = 173 - mfcc.shape[1]
                        mfcc = np.pad(
                            mfcc,
                            pad_width=((0, 0), (0, pad_width)),
                            mode='constant'
                        )
                    else:
                        mfcc = mfcc[:, :173]

                    features.append(mfcc)
                    labels.append(
                        emotion_map[emotion_code]
                    )

X = np.array(features)

X = X.reshape(
    X.shape[0],
    X.shape[1],
    X.shape[2],
    1
)

encoder = LabelEncoder()
y = encoder.fit_transform(labels)
y = to_categorical(y)

print("X Shape:", X.shape)
print("y Shape:", y.shape)

np.save("X.npy", X)
np.save("y.npy", y)

print("Data Saved Successfully")
