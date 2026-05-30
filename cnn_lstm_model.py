import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    TimeDistributed,
    Conv1D,
    MaxPooling1D,
    Flatten,
    LSTM,
    Dense,
    Dropout
)

# Load Augmented Dataset
X = np.load("X_aug.npy")
y = np.load("y_aug.npy")

print("Original Shape:", X.shape)

# Convert:
# (2592, 40, 173, 1)
# ->
# (2592, 40, 173)

X = X.squeeze(-1)

print("New Shape:", X.shape)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y, axis=1)
)

model = Sequential()

# CNN Feature Extraction
model.add(
    TimeDistributed(
        Conv1D(
            filters=32,
            kernel_size=3,
            activation='relu'
        ),
        input_shape=(40, 173, 1)
    )
)

model.add(
    TimeDistributed(
        MaxPooling1D(pool_size=2)
    )
)

model.add(
    TimeDistributed(
        Flatten()
    )
)

# LSTM
model.add(
    LSTM(
        64,
        return_sequences=False
    )
)

model.add(
    Dense(
        64,
        activation='relu'
    )
)

model.add(
    Dropout(0.5)
)

model.add(
    Dense(
        5,
        activation='softmax'
    )
)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train.reshape(
        X_train.shape[0],
        40,
        173,
        1
    ),
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(
        X_test.reshape(
            X_test.shape[0],
            40,
            173,
            1
        ),
        y_test
    )
)

loss, accuracy = model.evaluate(
    X_test.reshape(
        X_test.shape[0],
        40,
        173,
        1
    ),
    y_test
)

print("\nTest Accuracy:", accuracy)

model.save(
    "models/cnn_lstm_emotion.keras"
)
