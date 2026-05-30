import streamlit as st
import joblib
import librosa
import numpy as np
import tempfile

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="centered"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
model = joblib.load("models/emotion_rf_model.pkl")

# --------------------------------------------------
# Emotion Emojis
# --------------------------------------------------
emoji_map = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fearful": "😨",
    "neutral": "😐"
}

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🎤 Speech Emotion Recognition System")

st.markdown("""
This application analyzes speech audio and predicts the speaker's emotional state using a Machine Learning model trained on the **RAVDESS Speech Emotion Dataset**.

### Supported Emotions
- 😊 Happy
- 😢 Sad
- 😠 Angry
- 😨 Fearful
- 😐 Neutral
""")

# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload a WAV Audio File",
    type=["wav"]
)

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
if uploaded_file is not None:

    st.subheader("🎵 Uploaded Audio")
    st.audio(uploaded_file)

    try:

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        # Load Audio
        audio, sr = librosa.load(
            temp_path,
            duration=3,
            offset=0.5
        )

        # MFCC Feature Extraction
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

        # Prediction
        prediction = model.predict(mfcc_mean)
        emotion = prediction[0]

        # Probability Scores
        probabilities = model.predict_proba(mfcc_mean)

        confidence = np.max(probabilities) * 100

        # Result
        st.subheader("🎯 Prediction Result")

        st.success(
            f"{emoji_map.get(emotion, '🎤')} Predicted Emotion: {emotion.upper()}"
        )

        st.info(
            f"Confidence Score: {confidence:.2f}%"
        )

        # Probability Breakdown
        st.subheader("📊 Emotion Probabilities")

        emotion_labels = model.classes_

        for label, prob in zip(
            emotion_labels,
            probabilities[0]
        ):

            st.progress(float(prob))

            st.write(
                f"**{label.capitalize()}** : {prob*100:.2f}%"
            )

    except Exception as e:

        st.error(
            f"Error Processing Audio: {str(e)}"
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")

st.markdown("""
### 🛠 Technologies Used

- Python
- Librosa
- NumPy
- Scikit-Learn
- Streamlit
- Random Forest Classifier
- RAVDESS Dataset

---

Developed as part of an **AI/ML Internship Project on Speech Emotion Recognition**.
""")
