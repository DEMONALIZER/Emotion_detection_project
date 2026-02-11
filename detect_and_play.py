import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import random
from PIL import Image

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="Emotion Music Player", page_icon="😊")

st.title("😊 Emotion Detection Music Player")
st.write("Capture your mood and get a matching song!")

# ------------------------------
# Load Model (cached for speed)
# ------------------------------
@st.cache_resource
def load_emotion_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "mood_model.h5")
    return load_model(model_path)

model = load_emotion_model()
moods = ['happy', 'sad', 'angry']

# ------------------------------
# Prediction Function
# ------------------------------
def predict_mood_from_frame(frame):
    try:
        img = cv2.resize(frame, (48, 48))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        mood_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        return moods[mood_index], confidence
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

# ------------------------------
# Play Song Function
# ------------------------------
def play_song(mood):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(BASE_DIR, "songs", mood)

    if not os.path.exists(folder):
        st.error(f"Folder not found: {folder}")
        return

    songs = [f for f in os.listdir(folder) if f.endswith(".mp3") or f.endswith(".wav")]

    if not songs:
        st.error(f"No songs found in {folder}")
        return

    song_path = os.path.join(folder, random.choice(songs))
    st.audio(song_path)

# ------------------------------
# Camera Input
# ------------------------------
img_file = st.camera_input("📷 Take a picture")

if img_file is not None:
    image = Image.open(img_file)
    frame = np.array(image)

    st.image(frame, caption="Captured Image", channels="RGB")

    with st.spinner("Detecting mood..."):
        mood, confidence = predict_mood_from_frame(frame)

    if mood:
        st.success(f"Detected Mood: {mood}")
        st.write(f"Confidence: {confidence:.2f}")

        # Optional visual effect
        if mood == "happy":
            st.balloons()

        play_song(mood)
