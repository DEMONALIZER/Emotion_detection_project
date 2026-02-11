import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import random
from PIL import Image

# Load model
model = load_model("mood_model.h5")
moods = ['happy', 'sad', 'angry']

st.title("😊 Emotion Detection Music Player")

def predict_mood_from_frame(frame):
    try:
        img = cv2.resize(frame, (48, 48))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        mood_index = np.argmax(prediction)
        return moods[mood_index]
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def play_song(mood):
    folder = f"songs/{mood}/"
    if not os.path.exists(folder):
        st.error(f"Folder not found: {folder}")
        return

    songs = [f for f in os.listdir(folder) if f.endswith(".mp3") or f.endswith(".wav")]
    if not songs:
        st.error(f"No songs found in {folder}")
        return

    song_path = os.path.join(folder, random.choice(songs))
    st.audio(song_path)

# Camera input
img_file = st.camera_input("Take a picture")

if img_file is not None:
    image = Image.open(img_file)
    frame = np.array(image)

    st.image(frame, caption="Captured Image", channels="RGB")

    mood = predict_mood_from_frame(frame)

    if mood:
        st.success(f"Detected Mood: {mood}")
        play_song(mood)
