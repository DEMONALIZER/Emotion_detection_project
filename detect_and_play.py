import cv2
import numpy as np
from tensorflow.keras.models import load_model
import random
import pygame
import os


model = load_model("mood_model.h5")
moods = ['happy', 'sad', 'angry']

def play_song(mood):
    folder = f"songs/{mood}/"
    if not os.path.exists(folder):
        print(f"❌ Folder not found: {folder}")
        return

    songs = [f for f in os.listdir(folder) if f.endswith(".mp3") or f.endswith(".wav")]
    if not songs:
        print(f"❌ No songs in {folder}")
        return

    pygame.init()
    pygame.mixer.init()
    song = os.path.join(folder, random.choice(songs))
    pygame.mixer.music.load(song)
    pygame.mixer.music.play()
    print(f"🎵 Playing {mood} song: {song}")
    while pygame.mixer.music.get_busy():
        continue

def predict_mood_from_frame(frame):
    try:
        img = cv2.resize(frame, (48, 48))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)[0]
        mood_index = np.argmax(prediction)
        return moods[mood_index]
    except Exception as e:
        print(f"⚠️ Error in processing frame: {e}")
        return None

def capture_and_predict():
    cap = cv2.VideoCapture(0)
    print("📷 Press SPACE to capture mood... (ESC to exit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from camera.")
            break

    
        cv2.imshow("Mood Detection - Press SPACE to Capture", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key to quit
            print("👋 Exiting.")
            break
        elif key == 32:  # SPACE key to capture
            mood = predict_mood_from_frame(frame)
            if mood:
                print(f"✅ Detected Mood: {mood}")
                play_song(mood)
            else:
                print("⚠️ Couldn't detect mood.")

    cap.release()
    cv2.destroyAllWindows()

capture_and_predict()