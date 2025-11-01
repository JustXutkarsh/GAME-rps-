import streamlit as st
import cv2
import numpy as np
import tempfile
from tensorflow.keras.models import load_model
import mediapipe as mp
import random
import time

# Set page config
st.set_page_config(page_title="Rock Paper Scissors AI", layout="centered")

# Load model
model = load_model("model/rps_model.h5")
class_names = ["rock", "paper", "scissors"]

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Title
st.title("🪨📄✂️ Rock Paper Scissors - AI vs You")
st.write("Show your move to the camera! The AI will predict and play against you.")

# Start camera
run = st.checkbox("Start Game")
FRAME_WINDOW = st.image([])

player_score = 0
ai_score = 0
draws = 0
last_move_time = 0
move_delay = 3
player_move = None
ai_move = None
winner = None

cap = cv2.VideoCapture(0)

def get_winner(player, ai):
    if player == ai:
        return "Draw"
    elif (player == "rock" and ai == "scissors") or \
         (player == "paper" and ai == "rock") or \
         (player == "scissors" and ai == "paper"):
        return "Player"
    else:
        return "AI"

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("⚠️ Unable to access webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    current_time = time.time()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            margin = 20
            x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
            x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            hand_img = cv2.resize(hand_img, (128, 128))
            hand_img = hand_img / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            pred = model.predict(hand_img)
            player_move = class_names[np.argmax(pred)]

            if current_time - last_move_time > move_delay:
                ai_move = random.choice(class_names)
                winner = get_winner(player_move, ai_move)
                last_move_time = current_time

                if winner == "Player":
                    player_score += 1
                elif winner == "AI":
                    ai_score += 1
                else:
                    draws += 1

    # Overlay text on frame
    cv2.putText(frame, f"Your Move: {player_move if player_move else '-'}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"AI Move: {ai_move if ai_move else '-'}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Winner: {winner if winner else '-'}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
st.success(f"✅ Final Scores — You: {player_score}, AI: {ai_score}, Draws: {draws}")
