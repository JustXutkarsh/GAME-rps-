import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import random
import time

# Load model and define classes
model = load_model("rps_model.h5")
class_names = ["rock", "paper", "scissors"]

# Initialize Mediapipe and camera
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

# Game variables
player_move = None
ai_move = None
winner = None
player_score = 0
ai_score = 0
draws = 0
last_move_time = 0
move_delay = 3  # seconds between moves

# Function to decide winner
def get_winner(player, ai):
    if player == ai:
        return "Draw"
    elif (player == "rock" and ai == "scissors") or \
         (player == "paper" and ai == "rock") or \
         (player == "scissors" and ai == "paper"):
        return "Player"
    else:
        return "AI"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_time = time.time()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_min = w
            y_min = h
            x_max = y_max = 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            hand_img = cv2.resize(hand_img, (128, 128))
            hand_img = hand_img / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            pred = model.predict(hand_img)
            player_move = class_names[np.argmax(pred)]

            # Update AI move and scores every few seconds
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

    # Display scoreboard and info
    cv2.putText(frame, f"Your Move: {player_move if player_move else '-'}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"AI Move: {ai_move if ai_move else '-'}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Winner: {winner if winner else '-'}", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show scores
    cv2.putText(frame, f"Player: {player_score}", (20, h - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"AI: {ai_score}", (220, h - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(frame, f"Draws: {draws}", (380, h - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Timer indicator for next move
    cv2.putText(frame, f"Next move in: {max(0, int(move_delay - (current_time - last_move_time)))}s",
                (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

    cv2.imshow("Rock Paper Scissors - AI vs You", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
