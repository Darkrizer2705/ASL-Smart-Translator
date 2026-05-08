import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

DATA = []

# -------- FUNCTION TO PROCESS EACH VIDEO --------
def extract_from_video(video_path, label):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # OPTIONAL: skip frames to reduce duplicates
        if frame_count % 2 != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                landmarks = []

                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                    landmarks.append(lm.z)

                # -------- NORMALIZATION --------
                base_x, base_y, base_z = landmarks[0], landmarks[1], landmarks[2]

                normalized = []
                for i in range(0, len(landmarks), 3):
                        normalized.append(landmarks[i] - base_x)
                        normalized.append(landmarks[i+1] - base_y)
                        normalized.append(landmarks[i+2] - base_z)

                # Add label at the end
                normalized.append(label)

                DATA.append(normalized)

    cap.release()


# -------- MAIN LOOP --------
dataset_path = "gesture_datapath"

for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)

    if not os.path.isdir(category_path):
        continue

    for label in os.listdir(category_path):
        gesture_path = os.path.join(category_path, label)

        if not os.path.isdir(gesture_path):
            continue

        print(f"Processing gesture: {label}")

        for video in os.listdir(gesture_path):
            video_path = os.path.join(gesture_path, video)

            if video.endswith(".mp4") or video.endswith(".avi"):
                extract_from_video(video_path, label)


# -------- SAVE TO CSV --------
columns = []

for i in range(21):
    columns.append(f"x{i}")
    columns.append(f"y{i}")
    columns.append(f"z{i}")

columns.append("label")

df = pd.DataFrame(DATA, columns=columns)

df.to_csv("gesture_dataset.csv", index=False)

print("✅ Dataset created: gesture_dataset.csv")