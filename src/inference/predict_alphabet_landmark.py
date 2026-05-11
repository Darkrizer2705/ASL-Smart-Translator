# src/inference/predict_alphabet_landmark.py
# Uses MediaPipe landmarks instead of raw image — much more accurate
import cv2
import numpy as np
import pickle
import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.utils.mediapipe_utils import (
    create_hands_detector,
    extract_landmark_vector,
    frame_to_mp_image,
)

# ── Check if landmark-based alphabet model exists ──
LANDMARK_MODEL = "models/alphabet_landmark_classifier.pkl"

if not os.path.exists(LANDMARK_MODEL):
    print("Landmark alphabet model not found.")
    print("Run: python src/models/train_alphabet_landmarks.py first")
    exit()

with open(LANDMARK_MODEL, "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    encoder = data["encoder"]

# ── MediaPipe Tasks API ────────────────────────────
hands = create_hands_detector(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(0)
print(" Landmark-based alphabet recognition. Press Q to quit.")

word = []
last_prediction = ""
stable_count = 0
STABLE_THRESHOLD = 20
CONFIDENCE_THRESHOLD = 0.65

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    prediction = ""
    confidence = 0.0

    try:
        mp_image = frame_to_mp_image(frame)
        detection_result = hands.detect(mp_image)

        if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
            hand_landmarks = detection_result.hand_landmarks[0]
            features = extract_landmark_vector(hand_landmarks)
            
            if features is not None and len(features) == 63:
                features = np.array(features).reshape(1, -1)
                probs = model.predict_proba(features)[0]
                confidence = np.max(probs)
                pred_idx = np.argmax(probs)
                prediction = encoder.classes_[pred_idx]

                if prediction == last_prediction:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_prediction = prediction

                if stable_count == STABLE_THRESHOLD and confidence >= CONFIDENCE_THRESHOLD:
                    if prediction not in ["del", "nothing", "space"]:
                        word.append(prediction.upper())
                    elif prediction == "del" and word:
                        word.pop()
                    stable_count = 0
            else:
                prediction = "No hand"
                confidence = 0.0
        else:
            prediction = "No hand"
            confidence = 0.0

    except Exception as e:
        pass

    # ── UI ─────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
    color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
    cv2.putText(frame, f"Letter: {prediction.upper()}",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.0%}",
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.rectangle(frame, (0, h-60), (w, h), (0,0,0), -1)
    cv2.putText(frame, f"Word: {''.join(word)}",
                (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(frame, "C=Clear  U=Undo  Q=Quit",
                (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

    cv2.imshow("ASL Alphabet (Landmark)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        word = []
    elif key == ord('u') and word:
        word.pop()

cap.release()
cv2.destroyAllWindows()
print(f"\nSpelled: {''.join(word)}")
