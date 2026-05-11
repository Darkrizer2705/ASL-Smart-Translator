# src/inference/predict_alphabet.py
import cv2
import numpy as np
import pickle
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

# ── Load Model ─────────────────────────────────────
print("Loading alphabet model...")
with open("models/alphabet_landmark_classifier.pkl", "rb") as f:
    model_data = pickle.load(f)
    model = model_data["model"]
    encoder = model_data["encoder"]

print(f"Model loaded — {len(encoder.classes_)} alphabet classes")
print(f"Classes: {list(encoder.classes_)}")

from src.utils.mediapipe_utils import (
    create_hands_detector,
    extract_landmark_vector,
    frame_to_mp_image,
)

hands = create_hands_detector(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
print("MediaPipe Tasks API loaded")
cap = cv2.VideoCapture(0)
print(" Alphabet recognition started. Press Q to quit.")

word = []
last_prediction = ""
stable_count = 0
STABLE_THRESHOLD = 20
CONFIDENCE_THRESHOLD = 0.6

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
            else:
                prediction = "Unknown"
                confidence = 0.0
        else:
            prediction = "No hand"
            confidence = 0.0

        if prediction == last_prediction and prediction not in ["No hand", "Unknown"]:
            stable_count += 1
        else:
            stable_count = 0
            last_prediction = prediction

        if stable_count == STABLE_THRESHOLD and confidence >= CONFIDENCE_THRESHOLD and prediction not in ["No hand", "Unknown"]:
            if prediction not in ["del", "nothing", "space"]:
                word.append(prediction.upper())
            elif prediction == "space":
                word.append(" ")
            elif prediction == "del" and word:
                word.pop()
            stable_count = 0

    except Exception as e:
        pass

    # ── Top bar ────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
    color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
    cv2.putText(frame, f"Letter: {prediction.upper()}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.0%}  Stable: {stable_count}/{STABLE_THRESHOLD}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # ── Bottom bar ─────────────────────────────────
    cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Word: {''.join(word)}",
                (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(frame, "C=Clear  U=Undo  Q=Quit",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("ASL Alphabet Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        word = []
        print("🗑️ Cleared")
    elif key == ord('u') and word:
        removed = word.pop()
        print(f"↩️ Removed: {removed}")

cap.release()
cv2.destroyAllWindows()
print(f"\n📝 Spelled: {''.join(word)}")
print("✅ Alphabet recognition ended")