"""
Test real-time predictions with debugging to identify issues
"""
import cv2
import numpy as np
import pickle
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

# ── Load Model ──────────────────────────────────────
print("Loading model...")
with open("models/phrase_classifier.pkl", "rb") as f:
    data = pickle.load(f)
    model   = data["model"]
    encoder = data["encoder"]

print(f"Model expects: {model.n_features_in_} features")
print(f"Classes: {list(encoder.classes_)}")

# ── MediaPipe Setup ─────────────────────────────────
import mediapipe as mp
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

# ── Test Settings ───────────────────────────────────
cap = cv2.VideoCapture(0)
print("\nSettings to test:")
print("  - Press SPACE to capture a hand and test prediction")
print("  - Press 'R' to toggle real-time mode")
print("  - Press 'Q' to quit\n")

real_time_mode = True
frame_count = 0
detection_count = 0
successful_features = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    frame_count += 1
    
    results = hands.detect(frame_to_mp_image(frame))

    # Draw frame info
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Frames: {frame_count} | Detections: {detection_count}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Valid features: {successful_features}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if results.hand_landmarks and len(results.hand_landmarks) > 0:
        detection_count += 1
        hand_lms = results.hand_landmarks[0]

        # Draw landmarks
        for lm in hand_lms:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # Try feature extraction
        try:
            row = extract_landmark_vector(hand_lms)
            
            if len(row) == model.n_features_in_:
                successful_features += 1
                
                # Make prediction
                features = np.array(row, dtype=np.float32).reshape(1, -1)
                probs = model.predict_proba(features)[0]
                confidence = float(np.max(probs))
                pred_idx = int(np.argmax(probs))
                prediction = encoder.classes_[pred_idx]

                # Display prediction
                color = (0, 255, 0) if confidence >= 0.70 else (0, 165, 255)
                cv2.putText(frame, f"{prediction} ({confidence:.0%})",
                            (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, color, 2)

                # Show top 3 predictions
                top_idx = np.argsort(probs)[::-1][:3]
                for i, idx in enumerate(top_idx):
                    lbl = encoder.classes_[idx]
                    prob = probs[idx]
                    cv2.putText(frame, f"{i+1}. {lbl}: {prob:.1%}",
                                (10, h - 10 - i*25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (200, 200, 200), 1)
            else:
                cv2.putText(frame, f"⚠️  Feature mismatch: {len(row)} vs {model.n_features_in_}",
                            (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
        except Exception as e:
            cv2.putText(frame, f"⚠️  Error: {str(e)[:50]}",
                        (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No hand detected",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (100, 100, 100), 2)

    cv2.imshow("Prediction Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        real_time_mode = not real_time_mode
        print(f"Real-time mode: {'ON' if real_time_mode else 'OFF'}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print(f"Total frames: {frame_count}")
print(f"Hand detections: {detection_count} ({detection_count/max(frame_count, 1)*100:.1f}%)")
print(f"Successful features: {successful_features}")

if detection_count > 0:
    print(f"Feature extraction success rate: {successful_features/detection_count*100:.1f}%")