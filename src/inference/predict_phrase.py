# src/inference/predict_phrase.py
from pathlib import Path
import sys

import cv2
import numpy as np
import pickle

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

# ── Load Model ─────────────────────────────────────
print("Loading phrase model...")
PHRASE_MODEL = ROOT_DIR / "models" / "phrase_classifier.pkl"
if not PHRASE_MODEL.exists():
    print("Phrase model not found.")
    print(f"Expected it at: {PHRASE_MODEL}")
    print("Run: python src/models/train_phrases.py first")
    sys.exit(1)

with open(PHRASE_MODEL, "rb") as f:
    data = pickle.load(f)
    model   = data["model"]
    encoder = data["encoder"]

print(f"Model loaded — {len(encoder.classes_)} phrases")
print(f"Classes: {list(encoder.classes_)}")
print(f"Expected features: {model.n_features_in_}")

from src.utils.mediapipe_utils import (
    create_hands_detector,
    draw_hand_landmarks,
    extract_landmark_vector,
    frame_to_mp_image,
)

hands = create_hands_detector(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
print("MediaPipe Tasks API loaded")


def build_feature_vector(hand_landmarks_list, expected_features):
    """Build a feature row from one or two detected hands.

    Current models in this repo are mostly trained on 63 features (one hand).
    When two hands are available, we still capture both sets of features and
    concatenate them when the model can consume the larger vector.
    """
    if not hand_landmarks_list:
        return None

    hand_rows = [extract_landmark_vector(hand_lms) for hand_lms in hand_landmarks_list]

    # If the model was trained on a single hand, keep the most prominent hand
    # so inference remains compatible, but still capture both hands above.
    if expected_features == 63:
        best_row = max(hand_rows, key=len)
        return best_row if len(best_row) == expected_features else None

    # If the model expects more features, concatenate all detected hands.
    combined_row = [value for row in hand_rows for value in row]
    return combined_row if len(combined_row) == expected_features else None

# ── State ──────────────────────────────────────────
sentence        = []
last_pred       = ""
stable_count    = 0
probs           = np.zeros(len(encoder.classes_))

# TUNED THRESHOLDS for better real-time performance
STABLE_THRESHOLD     = 15  # Reduced from 20 for faster response
CONFIDENCE_THRESHOLD = 0.60  # Lowered from 0.70 for more predictions
UNKNOWN_THRESHOLD    = 0.50  # Below this = "Unknown"

cap = cv2.VideoCapture(0)
print("\nStarted. Controls: A=Add word  C=Clear  U=Undo  Q=Quit\n")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame      = cv2.flip(frame, 1)
    h, w       = frame.shape[:2]
    results = hands.detect(frame_to_mp_image(frame))
    draw_hand_landmarks(frame, results)

    prediction  = ""
    confidence  = 0.0
    hand_found  = False

    if results.hand_landmarks and len(results.hand_landmarks) > 0:
        # Build features from one or two hands.
        row = build_feature_vector(results.hand_landmarks, model.n_features_in_)

        # Verify feature count
        if row is None or len(row) != model.n_features_in_:
            print(
                f"Feature mismatch: got {0 if row is None else len(row)}, expected {model.n_features_in_}"
            )
            continue

        features   = np.array(row, dtype=np.float32).reshape(1, -1)
        probs      = model.predict_proba(features)[0]
        confidence = float(np.max(probs))
        pred_idx   = int(np.argmax(probs))
        prediction = encoder.classes_[pred_idx]
        if confidence < UNKNOWN_THRESHOLD:
            prediction = "Unknown"
        hand_found = True

        # ── Stability check ───────────────────────
        if prediction == last_pred:
            stable_count += 1
        else:
            stable_count = 0
            last_pred    = prediction

        # Auto-add when stable + confident
        if prediction != "Unknown" and (
            stable_count == STABLE_THRESHOLD and
            confidence >= CONFIDENCE_THRESHOLD):
            if not sentence or sentence[-1] != prediction:
                sentence.append(prediction)
                print(f"Added: {prediction}")
            stable_count = 0

    else:
        stable_count = 0

    # ══════════════ UI ═══════════════════════════

    # Top bar — current prediction
    cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)

    if hand_found:
        color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
        cv2.putText(frame, f"{prediction.upper()}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, color, 3)
        # Stability progress bar
        bar_w = int((stable_count / STABLE_THRESHOLD) * 300)
        cv2.rectangle(frame, (10, 70), (310, 88), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, 70), (10 + bar_w, 88), color, -1)
        cv2.putText(frame, f"{confidence:.0%}",
                    (320, 85), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1)
    else:
        cv2.putText(frame, "Show your hand...",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (100, 100, 100), 2)

    # Probability bars — top 5
    sorted_idx = np.argsort(probs)[::-1][:5]
    for i, idx in enumerate(sorted_idx):
        lbl  = encoder.classes_[idx]
        prob = probs[idx]
        by   = 110 + i * 28
        bw   = int(prob * 220)
        col  = (0, 200, 100) if i == 0 else (0, 120, 60)
        cv2.rectangle(frame, (10, by), (10 + bw, by + 20), col, -1)
        cv2.putText(frame, f"{lbl}: {prob:.0%}",
                    (240, by + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    # Sentence bar — bottom
    cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 0), -1)
    sentence_display = " > ".join(sentence[-6:]) if sentence else "—"
    cv2.putText(frame, sentence_display,
                (10, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, "A=Add  C=Clear  U=Undo  Q=Quit",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("ASL Phrase Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = []
        print("Cleared")
    elif key == ord('u') and sentence:
        removed = sentence.pop()
        print(f"Removed: {removed}")
    elif key == ord('a') and prediction and confidence >= CONFIDENCE_THRESHOLD:
        if not sentence or sentence[-1] != prediction:
            sentence.append(prediction)
            print(f"Manually added: {prediction}")

cap.release()
cv2.destroyAllWindows()

if sentence:
    print(f"\nFinal sentence words: {sentence}")
    print(f"Raw sentence: {' '.join(sentence).upper()}")
