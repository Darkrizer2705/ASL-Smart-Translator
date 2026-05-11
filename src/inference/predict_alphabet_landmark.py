# src/inference/predict_alphabet_landmark.py
# Uses MediaPipe landmarks instead of raw image — much more accurate
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

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

# ── MediaPipe ──────────────────────────────────────
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
except:
    import mediapipe.python.solutions.hands as mp_hands_module
    hands = mp_hands_module.Hands(static_image_mode=False, max_num_hands=1,
                                  min_detection_confidence=0.7)
    mp_draw = None

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
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction = ""
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            if mp_draw:
                mp_draw.draw_landmarks(frame, hand_lms,
                    mp.solutions.hands.HAND_CONNECTIONS)

            row = []
            for lm in hand_lms.landmark:
                row.extend([lm.x, lm.y, lm.z])

            features = np.array(row).reshape(1, -1)
            probs = model.predict_proba(features)[0]
            confidence = np.max(probs)
            pred_idx = np.argmax(probs)
            prediction = encoder.inverse_transform([pred_idx])[0]

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
