# src/inference/predict_number.py
import cv2
import mediapipe as mp
import numpy as np
import pickle

# ── Load landmark model ────────────────────────────
with open("models/number_landmark_classifier.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    encoder = data["encoder"]

print(f"✅ Classes: {list(encoder.classes_)}")

# ── MediaPipe ──────────────────────────────────────
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
except:
    import mediapipe.python.solutions.hands as mp_hands_module
    hands = mp_hands_module.Hands(min_detection_confidence=0.7)
    mp_draw = None

cap = cv2.VideoCapture(0)
print("✅ Number recognition started. Press Q to quit.")

last_prediction = ""
stable_count = 0
STABLE_THRESHOLD = 15
CONFIDENCE_THRESHOLD = 0.6
probs = np.zeros(len(encoder.classes_))

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

            row = [coord for lm in hand_lms.landmark
                   for coord in [lm.x, lm.y, lm.z]]
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

    # ── Top bar ────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
    color = (0, 255, 255) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
    cv2.putText(frame, f"Number: {prediction}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.0%}   Stable: {stable_count}/{STABLE_THRESHOLD}",
                (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # ── Probability bars (top 5) ───────────────────
    sorted_idx = np.argsort(probs)[::-1][:5]
    for i, idx in enumerate(sorted_idx):
        label = encoder.classes_[idx]
        prob  = probs[idx]
        bar_y = 100 + i * 26
        bar_w = int(prob * 180)
        cv2.rectangle(frame, (10, bar_y), (10 + bar_w, bar_y + 18),
                      (0, 200, 200), -1)
        cv2.putText(frame, f"{label}: {prob:.0%}",
                    (200, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    cv2.imshow("ASL Number Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()