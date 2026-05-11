# src/inference/predict_alphabet.py
import cv2
import numpy as np
import json
import tensorflow as tf

# ── Load Model ─────────────────────────────────────
model = tf.keras.models.load_model("models/alphabet_cnn.h5")
with open("models/alphabet_classes.json") as f:
    class_indices = json.load(f)
    classes = {v: k for k, v in class_indices.items()}

IMG_SIZE = (64, 64)
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

    # ── ROI — bottom RIGHT corner (where your hand naturally is) ──
    roi_size = 220
    roi_x = w - roi_size - 20   # right side
    roi_y = h - roi_size - 20   # bottom
    roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]

    # Draw ROI box in GREEN
    cv2.rectangle(frame, (roi_x, roi_y),
                  (roi_x+roi_size, roi_y+roi_size), (0, 255, 0), 2)
    cv2.putText(frame, "Show hand here",
                (roi_x, roi_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    prediction = ""
    confidence = 0.0

    try:
        img = cv2.resize(roi, IMG_SIZE)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        probs = model.predict(img, verbose=0)[0]
        confidence = np.max(probs)
        pred_idx = np.argmax(probs)
        prediction = classes[pred_idx]

        if prediction == last_prediction:
            stable_count += 1
        else:
            stable_count = 0
            last_prediction = prediction

        if stable_count == STABLE_THRESHOLD and confidence >= CONFIDENCE_THRESHOLD:
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