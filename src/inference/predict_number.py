import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.utils.mediapipe_utils import (
    create_hands_detector,
    draw_hand_landmarks,
    extract_landmark_vector,
    frame_to_mp_image,
)

LANDMARK_MODEL = ROOT_DIR / "models" / "number_landmark_classifier.pkl"
STABLE_THRESHOLD = 12
CONFIDENCE_THRESHOLD = 0.65
FEATURE_COUNT = 63


def open_camera(camera_index=0):
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("Default", cv2.CAP_ANY),
    ]

    for backend_name, backend in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Camera opened with {backend_name} backend.")
                return cap

        cap.release()

    return None


def load_model():
    if not os.path.exists(LANDMARK_MODEL):
        print("Number landmark model not found.")
        print(f"Expected it at: {LANDMARK_MODEL}")
        print("Run: python src/models/train_number_landmarks.py first")
        sys.exit(1)

    with open(LANDMARK_MODEL, "rb") as f:
        data = pickle.load(f)

    return data["model"], data["encoder"]


def predict_number(model, encoder, hand_landmarks):
    features = extract_landmark_vector(hand_landmarks)
    if len(features) != FEATURE_COUNT:
        return "", 0.0, np.zeros(len(encoder.classes_))

    features = np.array(features, dtype=np.float32).reshape(1, -1)
    probs = model.predict_proba(features)[0]
    confidence = float(np.max(probs))
    pred_idx = int(np.argmax(probs))
    prediction = str(encoder.classes_[pred_idx])
    return prediction, confidence, probs


def main():
    model, encoder = load_model()
    print(f"Classes: {list(encoder.classes_)}")

    hands = create_hands_detector(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    cap = open_camera(0)
    if cap is None:
        hands.close()
        print("Could not open camera 0.")
        print("Close other apps using the webcam, check Windows camera permissions, or try another camera index.")
        return

    print("Landmark-based number recognition started. Press Q to quit.")

    last_prediction = ""
    stable_count = 0
    all_probs = np.zeros(len(encoder.classes_))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera opened, but no frame was received.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            prediction = "No hand"
            confidence = 0.0
            hand_found = False

            try:
                detection_result = hands.detect(frame_to_mp_image(frame))
                draw_hand_landmarks(frame, detection_result)

                if detection_result.hand_landmarks:
                    hand_found = True
                    prediction, confidence, all_probs = predict_number(
                        model,
                        encoder,
                        detection_result.hand_landmarks[0],
                    )

                    if prediction == last_prediction:
                        stable_count += 1
                    else:
                        stable_count = 0
                        last_prediction = prediction
                else:
                    stable_count = 0

            except Exception as exc:
                print(f"Prediction error: {exc}", file=sys.stderr)
                prediction = "Error"
                confidence = 0.0

            color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
            shown_prediction = prediction if hand_found else "Show your hand"
            if hand_found and confidence < CONFIDENCE_THRESHOLD:
                shown_prediction = f"{prediction}?"

            cv2.rectangle(frame, (0, 0), (width, 90), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"Number: {shown_prediction}",
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"Confidence: {confidence:.0%}   Stable: {stable_count}/{STABLE_THRESHOLD}",
                (10, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
            )

            top_idx = np.argsort(all_probs)[::-1][:5]
            for i, idx in enumerate(top_idx):
                label = str(encoder.classes_[idx])
                prob = float(all_probs[idx])
                bar_y = 105 + i * 26
                bar_w = int(prob * 180)
                cv2.rectangle(frame, (10, bar_y), (10 + bar_w, bar_y + 18), (0, 200, 200), -1)
                cv2.putText(
                    frame,
                    f"{label}: {prob:.0%}",
                    (200, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (220, 220, 220),
                    1,
                )

            cv2.putText(
                frame,
                "Q=Quit",
                (10, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )

            cv2.imshow("ASL Number Recognition (Landmark)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
