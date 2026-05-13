import cv2
import numpy as np
import pickle
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.utils.mediapipe_utils import (
    create_hands_detector,
    draw_hand_landmarks,
    extract_landmark_vector,
    frame_to_mp_image,
)


LANDMARK_MODEL = ROOT_DIR / "models" / "alphabet_landmark_classifier.pkl"
STABLE_THRESHOLD = 20
CONFIDENCE_THRESHOLD = 0.65


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


if not os.path.exists(LANDMARK_MODEL):
    print("Landmark alphabet model not found.")
    print(f"Expected it at: {LANDMARK_MODEL}")
    print("Run: python src/models/train_alphabet_landmarks.py first")
    sys.exit(1)

with open(LANDMARK_MODEL, "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    encoder = data["encoder"]


def main():
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

    print(" Landmark-based alphabet recognition. Press Q to quit.")

    word = []
    last_prediction = ""
    stable_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera opened, but no frame was received.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            prediction = ""
            confidence = 0.0

            try:
                mp_image = frame_to_mp_image(frame)
                detection_result = hands.detect(mp_image)
                draw_hand_landmarks(frame, detection_result)

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

            except Exception as exc:
                print(f"Prediction error: {exc}", file=sys.stderr)

            cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
            color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.putText(
                frame,
                f"Letter: {prediction.upper()}",
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"Confidence: {confidence:.0%}",
                (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
            )

            cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"Word: {''.join(word)}",
                (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "C=Clear  U=Undo  Q=Quit",
                (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (150, 150, 150),
                1,
            )

            cv2.imshow("ASL Alphabet (Landmark)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                word = []
            elif key == ord("u") and word:
                word.pop()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print(f"\nSpelled: {''.join(word)}")


if __name__ == "__main__":
    main()
