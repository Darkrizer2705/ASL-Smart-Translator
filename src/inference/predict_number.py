import cv2
import json
import numpy as np
import os
import sys
import tensorflow as tf
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.utils.mediapipe_utils import (
    create_hands_detector,
    draw_hand_landmarks,
    frame_to_mp_image,
    get_hand_bbox,
)


CNN_MODEL = ROOT_DIR / "models" / "number_cnn.h5"
CLASS_INDICES_FILE = ROOT_DIR / "models" / "number_classes.json"
IMG_SIZE = (64, 64)
STABLE_THRESHOLD = 15
CONFIDENCE_THRESHOLD = 0.6


# ── Check models exist ─────────────────────────────
if not os.path.exists(CNN_MODEL):
    print("❌ CNN model not found.")
    print(f"Expected it at: {CNN_MODEL}")
    print("Run: python src/models/train_numbers.py first")
    sys.exit(1)

if not os.path.exists(CLASS_INDICES_FILE):
    print("❌ Class indices file not found.")
    print(f"Expected it at: {CLASS_INDICES_FILE}")
    print("Run: python src/models/train_numbers.py first")
    sys.exit(1)

# ── Load model & classes ───────────────────────────
print("Loading number CNN model...")
model = tf.keras.models.load_model(CNN_MODEL)

with open(CLASS_INDICES_FILE, "r") as f:
    class_indices = json.load(f)
    # Reverse mapping: index -> class name
    idx_to_class = {v: k for k, v in class_indices.items()}
    num_classes = len(idx_to_class)

print(f"✅ Classes: {sorted(class_indices.keys())}")


def open_camera(camera_index=0):
    """Open camera with multiple backend fallbacks."""
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


def preprocess_frame(frame):
    """Resize and normalize frame for CNN."""
    # Convert BGR (OpenCV) to RGB (model expects RGB from PIL training)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, IMG_SIZE)
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)


def square_crop_from_bbox(frame, bbox, padding=25):
    """Return a square crop around the detected hand so prediction follows it."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    side = max(x2 - x1, y2 - y1) + padding * 2
    side = min(side, w, h)

    crop_x1 = max(0, min(w - side, cx - side // 2))
    crop_y1 = max(0, min(h - side, cy - side // 2))
    crop_x2 = crop_x1 + side
    crop_y2 = crop_y1 + side

    return frame[crop_y1:crop_y2, crop_x1:crop_x2], (
        int(crop_x1),
        int(crop_y1),
        int(crop_x2),
        int(crop_y2),
    )


def main():
    hands = create_hands_detector(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    cap = open_camera(0)
    if cap is None:
        hands.close()
        print("❌ Could not open camera 0.")
        print("Close other apps using the webcam, check Windows camera permissions, or try another camera index.")
        return

    print("✅ Number recognition started. Press Q to quit.")

    last_prediction = ""
    stable_count = 0
    all_probs = np.zeros(num_classes)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera opened, but no frame was received.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            prediction = ""
            confidence = 0.0
            hand_found = False
            crop_box = None

            try:
                clean_frame = frame.copy()
                detection_result = hands.detect(frame_to_mp_image(frame))
                draw_hand_landmarks(frame, detection_result)

                if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
                    hand_found = True
                    bbox = get_hand_bbox(detection_result.hand_landmarks[0], w, h, padding=20)

                    if bbox:
                        box_region, crop_box = square_crop_from_bbox(clean_frame, bbox)
                        input_frame = preprocess_frame(box_region)

                        probs = model.predict(input_frame, verbose=0)[0]
                        all_probs = probs
                        confidence = np.max(probs)
                        pred_idx = np.argmax(probs)
                        prediction = idx_to_class[pred_idx]

                        if prediction == last_prediction:
                            stable_count += 1
                        else:
                            stable_count = 0
                            last_prediction = prediction
                    else:
                        prediction = "No hand"
                        stable_count = 0
                else:
                    prediction = "No hand"
                    stable_count = 0

            except Exception as exc:
                print(f"❌ Prediction error: {exc}", file=sys.stderr)
                prediction = "Error"
                confidence = 0.0

            # ── Draw moving prediction box around the tracked hand ────────────
            box_color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
            if crop_box:
                box_x1, box_y1, box_x2, box_y2 = crop_box
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), box_color, 3)

            # ── Top bar ────────────────────────────────
            cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
            color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.putText(
                frame,
                f"Number: {prediction if hand_found else 'Show your hand'}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"Confidence: {confidence:.0%}   Stable: {stable_count}/{STABLE_THRESHOLD}",
                (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )

            # ── Probability bars (top 5) ───────────────
            sorted_idx = np.argsort(all_probs)[::-1][:5]
            for i, idx in enumerate(sorted_idx):
                label = idx_to_class[idx]
                prob = all_probs[idx]
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

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
