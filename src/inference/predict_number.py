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
    extract_landmark_vector,
    frame_to_mp_image,
)


CNN_MODEL = "models/number_cnn.h5"
CLASS_INDICES_FILE = "models/number_classes.json"
IMG_SIZE = (64, 64)
STABLE_THRESHOLD = 15
CONFIDENCE_THRESHOLD = 0.6


# ── Check models exist ─────────────────────────────
if not os.path.exists(CNN_MODEL):
    print("❌ CNN model not found.")
    print("Run: python src/models/train_numbers.py first")
    sys.exit(1)

if not os.path.exists(CLASS_INDICES_FILE):
    print("❌ Class indices file not found.")
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


def main():
    cap = open_camera(0)
    if cap is None:
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

            # ── Define prediction box (right side, 300x300) ────────────
            box_size = 300
            box_x1 = w - box_size - 20  # 20px margin from right edge
            box_y1 = (h - box_size) // 2  # Centered vertically
            box_x2 = box_x1 + box_size
            box_y2 = box_y1 + box_size

            prediction = ""
            confidence = 0.0

            try:
                # Extract only the box region
                box_region = frame[box_y1:box_y2, box_x1:box_x2]
                
                # Preprocess box region for CNN
                input_frame = preprocess_frame(box_region)
                
                # Get predictions
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

            except Exception as exc:
                print(f"❌ Prediction error: {exc}", file=sys.stderr)
                prediction = "Error"
                confidence = 0.0

            # ── Draw prediction box ────────────────────────────────
            box_color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), box_color, 3)

            # ── Top bar ────────────────────────────────
            cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
            color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.putText(
                frame,
                f"Number: {prediction}",
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


if __name__ == "__main__":
    main()