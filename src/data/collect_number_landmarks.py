# src/data/collect_number_landmarks.py
import csv
import os
import sys
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.utils.mediapipe_utils import (
    create_hands_detector,
    extract_landmark_vector,
    frame_to_mp_image,
)

NUMBERS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
SAMPLES = 200
FEATURE_COUNT = 63
OUTPUT_CSV = "datasets/number_landmarks.csv"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def draw_hand_overlay(frame, hand_landmarks):
    height, width = frame.shape[:2]
    points = []

    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    if len(points) != 21:
        return

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 220, 255), 2)

    for x, y in points:
        cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 8, (0, 0, 0), 1)


def get_hand_features(detection_result):
    if not detection_result.hand_landmarks:
        return None, None

    hand_landmarks = detection_result.hand_landmarks[0]
    features = extract_landmark_vector(hand_landmarks)
    if len(features) != FEATURE_COUNT:
        return None, hand_landmarks

    return features, hand_landmarks


def draw_roi(frame, color):
    height, width = frame.shape[:2]
    roi_size = 220
    roi_x = width - roi_size - 20
    roi_y = height - roi_size - 20
    cv2.rectangle(
        frame,
        (roi_x, roi_y),
        (roi_x + roi_size, roi_y + roi_size),
        color,
        2,
    )


def read_detect_and_draw(cap, hands, roi_color=(0, 255, 255), status_y=80):
    ret, frame = cap.read()
    if not ret:
        return False, None, None

    frame = cv2.flip(frame, 1)
    features = None
    hand_landmarks = None

    try:
        detection_result = hands.detect(frame_to_mp_image(frame))
        features, hand_landmarks = get_hand_features(detection_result)
    except Exception as exc:
        print(f"Detection error: {exc}", file=sys.__stderr__)

    draw_roi(frame, roi_color)

    if hand_landmarks:
        draw_hand_overlay(frame, hand_landmarks)
        cv2.putText(frame, "Hand detected", (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return True, frame, features


def main():
    os.makedirs("datasets", exist_ok=True)

    hands = create_hands_detector(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    cap = cv2.VideoCapture(0)

    try:
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            header = [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
            writer.writerow(header + ["label"])

            for number in NUMBERS:
                print(f"\nGet ready to sign: '{number}'")
                print("Press SPACE to start recording...")

                while True:
                    ok, frame, _ = read_detect_and_draw(cap, hands)
                    if not ok:
                        return

                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
                    cv2.putText(frame, f"Next: {number} - Press SPACE to start",
                                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 2)
                    cv2.imshow("Collect Number Landmarks", frame)
                    if cv2.waitKey(1) & 0xFF == ord(" "):
                        break

                count = 0
                print(f"Recording '{number}'...")

                while count < SAMPLES:
                    ok, frame, features = read_detect_and_draw(
                        cap,
                        hands,
                        roi_color=(0, 255, 0),
                    )
                    if not ok:
                        return

                    if features is not None:
                        writer.writerow(features + [number])
                        count += 1

                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
                    cv2.putText(frame, f"Sign '{number}': {count}/{SAMPLES}",
                                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                                (0, 255, 0), 2)
                    cv2.imshow("Collect Number Landmarks", frame)
                    cv2.waitKey(20)

                print(f"'{number}' done - {count} samples")
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nSaved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
