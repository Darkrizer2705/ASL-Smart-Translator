"""
landmarks.py — Extract hand landmarks from gesture videos and save to CSV.

Uses the same MediaPipe Tasks API as the rest of the project (mediapipe_utils).
Run from the project root:
    python -m src.data.landmarks
"""
from __future__ import annotations

import csv
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

DATASET_PATH = ROOT_DIR / "datasets" / "wlasl"
OUTPUT_CSV   = ROOT_DIR / "datasets" / "gesture_dataset.csv"


def extract_from_video(video_path: Path, label: str, hands, writer: csv.writer) -> int:
    """Extract per-frame landmarks from a video and write rows to CSV."""
    cap = cv2.VideoCapture(str(video_path))
    count = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Skip every other frame to reduce duplicates
        if frame_count % 2 != 0:
            continue

        results = hands.detect(frame_to_mp_image(frame))

        if results.hand_landmarks:
            for hand_lms in results.hand_landmarks:
                normalized = extract_landmark_vector(hand_lms)
                if len(normalized) == 63:
                    writer.writerow(normalized + [label])
                    count += 1

    cap.release()
    return count


def main() -> None:
    if not DATASET_PATH.exists():
        print(f"Dataset path not found: {DATASET_PATH}")
        print("Place your gesture video folders under datasets/wlasl/")
        sys.exit(1)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Build column headers: x0,y0,z0, x1,y1,z1, ... x20,y20,z20, label
    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    header.append("label")

    hands = create_hands_detector(max_num_hands=1, min_detection_confidence=0.5)
    total = 0

    try:
        with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for category_dir in sorted(DATASET_PATH.iterdir()):
                if not category_dir.is_dir():
                    continue

                for gesture_dir in sorted(category_dir.iterdir()):
                    if not gesture_dir.is_dir():
                        continue

                    label = gesture_dir.name
                    print(f"Processing gesture: {label}")

                    for video_file in gesture_dir.iterdir():
                        if video_file.suffix.lower() in (".mp4", ".avi"):
                            n = extract_from_video(video_file, label, hands, writer)
                            total += n
    finally:
        hands.close()

    print(f"\n✅ Dataset created: {OUTPUT_CSV}  ({total} samples)")


if __name__ == "__main__":
    main()