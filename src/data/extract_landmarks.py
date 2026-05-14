from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

# Keep TensorFlow/MediaPipe startup logs quieter. setdefault lets callers override.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.data.config import PHRASE_CSV, WLASL_DIR
from src.utils.mediapipe_utils import (
    create_hands_detector,
    extract_landmark_vector,
    frame_to_mp_image,
)

PHRASES = [
    "hello",
    "goodbye",
    "please",
    "sorry",
    "thankyou",
    "yes",
    "no",
    "help",
    "stop",
    "more",
    "again",
    "wait",
    "understand",
    "water",
    "food",
    "bathroom",
    "home",
    "school",
    "work",
    "hospital",
    "me",
    "you",
    "family",
    "friend",
    "name",
    "what",
    "where",
    "when",
    "why",
    "how",
    "happy",
    "sad",
    "angry",
]

GLOSS_ALIASES = {
    "thankyou": "thank you",
}


def load_wlasl_index() -> dict[str, list[dict[str, object]]]:
    wlasl_root = Path(WLASL_DIR) / "wlasl-complete"
    json_path = wlasl_root / "WLASL_v0.3.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Could not find WLASL metadata at {json_path}. "
            "Run src/data/download_datasets.py first."
        )

    with json_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    return {entry["gloss"]: entry["instances"] for entry in entries}


def extract_video_landmarks(video_path: Path, hands, label: str, writer: csv.writer) -> int:
    cap = cv2.VideoCapture(str(video_path))
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = hands.detect(frame_to_mp_image(frame))
        if results.hand_landmarks:
            row = extract_landmark_vector(results.hand_landmarks[0])
            row.append(label)
            writer.writerow(row)
            count += 1

    cap.release()
    return count


def main() -> None:
    output_path = Path(PHRASE_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wlasl_root = Path(WLASL_DIR) / "wlasl-complete"
    videos_dir = wlasl_root / "videos"
    index = load_wlasl_index()

    hands = create_hands_detector(max_num_hands=1)
    try:
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]] + ["label"]
            writer.writerow(header)

            for phrase in PHRASES:
                gloss = GLOSS_ALIASES.get(phrase, phrase)
                instances = index.get(gloss)
                if not instances:
                    print(f"Missing WLASL gloss: {gloss} (label: {phrase})")
                    continue

                count = 0
                missing_videos = 0
                for instance in instances:
                    video_id = str(instance["video_id"])
                    video_path = videos_dir / f"{video_id}.mp4"
                    if not video_path.exists():
                        missing_videos += 1
                        continue
                    count += extract_video_landmarks(video_path, hands, phrase, writer)

                status = f"{phrase}: {count} samples"
                if missing_videos:
                    status += f" ({missing_videos} videos missing)"
                print(status)
    finally:
        hands.close()

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
