# src/data/collect_alphabet_landmarks.py
import cv2
import csv
import time
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.utils.mediapipe_utils import (
    create_hands_detector,
    extract_landmark_vector,
    frame_to_mp_image,
)

hands = create_hands_detector(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(0)

ALPHABETS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space", "nothing"]
SAMPLES = 100   # samples per letter

OUTPUT_CSV = "datasets/alphabet_landmarks.csv"
os.makedirs("datasets", exist_ok=True)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    header = [f"{a}{i}" for i in range(21) for a in ["x","y","z"]] + ["label"]
    writer.writerow(header)

    for letter in ALPHABETS:
        print(f"\n✋ Get ready to sign: '{letter}'")
        print("Press SPACE to start recording...")

        # Wait for space key
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Next: {letter} — Press SPACE to start",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("Collect Alphabet Data", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        count = 0
        print(f"🔴 Recording {letter}...")

        while count < SAMPLES:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            
            try:
                mp_image = frame_to_mp_image(frame)
                detection_result = hands.detect(mp_image)

                if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
                    hand_landmarks = detection_result.hand_landmarks[0]
                    features = extract_landmark_vector(hand_landmarks)
                    
                    if features is not None and len(features) == 63:
                        row = features + [letter]
                        writer.writerow(row)
                        count += 1
            except Exception as exc:
                print(f"Detection error: {exc}", file=sys.__stderr__)

            cv2.putText(frame, f"Recording '{letter}': {count}/{SAMPLES}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Collect Alphabet Data", frame)
            cv2.waitKey(30)

        print(f"✅ {letter} done ({count} samples)")

hands.close()
cap.release()
cv2.destroyAllWindows()
print(f"\n🎉 Saved to {OUTPUT_CSV}")
