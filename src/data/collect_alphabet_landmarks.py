# src/data/collect_alphabet_landmarks.py
import cv2
import mediapipe as mp
import csv
import time
import os

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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                lms = results.multi_hand_landmarks[0]
                if mp_draw:
                    mp_draw.draw_landmarks(frame, lms,
                        mp.solutions.hands.HAND_CONNECTIONS)
                row = [coord for lm in lms.landmark
                       for coord in [lm.x, lm.y, lm.z]]
                row.append(letter)
                writer.writerow(row)
                count += 1

            cv2.putText(frame, f"Recording '{letter}': {count}/{SAMPLES}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Collect Alphabet Data", frame)
            cv2.waitKey(30)

        print(f"✅ {letter} done ({count} samples)")

cap.release()
cv2.destroyAllWindows()
print(f"\n🎉 Saved to {OUTPUT_CSV}")
