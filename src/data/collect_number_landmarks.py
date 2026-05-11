# src/data/collect_number_landmarks.py
import cv2
import mediapipe as mp
import csv
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

NUMBERS = ["0","1","2","3","4","5","6","7","8","9"]
SAMPLES = 200   # 200 samples per number

OUTPUT_CSV = "datasets/number_landmarks.csv"
os.makedirs("datasets", exist_ok=True)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    header = [f"{a}{i}" for i in range(21) for a in ["x","y","z"]] + ["label"]
    writer.writerow(header)

    for number in NUMBERS:
        print(f"\n Get ready to sign: '{number}'")
        print("Press SPACE to start recording...")

        # Wait for spacebar
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Show ROI box
            roi_size = 220
            roi_x = w - roi_size - 20
            roi_y = h - roi_size - 20
            cv2.rectangle(frame, (roi_x, roi_y),
                          (roi_x+roi_size, roi_y+roi_size), (0, 255, 255), 2)

            cv2.rectangle(frame, (0,0), (w, 70), (0,0,0), -1)
            cv2.putText(frame, f"Next: {number}  — Press SPACE to start",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Collect Number Landmarks", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        count = 0
        print(f" Recording '{number}'...")

        while count < SAMPLES:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            roi_size = 220
            roi_x = w - roi_size - 20
            roi_y = h - roi_size - 20
            cv2.rectangle(frame, (roi_x, roi_y),
                          (roi_x+roi_size, roi_y+roi_size), (0, 255, 0), 2)

            if results.multi_hand_landmarks:
                lms = results.multi_hand_landmarks[0]
                if mp_draw:
                    mp_draw.draw_landmarks(frame, lms,
                        mp.solutions.hands.HAND_CONNECTIONS)
                row = [coord for lm in lms.landmark
                       for coord in [lm.x, lm.y, lm.z]]
                row.append(number)
                writer.writerow(row)
                count += 1

            cv2.rectangle(frame, (0,0), (w, 70), (0,0,0), -1)
            cv2.putText(frame, f"Sign '{number}': {count}/{SAMPLES}",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            cv2.imshow("Collect Number Landmarks", frame)
            cv2.waitKey(20)

        print(f"'{number}' done — {count} samples")

cap.release()
cv2.destroyAllWindows()
print(f"\n Saved to: {OUTPUT_CSV}")
