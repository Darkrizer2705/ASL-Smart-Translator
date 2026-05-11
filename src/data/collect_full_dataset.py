import cv2
import os
import time

# Label mapping
labels = {
    '1': ('BASIC', 'YES'),
    '2': ('BASIC', 'NO'),
    '3': ('BASIC', 'HELP'),
    '4': ('BASIC', 'STOP'),
    '5': ('BASIC', 'MORE'),
    '6': ('BASIC', 'WAIT'),
    '7': ('BASIC', 'AGAIN'),
    '8': ('BASIC', 'UNDERSTAND'),

    'q': ('NEEDS', 'WATER'),
    'w': ('NEEDS', 'FOOD'),
    'e': ('NEEDS', 'BATHROOM'),
    'r': ('NEEDS', 'HOME'),
    't': ('NEEDS', 'SCHOOL'),
    'y': ('NEEDS', 'WORK'),
    'u': ('NEEDS', 'HOSPITAL'),

    'a': ('PERSONAL', 'ME'),
    's': ('PERSONAL', 'YOU'),
    'd': ('PERSONAL', 'HE_SHE'),
    'f': ('PERSONAL', 'FAMILY'),
    'g': ('PERSONAL', 'FRIEND'),

    'z': ('QUESTIONS', 'WHAT'),
    'x': ('QUESTIONS', 'WHERE'),
    'c': ('QUESTIONS', 'WHEN'),
    'v': ('QUESTIONS', 'HOW'),
    'b': ('QUESTIONS', 'WHY'),
}

base_path = "gesture_dataset"

# Create folders
for category, label in labels.values():
    path = os.path.join(base_path, category, label)
    os.makedirs(path, exist_ok=True)

# Function to get next video number safely
def get_next_index(folder):
    existing_files = os.listdir(folder)
    return len(existing_files)

cap = cv2.VideoCapture(0)

print("Press assigned keys to record")
print("ESC → Quit safely anytime")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow("Dataset Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # EXIT safely
    if key == 27:
        print("Exiting safely...")
        break

    key_char = chr(key)

    if key_char in labels:
        category, label = labels[key_char]

        folder = os.path.join(base_path, category, label)

        # 🔥 SAFE numbering
        video_index = get_next_index(folder)

        video_name = f"{label}_{video_index}.avi"
        video_path = os.path.join(folder, video_name)

        print(f"Recording {label}...")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

        start_time = time.time()

        while int(time.time() - start_time) < 3:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            out.write(frame)
            cv2.imshow("Recording...", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        out.release()

        print(f"Saved safely: {video_path}")

cap.release()
cv2.destroyAllWindows()