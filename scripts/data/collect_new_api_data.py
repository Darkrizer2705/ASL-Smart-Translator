"""
Collect new training data using the SAME MediaPipe Tasks API as prediction
This ensures feature consistency between training and inference
"""
import cv2
import numpy as np
import pickle
import sys
import os
from pathlib import Path
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.utils.mediapipe_utils import (
    create_hands_detector,
    extract_landmark_vector,
    frame_to_mp_image,
)

# Labels for data collection
PHRASE_LABELS = [
    'YES', 'NO', 'HELP', 'STOP', 'MORE', 'WAIT', 'AGAIN', 'UNDERSTAND',
    'WATER', 'FOOD', 'BATHROOM', 'HOME', 'SCHOOL', 'WORK', 'HOSPITAL',
    'ME', 'YOU', 'HE_SHE', 'FAMILY', 'FRIEND',
    'WHAT', 'WHERE', 'WHEN', 'HOW', 'WHY'
]

def collect_phrase_data():
    """Collect hand landmark data using the NEW Tasks API"""
    
    hands = create_hands_detector(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    
    cap = cv2.VideoCapture(0)
    collected_data = defaultdict(list)
    
    print("\n" + "="*60)
    print("HAND LANDMARK DATA COLLECTION - Using NEW MediaPipe API")
    print("="*60)
    print("\nKey Mapping:")
    for i, label in enumerate(PHRASE_LABELS):
        key = str(i % 10) if i < 10 else chr(ord('a') + (i - 10))
        print(f"  {key} → {label}")
    print("\nPress keys to record, ESC to quit")
    print("="*60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Detect hands
        results = hands.detect(frame_to_mp_image(frame))
        
        # Draw instructions
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        collected_count = sum(len(v) for v in collected_data.values())
        cv2.putText(frame, f"Collected: {collected_count} samples",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press key to add current hand to label | ESC=Quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw hand if detected
        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            hand_lms = results.hand_landmarks[0]
            
            # Draw landmarks
            for lm in hand_lms:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            cv2.putText(frame, "Hand detected - ready to record",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand detected",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow("Data Collection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        
        # Check if a label key was pressed
        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            label_idx = None
            
            if key >= ord('0') and key <= ord('9'):
                label_idx = int(chr(key))
            elif key >= ord('a') and key <= ord('z'):
                label_idx = 10 + (key - ord('a'))
            
            if label_idx is not None and label_idx < len(PHRASE_LABELS):
                hand_lms = results.hand_landmarks[0]
                features = extract_landmark_vector(hand_lms)
                label = PHRASE_LABELS[label_idx]
                
                collected_data[label].append(features)
                print(f"✓ Added to {label}: {len(collected_data[label])} samples")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save collected data
    if collected_data:
        import pandas as pd
        
        rows = []
        for label, feature_lists in collected_data.items():
            for features in feature_lists:
                row = features + [label]
                rows.append(row)
        
        columns = [f"x{i//3}_{['x','y','z'][i%3]}" for i in range(63)]
        columns.append("label")
        
        df = pd.DataFrame(rows, columns=columns)
        csv_path = "datasets/new_task_api_landmarks.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"\n✓ Saved {len(rows)} samples to {csv_path}")
        print(f"  Phrases: {list(collected_data.keys())}")
    else:
        print("\n⚠️  No data collected")

if __name__ == "__main__":
    collect_phrase_data()