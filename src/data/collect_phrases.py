"""
Collect data for all phrase classes with natural hand movement.
The original data was too static - collect with more variation.
"""
import cv2
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.utils.mediapipe_utils import (
    create_hands_detector,
    extract_landmark_vector,
    frame_to_mp_image,
)

# Phrases that previously needed recollection.
BROKEN_PHRASES = [
    'MORE', 'NO', 'STOP', 'WAIT', 'BATHROOM', 'FOOD', 'HOME',
    'HOSPITAL', 'SCHOOL', 'WATER', 'WORK', 'FRIEND', 'HE_SHE', 'YOU', 'WHEN', 'WHERE', 'WHY'
]

# Phrases that already worked, but are still part of the full phrase set.
GOOD_PHRASES = ['AGAIN', 'FAMILY', 'HELP', 'HOW', 'ME', 'UNDERSTAND', 'WHAT', 'YES']

ALL_PHRASES = sorted(BROKEN_PHRASES + GOOD_PHRASES)

def collect_with_variation():
    """
    Collect phrase data with natural hand movement and variation.
    Record 3-5 second videos where hand moves naturally.
    """
    # Allow detecting up to two hands so we can capture both-hand features
    hands = create_hands_detector(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    
    cap = cv2.VideoCapture(0)
    collected_samples = {phrase: [] for phrase in ALL_PHRASES}
    
    print("\n" + "="*70)
    print("PHRASE COLLECTION: All Phrase Classes")
    print("="*70)
    print("\n  IMPORTANT INSTRUCTIONS:")
    print("  1. Do NOT hold your hand in the same position")
    print("  2. MOVE your hand naturally while doing the sign")
    print("  3. Make different hand shapes and positions")
    print("  4. Collect as many samples as you want for each phrase")
    print("  5. The detector captures up to TWO hands; each hand is saved as a separate sample")
    print("="*70)
    
    phrase_idx = 0
    current_phrase = ALL_PHRASES[phrase_idx]
    recording = False
    frame_buffer = []
    
    print(f"\nStarting with: {current_phrase}")
    print("Press SPACE to START recording this phrase")
    print("Press SPACE again to STOP and save")
    print("Press 'N' for NEXT phrase | 'P' for PREVIOUS | 'Q' to QUIT")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Detect hands
        results = hands.detect(frame_to_mp_image(frame))
        
        # Draw UI
        cv2.rectangle(frame, (0, 0), (w, 140), (0, 0, 0), -1)
        
        status_color = (0, 255, 0) if results.hand_landmarks else (0, 0, 255)
        recording_text = "RECORDING" if recording else "PAUSED"
        
        cv2.putText(frame, f"Phrase {phrase_idx + 1}/{len(ALL_PHRASES)}: {current_phrase}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        cv2.putText(frame, recording_text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Samples: {len(collected_samples[current_phrase])}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, "SPACE=Record | N=Next | P=Prev | Q=Quit", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # If recording and hand(s) detected, capture landmarks for ALL hands
        if recording and results.hand_landmarks and len(results.hand_landmarks) > 0:
            # Iterate over each detected hand and save its features as a separate sample
            for hand_lms in results.hand_landmarks:
                features = extract_landmark_vector(hand_lms)

                # Draw landmarks on frame for this hand
                for lm in hand_lms:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                # Buffer features
                frame_buffer.append(features)

            # Add to dataset every 5 feature-entries to avoid duplicates
            if len(frame_buffer) >= 5:
                # take the last N entries (may include both-hands samples)
                collected_samples[current_phrase].extend(frame_buffer[-5:])
                frame_buffer = []
        elif not recording:
            # Draw landmarks even when not recording for preview (all hands)
            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                for hand_lms in results.hand_landmarks:
                    for lm in hand_lms:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(frame, (x, y), 3, (100, 150, 100), -1)
        
        cv2.imshow("Phrase Collection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # SPACE
            if not recording:
                # Start recording
                recording = True
                frame_buffer = []
                print(f"  Recording {current_phrase}...")
            else:
                # Stop recording
                recording = False
                if frame_buffer:
                    collected_samples[current_phrase].extend(frame_buffer)
                frame_buffer = []
                print(f"  Saved {len(collected_samples[current_phrase])} samples for {current_phrase}")
        elif key == ord('n'):  # Next phrase
            recording = False
            phrase_idx = (phrase_idx + 1) % len(ALL_PHRASES)
            current_phrase = ALL_PHRASES[phrase_idx]
            print(f"\n-> Now collecting: {current_phrase}")
            print("  Press SPACE to START recording")
        elif key == ord('p'):  # Previous phrase
            recording = False
            phrase_idx = (phrase_idx - 1) % len(ALL_PHRASES)
            current_phrase = ALL_PHRASES[phrase_idx]
            print(f"\n<- Now collecting: {current_phrase}")
            print("  Press SPACE to START recording")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save collected data
    print("\n" + "="*70)
    print("SAVING COLLECTED DATA")
    print("="*70)
    
    # Load existing gesture dataset
    existing_df = pd.read_csv('datasets/gesture_dataset.csv')
    
    # Create new rows from collected samples
    new_rows = []
    total_collected = 0
    
    for phrase, samples in collected_samples.items():
        if samples:
            total_collected += len(samples)
            for features in samples:
                row = features + [phrase]
                new_rows.append(row)
            print(f"  {phrase:15} : {len(samples):4} new samples")
    
    if new_rows:
        # Append to existing dataset
        columns = [f"x{i//3}" if i % 3 == 0 else f"y{i//3}" if i % 3 == 1 else f"z{i//3}" 
                   for i in range(63)] + ['label']
        
        new_df = pd.DataFrame(new_rows, columns=columns)
        
        # Get proper column names from existing data
        columns = existing_df.columns.tolist()
        
        new_df = pd.DataFrame(new_rows, columns=columns)
        # Combine with existing
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv('datasets/gesture_dataset.csv', index=False)
        
        print(f"\n Appended {total_collected} new samples to gesture_dataset.csv")
        print(f"   Total dataset size: {len(combined_df)} samples")
        
        # Show new distribution
        print(f"\nUpdated distribution:")
        for phrase in sorted(combined_df['label'].unique()):
            count = (combined_df['label'] == phrase).sum()
            status = "OK" if count > 500 else "LOW" if count > 100 else "VERY_LOW"
            print(f"  {status} {phrase:15} : {count:4} samples")
    else:
        print("\n  No new samples collected")

if __name__ == "__main__":
    collect_with_variation()
    
    print("\n" + "="*70)
    print("NEXT STEP: Retrain model")
    print("="*70)
    print("Run: python src/models/train_phrases.py")
