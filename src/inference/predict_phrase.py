import argparse
import pickle
from pathlib import Path
import numpy as np


def run_test(model_path: Path) -> None:
    with model_path.open("rb") as f:
        data = pickle.load(f)
    model = data["model"]
    encoder = data["encoder"]
    print(f"Loaded phrase model; classes: {list(encoder.classes_)}")

    # sanity predict with zeros (63 features)
    import cv2
    import numpy as np
    import time

    # ── Load Model ─────────────────────────────────────
    print("Loading phrase model...")
    with open("models/phrase_classifier.pkl", "rb") as f:
        data = pickle.load(f)
        model   = data["model"]
        encoder = data["encoder"]

    print(f"Model loaded — {len(encoder.classes_)} phrases")
    print(f"Classes: {list(encoder.classes_)}")
    print(f"Expected features: {model.n_features_in_}")

    # ── MediaPipe ──────────────────────────────────────
    # Support both the old `solutions` API and the newer `tasks` API.
    mp_drawing = None
    hands = None
    hand_landmarker = None
    use_tasks = False
    connections = None
    try:
        import mediapipe as mp
        # Prefer the Tasks API when available (mp.tasks)
        if hasattr(mp, 'tasks'):
            from mediapipe.tasks import python as tasks
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.vision.core import image as mp_image
            # Create a HandLandmarker in IMAGE mode (we call detect() per frame)
            options = vision.HandLandmarkerOptions(
                base_options=tasks.BaseOptions(model_asset_path='models/hand_landmarker.task'),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
            )
            hand_landmarker = vision.HandLandmarker.create_from_options(options)
            mp_drawing = vision.drawing_utils
            connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
            use_tasks = True
            print('MediaPipe tasks API loaded')
        else:
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            connections = mp_hands.HAND_CONNECTIONS
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5
            )
            print('MediaPipe solutions API loaded')
    except Exception as e1:
        # Fallback: some installs expose the python package path for solutions
        try:
            from mediapipe.python.solutions import hands as mp_hands_module, drawing_utils as mp_drawing
            connections = mp_hands_module.HAND_CONNECTIONS
            hands = mp_hands_module.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6
            )
            print('MediaPipe (python.solutions) loaded')
        except Exception as e2:
            print(f"MediaPipe import failed: {e1}; {e2}")
            print("Please ensure mediapipe is installed and accessible in this Python environment.")
            return

    # ── State ──────────────────────────────────────────
    sentence        = []
    last_pred       = ""
    stable_count    = 0
    probs           = np.zeros(len(encoder.classes_))

    STABLE_THRESHOLD     = 20
    CONFIDENCE_THRESHOLD = 0.70

    cap = cv2.VideoCapture(0)
    print("\nStarted. Controls: A=Add word  C=Clear  U=Undo  Q=Quit\n", flush=True)

    if not cap.isOpened():
        print("ERROR: unable to open camera (cap.isOpened() == False)", flush=True)
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: frame read returned False", flush=True)
                break

            frame      = cv2.flip(frame, 1)
            h, w       = frame.shape[:2]
            rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            if use_tasks:
                # Tasks API: wrap numpy array into mediapipe Image
                from mediapipe.tasks.python.vision.core import image as mp_image
                img = mp_image.Image(mp_image.ImageFormat.SRGB, rgb)
                results = hand_landmarker.detect(img)
            else:
                results    = hands.process(rgb)
            rgb.flags.writeable = True

            prediction  = ""
            confidence  = 0.0
            hand_found  = False

            # Extract landmarks for both APIs
            hand_lms = None
            if not use_tasks and getattr(results, 'multi_hand_landmarks', None):
                hand_lms = results.multi_hand_landmarks[0]
            elif use_tasks and getattr(results, 'hand_landmarks', None):
                if len(results.hand_landmarks) > 0:
                    hand_lms = results.hand_landmarks[0]

            if hand_lms is not None:

                # Normalize access to the landmark sequence for both APIs
                if hasattr(hand_lms, 'landmark'):
                    landmarks_iter = hand_lms.landmark
                elif hasattr(hand_lms, 'landmarks'):
                    landmarks_iter = hand_lms.landmarks
                else:
                    landmarks_iter = hand_lms

                # ── Draw landmarks ────────────────────────
                if mp_drawing:
                    mp_drawing.draw_landmarks(
                        frame, landmarks_iter,
                        connections,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255,255,255), thickness=1)
                    )

                # ── Extract features in EXACT same order ──
                # Same as collect_my_data.py: x0,y0,z0,x1,y1,z1...
                row = []
                for lm in landmarks_iter:
                    x = getattr(lm, 'x', None)
                    y = getattr(lm, 'y', None)
                    z = getattr(lm, 'z', 0.0)
                    if x is None or y is None:
                        continue
                    row.extend([x, y, z])

                # Verify feature count
                if len(row) != model.n_features_in_:
                    print(f"⚠️ Feature mismatch: got {len(row)}, expected {model.n_features_in_}")
                    continue

                features   = np.array(row, dtype=np.float32).reshape(1, -1)
                probs      = model.predict_proba(features)[0]
                confidence = float(np.max(probs))
                pred_idx   = int(np.argmax(probs))
                prediction = encoder.classes_[pred_idx]
                hand_found = True

                # ── Stability check ───────────────────────
                if prediction == last_pred:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_pred    = prediction

                # Auto-add when stable + confident
                if (stable_count == STABLE_THRESHOLD and
                        confidence >= CONFIDENCE_THRESHOLD):
                    if not sentence or sentence[-1] != prediction:
                        sentence.append(prediction)
                        print(f"Added: {prediction}")
                    stable_count = 0

            else:
                stable_count = 0

            # ══════════════ UI ═══════════════════════════

            # Top bar — current prediction
            cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)

            if hand_found:
                color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
                cv2.putText(frame, f"{prediction.upper()}",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                            1.6, color, 3)
                # Stability progress bar
                bar_w = int((stable_count / STABLE_THRESHOLD) * 300)
                cv2.rectangle(frame, (10, 70), (310, 88), (40, 40, 40), -1)
                cv2.rectangle(frame, (10, 70), (10 + bar_w, 88), color, -1)
                cv2.putText(frame, f"{confidence:.0%}",
                            (320, 85), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color, 1)
            else:
                cv2.putText(frame, "Show your hand...",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (100, 100, 100), 2)

            # Probability bars — top 5
            sorted_idx = np.argsort(probs)[::-1][:5]
            for i, idx in enumerate(sorted_idx):
                lbl  = encoder.classes_[idx]
                prob = probs[idx]
                by   = 110 + i * 28
                bw   = int(prob * 220)
                col  = (0, 200, 100) if i == 0 else (0, 120, 60)
                cv2.rectangle(frame, (10, by), (10 + bw, by + 20), col, -1)
                cv2.putText(frame, f"{lbl}: {prob:.0%}",
                            (240, by + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

            # Sentence bar — bottom
            cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 0), -1)
            sentence_display = " › ".join(sentence[-6:]) if sentence else "—"
            cv2.putText(frame, sentence_display,
                        (10, h - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, "A=Add  C=Clear  U=Undo  Q=Quit",
                        (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

            cv2.imshow("ASL Phrase Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence = []
                print("Cleared")
            elif key == ord('u') and sentence:
                removed = sentence.pop()
                print(f"Removed: {removed}")
            elif key == ord('a') and prediction and confidence >= CONFIDENCE_THRESHOLD:
                if not sentence or sentence[-1] != prediction:
                    sentence.append(prediction)
                    print(f"Manually added: {prediction}")

    except Exception as e:
        import traceback
        print("EXCEPTION in main loop:", e, flush=True)
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if sentence:
        print(f"\nFinal sentence words: {sentence}")
        print(f"Raw sentence: {' '.join(sentence).upper()}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run phrase landmark-based predictor')
    parser.add_argument('--model', '-m', default='models/phrase_classifier.pkl', help='Path to phrase classifier pickle')
    args = parser.parse_args()
    run_test(Path(args.model))
