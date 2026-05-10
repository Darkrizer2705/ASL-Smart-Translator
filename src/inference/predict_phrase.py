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
    try:
        sample = np.zeros((1, 63))
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(sample)[0]
            idx = int(np.argmax(probs))
            print(f"Test prediction (zeros): {encoder.inverse_transform([idx])[0]} (conf={probs[idx]:.3f})")
        else:
            pred = model.predict(sample)
            print(f"Test prediction (zeros): {pred}")
    except Exception as e:
        print("Warning: model predict failed during test:", e)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Load model and run a quick test then exit")
    parser.add_argument("--model", default="models/phrase_classifier.pkl")
    args = parser.parse_args()

    model_path = Path(args.model)
    if args.test:
        if not model_path.exists():
            raise SystemExit(f"Model not found: {model_path}")
        run_test(model_path)
        return

    # Live webcam mode — keep original behavior if user runs without --test
    import cv2
    import mediapipe as mp

    with model_path.open("rb") as f:
        data = pickle.load(f)
    model = data["model"]
    encoder = data["encoder"]

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    print("✅ Phrase recognition started. Press Q to quit.")

    sentence = []
    last_prediction = ""
    stable_count = 0
    STABLE_THRESHOLD = 15
    CONFIDENCE_THRESHOLD = 0.6

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        prediction = ""
        confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp.solutions.hands.HAND_CONNECTIONS)
                row = []
                for lm in hand_lms.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                features = np.array(row).reshape(1, -1)
                probs = model.predict_proba(features)[0]
                confidence = float(np.max(probs))
                pred_idx = int(np.argmax(probs))
                prediction = encoder.inverse_transform([pred_idx])[0]

                if prediction == last_prediction:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_prediction = prediction

                if stable_count == STABLE_THRESHOLD and confidence >= CONFIDENCE_THRESHOLD:
                    if not sentence or sentence[-1] != prediction:
                        sentence.append(prediction)
                    stable_count = 0

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Sign: {prediction.upper()}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.0%}  Stable: {stable_count}/{STABLE_THRESHOLD}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 0), -1)
        sentence_text = " ".join(sentence[-6:])
        cv2.putText(frame, f"Sentence: {sentence_text}", (10, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(frame, "C=Clear  U=Undo  Q=Quit", (10, h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        cv2.imshow("ASL Phrase Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = []
            print("🗑️  Sentence cleared")
        elif key == ord('u'):
            if sentence:
                removed = sentence.pop()
                print(f"↩️  Removed: {removed}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n📝 Final sentence: {' '.join(sentence)}")


if __name__ == "__main__":
    main()
