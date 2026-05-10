import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2


def run_test(model_path: Path, classes_path: Path) -> None:
    model = tf.keras.models.load_model(str(model_path))
    with classes_path.open() as f:
        class_indices = json.load(f)
    classes = {int(v): k for k, v in class_indices.items()}
    print(f"Loaded alphabet model; classes_count={len(classes)} sample={list(classes.values())[:5]}")

    sample = np.zeros((1, * (64, 64, 3)))
    probs = model.predict(sample, verbose=0)[0]
    idx = int(np.argmax(probs))
    print(f"Test prediction (zeros): {classes.get(idx)} (conf={probs[idx]:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--model", default="models/alphabet_cnn.h5")
    parser.add_argument("--classes", default="models/alphabet_classes.json")
    args = parser.parse_args()

    model_path = Path(args.model)
    classes_path = Path(args.classes)
    if args.test:
        if not model_path.exists() or not classes_path.exists():
            raise SystemExit("Model or classes file missing")
        run_test(model_path, classes_path)
        return

    model = tf.keras.models.load_model(str(model_path))
    with classes_path.open() as f:
        class_indices = json.load(f)
    classes = {int(v): k for k, v in class_indices.items()}

    IMG_SIZE = (64, 64)
    cap = cv2.VideoCapture(0)
    print("✅ Alphabet recognition started. Press Q to quit.")

    word = []
    last_prediction = ""
    stable_count = 0
    STABLE_THRESHOLD = 20

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        roi_x, roi_y, roi_size = w//2 - 100, 50, 200
        roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        cv2.rectangle(frame, (roi_x, roi_y),(roi_x+roi_size, roi_y+roi_size), (255, 0, 0), 2)

        prediction = ""
        confidence = 0.0
        try:
            img = cv2.resize(roi, IMG_SIZE)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            probs = model.predict(img, verbose=0)[0]
            confidence = np.max(probs)
            pred_idx = int(np.argmax(probs))
            prediction = classes.get(pred_idx, "")

            if prediction == last_prediction:
                stable_count += 1
            else:
                stable_count = 0
                last_prediction = prediction

            if stable_count == STABLE_THRESHOLD and confidence > 0.7:
                if prediction not in ["del", "nothing", "space"]:
                    word.append(prediction)
                elif prediction == "space":
                    word.append(" ")
                elif prediction == "del" and word:
                    word.pop()
                stable_count = 0
        except Exception:
            pass

        cv2.rectangle(frame, (0, 0), (w, 80), (0,0,0), -1)
        cv2.putText(frame, f"Letter: {prediction.upper()}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.0%}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.rectangle(frame, (0, h-50), (w, h), (0,0,0), -1)
        cv2.putText(frame, f"Word: {''.join(word)}", (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.imshow("ASL Alphabet Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            word = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
