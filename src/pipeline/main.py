# src/pipeline/main.py
# -------------------------------------------------------
# This is the MAIN file that connects everything together:
#
#   Camera → Phrase Model → Smoother → Sentence Builder → LLM Refiner
#
# How to run (from your git repo root):
#   python -m src.pipeline.main
#
# Controls:
#   R = Refine current sentence with Gemini (English + Hindi)
#   C = Clear sentence
#   U = Undo last word
#   Q = Quit
# -------------------------------------------------------

import sys
import cv2
import numpy as np
import pickle
from pathlib import Path

# ── Path setup ─────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]  # goes up to git repo root
sys.path.insert(0, str(ROOT_DIR))

# ── Import your friend's utilities ─────────────────
from src.utils.mediapipe_utils import (
    create_hands_detector,
    draw_hand_landmarks,
    extract_landmark_vector,
    frame_to_mp_image,
)

# ── Import our pipeline ────────────────────────────
from src.pipeline.smoother import Smoother
from src.pipeline.sentence_builder import SentenceBuilder
from src.pipeline.llm_refiner import LLMRefiner

# ── Load phrase model ───────────────────────────────
print("Loading phrase model...")
PHRASE_MODEL = ROOT_DIR / "models" / "phrase_classifier.pkl"
if not PHRASE_MODEL.exists():
    print(f"Phrase model not found at: {PHRASE_MODEL}")
    print("Run: python src/models/train_phrases.py first")
    sys.exit(1)

with open(PHRASE_MODEL, "rb") as f:
    data    = pickle.load(f)
    model   = data["model"]
    encoder = data["encoder"]

print(f"Model loaded — {len(encoder.classes_)} phrases")
print(f"Classes: {list(encoder.classes_)}")

# ── MediaPipe setup ─────────────────────────────────
hands = create_hands_detector(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ── Our pipeline setup ──────────────────────────────
# We set min_confidence lower here because the phrase model
# already does its own stability check — we just smooth on top
smoother         = Smoother(window_size=10, min_count=5, min_confidence=0.60)
sentence_builder = SentenceBuilder()
refiner          = LLMRefiner()

# ── State ───────────────────────────────────────────
refined_english = ""
refined_hindi   = ""
probs           = np.zeros(len(encoder.classes_))

# ── Feature builder (copied from your friend's code) ─
def build_feature_vector(hand_landmarks_list, expected_features):
    if not hand_landmarks_list:
        return None
    hand_rows = [extract_landmark_vector(hand_lms) for hand_lms in hand_landmarks_list]
    if expected_features == 63:
        best_row = max(hand_rows, key=len)
        return best_row if len(best_row) == expected_features else None
    combined_row = [value for row in hand_rows for value in row]
    return combined_row if len(combined_row) == expected_features else None


def draw_ui(frame, prediction, confidence, hand_found, sentence, refined_english, refined_hindi):
    """Draw all the UI elements on the frame."""
    h, w = frame.shape[:2]

    # ── Top bar: current prediction ─────────────────
    cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
    if hand_found:
        color = (0, 255, 0) if confidence >= 0.60 else (0, 165, 255)
        cv2.putText(frame, prediction.upper(),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3)
        cv2.putText(frame, f"{confidence:.0%}",
                    (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    else:
        cv2.putText(frame, "Show your hand...",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

    # ── Raw sentence bar ────────────────────────────
    cv2.rectangle(frame, (0, h - 160), (w, h - 110), (30, 30, 30), -1)
    cv2.putText(frame, "Raw:", (10, h - 138),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    raw_display = sentence_builder.get() if sentence_builder.get() else "—"
    cv2.putText(frame, raw_display,
                (70, h - 138), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # ── Refined English bar ─────────────────────────
    cv2.rectangle(frame, (0, h - 110), (w, h - 60), (20, 20, 20), -1)
    cv2.putText(frame, "EN:", (10, h - 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(frame, refined_english if refined_english else "Press R to refine",
                (70, h - 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    # ── Hindi bar ───────────────────────────────────
    cv2.rectangle(frame, (0, h - 60), (w, h), (10, 10, 10), -1)
    cv2.putText(frame, "HI:", (10, h - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    # Note: OpenCV can't render Hindi script natively.
    # Hindi will print to terminal instead.
    cv2.putText(frame, refined_hindi if refined_hindi else "—",
                (70, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 50), 2)

    # ── Controls reminder ───────────────────────────
    cv2.putText(frame, "R=Refine  C=Clear  U=Undo  Q=Quit",
                (w - 310, h - 170), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)


# ── Main camera loop ────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera. Try closing other apps using it.")
    sys.exit(1)

print("\nCamera started!")
print("Controls: R=Refine  C=Clear  U=Undo  Q=Quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    results = hands.detect(frame_to_mp_image(frame))
    draw_hand_landmarks(frame, results)

    prediction = ""
    confidence = 0.0
    hand_found = False

    if results.hand_landmarks:
        row = build_feature_vector(results.hand_landmarks, model.n_features_in_)

        if row and len(row) == model.n_features_in_:
            features   = np.array(row, dtype=np.float32).reshape(1, -1)
            probs      = model.predict_proba(features)[0]
            confidence = float(np.max(probs))
            pred_idx   = int(np.argmax(probs))
            prediction = encoder.classes_[pred_idx]
            hand_found = True

            # ── Feed into our smoother ──────────────
            accepted = smoother.add(prediction, confidence)
            if accepted:
                sentence_builder.add(accepted)
                # Clear refined output since sentence changed
                refined_english = ""
                refined_hindi   = ""
                print(f"Added: {accepted} | Sentence: [{sentence_builder.get()}]")

    else:
        smoother.reset()  # reset when hand disappears

    # ── Draw UI ─────────────────────────────────────
    draw_ui(frame, prediction, confidence, hand_found,
            sentence_builder.get(), refined_english, refined_hindi)

    cv2.imshow("ASL Translator", frame)

    # ── Key controls ────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('c'):
        sentence_builder.reset()
        smoother.reset()
        refined_english = ""
        refined_hindi   = ""
        print("Cleared")

    elif key == ord('u'):
        # Undo last word — remove last token from sentence
        current = sentence_builder.get()
        words   = current.strip().split(" ")
        if words and words[-1]:
            words.pop()
            sentence_builder.reset()
            for w in words:
                sentence_builder.add(w)
                sentence_builder.add("space")
            print(f"Undone | Sentence: [{sentence_builder.get()}]")

    elif key == ord('r'):
        raw = sentence_builder.get().strip()
        if not raw:
            print("Nothing to refine yet.")
        else:
            print(f"\nRefining: '{raw}' ...")
            refined_english, refined_hindi = refiner.refine(raw)
            print(f"English : {refined_english}")
            print(f"Hindi   : {refined_hindi}\n")
            # Hindi also prints to terminal since OpenCV can't render it natively

cap.release()
cv2.destroyAllWindows()

# ── Final output ────────────────────────────────────
print("\n" + "=" * 50)
print(f"Raw sentence : {sentence_builder.get()}")
print(f"English      : {refined_english}")
print(f"Hindi        : {refined_hindi}")
print("=" * 50)
