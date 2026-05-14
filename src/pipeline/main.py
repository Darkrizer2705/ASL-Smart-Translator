# src/pipeline/main.py
# -------------------------------------------------------
# Main pipeline — Camera → Model (Phrase/Alphabet/Number)
#               → Smoother → Sentence Builder → LLM Refiner
#
# How to run (from your git repo root):
#   python -m src.pipeline.main
#
# Controls:
#   1 = Phrase mode    2 = Alphabet mode   3 = Number mode
#   R = Refine with Gemini (English + Hindi)
#   C = Clear sentence
#   U = Undo last word / last letter (alphabet mode)
#   Q = Quit
# -------------------------------------------------------

import sys
import cv2
import numpy as np
import pickle
from pathlib import Path

# ── Path setup ─────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.utils.mediapipe_utils import (
    create_hands_detector,
    draw_hand_landmarks,
    extract_landmark_vector,
    frame_to_mp_image,
)
from src.pipeline.smoother import Smoother
from src.pipeline.sentence_builder import SentenceBuilder
from src.pipeline.llm_refiner import LLMRefiner


# ── Load all three models ───────────────────────────
def load_pkl(path: Path, label: str):
    if not path.exists():
        print(f"{label} model not found at: {path}")
        sys.exit(1)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["encoder"]

print("Loading models...")
phrase_model,   phrase_enc   = load_pkl(ROOT_DIR / "models" / "phrase_classifier.pkl",           "Phrase")
alphabet_model, alphabet_enc = load_pkl(ROOT_DIR / "models" / "alphabet_landmark_classifier.pkl","Alphabet")
number_model,   number_enc   = load_pkl(ROOT_DIR / "models" / "number_landmark_classifier.pkl",  "Number")
print(f"  Phrases  : {len(phrase_enc.classes_)} classes")
print(f"  Alphabet : {len(alphabet_enc.classes_)} classes")
print(f"  Numbers  : {len(number_enc.classes_)} classes")

# ── MediaPipe setup ─────────────────────────────────
hands = create_hands_detector(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ── Pipeline objects ────────────────────────────────
smoother         = Smoother(window_size=10, min_count=5, min_confidence=0.60)
sentence_builder = SentenceBuilder()
refiner          = LLMRefiner()

# ── Mode definitions ────────────────────────────────
MODES = {
    "1": ("Phrase",   phrase_model,   phrase_enc,   (0, 200, 255)),   # cyan
    "2": ("Alphabet", alphabet_model, alphabet_enc, (120, 255, 120)), # green
    "3": ("Number",   number_model,   number_enc,   (255, 180, 60)),  # orange
}
current_mode = "1"  # start in Phrase mode

# State
refined_english  = ""
refined_hindi    = ""
current_word     = []      # accumulates letters in Alphabet mode
probs            = np.zeros(len(phrase_enc.classes_))


# ── Feature builder ─────────────────────────────────
def build_feature_vector(hand_landmarks_list, expected_features):
    if not hand_landmarks_list:
        return None
    hand_rows = [extract_landmark_vector(h) for h in hand_landmarks_list]
    if expected_features == 63:
        best_row = max(hand_rows, key=len)
        return best_row if len(best_row) == expected_features else None
    combined = [v for row in hand_rows for v in row]
    return combined if len(combined) == expected_features else None


# ── UI drawing ──────────────────────────────────────
def draw_ui(frame, prediction, confidence, hand_found,
            mode_name, mode_color, sentence, word_buf,
            refined_english, refined_hindi):
    h, w = frame.shape[:2]

    # ── Top bar ──────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 105), (0, 0, 0), -1)

    # Mode badge (top-right)
    badge_text = f"MODE: {mode_name}"
    cv2.putText(frame, badge_text,
                (w - 220, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

    if hand_found:
        color = mode_color if confidence >= 0.60 else (0, 120, 180)
        cv2.putText(frame, prediction.upper(),
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)
        cv2.putText(frame, f"{confidence:.0%}",
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)
    else:
        cv2.putText(frame, "Show your hand...",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)

    # ── Alphabet word-buffer bar ─────────────────────
    if mode_name == "Alphabet":
        cv2.rectangle(frame, (0, h - 200), (w, h - 160), (25, 25, 50), -1)
        cv2.putText(frame, "Spelling:", (10, h - 178),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 200), 1)
        spell_display = "".join(word_buf) if word_buf else "—"
        cv2.putText(frame, spell_display,
                    (100, h - 178), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2)

    # ── Raw sentence bar ─────────────────────────────
    cv2.rectangle(frame, (0, h - 160), (w, h - 110), (30, 30, 30), -1)
    cv2.putText(frame, "Raw:", (10, h - 138),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    raw_display = sentence if sentence else "—"
    cv2.putText(frame, raw_display,
                (70, h - 138), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # ── Refined English bar ──────────────────────────
    cv2.rectangle(frame, (0, h - 110), (w, h - 60), (20, 20, 20), -1)
    cv2.putText(frame, "EN:", (10, h - 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(frame, refined_english if refined_english else "Press R to refine",
                (70, h - 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    # ── Hindi bar ────────────────────────────────────
    cv2.rectangle(frame, (0, h - 60), (w, h), (10, 10, 10), -1)
    cv2.putText(frame, "HI:", (10, h - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(frame, refined_hindi if refined_hindi else "—",
                (70, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 50), 2)

    # ── Controls reminder ────────────────────────────
    cv2.putText(frame, "1=Phrase  2=Alphabet  3=Number  |  R=Refine  C=Clear  U=Undo  Q=Quit",
                (10, h - 170), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)


# ── Camera loop ──────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera. Try closing other apps using it.")
    sys.exit(1)

print("\nCamera started!")
print("Controls: 1=Phrase  2=Alphabet  3=Number  |  R=Refine  C=Clear  U=Undo  Q=Quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    results = hands.detect(frame_to_mp_image(frame))
    draw_hand_landmarks(frame, results)

    mode_name, model, enc, mode_color = MODES[current_mode]

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
            prediction = enc.classes_[pred_idx]
            hand_found = True

            accepted = smoother.add(prediction, confidence)
            if accepted:
                if mode_name == "Phrase":
                    # Phrases go straight into the sentence as words
                    sentence_builder.add(accepted)
                    sentence_builder.add("space")
                    refined_english = ""
                    refined_hindi   = ""
                    print(f"[Phrase] Added: {accepted} → [{sentence_builder.get()}]")

                elif mode_name == "Alphabet":
                    # Skip noise labels
                    if accepted.lower() == "nothing":
                        pass
                    elif accepted.lower() == "del":
                        if current_word:
                            current_word.pop()
                    elif accepted.lower() == "space":
                        # Commit accumulated letters as one word
                        if current_word:
                            word = "".join(current_word)
                            sentence_builder.add(word)
                            sentence_builder.add("space")
                            print(f"[Alphabet] Word committed: {word} → [{sentence_builder.get()}]")
                            current_word = []
                            refined_english = ""
                            refined_hindi   = ""
                    else:
                        current_word.append(accepted.upper())
                        print(f"[Alphabet] Spelling: {''.join(current_word)}")

                elif mode_name == "Number":
                    # Each digit goes in as its own token
                    sentence_builder.add(accepted)
                    sentence_builder.add("space")
                    refined_english = ""
                    refined_hindi   = ""
                    print(f"[Number] Added: {accepted} → [{sentence_builder.get()}]")

    else:
        smoother.reset()

    # ── Draw UI ─────────────────────────────────────
    draw_ui(frame, prediction, confidence, hand_found,
            mode_name, mode_color, sentence_builder.get(), current_word,
            refined_english, refined_hindi)

    cv2.imshow("ASL Translator", frame)

    # ── Key controls ────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key in (ord('1'), ord('2'), ord('3')):
        new_mode = chr(key)
        if new_mode != current_mode:
            current_mode = new_mode
            smoother.reset()
            current_word = []
            _, _, _, mode_color = MODES[current_mode]
            print(f"\nSwitched to {MODES[current_mode][0]} mode")

    elif key == ord('c'):
        sentence_builder.reset()
        smoother.reset()
        current_word    = []
        refined_english = ""
        refined_hindi   = ""
        print("Cleared")

    elif key == ord('u'):
        mode_name = MODES[current_mode][0]
        if mode_name == "Alphabet" and current_word:
            # Undo last letter in the spelling buffer
            removed = current_word.pop()
            print(f"[Alphabet] Removed letter: {removed} | Spelling: {''.join(current_word)}")
        else:
            # Undo last word in the sentence
            current = sentence_builder.get()
            words   = current.strip().split(" ")
            if words and words[-1]:
                words.pop()
                sentence_builder.reset()
                for i, w in enumerate(words):
                    sentence_builder.add(w)
                    if i < len(words) - 1:
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

    # Commit spelling buffer with ENTER key
    elif key == 13:  # Enter
        mode_name = MODES[current_mode][0]
        if mode_name == "Alphabet" and current_word:
            word = "".join(current_word)
            sentence_builder.add(word)
            sentence_builder.add("space")
            print(f"[Alphabet] Word committed (Enter): {word} → [{sentence_builder.get()}]")
            current_word    = []
            refined_english = ""
            refined_hindi   = ""

cap.release()
cv2.destroyAllWindows()

# ── Final output ─────────────────────────────────────
print("\n" + "=" * 50)
print(f"Raw sentence : {sentence_builder.get()}")
print(f"English      : {refined_english}")
print(f"Hindi        : {refined_hindi}")
print("=" * 50)
