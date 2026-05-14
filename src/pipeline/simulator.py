# simulator.py
# -------------------------------------------------------
# What this does:
#   Lets you test the ENTIRE pipeline by typing letters/words
#   in the terminal — no camera needed.
#
#   You type  →  smoother  →  sentence builder  →  LLM refiner + translator
#
#   When your friend's camera model is ready, you only need to
#   change ONE thing: replace the input() line with camera output.
#   Everything else stays exactly the same.
#
# How to run:
#   python simulator.py
#
# Commands you can type:
#   - Any letter (A, B, C...)  → adds that letter
#   - Any word (HELLO, THANKS) → adds that word
#   - space                    → adds a space between words
#   - del                      → deletes last character
#   - clear                    → wipes the whole sentence
#   - refine                   → sends sentence to Gemini (English + Hindi)
#   - quit                     → exit
# -------------------------------------------------------

from smoother import Smoother
from sentence_builder import SentenceBuilder
from llm_refiner import LLMRefiner

def run_simulator():
    smoother         = Smoother(window_size=10, min_count=7, min_confidence=0.80)
    sentence_builder = SentenceBuilder()
    refiner          = LLMRefiner()

    # Always start fresh
    smoother.reset()
    sentence_builder.reset()

    print("=" * 55)
    print("  ASL Pipeline Simulator")
    print("=" * 55)
    print("  Type letters, words, or commands.")
    print("  Commands: space | del | clear | refine | quit")
    print("=" * 55)
    print()

    while True:

        # ── THIS IS THE ONLY LINE YOU CHANGE WHEN INTEGRATING ──
        # Right now: you type the input manually
        # Later:     replace with your friend's model output
        #            e.g.  result = predict_from_camera()
        #                  user_input = result["prediction"]
        #                  confidence = result["confidence"]
        user_input = input("Enter prediction: ").strip()

        if not user_input:
            continue

        # --- quit ---
        if user_input.lower() == "quit":
            print("Bye!")
            break

        # --- refine: send to Gemini for English + Hindi ---
        if user_input.lower() == "refine":
            raw = sentence_builder.get()
            if not raw.strip():
                print("  [Nothing to refine yet]\n")
                continue
            print(f"\n  Sending to Gemini...")
            english, hindi = refiner.refine(raw)
            print(f"\n  Raw sentence : {raw}")
            print(f"  English      : {english}")
            print(f"  Hindi        : {hindi}")
            print()
            continue

        # --- clear ---
        if user_input.lower() == "clear":
            sentence_builder.reset()
            smoother.reset()
            print("  [Sentence cleared]\n")
            continue

        # --- space (goes directly to sentence builder, skip smoother) ---
        if user_input.lower() == "space":
            current_sentence = sentence_builder.add("space")
            print(f"  Added space")
            print(f"  Sentence : [{current_sentence}]\n")
            continue

        # --- del (goes directly to sentence builder, skip smoother) ---
        if user_input.lower() == "del":
            current_sentence = sentence_builder.add("del")
            print(f"  Deleted last character")
            print(f"  Sentence : [{current_sentence}]\n")
            continue

        # ── SIMULATE SMOOTHER ──
        # In real use, confidence comes from the model.
        # In simulator, we assume high confidence for whatever you type.
        # To test low-confidence filtering, type like: "A:0.5"

        confidence = 0.95  # default simulated confidence
        prediction = user_input

        if ":" in user_input:
            parts = user_input.split(":")
            prediction = parts[0].strip()
            try:
                confidence = float(parts[1].strip())
            except ValueError:
                pass

        # Feed into smoother — simulate N repeated frames
        accepted = None
        for _ in range(smoother.min_count):
            accepted = smoother.add(prediction, confidence)
            if accepted:
                break

        if accepted:
            current_sentence = sentence_builder.add(accepted)
            print(f"  Accepted : {accepted}")
            print(f"  Sentence : [{current_sentence}]")
            print()
        else:
            print(f"  Filtered out (low confidence or flickering)\n")


if __name__ == "__main__":
    run_simulator()
