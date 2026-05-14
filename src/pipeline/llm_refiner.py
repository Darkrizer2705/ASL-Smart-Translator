# llm_refiner.py
# -------------------------------------------------------
# What this does:
#   Takes a raw ASL-style sentence like "I GO SCHOOL TOMORROW"
#   and uses Gemini to:
#     1. Fix it into proper English
#     2. Translate it to Hindi
#   Both come back in one single API call.
#
# Usage:
#   english, hindi = refiner.refine("I GO SCHOOL TOMORROW")
#
# Requirements (already installed):
#   pip install google-generativeai python-dotenv
# -------------------------------------------------------

import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load the GEMINI_API_KEY from your .env file automatically
load_dotenv()

class LLMRefiner:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Make sure your .env file has it.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash-lite")  # free tier model

    def refine(self, raw_sentence: str):
        """
        Takes a raw ASL-style sentence and returns (english, hindi) as a tuple.

        Example:
            Input:  "I GO SCHOOL TOMORROW"
            Output: ("I will go to school tomorrow.", "मैं कल स्कूल जाऊंगा।")
        """
        if not raw_sentence.strip():
            return "", ""

        prompt = f"""You are a grammar correction assistant for an ASL (American Sign Language) translator.

ASL grammar skips articles, helper verbs, and prepositions.
Do two things with the ASL text below:
1. Convert it into natural, grammatically correct English
2. Translate that English sentence into Hindi

Rules:
- Keep the meaning exactly the same, just fix the grammar
- Add missing words (a, the, is, will, to, etc.)
- Capitalize properly and add punctuation
- If the input is a single word, just clean it up and translate it
- Reply in EXACTLY this format, two lines, nothing else:
ENGLISH: <corrected English sentence>
HINDI: <Hindi translation>

ASL text: {raw_sentence}"""

        try:
            response = self.model.generate_content(prompt)
            raw_output = response.text.strip()

            # Parse the two lines Gemini returns
            english = raw_sentence  # fallback to original if parsing fails
            hindi   = ""

            for line in raw_output.splitlines():
                if line.upper().startswith("ENGLISH:"):
                    english = line.split(":", 1)[1].strip()
                elif line.upper().startswith("HINDI:"):
                    hindi = line.split(":", 1)[1].strip()

            return english, hindi

        except Exception as e:
            print(f"[LLM Error] {e}")
            return raw_sentence, ""  # if something goes wrong, return original + empty hindi


# -------------------------------------------------------
# Quick test — run this file directly to check it works:
#   python llm_refiner.py
# -------------------------------------------------------
if __name__ == "__main__":
    refiner = LLMRefiner()

    test_sentences = [
        "I GO SCHOOL TOMORROW",
        "YOU HELP ME PLEASE",
        "WATER WHERE",
        "I LOVE YOU",
        "SHE SICK YESTERDAY",
    ]

    print("Testing LLM Refiner...\n")
    for raw in test_sentences:
        english, hindi = refiner.refine(raw)
        print(f"  Raw     : {raw}")
        print(f"  English : {english}")
        print(f"  Hindi   : {hindi}")
        print()
