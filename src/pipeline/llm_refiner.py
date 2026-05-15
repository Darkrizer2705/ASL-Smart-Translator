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
#   pip install google-generativeai python-dotenv deep-translator
# -------------------------------------------------------

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import google.generativeai as genai
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

# Load the GEMINI_API_KEY from the repo root automatically.
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _fallback_hindi_translation(text: str) -> str:
    try:
        return GoogleTranslator(source="auto", target="hi").translate(text).strip()
    except Exception:
        return ""

class LLMRefiner:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        self.last_error = ""

        if not api_key or api_key.strip().lower().startswith("your_"):
            self.model = None
            self.last_error = (
                "GEMINI_API_KEY is missing or still set to a placeholder. Set a real key in .env to enable Gemini refinement."
            )
            return

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")  # free tier model

    def _parse_response(self, raw_output: str, fallback_english: str) -> tuple[str, str]:
        cleaned = _strip_code_fences(raw_output)

        try:
            payload = json.loads(cleaned)
            english = str(payload.get("english") or payload.get("English") or fallback_english).strip()
            hindi = str(payload.get("hindi") or payload.get("Hindi") or "").strip()
            return english or fallback_english, hindi
        except Exception:
            pass

        english = fallback_english
        hindi = ""

        english_match = re.search(
            r"(?ims)^\s*(?:english|en)\s*[:=-]\s*(.+?)(?=^\s*(?:hindi|hi)\s*[:=-]|\Z)",
            cleaned,
        )
        hindi_match = re.search(
            r"(?ims)^\s*(?:hindi|hi)\s*[:=-]\s*(.+)$",
            cleaned,
        )

        if english_match:
            english = english_match.group(1).strip()
        elif cleaned and cleaned != fallback_english:
            first_line = cleaned.splitlines()[0].strip()
            if first_line:
                english = first_line

        if hindi_match:
            hindi = hindi_match.group(1).strip()
        elif not english_match and len(cleaned.splitlines()) > 1:
            last_line = cleaned.splitlines()[-1].strip()
            if last_line and last_line != english:
                hindi = last_line

        return english or fallback_english, hindi

    def refine(self, raw_sentence: str):
        """
        Takes a raw ASL-style sentence and returns (english, hindi) as a tuple.

        Example:
            Input:  "I GO SCHOOL TOMORROW"
            Output: ("I will go to school tomorrow.", "मैं कल स्कूल जाऊंगा।")
        """
        if not raw_sentence.strip():
            return "", ""

        self.last_error = ""
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
- Reply with a JSON object only. No markdown, no code fences, no extra text.
- The JSON must have exactly these keys: english, hindi
- Example: {{"english": "I will go to school tomorrow.", "hindi": "मैं कल स्कूल जाऊंगा।"}}

ASL text: {raw_sentence}"""

        try:
            if self.model is None:
                self.last_error = (
                    "Gemini is not configured, so the app is using a Hindi-only fallback. Add a real GEMINI_API_KEY to enable English refinement."
                )
                hindi = _fallback_hindi_translation(raw_sentence.strip())
                return raw_sentence.strip(), hindi

            response = self.model.generate_content(prompt)
            raw_output = (response.text or "").strip()
            english, hindi = self._parse_response(raw_output, raw_sentence.strip())

            if not hindi:
                hindi = _fallback_hindi_translation(english)

            return english, hindi

        except Exception as e:
            self.last_error = f"Gemini refinement failed: {e}"
            print(f"[LLM Error] {e}")
            hindi = _fallback_hindi_translation(raw_sentence.strip())
            return raw_sentence.strip(), hindi


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
