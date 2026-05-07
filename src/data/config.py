# config.py
import os

# ── Paths ──────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR     = os.path.join(BASE_DIR, "datasets")

ALPHABET_DIR    = os.path.join(DATASET_DIR, "alphabets")
NUMBERS_DIR     = os.path.join(DATASET_DIR, "numbers")
WLASL_DIR       = os.path.join(DATASET_DIR, "wlasl")

PHRASE_CSV      = os.path.join(DATASET_DIR, "phrase_landmarks.csv")
MY_CSV          = os.path.join(DATASET_DIR, "my_phrase_landmarks.csv")
COMBINED_CSV    = os.path.join(DATASET_DIR, "combined_landmarks.csv")

MODEL_DIR       = os.path.join(BASE_DIR, "models")
ALPHABET_MODEL  = os.path.join(MODEL_DIR, "alphabet_cnn.h5")
NUMBER_MODEL    = os.path.join(MODEL_DIR, "number_cnn.h5")
PHRASE_MODEL    = os.path.join(MODEL_DIR, "phrase_classifier.pkl")
HAND_LANDMARKER_MODEL = os.path.join(MODEL_DIR, "hand_landmarker.task")
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ── Model Settings ─────────────────────────────────
IMG_SIZE        = (64, 64)
BATCH_SIZE      = 32
EPOCHS          = 25
NUM_ALPHABETS   = 29        # A-Z + SPACE + DELETE + NOTHING
NUM_NUMBERS     = 10        # 0-9
NUM_PHRASES     = 30

# ── MediaPipe ──────────────────────────────────────
NUM_LANDMARKS   = 21
LANDMARK_DIMS   = 3         # x, y, z
FEATURE_SIZE    = NUM_LANDMARKS * LANDMARK_DIMS   # = 63

# ── LLM ────────────────────────────────────────────
LLM_MODEL       = "claude-sonnet-4-20250514"
MAX_TOKENS      = 200

# ── Translation ────────────────────────────────────
TARGET_LANGUAGES = ["hi", "te", "fr", "es"]   # Hindi, Telugu, French, Spanish