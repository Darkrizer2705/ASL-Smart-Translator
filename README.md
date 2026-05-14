# ASL Smart Translator

Real-time American Sign Language recognition pipeline with LLM-powered sentence refinement and Hindi translation.

## What it does

- **Alphabet recognition** — Spell out words letter by letter using ASL hand signs
- **Number recognition** — Recognise ASL numbers 0–9
- **Phrase recognition** — Instantly recognise common ASL words/phrases (HELLO, HELP, WATER, etc.)
- **Sentence builder** — Assembles recognised tokens into a sentence with a stability smoother to reduce flickering
- **LLM refinement** — Presses `R` to fix ASL-style grammar into natural English *and* translate to Hindi using Gemini
- **Streamlit Web App** — A modern, responsive web interface for easy multi-mode usage
- **Model Evaluation** — Auto-generates accuracy reports and confusion matrices during training

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create your .env file with your Gemini API key
#    (Get one free at https://aistudio.google.com/app/apikey)
echo GEMINI_API_KEY=your_key_here > .env
```

## Running

| Script | What it runs |
|---|---|
| `streamlit run src/pipeline/app.py` | **Streamlit Web App** — UI-based multi-mode ASL translation |
| `python -m src.pipeline.main` | **Main pipeline** — webcam multi-mode recognition + LLM refine |
| `python -m src.inference.predict_phrase` | Phrase-only recognition |
| `python -m src.inference.predict_alphabet_landmark` | Alphabet spelling |
| `python -m src.inference.predict_number` | Number recognition |
| `python -m src.pipeline.simulator` | Keyboard simulator (no camera needed) |

## Controls (main pipeline)

| Key | Action |
|---|---|
| `1` / `2` / `3` | Switch to Phrase / Alphabet / Number mode |
| `R` | Refine current sentence with Gemini (English + Hindi) |
| `C` | Clear sentence |
| `U` | Undo last word |
| `Q` | Quit |

## Layout

```
datasets/     Raw landmark CSVs and video data
models/       Trained model artifacts (.pkl, .h5, .task)
results/      Evaluation metrics (accuracy, classification reports, confusion matrices)
src/
  data/       Data collection and landmark extraction scripts
  models/     Training scripts (alphabet, numbers, phrases)
  inference/  Real-time webcam inference scripts
  pipeline/   Main app pipeline (smoother, sentence builder, LLM refiner)
  utils/      Shared MediaPipe utilities
scripts/      Analysis and maintenance utilities
```

## Re-training models

```bash
# Phrase classifier (XGBoost on hand landmarks)
python -m src.models.train_phrases

# Alphabet classifier
python -m src.models.train_alphabet --mode landmarks

# Number classifier
python -m src.models.train_numbers --mode landmarks
```

## Model Evaluation

Training any of the models using the scripts above will automatically evaluate their accuracy and save the following artifacts to the `results/` folder:
- **`[model]_metrics.txt`**: Contains the overall accuracy score and a detailed per-class classification report.
- **`[model]_confusion_matrix.png`**: A plotted visualization of the confusion matrix to compare model predictions against true labels.
