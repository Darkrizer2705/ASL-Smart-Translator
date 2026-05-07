# ASL Smart Translator

This repository is a scaffold for an ASL translation pipeline covering:

- Alphabet classification from images
- Number classification from images
- Phrase classification from WLASL-derived landmarks
- Webcam capture and dataset preparation
- Optional LLM-based sentence refinement
- A Streamlit app for interactive inference

## Layout

- `datasets/` stores raw and processed data.
- `models/` stores trained artifacts.
- `src/` contains scripts for data, training, inference, LLM, and utilities.
- `notebooks/` contains exploratory and training notebooks.

## Next steps

1. Place your datasets under `datasets/`.
2. Implement the data extraction and training scripts in `src/`.
3. Train the models and save them in `models/`.
4. Run the app with Streamlit after the inference pipeline is ready.
