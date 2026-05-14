# src/pipeline/app.py
# -------------------------------------------------------
# Streamlit UI for the ASL Translator
# Uses OpenCV directly (no streamlit-webrtc needed)
#
# How to run (from your git repo root):
#   streamlit run src/pipeline/app.py
# -------------------------------------------------------

import sys
import cv2
import pickle
import numpy as np
import streamlit as st
from pathlib import Path

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

# ── Page config ─────────────────────────────────────
st.set_page_config(
    page_title="ASL Smart Translator",
    page_icon="🤟",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────
st.markdown("""
<style>
    .title { text-align: center; font-size: 2.5rem; font-weight: bold; color: #ffffff; margin-bottom: 0; }
    .subtitle { text-align: center; color: #888; margin-bottom: 1rem; }
    .label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
    .value-box {
        background: #1e2130;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-left: 4px solid #4CAF50;
        min-height: 55px;
    }
    .value-box.hindi { border-left-color: #FF9800; }
    .value-box.raw   { border-left-color: #2196F3; }
    .value-box.pred  { border-left-color: #9C27B0; }
    .value-text { font-size: 1.15rem; color: #ffffff; font-weight: 500; }
    .confidence { font-size: 0.8rem; color: #4CAF50; margin-top: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Load model once (cached so it doesn't reload every frame) ──
@st.cache_resource
def load_phrase_model():
    model_path = ROOT_DIR / "models" / "phrase_classifier.pkl"
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["encoder"]

@st.cache_resource
def load_hands():
    return create_hands_detector(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

@st.cache_resource
def load_refiner():
    return LLMRefiner()

model, encoder = load_phrase_model()
hands          = load_hands()
refiner        = load_refiner()

# ── Feature builder ──────────────────────────────────
def build_feature_vector(hand_landmarks_list, expected_features):
    if not hand_landmarks_list:
        return None
    hand_rows = [extract_landmark_vector(h) for h in hand_landmarks_list]
    if expected_features == 63:
        best_row = max(hand_rows, key=len)
        return best_row if len(best_row) == expected_features else None
    combined = [v for row in hand_rows for v in row]
    return combined if len(combined) == expected_features else None

# ── Session state (persists across reruns) ───────────
if "smoother"         not in st.session_state:
    st.session_state.smoother         = Smoother(window_size=10, min_count=5, min_confidence=0.60)
if "sentence_builder" not in st.session_state:
    st.session_state.sentence_builder = SentenceBuilder()
if "sentence"         not in st.session_state:
    st.session_state.sentence         = ""
if "english"          not in st.session_state:
    st.session_state.english          = ""
if "hindi"            not in st.session_state:
    st.session_state.hindi            = ""
if "running"          not in st.session_state:
    st.session_state.running          = False

smoother         = st.session_state.smoother
sentence_builder = st.session_state.sentence_builder

# ── Title ────────────────────────────────────────────
st.markdown('<div class="title">🤟 ASL Smart Translator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sign → Sentence → English → Hindi</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Layout: camera left, outputs right ───────────────
col_cam, col_out = st.columns([3, 2])

with col_cam:
    st.markdown("### 📷 Camera Feed")
    frame_placeholder = st.empty()  # this is where camera frames go

with col_out:
    st.markdown("### 📊 Live Output")

    pred_placeholder = st.empty()
    raw_placeholder  = st.empty()
    eng_placeholder  = st.empty()
    hin_placeholder  = st.empty()

    st.markdown("---")

    btn1, btn2 = st.columns(2)
    with btn1:
        refine_btn = st.button("✨ Refine with Gemini", use_container_width=True)
    with btn2:
        clear_btn  = st.button("🗑️ Clear", use_container_width=True)

    undo_btn = st.button("↩️ Undo Last Word", use_container_width=True)
    stop_btn = st.button("⏹️ Stop Camera", use_container_width=True)

# ── Button actions ───────────────────────────────────
if refine_btn:
    raw = st.session_state.sentence.strip()
    if raw:
        with st.spinner("Asking Gemini..."):
            english, hindi = refiner.refine(raw)
        st.session_state.english = english
        st.session_state.hindi   = hindi
    else:
        st.warning("Nothing to refine yet — start signing!")

if clear_btn:
    sentence_builder.reset()
    smoother.reset()
    st.session_state.sentence = ""
    st.session_state.english  = ""
    st.session_state.hindi    = ""

if undo_btn:
    current = sentence_builder.get().strip()
    words   = current.split()
    if words:
        words.pop()
        sentence_builder.reset()
        for w in words:
            sentence_builder.add(w)
            sentence_builder.add("space")
        st.session_state.sentence = sentence_builder.get()

if stop_btn:
    st.session_state.running = False

# ── Helper to update output panels ───────────────────
def update_outputs(prediction, confidence, hand_found):
    pred_display = prediction.upper() if hand_found and prediction else "Waiting for hand..."
    conf_display = f"{confidence:.0%} confidence" if hand_found and prediction else ""

    pred_placeholder.markdown(f"""
    <div class="value-box pred">
        <div class="label">Current Prediction</div>
        <div class="value-text">{pred_display}</div>
        <div class="confidence">{conf_display}</div>
    </div>""", unsafe_allow_html=True)

    raw_display = st.session_state.sentence.strip() or "Start signing..."
    raw_placeholder.markdown(f"""
    <div class="value-box raw">
        <div class="label">Raw Sentence</div>
        <div class="value-text">{raw_display}</div>
    </div>""", unsafe_allow_html=True)

    eng_placeholder.markdown(f"""
    <div class="value-box">
        <div class="label">Refined English</div>
        <div class="value-text">{st.session_state.english or "Press Refine ↓"}</div>
    </div>""", unsafe_allow_html=True)

    hin_placeholder.markdown(f"""
    <div class="value-box hindi">
        <div class="label">Hindi Translation</div>
        <div class="value-text">{st.session_state.hindi or "—"}</div>
    </div>""", unsafe_allow_html=True)

# ── Camera loop ──────────────────────────────────────
update_outputs("", 0.0, False)  # show initial state

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Could not open camera. Close other apps using it and refresh.")
else:
    st.session_state.running = True
    try:
        while st.session_state.running:
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

                    accepted = smoother.add(prediction, confidence)
                    if accepted:
                        sentence_builder.add(accepted)
                        sentence_builder.add("space")
                        st.session_state.sentence = sentence_builder.get()
            else:
                smoother.reset()

            # Show frame in Streamlit (convert BGR to RGB first)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Update output panels
            update_outputs(prediction, confidence, hand_found)

    finally:
        cap.release()

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555; font-size:0.8rem'>"
    "ASL Smart Translator · Powered by MediaPipe + Gemini"
    "</div>",
    unsafe_allow_html=True
)
