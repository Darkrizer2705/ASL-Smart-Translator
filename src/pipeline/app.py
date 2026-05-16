# src/pipeline/app.py
# -------------------------------------------------------
# Streamlit UI for the ASL Translator
# Supports three recognition modes: Phrase / Alphabet / Number
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
from src.pipeline.llm_refiner import LLMRefiner, _fallback_hindi_translation
from src.llm.rag_pipeline import rag_refine
from src.llm.gan_augment import generate_samples

# ── Page config ─────────────────────────────────────
st.set_page_config(
    page_title="ASL Smart Translator",
    page_icon="🤟",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .title { text-align: center; font-size: 2.5rem; font-weight: 700; color: #ffffff; margin-bottom: 0; }
    .subtitle { text-align: center; color: #888; margin-bottom: 1rem; font-size: 1rem; }
    .mode-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .mode-phrase   { background: #1a3a4a; color: #00d4ff; border: 1px solid #00d4ff; }
    .mode-alphabet { background: #1a3a2a; color: #66ff88; border: 1px solid #66ff88; }
    .mode-number   { background: #3a2a1a; color: #ffaa33; border: 1px solid #ffaa33; }
    .label { font-size: 0.72rem; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
    .value-box {
        background: #1e2130;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-left: 4px solid #4CAF50;
        min-height: 55px;
    }
    .value-box.hindi  { border-left-color: #FF9800; }
    .value-box.raw    { border-left-color: #2196F3; }
    .value-box.pred   { border-left-color: #9C27B0; }
    .value-box.spell  { border-left-color: #66ff88; background: #111e14; }
    .value-text { font-size: 1.15rem; color: #ffffff; font-weight: 500; }
    .confidence { font-size: 0.8rem; color: #4CAF50; margin-top: 3px; }
    .spell-text { font-size: 1.4rem; color: #66ff88; font-weight: 700; letter-spacing: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Cached resource loaders ──────────────────────────
@st.cache_resource
def load_all_models():
    def _load(path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return d["model"], d["encoder"]

    pm, pe = _load(ROOT_DIR / "models" / "phrase_classifier.pkl")
    am, ae = _load(ROOT_DIR / "models" / "alphabet_landmark_classifier.pkl")
    nm, ne = _load(ROOT_DIR / "models" / "number_landmark_classifier.pkl")
    return (pm, pe), (am, ae), (nm, ne)

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

(phrase_model, phrase_enc), (alpha_model, alpha_enc), (num_model, num_enc) = load_all_models()
hands   = load_hands()
refiner = load_refiner()

MODELS = {
    "Phrase":   (phrase_model, phrase_enc),
    "Alphabet": (alpha_model,  alpha_enc),
    "Number":   (num_model,    num_enc),
}
MODE_BADGE = {
    "Phrase":   "mode-phrase",
    "Alphabet": "mode-alphabet",
    "Number":   "mode-number",
}
MODE_ICON = {"Phrase": "💬", "Alphabet": "🔤", "Number": "🔢"}


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


# ── Session state ────────────────────────────────────
defaults = {
    "smoother":         Smoother(window_size=10, min_count=5, min_confidence=0.60),
    "sentence_builder": SentenceBuilder(),
    "sentence":         "",
    "english":          "",
    "hindi":            "",
    "running":          False,
    "mode":             "Phrase",
    "current_word":     [],      # alphabet letter buffer
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

smoother         = st.session_state.smoother
sentence_builder = st.session_state.sentence_builder


# ── Title ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧬 GAN Augmentation")
    st.markdown("Generate synthetic data to augment the landmark dataset.")
    gan_label = st.selectbox("Select phrase", MODELS["Phrase"][1].classes_)
    gan_n = st.slider("Samples to generate", 10, 500, 50)
    if st.button("Generate Synthetic Data", use_container_width=True):
        try:
            with st.spinner(f"Generating {gan_n} samples for '{gan_label}'..."):
                fake_data = generate_samples(gan_label, gan_n)
            st.success(f"✅ Generated {fake_data.shape[0]} realistic fake samples!")
            st.dataframe(fake_data[:5])
            st.caption("Showing first 5 generated landmark vectors")
        except Exception as e:
            st.error(f"GAN not trained or error: {e}")

st.markdown('<div class="title">🤟 ASL Smart Translator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sign → Sentence → English → Hindi</div>', unsafe_allow_html=True)
st.markdown("---")

if getattr(refiner, "last_error", ""):
    st.warning(refiner.last_error)

# ── Mode selector ────────────────────────────────────
col_m1, col_m2, col_m3, col_m4 = st.columns([1, 1, 1, 2])
with col_m1:
    if st.button("💬 Phrase Mode",   use_container_width=True,
                 type="primary" if st.session_state.mode == "Phrase" else "secondary"):
        if st.session_state.mode != "Phrase":
            st.session_state.mode = "Phrase"
            smoother.reset()
            st.session_state.current_word = []
with col_m2:
    if st.button("🔤 Alphabet Mode", use_container_width=True,
                 type="primary" if st.session_state.mode == "Alphabet" else "secondary"):
        if st.session_state.mode != "Alphabet":
            st.session_state.mode = "Alphabet"
            smoother.reset()
            st.session_state.current_word = []
with col_m3:
    if st.button("🔢 Number Mode",   use_container_width=True,
                 type="primary" if st.session_state.mode == "Number" else "secondary"):
        if st.session_state.mode != "Number":
            st.session_state.mode = "Number"
            smoother.reset()
            st.session_state.current_word = []

mode         = st.session_state.mode
model, enc   = MODELS[mode]
badge_class  = MODE_BADGE[mode]

st.markdown(f'<span class="mode-badge {badge_class}">{MODE_ICON[mode]} {mode.upper()} MODE</span>',
            unsafe_allow_html=True)

# ── Layout ───────────────────────────────────────────
col_cam, col_out = st.columns([3, 2])

with col_cam:
    st.markdown("### 📷 Camera Feed")
    frame_placeholder = st.empty()

with col_out:
    st.markdown("### 📊 Live Output")

    pred_placeholder  = st.empty()
    spell_placeholder = st.empty()   # only visible in Alphabet mode
    raw_placeholder   = st.empty()
    eng_placeholder   = st.empty()
    hin_placeholder   = st.empty()

    st.markdown("---")

    # Action buttons
    btn1, btn2, btn3 = st.columns([1, 1, 1])
    with btn1:
        refine_btn = st.button("✨ Refine (Gemini)", use_container_width=True)
    with btn2:
        rag_btn = st.button("🧠 Refine (RAG)", use_container_width=True)
    with btn3:
        clear_btn  = st.button("🗑️ Clear", use_container_width=True)

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        undo_btn  = st.button("↩️ Undo Last Word",    use_container_width=True)
    with col_u2:
        commit_btn = st.button("✅ Commit Word (A→Z)", use_container_width=True,
                               help="Alphabet mode: commit the current spelling buffer as a word")

    stop_btn  = st.button("⏹️ Stop Camera", use_container_width=True)


# ── Button actions ───────────────────────────────────
if refine_btn:
    raw = st.session_state.sentence.strip()
    if raw:
        with st.spinner("Asking Gemini..."):
            english, hindi = refiner.refine(raw)
        st.session_state.english = english
        st.session_state.hindi   = hindi
        if getattr(refiner, "last_error", ""):
            st.warning(refiner.last_error)
    else:
        st.warning("Nothing to refine yet — start signing!")

if rag_btn:
    raw = st.session_state.sentence.strip()
    if raw:
        with st.spinner("RAG Pipeline: Retrieving context & asking Claude..."):
            try:
                res = rag_refine(raw.split())
                eng = res.get("refined", "")
                hin = _fallback_hindi_translation(eng) if eng else ""
                st.session_state.english = eng
                st.session_state.hindi = hin
                if res.get("retrieved"):
                    st.toast(f"Retrieved docs: {', '.join(res['retrieved'])}", icon="📚")
            except Exception as e:
                st.error(f"RAG failed: {e}")
    else:
        st.warning("Nothing to refine yet — start signing!")

if clear_btn:
    sentence_builder.reset()
    smoother.reset()
    st.session_state.sentence     = ""
    st.session_state.english      = ""
    st.session_state.hindi        = ""
    st.session_state.current_word = []

if undo_btn:
    if mode == "Alphabet" and st.session_state.current_word:
        st.session_state.current_word.pop()
    else:
        current = sentence_builder.get().strip()
        words   = current.split()
        if words:
            words.pop()
            sentence_builder.reset()
            for i, w in enumerate(words):
                sentence_builder.add(w)
                if i < len(words) - 1:
                    sentence_builder.add("space")
            st.session_state.sentence = sentence_builder.get()

if commit_btn and mode == "Alphabet" and st.session_state.current_word:
    word = "".join(st.session_state.current_word)
    sentence_builder.add(word)
    sentence_builder.add("space")
    st.session_state.sentence     = sentence_builder.get()
    st.session_state.current_word = []
    st.session_state.english      = ""
    st.session_state.hindi        = ""

if stop_btn:
    st.session_state.running = False


# ── Output panel helper ──────────────────────────────
def update_outputs(prediction, confidence, hand_found):
    pred_display = prediction.upper() if hand_found and prediction else "Waiting for hand..."
    conf_display = f"{confidence:.0%} confidence" if hand_found and prediction else ""

    pred_placeholder.markdown(f"""
    <div class="value-box pred">
        <div class="label">Current Prediction — {mode} Mode</div>
        <div class="value-text">{pred_display}</div>
        <div class="confidence">{conf_display}</div>
    </div>""", unsafe_allow_html=True)

    # Alphabet spelling buffer
    if mode == "Alphabet":
        spell_word = "".join(st.session_state.current_word) or "—"
        spell_placeholder.markdown(f"""
    <div class="value-box spell">
        <div class="label">✏️ Currently spelling</div>
        <div class="spell-text">{spell_word}</div>
    </div>""", unsafe_allow_html=True)
    else:
        spell_placeholder.empty()

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
update_outputs("", 0.0, False)

_, col_start, _ = st.columns([1, 2, 1])
with col_start:
    start_btn = st.button("▶️ Start Camera", use_container_width=True, type="primary")

if start_btn:
    st.session_state.running = True

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open camera. Close other apps using it and refresh.")
        st.session_state.running = False
    else:
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
                        prediction = enc.classes_[pred_idx]
                        hand_found = True

                        accepted = smoother.add(prediction, confidence)
                        if accepted:
                            if mode == "Phrase":
                                sentence_builder.add(accepted)
                                sentence_builder.add("space")
                                st.session_state.sentence = sentence_builder.get()
                                st.session_state.english  = ""
                                st.session_state.hindi    = ""

                            elif mode == "Alphabet":
                                if accepted.lower() == "nothing":
                                    pass
                                elif accepted.lower() == "del":
                                    if st.session_state.current_word:
                                        st.session_state.current_word.pop()
                                elif accepted.lower() == "space":
                                    if st.session_state.current_word:
                                        word = "".join(st.session_state.current_word)
                                        sentence_builder.add(word)
                                        sentence_builder.add("space")
                                        st.session_state.sentence     = sentence_builder.get()
                                        st.session_state.current_word = []
                                        st.session_state.english      = ""
                                        st.session_state.hindi        = ""
                                else:
                                    st.session_state.current_word.append(accepted.upper())

                            elif mode == "Number":
                                sentence_builder.add(accepted)
                                sentence_builder.add("space")
                                st.session_state.sentence = sentence_builder.get()
                                st.session_state.english  = ""
                                st.session_state.hindi    = ""
                else:
                    smoother.reset()

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                update_outputs(prediction, confidence, hand_found)

        finally:
            cap.release()

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555; font-size:0.8rem'>"
    "ASL Smart Translator · Phrase 💬 · Alphabet 🔤 · Number 🔢 · Powered by MediaPipe + Gemini"
    "</div>",
    unsafe_allow_html=True
)
