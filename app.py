import streamlit as st

from config import BASE_DIR

st.set_page_config(page_title="ASL Smart Translator", layout="wide")

st.title("ASL Smart Translator")
st.write("Project scaffold is ready. Add your datasets, models, and inference pipeline here.")
st.caption(f"Workspace root: {BASE_DIR}")
