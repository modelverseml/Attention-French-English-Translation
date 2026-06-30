"""
Tiny translation UI for the Attention seq2seq project: type French, see the
English translation from each trained model (PyTorch / TensorFlow / from-scratch
NumPy GRU+attention) side by side. Streamlit Cloud entry point.

    streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

import streamlit as st
import translator

ICON = {"pytorch": "🔥", "tensorflow": "🧠", "manual": "🧮"}
SAMPLES = ["je vous remercie .", "merci beaucoup .", "le parlement européen",
           "nous devons agir maintenant ."]


@st.cache_resource(show_spinner="Loading trained models…")
def get_models():
    return translator.load_models()


st.set_page_config(page_title="FR→EN Attention", page_icon="🌍", layout="centered")
st.title("🌍 French → English — GRU + Attention")
st.caption("Seq2seq with Bahdanau attention — compare the PyTorch, TensorFlow, and "
           "from-scratch NumPy models on the same sentence.")

models = get_models()
if not models:
    st.warning("No trained models found. Train at least one first, e.g.:\n\n"
               "```\ncd code\npython seq2seq_pytorch.py\npython seq2seq_manual.py train\n```")
    st.stop()

st.caption("Loaded models: " + " · ".join(models))

if "fr" not in st.session_state:
    st.session_state.fr = SAMPLES[0]
cols = st.columns(len(SAMPLES))
for c, s in zip(cols, SAMPLES):
    if c.button(s[:16] + "…", use_container_width=True):
        st.session_state.fr = s

fr = st.text_area("French", key="fr", height=80)
if st.button("Translate", type="primary") and fr.strip():
    with st.spinner("Translating…"):
        outs = translator.translate_all(fr.strip(), models)
    for name, en in outs.items():
        st.markdown(f"**{ICON.get(name,'')} {name}**")
        st.success(en if en else "—")
