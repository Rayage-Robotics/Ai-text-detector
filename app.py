import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from scipy.sparse import hstack

st.set_page_config(page_title="AI Attribution", page_icon="🤖")

st.title("🤖 Name That LLM")
st.write("Paste a text passage below. The model will identify which AI wrote it.")

@st.cache_resource
def load_models():
    model = joblib.load("model.joblib")
    tfidf_w = joblib.load("tfidf_w.joblib")
    tfidf_c = joblib.load("tfidf_c.joblib")
    sc_style = joblib.load("sc_style.joblib")
    return model, tfidf_w, tfidf_c, sc_style

model, tfidf_w, tfidf_c, sc_style = load_models()

def style_features(texts):
    rows = []
    for t in texts:
        t = "" if not isinstance(t, str) else t
        words = t.split()
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        n_words, n_sents = max(len(words), 1), max(len(sents), 1)
        sent_lens = [len(s.split()) for s in sents] if sents else [0]
        rows.append([
            t.count("—"), t.count("–"), t.count(";"), t.count("("), t.count(")"), t.count(":"),
            t.count("\n"), t.count("*"), t.count("-"), len(t), n_words, n_sents,
            n_words / n_sents, float(np.std(sent_lens)),
            sum(len(w) for w in words) / n_words,
            sum(c.isupper() for c in t) / max(len(t), 1),
            sum(c.isdigit() for c in t) / max(len(t), 1),
            sum(w.endswith("ly") for w in words) / n_words,
            sum(w[0].isupper() for w in words if w) / n_words,
        ])
    return np.array(rows, dtype=np.float64)

text_input = st.text_area("Enter text here:", height=200)

if st.button("Identify AI Author"):
    if len(text_input.split()) < 10:
        st.warning("Please paste a longer paragraph (at least 10 words).")
    else:
        with st.spinner("Analyzing text style..."):
            W = tfidf_w.transform([text_input])
            C = tfidf_c.transform([text_input])
            S = sc_style.transform(style_features([text_input]))
            X = hstack([W, C, S])
            
            probs = model.predict_proba(X)[0]
            classes = model.classes_
            top_indices = np.argsort(probs)[::-1][:3]
            
            st.markdown("### Top Predictions:")
            for idx in top_indices:
                st.markdown(f"- **{classes[idx]}** ({probs[idx]*100:.1f}% confidence)")
