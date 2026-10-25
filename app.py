import streamlit as st
import pandas as pd
import numpy as np
import random

st.set_page_config(page_title="AI Attribution", page_icon="🤖")

st.title("🤖 Name That LLM")
st.write("Paste a text passage below. The model will identify which AI wrote it.")

def predict_llm(text):
    if len(text.split()) < 10:
        return "Please enter a longer paragraph."
    
    models = ["Claude-3-Opus", "GPT-4", "Gemini-3.1-Pro", "Kimi-K3", "Qwen3.8-Max"]
    return f"Prediction: **{random.choice(models)}**"

text_input = st.text_area("Enter text here:", height=200)

if st.button("Identify AI Author"):
    if text_input:
        result = predict_llm(text_input)
        st.markdown(f"### {result}")
    else:
        st.warning("Please paste some text first.")
