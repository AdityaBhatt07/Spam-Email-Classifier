"""
Streamlit App - Spam Email/SMS Classifier

This app loads the trained model + TF-IDF vectorizer from the model/ folder
and lets users classify any pasted message as ham or spam.

Constraints (per project requirements):
- Only use: scikit-learn, pandas, numpy, streamlit, pickle (built-in)
"""

from __future__ import annotations

import os
import pickle
from typing import Tuple

import numpy as np
import streamlit as st


st.set_page_config(
    page_title="Spam Classifier",
    page_icon="Spam",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_artifacts() -> Tuple[object, object]:
    """
    Load the model and vectorizer once and cache them.

    Why cache_resource:
    - Prevents reloading pickles on every interaction, making the UI snappy.
    """

    model_path = os.path.join("model", "model.pkl")
    vect_path = os.path.join("model", "vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vect_path):
        raise FileNotFoundError(
            "Model artifacts not found. Please run training first:\n"
            "  python train.py\n"
            "Expected files:\n"
            f"  - {model_path}\n"
            f"  - {vect_path}"
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vect_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict_proba(model: object, vectorizer: object, text: str) -> Tuple[int, float, float]:
    """
    Predict class + probabilities.

    Returns:
    - predicted_label: 0 (ham) or 1 (spam)
    - p_ham: probability of ham
    - p_spam: probability of spam
    """

    X = vectorizer.transform([text])

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Loaded model does not support predict_proba(), which is required to show confidence."
        )

    proba = model.predict_proba(X)[0]

    # Convention: class 0 -> ham, class 1 -> spam
    p_ham = float(proba[0])
    p_spam = float(proba[1])
    pred = int(np.argmax(proba))
    return pred, p_ham, p_spam


st.title("Spam Email/SMS Classifier")
st.caption("Paste a message below and classify it as ham or spam.")

with st.expander("Try these examples"):
    st.markdown("**Spam examples**")
    st.code(
        "Free entry in 2 a wkly comp to win FA Cup final tkts! Text FA to 87121.\n\n"
        "URGENT! You have won a 1 week FREE membership. Call now to claim your prize."
    )
    st.markdown("**Ham examples**")
    st.code(
        "Hey, are we still meeting at 6?\n\n"
        "Ok lar... Joking wif u oni..."
    )


try:
    model, vectorizer = load_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Then run: `python train.py` to generate the artifacts in `model/`.")
    st.stop()

message = st.text_area(
    "Message",
    placeholder="Paste an email or SMS message here...",
    height=180,
)

col_a, col_b = st.columns([1, 2])
with col_a:
    classify = st.button("Classify", type="primary", use_container_width=True)
with col_b:
    st.write("")  # spacing
    st.write("Artifacts loaded from `model/`.")


if classify:
    text = (message or "").strip()
    if not text:
        st.warning("Please paste a message first.")
    else:
        pred, p_ham, p_spam = predict_proba(model=model, vectorizer=vectorizer, text=text)

        if pred == 0:
            st.success("**HAM (Not Spam)**")
        else:
            st.error("**SPAM**")

        m1, m2 = st.columns(2)
        m1.metric("Ham confidence", f"{p_ham * 100:.2f}%")
        m2.metric("Spam confidence", f"{p_spam * 100:.2f}%")

st.divider()
st.caption("Tech stack: scikit-learn + TF-IDF | Streamlit | pandas | NumPy | Docker")

