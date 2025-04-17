

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_resources():
    tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
    model = load_model('best_bilstm_model.keras')
    return tokenizer, model

tokenizer, model = load_resources()
max_len = 100 

st.set_page_config(page_title="📱 SMS Spam Detector", page_icon="📩")
st.title("📱 SMS Spam Detector")
st.subheader("Check if a message is spam or not!")
st.markdown("---")

user_input = st.text_area("Enter the SMS message here:", height=150)

def predict_spam(text):
    # Clean the text a little (optional)
    text = text.lower().strip()

    # Tokenize and pad
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

    # Predict
    pred = model.predict(padded, verbose=0)[0][0]

    # Return result
    if pred > 0.5:
        return "🚨 Spam", pred
    else:
        return "✅ Not spam", pred


if st.button("Predict"):
    if user_input.strip():
        result, probability = predict_spam(user_input)
        st.success(f"**Prediction:** {result}")
        st.info(f"**Spam Probability:** {probability:.2f}")
    else:
        st.warning("Please enter a message to classify.")


st.markdown("---")
st.caption("Built using BiLSTM + Streamlit")