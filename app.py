import streamlit as st
import joblib
import numpy as np
import re


model = joblib.load(r'D:\emotion_classifiaction\model.pkl')


def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  
    text = text.lower()  
    return text


st.title("Text Emotion Classification")
st.write("Enter a sentence to classify its emotion:")

user_input = st.text_area("Your text here:", "")

if st.button("Classify Emotion"):
    if user_input.strip():
        processed_text = preprocess_text(user_input)
        prediction = model.predict([processed_text])
        emotion_label = prediction[0]

        st.subheader("Predicted Emotion:")
        st.write(f"**{emotion_label}**")
    else:
        st.warning("Please enter some text before clicking classify!")

