import streamlit as st
import pandas as pd
from src.utils import load_object

# Load model, vectorizer, and label encoder
model = load_object("artifacts/model.pkl")
vectorizer = load_object("artifacts/preprocessor.pkl")
label_encoder = load_object("artifacts/label_encoder.pkl")

# Emotion label to emoji
emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨",
    "joy": "😂",
    "neutral": "😐",
    "sadness": "😔",
    "surprise": "😮"
}

# Prediction function
def predict_emotion(text):
    vectorized_text = vectorizer.transform([text])
    prediction_index = model.predict(vectorized_text)[0]
    prediction_label = label_encoder.inverse_transform([prediction_index])[0]
    return prediction_label

# Streamlit UI
def main():
    st.set_page_config(page_title="Emotion Detection", layout="centered")
    st.title("🔍 Emotion Detection from Text")

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Detect Emotion in Your Text")
        user_input = st.text_area("Enter your sentence:")

        if st.button("Detect"):
            if not user_input.strip():
                st.warning("⚠️ Please enter some text.")
            else:
                prediction = predict_emotion(user_input)
                emoji = emotions_emoji_dict.get(prediction.lower(), "❓")

                st.success("🎯 Prediction")
                st.markdown(f"**{prediction.capitalize()}** {emoji}")

    elif choice == "About":
        st.subheader("About This App")
        st.markdown("""
        This app is part of an **NLP-based Emotion Detection** project using machine learning.

        **Built With:**
        - TF-IDF Vectorization
        - Multiclass Classification (e.g., Logistic Regression / LinearSVC)
        - Label Encoding
        - Streamlit for UI
        """)

if __name__ == "__main__":
    main()
