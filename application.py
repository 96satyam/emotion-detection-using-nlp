import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("artifacts/model.pkl", "rb"))
vectorizer = pickle.load(open("artifacts/preprocessor.pkl", "rb"))  # Renamed for clarity
label_encoder = pickle.load(open("artifacts/label_encoder.pkl", "rb"))

# Emotion mapping (adjust if your labels differ)
emotion_map = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Streamlit UI
st.set_page_config(page_title="Emotion Detection", layout="centered")

st.title("🧠 Emotion Detection from Text")

text_input = st.text_area("Enter your text here")

if st.button("Detect Emotion"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize and predict
        transformed = vectorizer.transform([text_input])
        pred = model.predict(transformed)[0]
        emotion = emotion_map.get(pred, "Unknown")

        st.success(f"**Detected Emotion:** {emotion}")
