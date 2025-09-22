import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

# ---------------------------
# Load Models and Preprocessors
# ---------------------------
VECTORIZER_PATH = "/home/vishwesh/Documents/Labmentix_Internship/Project_7/tfidf_vectorizer.joblib"
LOGREG_PATH = "/home/vishwesh/Documents/Labmentix_Internship/Project_7/best_logreg_model.joblib"
CNN_PATH = "/home/vishwesh/Documents/Labmentix_Internship/Project_7/best_cnn_model-1.h5"
LSTM_PATH = "/home/vishwesh/Documents/Labmentix_Internship/Project_7/best_lstm_model-1.h5"
TOKENIZER_PATH = "/home/vishwesh/Documents/Labmentix_Internship/Project_7/tokenizer-old.pkl"  # saved using joblib.dump(tokenizer, "tokenizer.joblib")

# Load vectorizer & tokenizer
vectorizer = joblib.load(VECTORIZER_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)

# Load models
logreg_model = joblib.load(LOGREG_PATH)  # since it’s sklearn-based
cnn_model = tf.keras.models.load_model(CNN_PATH)
lstm_model = tf.keras.models.load_model(LSTM_PATH)

MAXLEN = 100  # padding length used during training

# Toxicity Labels
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# ---------------------------
# Helper Functions
# ---------------------------
def predict_logreg(text):
    X = vectorizer.transform([text])
    preds = logreg_model.predict(X)
    return dict(zip(LABELS, preds[0]))

def predict_dl(model, text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding="post")
    preds = model.predict(padded)
    preds = (preds > 0.5).astype(int)[0]
    return dict(zip(LABELS, preds))

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Toxic Comment Classifier", layout="wide")

st.title("🧠 Toxic Comment Classification App")
st.markdown("This app lets you test **Logistic Regression, CNN, and LSTM models** for toxicity detection in text.")

# Sidebar - Model selection
model_choice = st.sidebar.radio("Select Model", ["Logistic Regression", "CNN", "LSTM"])

# ---------------------------
# Real-time Prediction Section
# ---------------------------
st.header("🔮 Real-time Prediction")
user_input = st.text_area("Enter a comment:", "")

if st.button("Predict"):
    if user_input.strip():
        if model_choice == "Logistic Regression":
            preds = predict_logreg(user_input)
        elif model_choice == "CNN":
            preds = predict_dl(cnn_model, user_input)
        else:
            preds = predict_dl(lstm_model, user_input)

        st.subheader("Prediction Results")
        for label, val in preds.items():
            st.write(f"**{label}**: {'✅ Yes' if val == 1 else '❌ No'}")

# ---------------------------
# Bulk Prediction (CSV Upload)
# ---------------------------
st.header("📂 Bulk Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload a CSV with a 'comment_text' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "comment_text" not in df.columns:
        st.error("CSV must have a 'comment_text' column")
    else:
        st.success("File uploaded successfully!")
        if model_choice == "Logistic Regression":
            preds = logreg_model.predict(vectorizer.transform(df["comment_text"]))
        elif model_choice == "CNN":
            seq = tokenizer.texts_to_sequences(df["comment_text"])
            padded = pad_sequences(seq, maxlen=MAXLEN, padding="post")
            preds = (cnn_model.predict(padded) > 0.5).astype(int)
        else:
            seq = tokenizer.texts_to_sequences(df["comment_text"])
            padded = pad_sequences(seq, maxlen=MAXLEN, padding="post")
            preds = (lstm_model.predict(padded) > 0.5).astype(int)

        preds_df = pd.DataFrame(preds, columns=LABELS)
        results = pd.concat([df, preds_df], axis=1)
        st.dataframe(results.head(10))
        st.download_button("Download Predictions", results.to_csv(index=False), "predictions.csv", "text/csv")

# ---------------------------
# Sample Test Cases
# ---------------------------
st.header("📝 Sample Test Cases")
examples = [
    "I love this product, it’s amazing!",
    "You are so stupid and dumb.",
    "I will find you and hurt you.",
    "This is the best day of my life.",
]

selected_example = st.selectbox("Choose an example to test", examples)
if st.button("Run Example"):
    if model_choice == "Logistic Regression":
        preds = predict_logreg(selected_example)
    elif model_choice == "CNN":
        preds = predict_dl(cnn_model, selected_example)
    else:
        preds = predict_dl(lstm_model, selected_example)

    st.subheader(f"Results for: {selected_example}")
    for label, val in preds.items():
        st.write(f"**{label}**: {'✅ Yes' if val == 1 else '❌ No'}")
