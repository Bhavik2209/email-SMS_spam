import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained classifier and TF-IDF vectorizer
with open(r"C:\Users\SVI\Desktop\email-classifier\model.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

with open(r"C:\Users\SVI\Desktop\email-classifier\vector.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

def predict_spam(text):
    # Vectorize the input text using the tfidf vectorizer
    text_vectorized = tfidf_vectorizer.transform([text])
    # Predict whether the text is spam or not
    prediction = classifier.predict(text_vectorized)
    return prediction[0]

def main():
    st.title("Email/SMS Spam Classifier")
    st.write("Enter the text to check if it's spam or not:")

    # Get user input
    user_input = st.text_area("Enter your email/SMS text here:")
    if st.button("Check"):
        # Make prediction
        prediction = predict_spam(user_input)
        if prediction == 1:
            st.error("This text is classified as spam.")
        else:
            st.success("This text is not spam.")

if __name__ == "__main__":
    main()
