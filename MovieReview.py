import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline

# Download NLTK resources
#nltk.download('punkt')
#nltk.download('stopwords')

# Load the dataset
dataset = pd.read_csv('/Users/deepesh/Desktop/Wasim_MRS/IMDB_Dataset.csv')

# Preprocess function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Join tokens back to text
    text = ' '.join(tokens)

    return text

# Apply preprocessing to the 'review' column
dataset['clean_review'] = dataset['review'].apply(preprocess_text)

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the dataset
features = vectorizer.fit_transform(dataset['clean_review'])

# Initialize the sentiment analysis model
model = LogisticRegression()

# Train the model
model.fit(features, dataset['sentiment'])

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Define the Streamlit app
def main():
    st.title("Sentiment Analysis with Streamlit")
    st.subheader("Enter a movie review to predict its sentiment:")

    # User input for the movie review
    review = st.text_area("Review", value='', height=200)

    # Button to predict sentiment
    if st.button("Predict Sentiment"):
        predicted_sentiment, review_summary = predict_sentiment(review)
        st.write("Predicted Sentiment:", predicted_sentiment)
        st.write("Review Summary:", review_summary)

# Function to preprocess the review, generate summary, and predict sentiment
def predict_sentiment(review):
    # Preprocess the input review
    clean_review = preprocess_text(review)

    # Generate summary
    summary = summarizer(clean_review, max_length=150, min_length=30, do_sample=False)[0]['summary_text']

    # Transform the preprocessed review using the vectorizer
    review_features = vectorizer.transform([summary])

    # Predict sentiment label
    sentiment = model.predict(review_features)[0]

    return sentiment, summary

# Run the Streamlit app
if __name__ == '__main__':
    main()