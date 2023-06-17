# Cybersapient_Assignment03
# Sentiment Analysis with Streamlit
This is a Streamlit application that performs sentiment analysis on movie reviews. It uses the IMDB movie review dataset to train a logistic regression model for sentiment classification. The application also provides a summary of the review using a summarization pipeline.

# Dependencies
The following dependencies are required to run the code: <br>

Streamlit: Streamlit is an open-source Python library used for building web applications. <br>
Pandas: Pandas is a data manipulation and analysis library. <br>
re: The re module provides support for regular expressions in Python. <br>
NLTK: NLTK (Natural Language Toolkit) is a library used for natural language processing tasks. <br>
scikit-learn: Scikit-learn is a machine learning library that provides various algorithms and tools for data analysis. <br>
Transformers: Transformers is a library developed by Hugging Face that provides state-of-the-art natural language processing models and tools. <br> 

# Model Training
The sentiment analysis model is trained using logistic regression on the IMDB movie review dataset. The training process is handled by the following steps in the code: <br>

1. Load the IMDB movie review dataset from IMDB_Dataset.csv. <br>
2. Preprocess the text data by converting it to lowercase, removing special characters and digits, tokenizing the text, and removing stopwords. <br>
3. Initialize a TF-IDF vectorizer. <br>
4. Fit and transform the dataset using the vectorizer. <br>
5. Initialize a logistic regression model. <br>
6. Train the model using the transformed features and the sentiment labels from the dataset. <br>

# Summary Generation
The code also includes a summarization pipeline using the Hugging Face transformers library. The summarizer generates a summary of the review using the following steps: <br>

1.Preprocess the input review using the same preprocessing steps as in the model training. <br>
2. Use the summarization pipeline to generate a summary of the preprocessed review. <br>
3. Extract the summary text from the pipeline output. <br>

# Usage
1. Download NLTK resources: <br>
   nltk.download('punkt') <br>
   nltk.download('stopwords') <br>
2. Run the Streamlit app: <br>
   streamlit run app.py <br>
3. Open your web browser and navigate to the provided URL.<br>
4. Enter a movie review in the text area.<br>
5. Click on the "Predict Sentiment" button to predict the sentiment of the review.<br>
6. The predicted sentiment and a summary of the review will be displayed on the web page.<br>

