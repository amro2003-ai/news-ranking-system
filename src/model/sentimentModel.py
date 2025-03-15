import tensorflow as tf
import pickle
import os
import logging
from pydantic import BaseModel
from fastapi import HTTPException

import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(MAIN_DIR)

from config.configLoder import ConfigLoder

class Model:
    def __init__(self):
        """
        Initializes the Sentiment Analysis Model.
        Loads the TensorFlow model and tokenizer.
        """
        logging.info("Initializing Sentiment Analysis Model...")
        self.config = ConfigLoder().load_config()
        
        model_path = self.config['model']['sentiment_model']
        tokenizer_path = self.config['model']['tokenizer']

        # Check if model and tokenizer files exist
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(tokenizer_path):
            logging.error(f"Tokenizer file not found: {tokenizer_path}")
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        try:
            self.model = tf.keras.models.load_model(model_path)
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.exception("Failed to load model.")
            raise RuntimeError(f"Failed to load model: {e}")

        try:
            with open(tokenizer_path, "rb") as handle:
                self.tokenizer = pickle.load(handle)
                logging.info("Tokenizer loaded successfully.")
        except Exception as e:
            logging.exception("Failed to load tokenizer.")
            raise RuntimeError(f"Failed to load tokenizer: {e}")
        
        # Download required NLTK resources
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        logging.info("NLTK resources downloaded successfully.")
    
    def clean_text(self, text: str) -> str:
        """
        Preprocesses the input text by cleaning, tokenizing, removing stopwords, and lemmatizing.
        
        param text: The input text to be cleaned.
        task: Perform text normalization and preprocessing for sentiment analysis.
        return: A cleaned and preprocessed string.
        """
        logging.info("Cleaning input text...")
        text = text.lower()
        text = re.sub(r"\d+", "", text)  # Remove digits
        text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
        text = re.sub(r"\W", " ", text)  # Remove non-word characters
        text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
        text = re.sub(r"http\S+", "", text)  # Remove URLs

        tokens = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        filtered_tokens = [word for word in tokens if word not in stop_words]

        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        cleaned_text = " ".join(lemmatized_tokens)
        logging.info("Text cleaning completed.")
        return cleaned_text
    
    class SentimentRequest(BaseModel):
        text: str  # User input text for sentiment analysis

    def _predict_sentiment(self, request: SentimentRequest) -> dict:
        """
        Predicts the sentiment of the given text using the trained model.
        
        param request: A SentimentRequest object containing the input text.
        task: Process the input text, clean it, tokenize, pad sequences, and predict sentiment.
        return: A dictionary with text, sentiment label (positive/negative), and confidence score.
        """
        logging.info("Processing sentiment prediction request...")
        try:
            # Preprocess the input text
            cleaned_text = self.clean_text(request.text)
            seq = self.tokenizer.texts_to_sequences([cleaned_text])
            padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=40, truncating="post", padding="post")
            
            # Perform sentiment prediction
            prediction = self.model.predict(padded_seq)
            logging.info(f"Prediction score: {prediction[0][0]}")
            
            score_threshold = self.config['ranking']['score_threshold']
            sentiment = "positive" if prediction[0][0] > score_threshold else "negative"
            
            result = {
                "text": request.text,
                "sentiment": sentiment,
                "confidence": float(prediction[0][0]),
            }
            logging.info(f"Sentiment prediction result: {result}")
            return result
        
        except Exception as e:
            logging.exception("Error occurred during sentiment prediction.")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
