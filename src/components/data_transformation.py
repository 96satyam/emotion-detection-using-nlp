import sys
import os
import re
import string
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import neattext.functions as nfx
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    label_encoder_path = os.path.join("artifacts", "label_encoder.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def clean_text(self, text):
        try:
            text = text.lower()
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'<.*?>+', '', text)
            text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub(r'\n', '', text)
            text = re.sub(r'\w*\d\w*', '', text)
            return text
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, data_path: str):
        try:
            df = pd.read_csv(data_path)
            logging.info("Raw data loaded successfully.")

            # Normalize column names
            df.columns = df.columns.str.lower()
            logging.info(f"Normalized columns: {df.columns.tolist()}")

            if 'text' not in df.columns or 'emotion' not in df.columns:
                raise CustomException("Required columns ['text', 'emotion'] not found in dataset", sys)

            # Combine all cleaning steps
            df["clean_text"] = df["text"].astype(str)
            df["clean_text"] = df["clean_text"].apply(nfx.remove_userhandles)
            df["clean_text"] = df["clean_text"].apply(self.clean_text)
            df["clean_text"] = df["clean_text"].apply(nfx.remove_stopwords)

            X = df["clean_text"]
            y = df["emotion"]

            # Encode labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            logging.info("Labels encoded successfully.")

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            logging.info("Data split into train and test sets.")

            # TF-IDF Vectorization
            tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )

            X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
            X_test_tfidf = tfidf_vectorizer.transform(X_test)
            logging.info("TF-IDF vectorization completed.")

            # Save the TF-IDF vectorizer and label encoder
            save_object(self.config.preprocessor_obj_file_path, tfidf_vectorizer)
            save_object(self.config.label_encoder_path, label_encoder)
            logging.info("Preprocessing objects saved to artifacts.")

            return (
                X_train_tfidf,
                X_test_tfidf,
                y_train,
                y_test,
                self.config.preprocessor_obj_file_path,
                self.config.label_encoder_path
            )

        except Exception as e:
            raise CustomException(e, sys)
