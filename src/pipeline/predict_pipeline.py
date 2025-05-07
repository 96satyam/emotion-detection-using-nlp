import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.label_encoder_path = os.path.join('artifacts', 'label_encoder.pkl')

    def predict(self, features):
        try:
            logging.info(" Loading model and preprocessor...")
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            label_encoder = load_object(self.label_encoder_path)

            logging.info(" Model and vectorizer loaded successfully.")

            
            if isinstance(features, str):
                features = pd.DataFrame({"text": [features]})
            elif isinstance(features, dict):
                features = pd.DataFrame(features)

            logging.info(f" Raw Input: {features.head().to_dict()}")

            # Preprocess text and predict
            transformed_features = preprocessor.transform(features["text"])
            predictions = model.predict(transformed_features)

            # Decode label 
            if label_encoder:
                decoded_preds = label_encoder.inverse_transform(predictions)
                logging.info(f" Final Prediction: {decoded_preds[0]}")
                return decoded_preds
            else:
                logging.warning(" No label encoder found. Returning raw prediction.")
                return predictions

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, text: str):
        self.text = text

    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame({"text": [self.text]})
        except Exception as e:
            raise CustomException(e, sys)
