import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            # Ensure features is a DataFrame with a 'text' column
            if isinstance(features, str):
                features = pd.DataFrame({"text": [features]})
            elif isinstance(features, dict):
                features = pd.DataFrame(features)

            # Transform and predict
            transformed_features = preprocessor.transform(features)
            preds = model.predict(transformed_features)

            return preds

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
