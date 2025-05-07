import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    source_data_path: str = os.path.join('notebook', 'data', 'emotion_dataset_raw.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion process started.")
        try:
            # Data Loading
            df = pd.read_csv(self.ingestion_config.source_data_path)
            logging.info("Dataset loaded into DataFrame.")

            # Basic cleanup
            df.columns = df.columns.str.lower().str.strip()
            df.drop_duplicates(inplace=True)
            df.dropna(subset=['text', 'emotion'], inplace=True)

            # Log shape and class balance
            logging.info(f"Dataset shape after cleaning: {df.shape}")
            logging.info(f"Emotion class distribution:\n{df['emotion'].value_counts()}")

            # artifacts directory 
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw cleaned data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            # Stratified split to preserve class balance
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df['emotion']
            )
            logging.info("Train-test split completed with stratification.")

            # Save train/test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Train and test datasets saved successfully.")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

# Pipeline 
if __name__ == '__main__':
    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer

    ingestion = DataIngestion()
    train_path, test_path, raw_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    X_train, X_test, y_train, y_test, _, _ = transformation.initiate_data_transformation(raw_path)

    trainer = ModelTrainer()
    report = trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)

    print("Model evaluation report:")
    print(report)
