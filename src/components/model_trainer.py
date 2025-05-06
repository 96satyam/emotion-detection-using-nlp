import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Training and evaluating models...")

            models = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'MultinomialNB': MultinomialNB(),
            'LinearSVC': LinearSVC()
        }
        

            params = {
            'LogisticRegression': {
            'C': [0.1, 1, 10]
        },
            'MultinomialNB': {
            'alpha': [0.1, 1.0, 10.0]
        },
            'LinearSVC': {
            'C': [0.1, 1, 10]
        }
        }

            model_report = evaluate_models(X_train, X_test, y_train, y_test, models, params)

            # Save best model
            best_model_name = max(model_report, key=lambda x: model_report[x]['f1_score'])
            best_model = models[best_model_name].set_params(**model_report[best_model_name]['best_params'])
            best_model.fit(X_train, y_train)

            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info(f"Best model saved: {best_model_name} with F1 Score: {model_report[best_model_name]['f1_score']}")

            return model_report

        except Exception as e:
            raise CustomException(e, sys)
