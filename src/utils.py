import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f" Object saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            logging.info(f"Object loaded successfully from: {file_path}")
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, X_test, y_train, y_test, models, params):
    try:
        report = {}

        for model_name in models:
            logging.info(f"🔍 Tuning hyperparameters for: {model_name}")
            model = models[model_name]
            param_grid = params.get(model_name, {})

            gs = GridSearchCV(model, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_

            y_test_pred = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_test_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted')

            report[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'best_params': gs.best_params_
            }

            logging.info(f" {model_name} evaluation complete with F1 Score: {f1:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
