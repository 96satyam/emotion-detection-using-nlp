# Emotion Detection using NLP
### A robust and scalable machine learning pipeline for detecting emotions in text using Natural Language Processing (NLP) techniques. This project encompasses data preprocessing, model training, evaluation, and deployment through both Streamlit and Flask web applications.

## Features
#### Multi-Model Training: Implements Logistic Regression, Multinomial Naive Bayes, and Linear SVC classifiers.

#### Hyperparameter Tuning: Utilizes GridSearchCV for optimal model performance.

#### Comprehensive Evaluation: Assesses models using accuracy, precision, recall, and F1-score metrics.

#### Serialization: Saves trained models and preprocessors using dill for seamless deployment.

##### Dual Deployment: Offers user-friendly interfaces via both Streamlit and Flask applications.

## Models Implemented
#### Logistic Regression

##### Multinomial Naive Bayes

#### Linear Support Vector Classifier (SVC)

Each model undergoes hyperparameter tuning using GridSearchCV to identify the best-performing parameters based on F1-score.

## Evaluation Metrics
#### Accuracy
#### Precision
#### Recall
#### F1-Score

The model with the highest F1-score is selected as the best model and is saved for deployment.

## Deployment

### Streamlit Application

#### Provides an interactive web interface for users to input text and receive emotion predictions.

#### Displays the predicted emotion along with an appropriate emoji representation.

## Flask Application
#### Offers a web interface with a textarea for user input.
#### Upon submission, displays the detected emotion.

##  Installation
### Clone the repository:
git clone https://github.com/96satyam/emotion-detection-using-nlp.git
cd emotion-detection-using-nlp

### Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### Install dependencies:
pip install -r requirements.txt

##  Running the Applications
### Streamlit App
streamlit run app.py
### Flask App
python application.py

Access the applications via http://localhost:8501 for Streamlit and http://localhost:5000 for Flask.

 ## Sample Inputs
### Try the following sentences to test the emotion detection:
#### "I'm feeling fantastic today!"
#### "This is so frustrating and annoying."
#### "I am scared of the dark."
#### "I feel so sad and lonely."
#### "It's just another regular day."
#### "Wow! That was unexpected!"

## License
This project is licensed under the MIT License. See the LICENSE file for details.
