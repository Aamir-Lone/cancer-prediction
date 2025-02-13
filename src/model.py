# src/model.py

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Logistic Regression Model
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Random Forest Model
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# # Support Vector Machine (SVM) Model
def train_svm(X_train, y_train):
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    return model

# Gradient Boosting Model (XGBoost)
def train_xgboost(X_train, y_train):
    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate Model Performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    return accuracy

# Save model to a file
def save_model(model, model_name='model.pkl'):
    joblib.dump(model, model_name)
    print(f'Model saved to {model_name}')

# Load model from a file
def load_model(model_name='model.pkl'):
    model = joblib.load(model_name)
    return model
