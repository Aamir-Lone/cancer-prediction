# Cancer Prediction

This project involves predicting cancer diagnosis using machine learning algorithms. It utilizes the **Cancer dataset** to compare different models and predict whether a tumor is malignant or benign.

## Project Overview

1. **Data Preprocessing**: 
   - The dataset (`cancer_dataset.csv`) is loaded and processed using various techniques:
     - Handling missing values using imputation (mean for numeric columns and most frequent for categorical columns).
     - Encoding categorical columns using Label Encoding.
     - Scaling numeric features using StandardScaler.
   
2. **Model Training**:
   - Various machine learning algorithms are used to train the model on the processed dataset:
     - **Logistic Regression**
     - **Random Forest**
     - **XGBoost**
   - The models are trained on a training set and evaluated on a separate testing set to compare their performance.

3. **Evaluation**:
   - After training, the models are evaluated using accuracy, and their performances are compared. The results are printed to the console.
   - The models with the highest accuracy are saved using **Pickle** for future use.

4. **Model Saving**:
   - The trained models are saved as `.pkl` files (`logreg_model.pkl`, `rf_model.pkl`, `xgb_model.pkl`) for future predictions or deployment.

5. **Final Predictions**:
   - Using the saved models, predictions are made on the test data, and their accuracy is printed.

## Technologies Used

- **Python**: The primary programming language.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For machine learning algorithms, preprocessing, and model evaluation.
- **XGBoost**: For training the XGBoost model.
- **Pickle**: For saving and loading trained models.

## Setup

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt


Place the cancer_dataset.csv in the data/raw folder.

Run the main script:

python main.py
This will perform data preprocessing, train the models, and evaluate their performance.

Results
The models are trained and evaluated on the dataset with the following accuracy:

Logistic Regression: 98.2%
Random Forest: 97.7%
XGBoost: 97.1%
The trained models are saved to .pkl files and can be loaded for future predictions.