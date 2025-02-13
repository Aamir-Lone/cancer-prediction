from src.preprocessing import load_data, handle_missing_values, encode_categorical_columns, scale_features, split_data
from src.model import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    evaluate_model,
    save_model
)

def main():
    # Load the dataset
    df = load_data('data/raw/cancer_dataset.csv')

    # Handle missing values
    df, categorical_columns = handle_missing_values(df)
    
    # Encode categorical columns (you'll need to specify which columns are categorical)
    categorical_columns = ['column1', 'column2']  # replace with actual categorical columns
    df = encode_categorical_columns(df, categorical_columns)

    # Scale the features (if needed)
    features = df.select_dtypes(include=['number']).columns.tolist()  # Select numeric features dynamically
    df = scale_features(df, features)  # replace with actual feature columns
    
    print("Available columns in DataFrame:", df.columns.tolist())
    print("Expected features:", features)

    # Map diagnosis to 1 (Malignant) and 0 (Benign)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # Convert to 1 and 0

    # Split the data into training and testing sets
    target_column = 'diagnosis'  # Replace with the correct target column in your dataset
    X_train, X_test, y_train, y_test = split_data(df, target_column)

    print("Data preprocessing complete.")
    print("Training and testing sets created.")

    # ************************************************************
    # Train and evaluate Logistic Regression
    logreg_model = train_logistic_regression(X_train, y_train)
    evaluate_model(logreg_model, X_test, y_test)
    save_model(logreg_model, 'logreg_model.pkl')

    # Train and evaluate Random Forest
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)
    save_model(rf_model, 'rf_model.pkl')



    # Train and evaluate XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test)
    save_model(xgb_model, 'xgb_model.pkl')
    # ************************************************************

if __name__ == "__main__":
    main()
