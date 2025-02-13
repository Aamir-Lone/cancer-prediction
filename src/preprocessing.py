import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    # Load the dataset from a CSV file
    df = pd.read_csv(file_path)
    return df




def handle_missing_values(df):
    """Handle missing values by dropping empty columns and imputing numeric and categorical columns."""
    # Drop columns with all missing values
    df = df.dropna(axis=1, how='all')

    # Identify numeric and categorical columns **after dropping empty columns**
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Impute numeric columns (mean)
    num_imputer = SimpleImputer(strategy='mean')
    df.loc[:, numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    # Impute categorical columns (most frequent)
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df.loc[:, categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    return df, categorical_cols  # Return updated categorical columns


def encode_categorical_columns(df, categorical_columns):
    """Encodes categorical columns using Label Encoding."""
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        if col in df.columns:  # Avoid KeyError
            df.loc[:, col] = label_encoder.fit_transform(df[col])
    
    return df


def scale_features(df, features):
    # Keep only existing features
    valid_features = [col for col in features if col in df.columns]
    
    if not valid_features:
        raise ValueError("No valid features found for scaling. Check column names.")

    scaler = StandardScaler()
    df[valid_features] = scaler.fit_transform(df[valid_features])
    
    return df

def preprocess_data(df):
    """Main preprocessing function that combines missing value handling, encoding, and scaling."""
    # Handle missing values
    df, categorical_cols = handle_missing_values(df)
    
    # Encode categorical columns
    df = encode_categorical_columns(df, categorical_cols)
    
    # Scale features (all columns except the target column)
    features = df.columns.tolist()
    df = scale_features(df, features)
    
    return df




def split_data(df, target_column):
    X = df.drop(target_column, axis=1)  # Features
    y = df[target_column]               # Target/Label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
