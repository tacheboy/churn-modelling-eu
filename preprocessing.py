# preprocessing.py

import pandas as pd
from sklearn.impute import SimpleImputer

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the bank churn data from a CSV file.
    """
    df = pd.read_csv(file_path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the data:
    - Drop columns that are not useful for modeling.
    - Impute missing values.
    - Rename columns (optional).
    """
    # Drop columns that are not useful. For example, customerid and surname may not help prediction.
    df = df.drop(columns=["customerid", "surname"], errors="ignore")

    # Convert column names to a standardized format (optional)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Impute missing values.
    # For numerical columns, we'll use the median; for categorical columns, we'll use the most frequent value.
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df

def preprocess_data(file_path: str):
    """
    Load and clean the data. Return X (features) and y (target).
    """
    df = load_data(file_path)
    df = clean_data(df)
    
    # Define target and features
    # Assume "exited" is our target variable (1: churn, 0: stay)
    y = df["exited"]
    X = df.drop(columns=["exited"], errors="ignore")
    return X, y

if __name__ == "__main__":
    # For testing purposes.
    file_path = "../Churn_Modelling.csv"
    X, y = preprocess_data(file_path)
    print("Features shape:", X.shape)
    print("Target distribution:\n", y.value_counts())
