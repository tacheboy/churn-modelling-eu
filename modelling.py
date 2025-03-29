# modeling.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from preprocessing import preprocess_data
from visualization import visualize_data

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plot a confusion matrix.
    """
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def create_preprocessor(X):
    """
    Create a preprocessing pipeline: scale numeric and one-hot encode categorical features.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    return preprocessor

def feature_importance_analysis(pipeline, X, model_name):
    """
    Analyze key features based on the model's feature importance or coefficients.
    """
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']

    # Construct full feature names from the ColumnTransformer
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'cat':
            try:
                new_names = list(transformer.get_feature_names_out(columns))
            except Exception:
                new_names = list(columns)
            feature_names.extend(new_names)
        else:
            feature_names.extend(list(columns))

    if model_name == "Random Forest":
        importance = classifier.feature_importances_
        sorted_idx = np.argsort(importance)[-10:]
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel("Feature Importance")
        plt.title("Top 10 Features - Random Forest")
        plt.show()

    elif model_name == "Logistic Regression":
        coef = classifier.coef_[0]
        sorted_idx = np.argsort(np.abs(coef))[-10:]
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(sorted_idx)), coef[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel("Coefficient Weight")
        plt.title("Top 10 Features - Logistic Regression")
        plt.show()

def train_and_evaluate(X, y):
    """
    Train models and evaluate performance, then output insights on key features.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create preprocessor and pipeline for two models
    preprocessor = create_preprocessor(X_train)

    pipelines = {
        "Logistic Regression": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ]),
        "Random Forest": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
        ])
    }

    for name, pipeline in pipelines.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, classes=["Stay", "Churn"], title=f"{name} Confusion Matrix")

        # Feature importance analysis to determine key features driving churn
        feature_importance_analysis(pipeline, X_train, name)

def main():
    """
    Main function:
      1. Visualize data
      2. Preprocess data
      3. Train and evaluate models
    """
    file_path = "..\Churn_Modelling.csv"

    # Visualize the dataset
    visualize_data(file_path)

    # Preprocess the data to get features and target
    X, y = preprocess_data(file_path)

    # Train models and evaluate performance + insights
    train_and_evaluate(X, y)

if __name__ == "__main__":
    main()
