# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from preprocessing import load_data, clean_data

def basic_eda(df: pd.DataFrame):
    """
    Display basic dataset info.
    """
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nFirst few rows:\n", df.head())

def plot_churn_distribution(df: pd.DataFrame):
    """
    Plot distribution of the target variable (churn).
    """
    plt.figure(figsize=(6,4))
    sns.countplot(x="exited", data=df, palette="viridis")
    plt.title("Churn Distribution")
    plt.xlabel("Exited (0: Stay, 1: Churn)")
    plt.ylabel("Count")
    plt.show()

def plot_usage_vs_churn(df: pd.DataFrame):
    """
    Plot the number of products (usage) against churn.
    """
    plt.figure(figsize=(8,5))
    sns.boxplot(x="exited", y="numofproducts", data=df, palette="Set2")
    plt.title("Products Used vs. Churn")
    plt.xlabel("Exited")
    plt.ylabel("Number of Products")
    plt.show()

def plot_credit_score_distribution(df: pd.DataFrame):
    """
    Plot credit score distribution by churn status.
    """
    plt.figure(figsize=(8,5))
    sns.histplot(data=df, x="creditscore", hue="exited", kde=True, palette="coolwarm")
    plt.title("Credit Score Distribution by Churn")
    plt.xlabel("Credit Score")
    plt.ylabel("Count")
    plt.show()

def plot_age_vs_churn(df: pd.DataFrame):
    """
    Plot age distribution for churn vs. non-churn.
    """
    plt.figure(figsize=(8,5))
    sns.boxplot(x="exited", y="age", data=df, palette="pastel")
    plt.title("Age Distribution by Churn")
    plt.xlabel("Exited")
    plt.ylabel("Age")
    plt.show()

def visualize_data(file_path: str):
    """
    Load data and generate visualizations.
    """
    df = load_data(file_path)
    df = clean_data(df)
    
    basic_eda(df)
    plot_churn_distribution(df)
    plot_usage_vs_churn(df)
    plot_credit_score_distribution(df)
    plot_age_vs_churn(df)

if __name__ == "__main__":
    file_path = "..\Churn_Modelling.csv"
    visualize_data(file_path)
