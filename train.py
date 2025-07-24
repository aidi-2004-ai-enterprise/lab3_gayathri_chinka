"""
Train an XGBoost model on the penguins dataset with proper preprocessing and evaluation.

This module loads the penguins dataset from Seaborn, applies one-hot encoding to categorical
features, label encoding to the target variable, trains an XGBoost classifier with parameters
to prevent overfitting, evaluates the model, and saves it for later use.
"""

import os
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, accuracy_score
from typing import Tuple
import joblib


def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Load the penguins dataset and preprocess it for training.
    
    Returns:
        Tuple[pd.DataFrame, pd.Series, LabelEncoder]: Processed features, encoded target, and label encoder
    """
    print("Loading penguins dataset...")
    
    # Load the penguins dataset
    penguins = sns.load_dataset('penguins')
    
    # Drop rows with missing values
    penguins = penguins.dropna()
    
    print(f"Dataset shape after removing missing values: {penguins.shape}")
    print(f"Dataset columns: {penguins.columns.tolist()}")
    
    # Separate features and target
    X = penguins.drop('species', axis=1)
    y = penguins['species']
    
    # Apply one-hot encoding to categorical features (sex, island)
    X_encoded = pd.get_dummies(X, columns=['sex', 'island'], prefix=['sex', 'island'])
    
    print(f"Features after one-hot encoding: {X_encoded.columns.tolist()}")
    
    # Apply label encoding to target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Target classes: {label_encoder.classes_}")
    
    return X_encoded, y_encoded, label_encoder


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier with parameters to prevent overfitting.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        xgb.XGBClassifier: Trained XGBoost model
    """
    print("Training XGBoost model...")
    
    # Initialize XGBoost classifier with parameters to prevent overfitting
    model = xgb.XGBClassifier(
        max_depth=3,              # Limit tree depth to prevent overfitting
        n_estimators=100,         # Number of boosting rounds
        learning_rate=0.1,        # Step size shrinkage
        random_state=42,          # For reproducibility
        eval_metric='mlogloss'    # Evaluation metric for multiclass
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("Model training completed.")
    return model


def evaluate_model(model: xgb.XGBClassifier, X_train: pd.DataFrame, y_train: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series, label_encoder: LabelEncoder) -> None:
    """
    Evaluate the trained model on both training and test sets.
    
    Args:
        model (xgb.XGBClassifier): Trained model
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        label_encoder (LabelEncoder): Label encoder for target classes
    """
    print("\nEvaluating model performance...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for training set
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Calculate metrics for test set
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Set Performance:")
    print(f"  Accuracy: {train_accuracy:.4f}")
    print(f"  F1-Score (weighted): {train_f1:.4f}")
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  F1-Score (weighted): {test_f1:.4f}")
    
    # Detailed classification report for test set
    print(f"\nDetailed Classification Report (Test Set):")
    target_names = label_encoder.classes_
    print(classification_report(y_test, y_test_pred, target_names=target_names))


def save_model_and_encoder(model: xgb.XGBClassifier, label_encoder: LabelEncoder, 
                          feature_columns: list) -> None:
    """
    Save the trained model, label encoder, and feature columns to files.
    
    Args:
        model (xgb.XGBClassifier): Trained model
        label_encoder (LabelEncoder): Label encoder for target classes
        feature_columns (list): List of feature column names after preprocessing
    """
    # Create app/data directory if it doesn't exist
    os.makedirs('app/data', exist_ok=True)
    
    # Save the model
    model.save_model('app/data/model.json')
    print("Model saved to app/data/model.json")
    
    # Save label encoder and feature columns for consistent preprocessing in API
    joblib.dump(label_encoder, 'app/data/label_encoder.pkl')
    joblib.dump(feature_columns, 'app/data/feature_columns.pkl')
    print("Label encoder and feature columns saved to app/data/")


def main() -> None:
    """
    Main function to execute the entire training pipeline.
    """
    print("Starting penguins classification training pipeline...")
    
    # Load and preprocess data
    X, y, label_encoder = load_and_preprocess_data()
    
    # Split data into training and test sets (80/20 split, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # Stratified split to handle class imbalance
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_train, y_train, X_test, y_test, label_encoder)
    
    # Save model and preprocessing artifacts
    save_model_and_encoder(model, label_encoder, X.columns.tolist())
    
    print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()



