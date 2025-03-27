import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Train a model and evaluate its performance.
    
    Args:
        model: Model to train
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        
    Returns:
        Trained model and performance metrics
    """
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    train_precision = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
    test_precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    
    train_recall = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
    
    train_f1 = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    
    # Calculate class distribution
    class_distribution = np.bincount(y_train) / len(y_train)
    
    # Calculate feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    else:
        feature_importance = None
    
    # Return metrics
    metrics = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'confusion_matrix': cm,
        'class_distribution': class_distribution,
        'feature_importance': feature_importance
    }
    
    return model, metrics

def save_model(model, scaler, metrics, symbol, timeframe):
    """
    Save a trained model and its metrics.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        metrics: Model performance metrics
        symbol: Trading symbol
        timeframe: Timeframe
    """
    # Create directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = f'models/{symbol}_{timeframe}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    scaler_path = f'models/{symbol}_{timeframe}_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metrics
    metrics_path = f'models/{symbol}_{timeframe}_metrics.pkl'
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    print(f"\nModel saved to {model_path}")

def load_model(symbol, timeframe):
    """
    Load a trained model and its metrics.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        
    Returns:
        Trained model, scaler, and metrics
    """
    # Load model
    model_path = f'models/{symbol}_{timeframe}_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    scaler_path = f'models/{symbol}_{timeframe}_scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load metrics
    metrics_path = f'models/{symbol}_{timeframe}_metrics.pkl'
    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    
    return model, scaler, metrics

def predict_signal(model, scaler, candle_data, feature_columns=None):
    """
    Predict trading signal for a candle.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        candle_data: Candle data
        feature_columns: Feature columns to use
        
    Returns:
        Predicted signal (-1, 0, or 1)
    """
    # If feature_columns is not provided, use the simplified set
    if feature_columns is None:
        feature_columns = [
            'Price_Change', 'High_Low_Range', 'Body_Size', 'Upper_Wick', 
            'Lower_Wick', 'Price_Position', 'RSI_Signal'
        ]
    
    # Extract features
    features = candle_data[feature_columns].values.reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    
    return prediction

def evaluate_model_performance(metrics):
    """
    Evaluate model performance based on metrics.
    
    Args:
        metrics: Model performance metrics
        
    Returns:
        Overall score
    """
    # Calculate overall score
    accuracy_weight = 0.3
    precision_weight = 0.3
    recall_weight = 0.2
    f1_weight = 0.2
    
    overall_score = (
        accuracy_weight * metrics['test_accuracy'] +
        precision_weight * metrics['test_precision'] +
        recall_weight * metrics['test_recall'] +
        f1_weight * metrics['test_f1']
    )
    
    return overall_score

def compare_models(model1_metrics, model2_metrics):
    """
    Compare two models based on their metrics.
    
    Args:
        model1_metrics: Metrics for model 1
        model2_metrics: Metrics for model 2
        
    Returns:
        1 if model 1 is better, 2 if model 2 is better, 0 if they are equal
    """
    # Calculate overall scores
    score1 = evaluate_model_performance(model1_metrics)
    score2 = evaluate_model_performance(model2_metrics)
    
    # Compare scores
    if score1 > score2:
        return 1
    elif score2 > score1:
        return 2
    else:
        return 0

def save_backtest_results(backtest_results, symbol, timeframe):
    """
    Save backtest results.
    
    Args:
        backtest_results: Backtest results
        symbol: Trading symbol
        timeframe: Timeframe
    """
    # Create directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save backtest results
    results_path = f'results/{symbol}_{timeframe}_backtest.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(backtest_results, f)
    
    print(f"\nBacktest results saved to {results_path}")
