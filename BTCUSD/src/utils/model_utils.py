# Updated model_utils.py
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(X_train, y_train):
    """Train a machine learning model for trading signals."""
    print("\nTraining model...")
    
    # Initialize and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions on training data for evaluation
    y_pred = model.predict(X_train)
    
    # Calculate metrics
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_train, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_train, y_pred, average='weighted', zero_division=0)
    
    print(f"\nTraining metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return model, metrics

def save_model(model, scaler, metrics, symbol, timeframe, force_save=False):
    """Save the trained model, scaler, and metrics to disk."""
    # Create directory if it doesn't exist
    model_dir = os.path.join('models', symbol, timeframe)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metrics
    metrics_path = os.path.join(model_dir, 'training_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    print(f"\nModel, scaler, and metrics saved to {model_dir}")
    return model_dir

def load_model(symbol, timeframe):
    """Load a trained model and scaler from disk."""
    model_dir = os.path.join('models', symbol, timeframe)
    
    # Check if model exists
    model_path = os.path.join(model_dir, 'model.pkl')
    if not os.path.exists(model_path):
        print(f"\nModel not found at {model_path}")
        return None, None
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        print(f"\nScaler not found at {scaler_path}")
        return model, None
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"\nModel and scaler loaded from {model_dir}")
    return model, scaler

def predict_signal(model, scaler, candle_data, feature_columns):
    """
    Make a prediction using the trained model on a single candle.
    This function is used consistently across backtesting and live trading.
    
    Args:
        model: Trained ML model
        scaler: Fitted scaler for feature normalization
        candle_data: Single row of data (can be pandas Series or dict-like)
        feature_columns: List of feature column names
        
    Returns:
        prediction: 1 for buy, -1 for sell, 0 for hold
    """
    if model is None:
        return 0
    
    # Extract features from candle data
    if isinstance(candle_data, pd.Series):
        features = candle_data[feature_columns].values.reshape(1, -1)
    else:
        features = np.array([candle_data[col] for col in feature_columns]).reshape(1, -1)
    
    # Scale features if scaler is provided
    if scaler is not None:
        features = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    return prediction

def evaluate_model_performance(backtest_results):
    """
    Evaluate model performance based on backtest results.
    Returns a score that can be used to compare models.
    
    Args:
        backtest_results: Dictionary with backtest metrics
        
    Returns:
        score: Overall performance score (higher is better)
    """
    metrics = backtest_results['metrics']
    
    # Extract key metrics
    return_pct = metrics['return_pct']
    win_rate = metrics['win_rate'] * 100  # Convert to percentage
    profit_factor = metrics['profit_factor']
    max_drawdown = metrics['max_drawdown']
    total_trades = metrics['total_trades']
    
    # Calculate score components
    return_score = return_pct / 10  # Normalize returns
    win_rate_score = win_rate / 2  # Win rate contributes up to 50 points
    profit_factor_score = min(profit_factor * 10, 50)  # Cap at 50 points
    drawdown_penalty = max_drawdown * 100  # Penalty for drawdown
    trade_count_score = min(total_trades / 10, 10)  # Reward for more trades, up to 10 points
    
    # Calculate overall score
    score = return_score + win_rate_score + profit_factor_score - drawdown_penalty + trade_count_score
    
    print("\nModel Performance Evaluation:")
    print(f"Return Score: {return_score:.2f}")
    print(f"Win Rate Score: {win_rate_score:.2f}")
    print(f"Profit Factor Score: {profit_factor_score:.2f}")
    print(f"Drawdown Penalty: {drawdown_penalty:.2f}")
    print(f"Trade Count Score: {trade_count_score:.2f}")
    print(f"Overall Score: {score:.2f}")
    
    return score

def compare_models(new_score, symbol, timeframe):
    """
    Compare new model performance with existing model.
    Returns True if new model is better or no existing model.
    
    Args:
        new_score: Performance score of new model
        symbol: Trading symbol
        timeframe: Trading timeframe
        
    Returns:
        is_better: Boolean indicating if new model is better
    """
    metrics_path = os.path.join('models', symbol, timeframe, 'backtest_metrics.pkl')
    
    # If no existing metrics, new model is better by default
    if not os.path.exists(metrics_path):
        return True
    
    # Load existing metrics
    with open(metrics_path, 'rb') as f:
        existing_metrics = pickle.load(f)
    
    existing_score = evaluate_model_performance(existing_metrics)
    
    print(f"\nExisting model score: {existing_score:.2f}")
    print(f"New model score: {new_score:.2f}")
    
    # Return True if new model is better
    return new_score > existing_score

def save_backtest_results(backtest_results, symbol, timeframe):
    """Save backtest results to disk."""
    model_dir = os.path.join('models', symbol, timeframe)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save backtest metrics
    metrics_path = os.path.join(model_dir, 'backtest_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(backtest_results, f)
    
    print(f"\nBacktest results saved to {metrics_path}")
