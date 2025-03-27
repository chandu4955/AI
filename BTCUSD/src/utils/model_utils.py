import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import os

class ModelWrapper:
    def __init__(self, model):
        self.model = model
        
    def predict(self, X):
        pred = self.model.predict(X)
        # Map predictions back to [-1, 0, 1]
        return np.array([{0: -1, 1: 0, 2: 1}[p] for p in pred])
        
    @property
    def feature_importances_(self):
        return self.model.feature_importances_
        
    @property
    def n_estimators(self):
        return self.model.n_estimators
        
    @property
    def max_depth(self):
        return self.model.max_depth

def predict_signal(model, scaler, candle_data, feature_columns):
    """
    Make a prediction using a trained model on a single completed candle.
    This function is used consistently across training, backtesting, and live trading.
    
    Args:
        model: Trained ML model
        scaler: Fitted scaler for feature normalization
        candle_data: DataFrame row or Series containing a single candle's data
        feature_columns: List of feature column names used by the model
        
    Returns:
        Prediction: 1 (buy), -1 (sell), or 0 (hold)
    """
    # Extract features
    features = candle_data[feature_columns].values.reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    
    return prediction


def train_model(df):
    """Train a model on the prepared features."""
    features = ['Price_Change', 'High_Low_Range', 'Body_Size', 'Upper_Wick', 
                'Lower_Wick', 'Volume_Change', 'Volume_Ratio', 'Momentum', 
                'Volatility', 'Price_Position', 'Trend', 'RSI_Signal', 'target']
    df_clean = df[features].dropna()
    
    feature_columns = [col for col in features if col != 'target']
    X = df_clean[feature_columns]
    y = df_clean['target']
    
    # Add feature scaling for better XGBoost performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Map target values from [-1, 0, 1] to [0, 1, 2]
    y_mapped = y.map({-1: 0, 0: 1, 1: 2})
    
    print(f"\nTraining model with {len(X)} samples")
    print("Feature statistics:")
    print(X.describe().round(4))
    
    # XGBoost model configuration for swing trading
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,            # Reduced to prevent overfitting
        learning_rate=0.03,     # Increased for more decisive predictions
        subsample=0.9,          # Increased for better trend capture
        colsample_bytree=0.9,   # Increased for feature utilization
        min_child_weight=5,     # Increased for more stable predictions
        gamma=0.2,              # Increased to reduce false signals
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_scaled, y_mapped)
    
    wrapped_model = ModelWrapper(model)
    return wrapped_model, df_clean.index, scaler

def save_model(model, scaler, results, symbol, timeframe):
    """Save model, scaler and results to disk."""
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(model, f'models/{symbol}_{timeframe}_model.joblib')
    joblib.dump(scaler, f'models/{symbol}_{timeframe}_scaler.joblib')
    joblib.dump(results, f'models/{symbol}_{timeframe}_results.pkl')
    
    print(f"\nModel saved for {symbol} {timeframe}")
    
def load_model(symbol, timeframe):
    """Load model and scaler from disk."""
    model_path = f'models/{symbol}_{timeframe}_model.joblib'
    scaler_path = f'models/{symbol}_{timeframe}_scaler.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"\nError: Model files not found in {os.path.abspath('models')} directory")
        print(f"Looking for: {model_path} and {scaler_path}")
        return None, None
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("\nModel loaded successfully")
        return model, scaler
    except Exception as e:
        print(f"\nError loading model: {str(e)}")
        return None, None
