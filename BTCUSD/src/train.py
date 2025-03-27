# Updated train.py with fixed max_features parameter
import argparse
import os
import pickle
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from utils.data_utils import get_historical_data, prepare_features
from utils.model_utils import train_model, save_model, load_model, evaluate_model_performance, compare_models, save_backtest_results
from utils.backtest_utils import run_backtest, plot_backtest_results
from utils.trading_utils import TradingParameters, setup_mt5_connection

def main():
    parser = argparse.ArgumentParser(description='Train a trading model')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., BTCUSD)')
    parser.add_argument('--timeframe', type=str, default='M15', help='Timeframe (M1, M5, M15, H1, D1)')
    parser.add_argument('--days', type=int, default=60, help='Number of days of historical data to use')
    parser.add_argument('--lotsize', type=float, default=0.01, help='Lot size for trading')
    parser.add_argument('--spread', type=float, default=1.0, help='Spread in price units')
    parser.add_argument('--capital', type=float, default=10000.0, help='Initial capital for backtesting')
    parser.add_argument('--risk', type=float, default=0.02, help='Risk per trade (as a decimal, e.g., 0.02 for 2%)')
    parser.add_argument('--max_trades', type=int, default=5, help='Maximum number of open trades')
    parser.add_argument('--commission', type=float, default=0.0, help='Commission per trade')
    parser.add_argument('--force-save', action='store_true', help='Force save best model even if target not reached')
    parser.add_argument('--iterations', type=int, default=50, help='Maximum number of optimization iterations')
    parser.add_argument('--min-profit', action='store_true', help='Only save profitable models')
    parser.add_argument('--target-pf', type=float, default=2.0, help='Target profit factor to achieve')
    
    args = parser.parse_args()
    
    print(f"\nTraining model for {args.symbol} on {args.timeframe} timeframe")

    
    # Setup MT5 connection
    if not setup_mt5_connection():
        print("\nFailed to connect to MT5. Exiting.")
        return
    
    # Set up trading parameters
    params = TradingParameters(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lotsize=args.lotsize,
        spread=args.spread,
        capital=args.capital,
        risk_per_trade=args.risk,
        max_trades=args.max_trades,
        commission=args.commission
    )
    
    # Get historical data
    print(f"\nFetching {args.days} days of historical data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    df = get_historical_data(args.symbol, args.timeframe, start_date, end_date)
    if df is None or len(df) == 0:
        print("\nFailed to get historical data. Exiting.")
        mt5.shutdown()
        return
    
    print(f"\nGot {len(df)} candles of historical data")
    
    # Prepare features
    df = prepare_features(df)
    
    # Define feature columns (updated to match the simplified feature set)
    feature_columns = [
        'Price_Change', 'High_Low_Range', 'Body_Size', 'Upper_Wick', 
        'Lower_Wick', 'Price_Position', 'RSI_Signal'
    ]
    
    # Create target variable (next candle direction)
    df['Next_Direction'] = np.where(df['close'].shift(-1) > df['close'], 1, 
                                   np.where(df['close'].shift(-1) < df['close'], -1, 0))
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Split data into training and testing sets
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    
    X_train = df_train[feature_columns].values
    y_train = df_train['Next_Direction'].values
    
    # Initialize best model tracking
    best_model = None
    best_scaler = None
    best_metrics = None
    best_backtest = None
    best_profit_factor = 0
    
    # Run optimization loop
    print(f"\nStarting optimization with up to {args.iterations} iterations...")
    print(f"Will stop when profit factor >= {args.target_pf} is achieved")
    
    for iteration in range(args.iterations):
        print(f"\n--- Iteration {iteration+1}/{args.iterations} ---")
        
        # Vary hyperparameters for each iteration - using Python's random instead of numpy
        import random
        n_estimators = random.choice([50, 100, 150, 200, 250, 300])
        max_depth = random.choice([3, 5, 8, 10, 12, 15, None])
        min_samples_split = random.choice([2, 5, 10, 15, 20])
        min_samples_leaf = random.choice([1, 2, 4, 5, 8, 10])
        class_weight = random.choice(['balanced', 'balanced_subsample', None])
        
        # Randomly select max_features (using Python's random)
        max_features_choice = random.choice(['sqrt', 'log2', 0.7, 0.8, 0.9, 1.0])
        
        print(f"Training with: n_estimators={n_estimators}, max_depth={max_depth}, "
              f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
              f"class_weight={class_weight}, max_features={max_features_choice}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model with current hyperparameters
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            max_features=max_features_choice,
            random_state=42 + iteration  # Different seed for each iteration
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions on training data for evaluation
        y_pred = model.predict(X_train_scaled)
        
        # Calculate training metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_train, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_train, y_pred, average='weighted', zero_division=0)
        
        print(f"\nTraining metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        training_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Run backtest on test data
        backtest_results = run_backtest(model, df_test, params, scaler)
        
        # Get profit factor from backtest
        profit_factor = backtest_results['metrics']['profit_factor']
        print(f"\nProfit Factor: {profit_factor:.2f}")
        print(f"\n****************************************\n")
        
        # Update best model if current model has better profit factor
        if profit_factor > best_profit_factor:
            best_model = model
            best_scaler = scaler
            best_metrics = training_metrics
            best_backtest = backtest_results
            best_profit_factor = profit_factor
            print(f"\n****************************************\n")
            print(f"\nNew best model found! Profit Factor: {best_profit_factor:.2f}")
            print(f"\n****************************************\n")
    
            # Save the current best model
            save_model(best_model, best_scaler, best_metrics, args.symbol, args.timeframe)
            save_backtest_results(best_backtest, args.symbol, args.timeframe)
            print("\nIntermediate best model saved")
            
            # Plot backtest results for the current best model
            plot_backtest_results(best_backtest, args.symbol, args.timeframe)
        
        # Check if we've reached the target profit factor
        if profit_factor >= args.target_pf:
            print(f"\nTarget profit factor of {args.target_pf} achieved! Stopping optimization.")
            break
    
    # Final summary
    if best_model is not None:
        print("\n--- Optimization Summary ---")
        print(f"Best Profit Factor: {best_profit_factor:.2f}")
        print(f"Target Profit Factor: {args.target_pf:.2f}")
        
        if best_profit_factor >= args.target_pf:
            print("\nSUCCESS: Target profit factor achieved!")
        else:
            print("\nNOTE: Target profit factor not achieved, but best model was saved.")
            
        # Ensure the best model is saved
        save_model(best_model, best_scaler, best_metrics, args.symbol, args.timeframe)
        save_backtest_results(best_backtest, args.symbol, args.timeframe)
        print("\nBest model saved successfully")
        
        # Plot final backtest results
        plot_backtest_results(best_backtest, args.symbol, args.timeframe)
    else:
        print("\nNo suitable model found during optimization.")
    
    # Shutdown MT5
    mt5.shutdown()
    print("\nTraining completed")

if __name__ == "__main__":
    main()
