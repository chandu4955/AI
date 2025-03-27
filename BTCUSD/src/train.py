# Updated train.py
import argparse
import os
import pickle
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
    parser.add_argument('--force-save', action='store_true', help='Force save model even if worse than existing')
    
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
    
    # Define feature columns
    feature_columns = ['Price_Change', 'High_Low_Range', 'Body_Size', 'Upper_Wick', 
                      'Lower_Wick', 'Volume_Change', 'Volume_Ratio', 'Momentum', 
                      'Volatility', 'Price_Position', 'Trend', 'RSI_Signal']
    
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
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model, training_metrics = train_model(X_train_scaled, y_train)
    
    # Run backtest on test data
    backtest_results = run_backtest(model, df_test, params, scaler)
    
    # Evaluate model performance
    model_score = evaluate_model_performance(backtest_results)
    
    # Compare with existing model
    is_better = compare_models(model_score, args.symbol, args.timeframe)
    
    # Save model if it's better or if force-save is specified
    if is_better or args.force_save:
        save_model(model, scaler, training_metrics, args.symbol, args.timeframe)
        save_backtest_results(backtest_results, args.symbol, args.timeframe)
        print("\nModel saved successfully")
    else:
        print("\nNew model is not better than existing model. Not saving.")
    
    # Plot backtest results
    plot_backtest_results(backtest_results, args.symbol, args.timeframe)
    
    # Shutdown MT5
    mt5.shutdown()
    print("\nTraining completed")

if __name__ == "__main__":
    main()

