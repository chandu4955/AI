# Updated backtest.py
import argparse
import os
import pickle
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta

from utils.data_utils import get_historical_data, prepare_features
from utils.model_utils import load_model, save_backtest_results
from utils.backtest_utils import run_backtest, plot_backtest_results
from utils.trading_utils import TradingParameters, setup_mt5_connection

def main():
    parser = argparse.ArgumentParser(description='Backtest a trained trading model')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., BTCUSD)')
    parser.add_argument('--timeframe', type=str, default='M15', help='Timeframe (M1, M5, M15, H1, D1)')
    parser.add_argument('--days', type=int, default=30, help='Number of days of historical data to use')
    parser.add_argument('--lotsize', type=float, default=0.01, help='Lot size for trading')
    parser.add_argument('--spread', type=float, default=1.0, help='Spread in price units')
    parser.add_argument('--capital', type=float, default=10000.0, help='Initial capital for backtesting')
    parser.add_argument('--risk', type=float, default=0.02, help='Risk per trade (as a decimal, e.g., 0.02 for 2%)')
    parser.add_argument('--max_trades', type=int, default=5, help='Maximum number of open trades')
    parser.add_argument('--commission', type=float, default=0.0, help='Commission per trade')
    
    args = parser.parse_args()
    
    print(f"\nBacktesting model for {args.symbol} on {args.timeframe} timeframe")
    
    # Load the trained model
    model, scaler = load_model(args.symbol, args.timeframe)
    if model is None or scaler is None:
        print("\nFailed to load model. Exiting.")
        return
    
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
    
    # Run backtest
    backtest_results = run_backtest(model, df, params, scaler)
    
    # Save backtest results
    save_backtest_results(backtest_results, args.symbol, args.timeframe)
    
    # Plot backtest results
    plot_backtest_results(backtest_results, args.symbol, args.timeframe)
    
    # Shutdown MT5
    mt5.shutdown()
    print("\nBacktest completed")

if __name__ == "__main__":
    main()
