import argparse
import os
import sys
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_mt5_history, prepare_features
from utils.model_utils import load_model
from utils.backtest_utils import run_backtest, save_backtest_results
from utils.trading_utils import TradingParameters

def main():
    parser = argparse.ArgumentParser(description='Backtest a trained trading model')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., EURUSD)')
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
    print(f"Using {args.days} days of historical data")
    
    # Load the trained model
    model, scaler = load_model(args.symbol, args.timeframe)
    if model is None or scaler is None:
        print("\nFailed to load model. Exiting.")
        return
    
    # Load historical data
    df = load_mt5_history(args.symbol, args.timeframe, args.days)
    if df is None:
        print("\nFailed to load historical data. Exiting.")
        return
    
    print(f"\nLoaded {len(df)} candles of historical data")
    
    # Prepare features
    df_features = prepare_features(df, args.timeframe)
    if df_features is None:
        print("\nFailed to prepare features. Exiting.")
        return
    
    print(f"\nPrepared features for {len(df_features)} candles")
    
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
    
    # Run backtest using the same methodology as in train.py
    backtest_results = run_backtest(model, df_features, params, scaler)
    
    # Save backtest results
    results_file = save_backtest_results(backtest_results, args.symbol, args.timeframe, args.capital)
    print(f"\nBacktest results saved to {results_file}")
    
    print("\nBacktesting completed successfully")

if __name__ == "__main__":
    main()
