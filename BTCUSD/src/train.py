import argparse
import os
import sys
import json
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_mt5_history, prepare_features
from utils.model_utils import train_model, save_model
from utils.backtest_utils import run_backtest, save_backtest_results
from utils.trading_utils import TradingParameters

def load_backtest_metrics(symbol, timeframe):
    """Load backtest metrics for an existing model."""
    metrics_path = os.path.join('models', f'{symbol}_{timeframe}_backtest_metrics.json')
    if not os.path.exists(metrics_path):
        return None
    
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        print(f"Error loading backtest metrics: {str(e)}")
        return None

def save_backtest_metrics(metrics, symbol, timeframe):
    """Save backtest metrics to a file."""
    metrics_path = os.path.join('models', f'{symbol}_{timeframe}_backtest_metrics.json')
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Backtest metrics saved to {metrics_path}")
    except Exception as e:
        print(f"Error saving backtest metrics: {str(e)}")

def compare_backtest_results(new_metrics, existing_metrics=None):
    """Compare new backtest metrics with existing backtest metrics."""
    if existing_metrics is None:
        return True  # No existing metrics to compare with
    
    # Define weights for different metrics
    weights = {
        'return_pct': 0.4,
        'win_rate': 0.2,
        'profit_factor': 0.3,
        'max_drawdown': -0.1  # Negative weight because lower is better
    }
    
    # Calculate weighted score for each model
    new_score = (
        new_metrics['return_pct'] * weights['return_pct'] + 
        new_metrics['win_rate'] * 100 * weights['win_rate'] + 
        new_metrics['profit_factor'] * weights['profit_factor'] + 
        new_metrics['max_drawdown'] * weights['max_drawdown']
    )
    
    existing_score = (
        existing_metrics['return_pct'] * weights['return_pct'] + 
        existing_metrics['win_rate'] * 100 * weights['win_rate'] + 
        existing_metrics['profit_factor'] * weights['profit_factor'] + 
        existing_metrics['max_drawdown'] * weights['max_drawdown']
    )
    
    # Return True if new model is better
    return new_score > existing_score

def main():
    parser = argparse.ArgumentParser(description='Train a trading model')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., EURUSD)')
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
    print(f"Using {args.days} days of historical data")
    
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
    
    # Train model
    model, valid_indices, scaler, train_metrics = train_model(df_features)
    
    # Run backtest on the trained model
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
    
    backtest_results = run_backtest(model, df_features, params, scaler, valid_indices)
    
    # Save backtest results to file for reference
    results_file = save_backtest_results(backtest_results, args.symbol, args.timeframe, args.capital)
    
    # Check existing backtest metrics
    existing_backtest_metrics = load_backtest_metrics(args.symbol, args.timeframe)
    
    if existing_backtest_metrics:
        print("\nExisting model backtest metrics:")
        print(f"Return: {existing_backtest_metrics['return_pct']:.2f}%")
        print(f"Win Rate: {existing_backtest_metrics['win_rate']:.2%}")
        print(f"Profit Factor: {existing_backtest_metrics['profit_factor']:.2f}")
        print(f"Max Drawdown: {existing_backtest_metrics['max_drawdown']:.2f}%")
    
    # Compare new backtest results with existing
    new_backtest_metrics = backtest_results['metrics']
    should_save = args.force_save or compare_backtest_results(new_backtest_metrics, existing_backtest_metrics)
    
    if should_save:
        print("\nNew model performs better in backtesting. Saving model...")
        # Save model, scaler, and backtest metrics
        save_model(model, scaler, train_metrics, args.symbol, args.timeframe, force_save=True)
        save_backtest_metrics(new_backtest_metrics, args.symbol, args.timeframe)
    else:
        print("\nExisting model performs better in backtesting. New model will not be saved.")
    
    print("\nTraining and backtesting completed successfully")

if __name__ == "__main__":
    main()
