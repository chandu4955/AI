import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .data_utils import get_minutes_per_candle

def calculate_performance_metrics(trades):
    """Calculate performance metrics from a list of trades."""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_trade_duration': timedelta(0)
        }
    
    wins = [t for t in trades if t['profit'] and t['profit'] > 0]
    losses = [t for t in trades if t['profit'] and t['profit'] <= 0]
    
    # Calculate win rate
    total_trades = len([t for t in trades if t['profit'] is not None])
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    
    # Calculate average win/loss
    avg_win = sum(t['profit'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['profit'] for t in losses) / len(losses) if losses else 0
    
    # Calculate profit factor
    total_wins = sum(t['profit'] for t in wins) if wins else 0
    total_losses = abs(sum(t['profit'] for t in losses)) if losses else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    # Calculate consecutive wins/losses
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_streak = 0
    
    for trade in trades:
        if trade['profit'] is None:
            continue
        
        if trade['profit'] > 0:
            if current_streak > 0:
                current_streak += 1
            else:
                current_streak = 1
            max_consecutive_wins = max(max_consecutive_wins, current_streak)
        else:
            if current_streak < 0:
                current_streak -= 1
            else:
                current_streak = -1
            max_consecutive_losses = max(max_consecutive_losses, abs(current_streak))
    
    # Calculate average trade duration
    durations = []
    for trade in trades:
        if trade['exit_time'] and trade['entry_time']:
            duration = trade['exit_time'] - trade['entry_time']
            durations.append(duration)
    
    avg_trade_duration = sum(durations, timedelta()) / len(durations) if durations else timedelta(0)
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'avg_trade_duration': avg_trade_duration
    }

def print_backtest_summary(trades, initial_capital, final_balance, drawdown, daily_profit):
    """Print a summary of backtest results."""
    if not trades:
        print("\nNo trades executed during backtest")
        return
        
    # Calculate trade statistics
    profits = [t['profit'] for t in trades if t['profit'] and t['profit'] > 0]
    losses = [t['profit'] for t in trades if t['profit'] and t['profit'] <= 0]
    
    win_rate = (len(profits) / len(trades) * 100) if len(trades) > 0 else 0
    
    summary = f"""
=== Backtest Summary ===
Total Trades: {len(trades)}
Profitable Trades: {len(profits)}
Loss-Making Trades: {len(losses)}
Win Rate: {win_rate:.2f}%

Profit/Loss:
- Total P/L: ${(final_balance - initial_capital):.2f}
- Average Daily Profit: ${daily_profit:.2f}
- Biggest Profit: ${max(profits) if profits else 0:.2f}
- Biggest Loss: ${min(losses) if losses else 0:.2f}
- Average Profit: ${(sum(profits) / len(profits) if profits else 0):.2f}
- Average Loss: ${(sum(losses) / len(losses) if losses else 0):.2f}

Risk Metrics:
- Profit Factor: {abs(sum(profits) / sum(losses)) if losses and sum(losses) != 0 else float('inf'):.2f}
- Maximum Drawdown: {drawdown:.2f}%
- Risk/Reward Ratio: {abs((sum(profits) / len(profits)) / (sum(losses) / len(losses))) if losses and profits and sum(losses) != 0 and len(losses) > 0 else 0:.2f}

Trade Duration:
- Average Hold Time: {sum((t['exit_time'] - t['entry_time'] for t in trades if t['exit_time'] and t['entry_time']), timedelta()) / len(trades) if trades else timedelta(0)}
- Longest Trade: {max((t['exit_time'] - t['entry_time'] for t in trades if t['exit_time'] and t['entry_time']), default=timedelta(0))}
- Shortest Trade: {min((t['exit_time'] - t['entry_time'] for t in trades if t['exit_time'] and t['entry_time']), default=timedelta(0))}
"""
    print(summary)
    return summary

def run_backtest(model, df, params, scaler, test_indices=None):
    """
    Run a backtest on historical data using the trained model.
    Only uses completed candles for prediction, matching live trading behavior.
    
    Args:
        model: Trained ML model
        df: DataFrame with historical data and features
        params: TradingParameters object
        scaler: Fitted scaler for feature normalization
        test_indices: Optional indices to use for testing (to avoid look-ahead bias)
        
    Returns:
        Dictionary with backtest results
    """
    print("\nRunning backtest...")
    
    # Define feature columns
    feature_columns = ['Price_Change', 'High_Low_Range', 'Body_Size', 'Upper_Wick', 
                      'Lower_Wick', 'Volume_Change', 'Volume_Ratio', 'Momentum', 
                      'Volatility', 'Price_Position', 'Trend', 'RSI_Signal']
    
    # If test_indices provided, use only those rows
    if test_indices is not None:
        df_test = df.loc[test_indices]
    else:
        df_test = df
    
    # Initialize variables
    balance = params.capital
    positions = []
    trades = []
    equity_curve = [balance]
    
    # Loop through each candle (excluding the first few that might not have all features)
    for i in range(50, len(df_test)):
        # Get current candle (this is a completed candle)
        current_candle = df_test.iloc[i]
        current_time = current_candle.name if isinstance(current_candle.name, pd.Timestamp) else pd.Timestamp(current_candle.name)
        current_price = current_candle['close']
        
        # Update open positions
        for pos in positions[:]:
            # Calculate current profit/loss
            if pos['type'] == 'buy':
                profit = (current_price - pos['entry_price']) * pos['size'] - params.spread
            else:
                profit = (pos['entry_price'] - current_price) * pos['size'] - params.spread
            
            # Check if we should close the position based on model prediction
            # Get prediction for the current completed candle
            prediction = predict_signal(model, scaler, current_candle, feature_columns)
            
            # Close position if signal is opposite to current position
            if (pos['type'] == 'buy' and prediction == -1) or (pos['type'] == 'sell' and prediction == 1):
                # Close the position
                balance += profit
                pos['exit_price'] = current_price
                pos['exit_time'] = current_time
                pos['profit'] = profit
                trades.append(pos)
                positions.remove(pos)
        
        # Make a trading decision for the next candle
        # Only if we have capacity for new positions
        if len(positions) < params.max_trades:
            # Get prediction for the current completed candle
            prediction = predict_signal(model, scaler, current_candle, feature_columns)
            
            # Open new position based on prediction
            if prediction == 1:  # Buy signal
                # Calculate position size
                position_size = params.lotsize
                
                # Open buy position
                positions.append({
                    'type': 'buy',
                    'entry_price': current_price + params.spread,
                    'entry_time': current_time,
                    'size': position_size
                })
                
            elif prediction == -1:  # Sell signal
                # Calculate position size
                position_size = params.lotsize
                
                # Open sell position
                positions.append({
                    'type': 'sell',
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'size': position_size
                })
        
        # Calculate current equity
        current_equity = balance
        for pos in positions:
            if pos['type'] == 'buy':
                profit = (current_price - pos['entry_price']) * pos['size'] - params.spread
            else:
                profit = (pos['entry_price'] - current_price) * pos['size'] - params.spread
            current_equity += profit
        
        equity_curve.append(current_equity)
    
    # Close any remaining positions at the end of the backtest
    final_price = df_test.iloc[-1]['close']
    for pos in positions:
        if pos['type'] == 'buy':
            profit = (final_price - pos['entry_price']) * pos['size'] - params.spread
        else:
            profit = (pos['entry_price'] - final_price) * pos['size'] - params.spread
        
        balance += profit
        pos['exit_price'] = final_price
        pos['exit_time'] = df_test.index[-1]
        pos['profit'] = profit
        trades.append(pos)
    
    # Calculate performance metrics
    total_trades = len(trades)
    winning_trades = sum(1 for trade in trades if trade['profit'] > 0)
    losing_trades = total_trades - winning_trades
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    total_profit = sum(trade['profit'] for trade in trades if trade['profit'] > 0)
    total_loss = sum(abs(trade['profit']) for trade in trades if trade['profit'] < 0)
    
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Calculate max drawdown
    peak = equity_curve[0]
    max_drawdown = 0
    
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak if peak > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    # Calculate consecutive wins/losses
    consecutive_wins = 0
    consecutive_losses = 0
    current_streak = 0
    
    for i in range(len(trades)):
        if trades[i]['profit'] > 0:
            if current_streak > 0:
                current_streak += 1
            else:
                current_streak = 1
            consecutive_wins = max(consecutive_wins, current_streak)
        else:
            if current_streak < 0:
                current_streak -= 1
            else:
                current_streak = -1
            consecutive_losses = min(consecutive_losses, current_streak)
    
    # Calculate average trade duration
    durations = []
    for trade in trades:
        duration = trade['exit_time'] - trade['entry_time']
        durations.append(duration.total_seconds() / 60)  # in minutes
    
    avg_duration = sum(durations) / len(durations) if durations else 0
    
    # Prepare results
    metrics = {
        'initial_capital': params.capital,
        'final_balance': balance,
        'total_profit': balance - params.capital,
        'return_pct': (balance / params.capital - 1) * 100,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown * 100,
        'max_consecutive_wins': consecutive_wins,
        'max_consecutive_losses': abs(consecutive_losses),
        'avg_trade_duration': avg_duration
    }
    
    results = {
        'metrics': metrics,
        'trades': trades,
        'equity_curve': equity_curve
    }
    
    print("\nBacktest completed:")
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"Final balance: ${balance:.2f}")
    print(f"Return: {metrics['return_pct']:.2f}%")
    print(f"Max drawdown: {max_drawdown:.2%}")
    
    return results


def save_backtest_results(results, symbol, timeframe, initial_capital):
    """Save backtest results to a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"backtest_{symbol}_{timeframe}_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write(f"Backtest Results for {symbol} {timeframe}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Initial Capital: ${initial_capital:.2f}\n\n")
        
        f.write("=== Performance Summary ===\n")
        f.write(f"Final Balance: ${initial_capital + results['total_profit']:.2f}\n")
        f.write(f"Total Profit: ${results['total_profit']:.2f}\n")
        f.write(f"Daily Profit: ${results['daily_profit']:.2f}\n")
        f.write(f"Maximum Drawdown: {results['drawdown']:.2f}%\n\n")
        
        metrics = results['metrics']
        f.write("=== Trade Metrics ===\n")
        f.write(f"Total Trades: {metrics['total_trades']}\n")
        f.write(f"Win Rate: {metrics['win_rate']:.2f}%\n")
        f.write(f"Average Win: ${metrics['avg_win']:.2f}\n")
        f.write(f"Average Loss: ${metrics['avg_loss']:.2f}\n")
        f.write(f"Profit Factor: {metrics['profit_factor']:.2f}\n")
        f.write(f"Max Consecutive Wins: {metrics['max_consecutive_wins']}\n")
        f.write(f"Max Consecutive Losses: {metrics['max_consecutive_losses']}\n")
        f.write(f"Average Trade Duration: {metrics['avg_trade_duration']}\n\n")
        
        f.write("=== Trade Details ===\n")
        for i, trade in enumerate(results['trades'], 1):
            f.write(f"Trade {i}:\n")
            f.write(f"  Type: {trade['type']}\n")
            f.write(f"  Entry Time: {trade['entry_time']}\n")
            f.write(f"  Entry Price: ${trade['entry_price']:.2f}\n")
            f.write(f"  Exit Time: {trade['exit_time']}\n")
            f.write(f"  Exit Price: ${trade['exit_price']:.2f}\n")
            f.write(f"  Profit: ${trade['profit']:.2f}\n\n")
    
    print(f"\nBacktest results saved to {results_file}")
    return results_file
