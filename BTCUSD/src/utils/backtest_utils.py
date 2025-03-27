# Updated backtest_utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pickle
from .model_utils import predict_signal

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
        
        # Make prediction using the consistent prediction function
        prediction = predict_signal(model, scaler, current_candle, feature_columns)
        
        # Execute trade decision using the standardized function
        from .trading_utils import execute_trade_decision
        new_positions, closed_positions = execute_trade_decision(
            prediction, params, positions, current_price, current_time, is_live=False
        )
        
        # Add new positions
        positions.extend(new_positions)
        
        # Process closed positions
        for closed_pos in closed_positions:
            balance += closed_pos['profit']
            trades.append(closed_pos)
            positions.remove(next(pos for pos in positions if pos['entry_time'] == closed_pos['entry_time'] and pos['type'] == closed_pos['type']))
        
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
    for pos in positions[:]:  # Use a copy to avoid modifying during iteration
        if pos['type'] == 'buy':
            profit = (final_price - pos['entry_price']) * pos['size'] - params.spread
        else:
            profit = (pos['entry_price'] - final_price) * pos['size'] - params.spread
        
        balance += profit
        closed_pos = pos.copy()
        closed_pos['exit_price'] = final_price
        closed_pos['exit_time'] = df_test.index[-1] if isinstance(df_test.index[-1], pd.Timestamp) else pd.Timestamp(df_test.index[-1])
        closed_pos['profit'] = profit
        trades.append(closed_pos)
        positions.remove(pos)
    
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
        'max_drawdown': max_drawdown,
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

def plot_backtest_results(results, symbol, timeframe):
    """Plot equity curve and trade distribution from backtest results."""
    metrics = results['metrics']
    equity_curve = results['equity_curve']
    trades = results['trades']
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot equity curve
    ax1.plot(equity_curve)
    ax1.set_title(f'Equity Curve - {symbol} {timeframe}')
    ax1.set_ylabel('Equity')
    ax1.grid(True)
    
    # Add key metrics as text
    text = (
        f"Initial Capital: ${metrics['initial_capital']:.2f}\n"
        f"Final Balance: ${metrics['final_balance']:.2f}\n"
        f"Return: {metrics['return_pct']:.2f}%\n"
        f"Total Trades: {metrics['total_trades']}\n"
        f"Win Rate: {metrics['win_rate']:.2%}\n"
        f"Profit Factor: {metrics['profit_factor']:.2f}\n"
        f"Max Drawdown: {metrics['max_drawdown']:.2%}"
    )
    ax1.text(0.02, 0.95, text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot trade distribution
    if trades:
        profits = [trade['profit'] for trade in trades]
        ax2.hist(profits, bins=20, alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_title('Trade Profit Distribution')
        ax2.set_xlabel('Profit')
        ax2.set_ylabel('Frequency')
        ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/backtest_{symbol}_{timeframe}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    
    plt.show()
