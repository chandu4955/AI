import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.model_utils import predict_signal

def run_backtest(model, df, params, scaler):
    """
    Run a backtest on historical data.
    
    Args:
        model: Trained model
        df: DataFrame with historical data
        params: Trading parameters
        scaler: Feature scaler
        
    Returns:
        Backtest results
    """
    # Define feature columns - use the simplified set
    feature_columns = [
        'Price_Change', 'High_Low_Range', 'Body_Size', 'Upper_Wick', 
        'Lower_Wick', 'Price_Position', 'RSI_Signal'
    ]
    
    # Initialize variables
    balance = params.capital
    equity = balance
    open_positions = []
    trades = []
    equity_curve = []
    
    # Run backtest
    print("\nRunning backtest...")
    
    for i in range(len(df)):
        # Get current candle
        current_candle = df.iloc[i]
        
        # Update equity
        equity = balance
        for pos in open_positions:
            equity += pos['profit_loss']
        
        # Record equity
        equity_curve.append(equity)
        
        # Check for closed positions
        closed_positions = []
        for j, pos in enumerate(open_positions):
            # Calculate profit/loss
            if pos['type'] == 'buy':
                pos['profit_loss'] = (current_candle['close'] - pos['entry_price']) * pos['lotsize'] * 100000
            else:
                pos['profit_loss'] = (pos['entry_price'] - current_candle['close']) * pos['lotsize'] * 100000
            
            # Check if position should be closed
            if pos['type'] == 'buy' and current_candle['close'] <= pos['stop_loss']:
                # Close position with loss
                balance += pos['profit_loss']
                pos['exit_price'] = current_candle['close']
                pos['exit_time'] = current_candle.name
                pos['profit'] = pos['profit_loss']
                trades.append(pos)
                closed_positions.append(j)
            elif pos['type'] == 'buy' and current_candle['close'] >= pos['take_profit']:
                # Close position with profit
                balance += pos['profit_loss']
                pos['exit_price'] = current_candle['close']
                pos['exit_time'] = current_candle.name
                pos['profit'] = pos['profit_loss']
                trades.append(pos)
                closed_positions.append(j)
            elif pos['type'] == 'sell' and current_candle['close'] >= pos['stop_loss']:
                # Close position with loss
                balance += pos['profit_loss']
                pos['exit_price'] = current_candle['close']
                pos['exit_time'] = current_candle.name
                pos['profit'] = pos['profit_loss']
                trades.append(pos)
                closed_positions.append(j)
            elif pos['type'] == 'sell' and current_candle['close'] <= pos['take_profit']:
                # Close position with profit
                balance += pos['profit_loss']
                pos['exit_price'] = current_candle['close']
                pos['exit_time'] = current_candle.name
                pos['profit'] = pos['profit_loss']
                trades.append(pos)
                closed_positions.append(j)
        
        # Remove closed positions
        for j in sorted(closed_positions, reverse=True):
            del open_positions[j]
        
        # Predict signal
        prediction = predict_signal(model, scaler, current_candle, feature_columns)
        
        # Open new position if signal is not 0 and we have less than max_trades open positions
        if prediction != 0 and len(open_positions) < params.max_trades:
            # Calculate position size
            risk_amount = balance * params.risk_per_trade
            
            if prediction == 1:  # Buy signal
                # Calculate stop loss and take profit
                stop_loss = current_candle['close'] * 0.99  # 1% stop loss
                take_profit = current_candle['close'] * 1.02  # 2% take profit
                
                # Calculate position size
                position_size = risk_amount / (current_candle['close'] - stop_loss) / 100000
                position_size = min(position_size, params.lotsize)
                
                # Open position
                position = {
                    'type': 'buy',
                    'entry_price': current_candle['close'],
                    'entry_time': current_candle.name,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'lotsize': position_size,
                    'profit_loss': 0
                }
                
                open_positions.append(position)
            
            elif prediction == -1:  # Sell signal
                # Calculate stop loss and take profit
                stop_loss = current_candle['close'] * 1.01  # 1% stop loss
                take_profit = current_candle['close'] * 0.98  # 2% take profit
                
                # Calculate position size
                position_size = risk_amount / (stop_loss - current_candle['close']) / 100000
                position_size = min(position_size, params.lotsize)
                
                # Open position
                position = {
                    'type': 'sell',
                    'entry_price': current_candle['close'],
                    'entry_time': current_candle.name,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'lotsize': position_size,
                    'profit_loss': 0
                }
                
                open_positions.append(position)
    
    # Close any remaining open positions
    for pos in open_positions:
        pos['exit_price'] = df.iloc[-1]['close']
        pos['exit_time'] = df.iloc[-1].name
        pos['profit'] = pos['profit_loss']
        trades.append(pos)
    
    # Calculate backtest metrics
    total_trades = len(trades)
    winning_trades = sum(1 for trade in trades if trade['profit'] > 0)
    losing_trades = sum(1 for trade in trades if trade['profit'] <= 0)
    
    if total_trades > 0:
        win_rate = winning_trades / total_trades
    else:
        win_rate = 0
    
    total_profit = sum(trade['profit'] for trade in trades if trade['profit'] > 0)
    total_loss = sum(trade['profit'] for trade in trades if trade['profit'] <= 0)
    
    if total_loss != 0:
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
    else:
        profit_factor = float('inf') if total_profit > 0 else 0
    
    net_profit = total_profit + total_loss
    
    # Calculate drawdown
    max_equity = 0
    max_drawdown = 0
    
    for equity_value in equity_curve:
        max_equity = max(max_equity, equity_value)
        drawdown = (max_equity - equity_value) / max_equity if max_equity > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    # Calculate Sharpe ratio
    if len(equity_curve) > 1:
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Calculate metrics
    metrics = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'net_profit': net_profit,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'final_balance': balance
    }
    
    # Return backtest results
    backtest_results = {
        'trades': trades,
        'equity_curve': equity_curve,
        'metrics': metrics
    }
    
    return backtest_results

def plot_backtest_results(backtest_results, symbol, timeframe):
    """
    Plot backtest results.
    
    Args:
        backtest_results: Backtest results
        symbol: Trading symbol
        timeframe: Timeframe
    """
    # Create directory if it doesn't exist
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Extract data
    equity_curve = backtest_results['equity_curve']
    metrics = backtest_results['metrics']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot equity curve
    ax1.plot(equity_curve)
    ax1.set_title(f'Equity Curve - {symbol} {timeframe}')
    ax1.set_xlabel('Candles')
    ax1.set_ylabel('Equity')
    ax1.grid(True)
    
    # Plot drawdown
    max_equity = np.maximum.accumulate(equity_curve)
    drawdown = (max_equity - equity_curve) / max_equity
    ax2.fill_between(range(len(drawdown)), 0, drawdown, alpha=0.3, color='red')
    ax2.set_title('Drawdown')
    ax2.set_xlabel('Candles')
    ax2.set_ylabel('Drawdown')
    ax2.grid(True)
    
    # Add metrics as text
    metrics_text = (
        f"Total Trades: {metrics['total_trades']}\n"
        f"Winning Trades: {metrics['winning_trades']}\n"
        f"Losing Trades: {metrics['losing_trades']}\n"
        f"Win Rate: {metrics['win_rate']:.2%}\n"
        f"Total Profit: ${metrics['total_profit']:.2f}\n"
        f"Total Loss: ${metrics['total_loss']:.2f}\n"
        f"Net Profit: ${metrics['net_profit']:.2f}\n"
        f"Profit Factor: {metrics['profit_factor']:.2f}\n"
        f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        f"Final Balance: ${metrics['final_balance']:.2f}"
    )
    
    plt.figtext(0.15, 0.01, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure
    plt.savefig(f'plots/{symbol}_{timeframe}_backtest.png')
    
    # Show figure
    plt.close()
    
    print(f"\nBacktest plot saved to plots/{symbol}_{timeframe}_backtest.png")
