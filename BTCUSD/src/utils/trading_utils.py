# Updated trading_utils.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime

class TradingParameters:
    def __init__(self, symbol, timeframe, lotsize=0.01, spread=1.0, capital=10000.0, 
                 risk_per_trade=0.02, max_trades=5, commission=0.0):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lotsize = lotsize
        self.spread = spread
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_trades = max_trades
        self.commission = commission

def setup_mt5_connection():
    """Initialize connection to MetaTrader 5."""
    if not mt5.initialize():
        print("\nFailed to initialize MT5")
        return False
    
    # Check if connected
    if not mt5.terminal_info().connected:
        print("\nNot connected to MT5 terminal")
        return False
    
    print("\nConnected to MT5 terminal")
    return True

def get_open_positions(symbol=None):
    """Get all open positions, optionally filtered by symbol."""
    if not mt5.initialize():
        print("\nFailed to initialize MT5")
        return []
    
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return []
    
    return positions

def place_market_order(symbol, order_type, volume, comment=""):
    """Place a market order in MT5 without SL/TP."""
    if not mt5.initialize():
        print("\nFailed to initialize MT5")
        return None
    
    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"\nSymbol {symbol} not found")
        return None
    
    if not symbol_info.visible:
        print(f"\nSymbol {symbol} is not visible, trying to switch on")
        if not mt5.symbol_select(symbol, True):
            print(f"\nFailed to select {symbol}")
            return None
    
    # Prepare the request
    price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # Send the order
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"\nOrder failed: {result.retcode}")
        print(f"Error: {result.comment}")
        return None
    
    print(f"\nOrder placed successfully: {result.order}")
    return result.order

def close_position(ticket):
    """Close a specific position by ticket number."""
    if not mt5.initialize():
        print("\nFailed to initialize MT5")
        return False
    
    # Get position info
    positions = mt5.positions_get(ticket=ticket)
    if positions is None or len(positions) == 0:
        print(f"\nPosition {ticket} not found")
        return False
    
    position = positions[0]
    
    # Prepare the request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "position": ticket,
        "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
        "deviation": 10,
        "magic": 234000,
        "comment": "Close position",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # Send the order
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"\nFailed to close position {ticket}: {result.retcode}")
        print(f"Error: {result.comment}")
        return False
    
    print(f"\nPosition {ticket} closed successfully")
    return True

def calculate_position_size(account_balance, risk_percent, symbol):
    """Calculate position size based on risk management."""
    if not mt5.initialize():
        print("\nFailed to initialize MT5")
        return 0.01  # Default minimum
    
    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"\nSymbol {symbol} not found")
        return 0.01
    
    # Calculate risk amount
    risk_amount = account_balance * risk_percent
    
    # Get current price
    current_price = mt5.symbol_info_tick(symbol).ask
    
    # Calculate position size based on 1% price movement risk
    # This is a simplified approach - adjust based on your risk model
    price_risk = current_price * 0.01  # 1% price movement
    
    if price_risk > 0:
        position_size = risk_amount / price_risk
    else:
        position_size = 0.01  # Default minimum
    
    # Round to standard lot sizes
    if position_size >= 1:
        position_size = round(position_size)
    elif position_size >= 0.1:
        position_size = round(position_size * 10) / 10
    else:
        position_size = round(position_size * 100) / 100
    
    # Ensure minimum lot size
    min_lot = symbol_info.volume_min
    if position_size < min_lot:
        position_size = min_lot
    
    return position_size

def execute_trade_decision(prediction, params, positions, current_price, current_time, is_live=False):
    """
    Execute a trade decision based on model prediction.
    This function is used consistently across backtesting and live trading.
    
    Args:
        prediction: Model prediction (1=buy, -1=sell, 0=hold)
        params: TradingParameters object
        positions: List of current open positions
        current_price: Current market price
        current_time: Current timestamp
        is_live: Boolean indicating if this is live trading
        
    Returns:
        new_positions: List of new positions opened
        closed_positions: List of positions closed
    """
    new_positions = []
    closed_positions = []
    
    # Check if we can open new positions
    if len(positions) < params.max_trades:
        if prediction == 1:  # Buy signal
            # Calculate position size
            if is_live:
                account_info = mt5.account_info()
                if account_info:
                    position_size = calculate_position_size(
                        account_info.balance, 
                        params.risk_per_trade,
                        params.symbol
                    )
                else:
                    position_size = params.lotsize
            else:
                position_size = params.lotsize
            
            # Open buy position
            if is_live:
                order_id = place_market_order(
                    params.symbol,
                    mt5.ORDER_TYPE_BUY,
                    position_size,
                    comment=f"ML Signal {current_time}"
                )
                
                if order_id:
                    new_position = {
                        'ticket': order_id,
                        'type': 'buy',
                        'entry_price': current_price + params.spread,
                        'entry_time': current_time,
                        'size': position_size
                    }
                    new_positions.append(new_position)
            else:
                # For backtesting
                new_position = {
                    'type': 'buy',
                    'entry_price': current_price + params.spread,
                    'entry_time': current_time,
                    'size': position_size
                }
                new_positions.append(new_position)
            
        elif prediction == -1:  # Sell signal
            # Calculate position size
            if is_live:
                account_info = mt5.account_info()
                if account_info:
                    position_size = calculate_position_size(
                        account_info.balance, 
                        params.risk_per_trade,
                        params.symbol
                    )
                else:
                    position_size = params.lotsize
            else:
                position_size = params.lotsize
            
            # Open sell position
            if is_live:
                order_id = place_market_order(
                    params.symbol,
                    mt5.ORDER_TYPE_SELL,
                    position_size,
                    comment=f"ML Signal {current_time}"
                )
                
                if order_id:
                    new_position = {
                        'ticket': order_id,
                        'type': 'sell',
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'size': position_size
                    }
                    new_positions.append(new_position)
            else:
                # For backtesting
                new_position = {
                    'type': 'sell',
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'size': position_size
                }
                new_positions.append(new_position)
    
    # Check if we need to close any positions based on model prediction
    for position in positions:
        if is_live:
            position_type = "buy" if position.type == mt5.ORDER_TYPE_BUY else "sell"
            position_ticket = position.ticket
            position_profit = position.profit
            position_volume = position.volume
            position_price_open = position.price_open
            position_time = pd.to_datetime(position.time, unit='s')
        else:
            position_type = position['type']
            position_ticket = position.get('ticket', None)
            position_profit = position.get('profit', 0)
            position_volume = position['size']
            position_price_open = position['entry_price']
            position_time = position['entry_time']
        
        # Close position if signal is opposite to current position
        if (position_type == "buy" and prediction == -1) or (position_type == "sell" and prediction == 1):
            if is_live:
                if close_position(position_ticket):
                    closed_position = {
                        'ticket': position_ticket,
                        'type': position_type,
                        'entry_price': position_price_open,
                        'entry_time': position_time,
                        'exit_price': current_price,
                        'exit_time': current_time,
                        'size': position_volume,
                        'profit': position_profit
                    }
                    closed_positions.append(closed_position)
            else:
                # For backtesting
                if position_type == "buy":
                    profit = (current_price - position['entry_price']) * position['size'] - params.spread
                else:
                    profit = (position['entry_price'] - current_price) * position['size'] - params.spread
                
                closed_position = position.copy()
                closed_position['exit_price'] = current_price
                closed_position['exit_time'] = current_time
                closed_position['profit'] = profit
                closed_positions.append(closed_position)
    
    return new_positions, closed_positions
