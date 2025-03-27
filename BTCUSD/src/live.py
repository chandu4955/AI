import argparse
import os
import sys
import time
import logging
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

from utils.model_utils import load_model, predict_signal
from utils.trading_utils import TradingParameters, setup_mt5_connection, get_open_positions
from utils.trading_utils import place_market_order, close_position, calculate_position_size

def main():
    parser = argparse.ArgumentParser(description='Run live trading with a trained model')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., EURUSD)')
    parser.add_argument('--timeframe', type=str, default='M15', help='Timeframe (M1, M5, M15, H1, D1)')
    parser.add_argument('--lotsize', type=float, default=0.01, help='Lot size for trading')
    parser.add_argument('--capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.02, help='Risk per trade (as a decimal, e.g., 0.02 for 2%)')
    parser.add_argument('--max_trades', type=int, default=5, help='Maximum number of open trades')
    
    args = parser.parse_args()
    
    print(f"\nStarting live trading for {args.symbol} on {args.timeframe} timeframe")
    
    # Load the trained model
    model, scaler = load_model(args.symbol, args.timeframe)
    if model is None or scaler is None:
        print("\nFailed to load model. Exiting.")
        return
    
    # Set up trading parameters
    params = TradingParameters(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lotsize=args.lotsize,
        spread=1.0,  # This will be determined from the market in live trading
        capital=args.capital,
        risk_per_trade=args.risk,
        max_trades=args.max_trades
    )
    
    # Define feature columns used by the model
    feature_columns = ['Price_Change', 'High_Low_Range', 'Body_Size', 'Upper_Wick', 
                      'Lower_Wick', 'Volume_Change', 'Volume_Ratio', 'Momentum', 
                      'Volatility', 'Price_Position', 'Trend', 'RSI_Signal']
    
    # Setup MT5 connection
    if not setup_mt5_connection():
        print("\nFailed to connect to MT5. Exiting.")
        return
    
    # Setup logging
    logging.basicConfig(
        filename=f"trading_{params.symbol}_{params.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info(f"Starting live trading for {params.symbol} on {params.timeframe}")
    
    timeframe_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1,
        'D1': mt5.TIMEFRAME_D1
    }
    
    mt5_timeframe = timeframe_map.get(params.timeframe)
    if mt5_timeframe is None:
        print(f"\nInvalid timeframe: {params.timeframe}")
        return
    
    try:
        last_processed_time = None
        
        while True:
            # Get current open positions
            positions = get_open_positions(params.symbol)
            
            # Get latest data
            rates = mt5.copy_rates_from_pos(params.symbol, mt5_timeframe, 0, 100)
            if rates is None:
                print("\nFailed to get market data. Retrying...")
                time.sleep(5)
                continue
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Prepare features
            df['Price_Change'] = df['close'].pct_change()
            df['High_Low_Range'] = (df['high'] - df['low']) / df['close']
            df['Body_Size'] = abs(df['close'] - df['open']) / df['close']
            df['Upper_Wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
            df['Lower_Wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
            df['Volume_Change'] = df['tick_volume'].pct_change()
            df['Volume_MA'] = df['tick_volume'].rolling(window=5).mean()
            df['Volume_Ratio'] = df['tick_volume'] / df['Volume_MA']
            df['Momentum'] = df['Price_Change'].rolling(window=3).sum()
            df['Volatility'] = df['Price_Change'].rolling(window=5).std()
            df['Price_Position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['RSI'] = 100 - (100 / (1 + (df['close'].diff().clip(lower=0).rolling(14).mean() / 
                                          df['close'].diff().clip(upper=0).abs().rolling(14).mean())))
            df['RSI_Signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
            df['Trend'] = np.where(
                (df['EMA20'] > df['EMA50']) & (df['close'] > df['EMA20']), 1,
                np.where((df['EMA20'] < df['EMA50']) & (df['close'] < df['EMA20']), -1, 0)
            )
            
            # Drop NaN values
            df = df.dropna()
            
            if len(df) < 2:
                print("\nNot enough data points after feature calculation")
                time.sleep(5)
                continue
            
            # Get the latest COMPLETED candle (second to last row)
            latest_completed_candle = df.iloc[-2]
            current_time = latest_completed_candle['time']
            
            # Skip if we've already processed this candle
            if last_processed_time is not None and current_time <= last_processed_time:
                # Calculate time to wait until next candle
                minutes_to_wait = 1  # Default for M1
                if params.timeframe == 'M5':
                    minutes_to_wait = 5
                elif params.timeframe == 'M15':
                    minutes_to_wait = 15
                elif params.timeframe == 'H1':
                    minutes_to_wait = 60
                elif params.timeframe == 'D1':
                    minutes_to_wait = 1440
                
                # Calculate seconds remaining until next candle
                now = datetime.now()
                next_candle_time = current_time + pd.Timedelta(minutes=minutes_to_wait)
                seconds_to_wait = max(1, (next_candle_time - pd.Timestamp(now)).total_seconds())
                
                print(f"\nWaiting {int(seconds_to_wait)} seconds for next candle...")
                time.sleep(min(60, seconds_to_wait))  # Wait at most 60 seconds before checking again
                continue
            
            # Update last processed time
            last_processed_time = current_time
            
            # Make prediction using the same method as in backtesting
            prediction = predict_signal(model, scaler, latest_completed_candle, feature_columns)
            
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Latest completed candle time: {current_time}")
            print(f"Current price: {latest_completed_candle['close']}")
            print(f"Prediction: {prediction} (1=Buy, -1=Sell, 0=Hold)")
            
            # Check if we can open new positions
            if len(positions) < params.max_trades:
                if prediction == 1:  # Buy signal
                    # Calculate position size based on risk management
                    account_info = mt5.account_info()
                    if account_info:
                        position_size = calculate_position_size(
                            account_info.balance, 
                            params.risk_per_trade,
                            params.symbol
                        )
                    else:
                        position_size = params.lotsize
                    
                    # Place buy order
                    order_id = place_market_order(
                        params.symbol,
                        mt5.ORDER_TYPE_BUY,
                        position_size,
                        comment=f"ML Signal {current_time}"
                    )
                    
                    if order_id:
                        logging.info(f"BUY order placed: {order_id}, Size: {position_size}")
                        print(f"\nBUY order placed: {order_id}, Size: {position_size}")
                
                elif prediction == -1:  # Sell signal
                    # Calculate position size based on risk management
                    account_info = mt5.account_info()
                    if account_info:
                        position_size = calculate_position_size(
                            account_info.balance, 
                            params.risk_per_trade,
                            params.symbol
                        )
                    else:
                        position_size = params.lotsize
                    
                    # Place sell order
                    order_id = place_market_order(
                        params.symbol,
                        mt5.ORDER_TYPE_SELL,
                        position_size,
                        comment=f"ML Signal {current_time}"
                    )
                    
                    if order_id:
                        logging.info(f"SELL order placed: {order_id}, Size: {position_size}")
                        print(f"\nSELL order placed: {order_id}, Size: {position_size}")
            
            # Check if we need to close any positions based on model prediction
            for position in positions:
                position_type = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
                
                # Close position if signal is opposite to current position
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime

class TradingParameters:
    def __init__(self, symbol, timeframe, lotsize, spread, capital, risk_per_trade=0.02, 
                 max_trades=5):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lotsize = lotsize
        self.spread = spread
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_trades = max_trades

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
