import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging

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
        print(f"Error: {mt5.last_error()}")
        return False
    
    print("\nMT5 initialized successfully")
    account_info = mt5.account_info()
    if account_info:
        print(f"Connected to account #{account_info.login} {account_info.server}")
        print(f"Balance: ${account_info.balance:.2f}, Equity: ${account_info.equity:.2f}")
        return True
    else:
        print("\nFailed to get account info")
        return False

def place_market_order(symbol, order_type, volume, sl=0.0, tp=0.0, comment=""):
    """Place a market order in MT5."""
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
        "sl": sl,
        "tp": tp,
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

def close_position(position_id):
    """Close a specific position by its ticket ID."""
    if not mt5.initialize():
        print("\nFailed to initialize MT5")
        return False
    
    # Get the position details
    position = mt5.positions_get(ticket=position_id)
    if not position:
        print(f"\nPosition {position_id} not found")
        return False
    
    position = position[0]
    
    # Determine the order type for closing
    close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    
    # Prepare the request
    price = mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": close_type,
        "position": position_id,
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": "Close position",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # Send the order
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"\nFailed to close position: {result.retcode}")
        print(f"Error: {result.comment}")
        return False
    
    print(f"\nPosition {position_id} closed successfully")
    return True

def get_open_positions(symbol=None):
    """Get all open positions, optionally filtered by symbol."""
    if not mt5.initialize():
        print("\nFailed to initialize MT5")
        return []
    
    if symbol:
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()
    
    if positions is None:
        print("\nNo positions found")
        return []
    
    return positions

def calculate_position_size(account_balance, risk_percent, stop_loss_pips, symbol):
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
    
    # Get pip value
    pip_value = symbol_info.trade_tick_value
    
    # Calculate position size
    if stop_loss_pips > 0 and pip_value > 0:
        position_size = risk_amount / (stop_loss_pips * pip_value)
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

def run_live_trading(model, scaler, params, feature_columns):
    """Run live trading with the given model and parameters."""
    if not setup_mt5_connection():
        print("\nFailed to connect to MT5. Exiting.")
        return
    
    print(f"\nStarting live trading for {params.symbol} on {params.timeframe} timeframe")
    print(f"Using lot size: {params.lotsize}")
    print(f"Maximum open trades: {params.max_trades}")
    
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
            
            # Get the latest completed candle
            latest_candle = df.iloc[-2]
            features = latest_candle[feature_columns].values.reshape(1, -1)
            
            # Scale features and predict
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n[{current_time}] Latest candle time: {latest_candle['time']}")
            print(f"Current price: {latest_candle['close']}")
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
                            params.stop_loss_pips,
                            params.symbol
                        )
                    else:
                        position_size = params.lotsize
                    
                    # Calculate stop loss and take profit levels
                    current_price = mt5.symbol_info_tick(params.symbol).ask
                    stop_loss = current_price - (params.stop_loss_pips * mt5.symbol_info(params.symbol).point)
                    take_profit = current_price + (params.take_profit_pips * mt5.symbol_info(params.symbol).point)
                    
                    # Place buy order
                    order_id = place_market_order(
                        params.symbol,
                        mt5.ORDER_TYPE_BUY,
                        position_size,
                        sl=stop_loss,
                        tp=take_profit,
                        comment=f"ML Signal {current_time}"
                    )
                    
                    if order_id:
                        logging.info(f"BUY order placed: {order_id}, Size: {position_size}, Price: {current_price}")
                        print(f"\nBUY order placed: {order_id}, Size: {position_size}, Price: {current_price}")
                
                elif prediction == -1:  # Sell signal
                    # Calculate position size based on risk management
                    account_info = mt5.account_info()
                    if account_info:
                        position_size = calculate_position_size(
                            account_info.balance, 
                            params.risk_per_trade,
                            params.stop_loss_pips,
                            params.symbol
                        )
                    else:
                        position_size = params.lotsize
                    
                    # Calculate stop loss and take profit levels
                    # Calculate stop loss and take profit levels
                    current_price = mt5.symbol_info_tick(params.symbol).bid
                    stop_loss = current_price + (params.stop_loss_pips * mt5.symbol_info(params.symbol).point)
                    take_profit = current_price - (params.take_profit_pips * mt5.symbol_info(params.symbol).point)
                    
                    # Place sell order
                    order_id = place_market_order(
                        params.symbol,
                        mt5.ORDER_TYPE_SELL,
                        position_size,
                        sl=stop_loss,
                        tp=take_profit,
                        comment=f"ML Signal {current_time}"
                    )
                    
                    if order_id:
                        logging.info(f"SELL order placed: {order_id}, Size: {position_size}, Price: {current_price}")
                        print(f"\nSELL order placed: {order_id}, Size: {position_size}, Price: {current_price}")
            
            # Check if we need to close any positions
            for position in positions:
                position_type = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
                
                # Close position if signal is opposite to current position
                if (position_type == "BUY" and prediction == -1) or (position_type == "SELL" and prediction == 1):
                    if close_position(position.ticket):
                        logging.info(f"Closed {position_type} position: {position.ticket}, Profit: {position.profit}")
                        print(f"\nClosed {position_type} position: {position.ticket}, Profit: {position.profit}")
            
            # Wait for next candle
            minutes_to_wait = 1  # Default for M1
            if params.timeframe == 'M5':
                minutes_to_wait = 5
            elif params.timeframe == 'M15':
                minutes_to_wait = 15
            elif params.timeframe == 'H1':
                minutes_to_wait = 60
            elif params.timeframe == 'D1':
                minutes_to_wait = 1440
            
            print(f"\nWaiting {minutes_to_wait} minutes for next candle...")
            time.sleep(minutes_to_wait * 60)
            
    except KeyboardInterrupt:
        print("\nTrading stopped by user")
        logging.info("Trading stopped by user")
    except Exception as e:
        print(f"\nError in trading loop: {str(e)}")
        logging.error(f"Error in trading loop: {str(e)}")
        raise
    finally:
        print("\nClosing MT5 connection")
        mt5.shutdown()
