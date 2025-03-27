# Updated data_utils.py with no trend indicators
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_historical_data(symbol, timeframe, start_date, end_date):
    """
    Get historical data from MT5.
    
    Args:
        symbol: Trading symbol (e.g., BTCUSD)
        timeframe: Timeframe (M1, M5, M15, H1, D1)
        start_date: Start date for historical data
        end_date: End date for historical data
        
    Returns:
        DataFrame with historical data
    """
    if not mt5.initialize():
        print("\nFailed to initialize MT5")
        return None
    
    # Map timeframe string to MT5 timeframe
    timeframe_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1,
        'D1': mt5.TIMEFRAME_D1
    }
    
    mt5_timeframe = timeframe_map.get(timeframe)
    if mt5_timeframe is None:
        print(f"\nInvalid timeframe: {timeframe}")
        return None
    
    # Get historical data
    rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        print("\nFailed to get historical data")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    return df

def prepare_features(df):
    """
    Prepare simplified features for the model.
    Removed: volume, momentum, volatility, and ALL trend indicators.
    Only keeping price action and RSI indicators.
    
    Args:
        df: DataFrame with historical data
        
    Returns:
        DataFrame with added features
    """
    # Calculate price-based features
    df['Price_Change'] = df['close'].pct_change()
    df['High_Low_Range'] = (df['high'] - df['low']) / df['close']
    df['Body_Size'] = abs(df['close'] - df['open']) / df['close']
    df['Upper_Wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['Lower_Wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    
    # Calculate price position
    df['Price_Position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # RSI signal
    df['RSI_Signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    
    return df
