import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_minutes_per_candle(timeframe):
    """Convert timeframe string to minutes."""
    timeframe_minutes = {
        'M1': 1,
        'M5': 5,
        'M15': 15,
        'H1': 60,
        'D1': 1440
    }
    return timeframe_minutes.get(timeframe, 1)

def load_mt5_history(symbol, timeframe, days=None):
    """Load historical data from MT5."""
    if not mt5.initialize():
        print("\nFailed to initialize MT5")
        return None
        
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
    
    # If days is specified, calculate number of bars
    if days:
        bars_per_day = {
            'M1': 1440,
            'M5': 288,
            'M15': 96,
            'H1': 24,
            'D1': 1
        }
        total_bars = bars_per_day[timeframe] * days
    else:
        # Default to 50000 candles if days not specified
        total_bars = 50000
    
    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, total_bars)
    
    if rates is None:
        print(f"\nFailed to get history data for {symbol}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_features(df, timeframe='M1'):
    """Prepare features for model training and prediction."""
    if 'close' not in df.columns:
        print("\nError: 'close' column not found in DataFrame")
        print("Available columns:", df.columns.tolist())
        return None
        
    print("\nPreparing price action features...")
    
    # Price action features
    df['Price_Change'] = df['close'].pct_change()
    df['High_Low_Range'] = (df['high'] - df['low']) / df['close']
    df['Body_Size'] = abs(df['close'] - df['open']) / df['close']
    df['Upper_Wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['Lower_Wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    
    # Volume features
    df['Volume_Change'] = df['tick_volume'].pct_change()
    df['Volume_MA'] = df['tick_volume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['tick_volume'] / df['Volume_MA']
    
    # Momentum and volatility
    df['Momentum'] = df['Price_Change'].rolling(window=3).sum()
    df['Volatility'] = df['Price_Change'].rolling(window=5).std()
    
    # Market structure
    df['Higher_High'] = df['high'] > df['high'].shift(1)
    df['Lower_Low'] = df['low'] < df['low'].shift(1)
    df['Price_Position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Add more technical indicators for better trend confirmation
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['RSI'] = calculate_rsi(df['close'], 14)
    df['ATR'] = (
        df['high'].rolling(14).max() - df['low'].rolling(14).min()
    ) / df['close'].rolling(14).mean()
    
    # Add RSI signal before using it in target calculation
    df['RSI_Signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    
    # Enhanced trend identification with more sensitive conditions
    df['Trend'] = np.where(
        (df['EMA20'] > df['EMA50']) & (df['close'] > df['EMA20']), 1,
        np.where((df['EMA20'] < df['EMA50']) & (df['close'] < df['EMA20']), -1, 0)
    )
    
    # Create a shifted volatility to avoid look-ahead bias
    df['Volatility_Prev'] = df['Volatility'].shift(1)
    
    # Modified trading signal conditions with more relaxed thresholds
    df['target'] = np.where(
        (df['Price_Change'].shift(-1) > 0.1 * df['Volatility_Prev']) & 
        (df['Volume_Ratio'] > 1.0) & 
        ((df['Trend'] == 1) | (df['RSI_Signal'] == 1) | (df['Price_Position'] < 0.3)), 1,
        np.where(
            (df['Price_Change'].shift(-1) < -0.1 * df['Volatility_Prev']) & 
            (df['Volume_Ratio'] > 1.0) & 
            ((df['Trend'] == -1) | (df['RSI_Signal'] == -1) | (df['Price_Position'] > 0.7)), -1, 0
        )
    )
    
    # Remove the last row since it has NaN target due to shift(-1)
    df = df.iloc[:-1]
    
    # Remove the first few rows that have NaN values due to indicators calculation
    df = df.dropna()
    
    # Ensure we have at least one instance of each class
    if len(df['target'].unique()) < 3:
        print("\nWarning: Not enough signal diversity. Adjusting thresholds...")
        # Force some signals if none exist
        if 1 not in df['target'].unique():
            strongest_buy = df[df['RSI'] == df['RSI'].min()].index[0]
            df.loc[strongest_buy, 'target'] = 1
        if -1 not in df['target'].unique():
            strongest_sell = df[df['RSI'] == df['RSI'].max()].index[0]
            df.loc[strongest_sell, 'target'] = -1
        if 0 not in df['target'].unique():
            neutral_point = df[abs(df['RSI'] - 50).idxmin()]
            df.loc[neutral_point.name, 'target'] = 0
    
    # Drop the temporary column used for calculation
    df = df.drop('Volatility_Prev', axis=1)
    
    return df
