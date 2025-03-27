# Updated live.py
import argparse
import os
import time
import logging
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta

from utils.data_utils import get_historical_data, prepare_features
from utils.model_utils import load_model, predict_signal
from utils.trading_utils import TradingParameters, setup_mt5_connection, get_open_positions, execute_trade_decision

def main():
    parser = argparse.ArgumentParser(description='Run live trading with a trained model')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., BTCUSD)')
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
            df.set_index('time', inplace=True)
            
            # Prepare features
            df = prepare_features(df)
            
            # Drop rows with NaN values
            df = df.dropna()
            
            if len(df) < 2:
                print("\nNot enough data points after feature calculation")
                time.sleep(5)
                continue
            
            # Get the latest COMPLETED candle (second to last row)
            latest_completed_candle = df.iloc[-2]
            current_time = latest_completed_candle.name
            current_price = latest_completed_candle['close']
            
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
            
            # Make prediction using the consistent prediction function
            prediction = predict_signal(model, scaler, latest_completed_candle, feature_columns)
            
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Latest completed candle time: {current_time}")
            print(f"Current price: {current_price}")
            print(f"Prediction: {prediction} (1=Buy, -1=Sell, 0=Hold)")
            
            # Execute trade decision using the standardized function
            new_positions, closed_positions = execute_trade_decision(
                prediction, params, positions, current_price, current_time, is_live=True
            )
            
            # Log new positions
            for pos in new_positions:
                logging.info(f"New {pos['type'].upper()} position: Ticket {pos['ticket']}, Size: {pos['size']}")
                print(f"\nNew {pos['type'].upper()} position: Ticket {pos['ticket']}, Size: {pos['size']}")
            
            # Log closed positions
            for pos in closed_positions:
                logging.info(f"Closed {pos['type'].upper()} position: Ticket {pos['ticket']}, Profit: {pos['profit']}")
                print(f"\nClosed {pos['type'].upper()} position: Ticket {pos['ticket']}, Profit: {pos['profit']}")
            
            # Wait for a short time before checking again
            time.sleep(10)
            
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

if __name__ == "__main__":
    main()
