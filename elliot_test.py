import requests
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

# Configuration Parameters
SYMBOL = "XLM-USD"
GRANULARITY = 86400  # Daily data
LOOKBACK = 100  # Number of data points to analyze
MIN_DIST = 5  # Minimum distance between local extrema for Elliott Waves
WAVE_DEVIATION = 0.05  # Deviation to qualify Elliott Wave structure
TRADE_SIZE = 10000  # $10,000 per trade

# Fetch Historical Data
def fetch_historical_data(symbol, granularity):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
    response = requests.get(url)
    if response.status_code == 200:
        columns = ["time", "low", "high", "open", "close", "volume"]
        data = pd.DataFrame(response.json(), columns=columns)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        numeric_columns = ["low", "high", "open", "close", "volume"]
        data[numeric_columns] = data[numeric_columns].astype(float)
        data = data.sort_values(by="time").reset_index(drop=True)
        return data
    else:
        raise Exception(f"Failed to fetch data for {symbol}: {response.status_code}")

# Identify Local Extrema
def identify_extrema(data):
    highs = argrelextrema(data['high'].values, np.greater, order=MIN_DIST)[0].astype(int)
    lows = argrelextrema(data['low'].values, np.less, order=MIN_DIST)[0].astype(int)
    return highs, lows

# Validate Elliott Wave
def validate_elliott_wave(data, highs, lows):
    waves = []
    all_points = sorted(list(highs) + list(lows))
    for i in range(len(all_points) - 4):
        wave_points = all_points[i:i+5]
        wave_prices = data['close'].iloc[wave_points].values

        if (
            wave_prices[1] > wave_prices[0] and
            wave_prices[2] < wave_prices[1] and
            wave_prices[3] > wave_prices[2] and
            wave_prices[4] < wave_prices[3] and
            abs((wave_prices[4] - wave_prices[0]) / wave_prices[0]) > WAVE_DEVIATION
        ):
            waves.append({
                "start": wave_points[0],
                "end": wave_points[4],
                "wave_prices": wave_prices
            })
    return waves

# Analyze Signals
def analyze_signals(data, waves):
    signals = []
    for wave in waves:
        start_price = data['close'].iloc[wave['start']]
        end_price = data['close'].iloc[wave['end']]
        if end_price < start_price:  # Downward Wave
            signals.append({
                "type": "SELL",
                "entry_index": wave['end'],
                "entry_price": end_price,
                "stop_loss": end_price + (end_price * 0.02),
                "take_profit": end_price - (end_price * 0.05)
            })
        else:  # Upward Wave
            signals.append({
                "type": "BUY",
                "entry_index": wave['end'],
                "entry_price": end_price,
                "stop_loss": end_price - (end_price * 0.02),
                "take_profit": end_price + (end_price * 0.05)
            })
    return signals

# Backtest Signals
def backtest_signals(data, signals):
    total_profit_loss = 0
    results = []

    for signal in signals:
        trade_data = data.iloc[signal['entry_index'] + 1:]  # Trade starts after signal index
        if trade_data.empty:
            break

        for _, row in trade_data.iterrows():
            if signal['type'] == "BUY":
                if row['high'] >= signal['take_profit']:
                    profit = (signal['take_profit'] - signal['entry_price']) / signal['entry_price'] * TRADE_SIZE
                    total_profit_loss += profit
                    results.append({'signal': 'BUY', 'result': 'TP', 'profit_loss': profit})
                    break
                elif row['low'] <= signal['stop_loss']:
                    loss = (signal['entry_price'] - signal['stop_loss']) / signal['entry_price'] * TRADE_SIZE
                    total_profit_loss -= loss
                    results.append({'signal': 'BUY', 'result': 'SL', 'profit_loss': -loss})
                    break
            elif signal['type'] == "SELL":
                if row['low'] <= signal['take_profit']:
                    profit = (signal['entry_price'] - signal['take_profit']) / signal['entry_price'] * TRADE_SIZE
                    total_profit_loss += profit
                    results.append({'signal': 'SELL', 'result': 'TP', 'profit_loss': profit})
                    break
                elif row['high'] >= signal['stop_loss']:
                    loss = (signal['stop_loss'] - signal['entry_price']) / signal['entry_price'] * TRADE_SIZE
                    total_profit_loss -= loss
                    results.append({'signal': 'SELL', 'result': 'SL', 'profit_loss': -loss})
                    break
    return results, total_profit_loss

# Main Function
def backtest():
    data = fetch_historical_data(SYMBOL, GRANULARITY)
    highs, lows = identify_extrema(data)
    waves = validate_elliott_wave(data, highs, lows)
    signals = analyze_signals(data, waves)
    results, total_profit_loss = backtest_signals(data, signals)
    return results, total_profit_loss

# Run Backtest
if __name__ == "__main__":
    trade_results, total_profit_loss = backtest()
    for trade in trade_results:
        print(f"Signal: {trade['signal']} | Result: {trade['result']} | Profit/Loss: ${trade['profit_loss']:.2f}")
    print(f"Total Profit/Loss: ${total_profit_loss:.2f}")
