import requests
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# Configuration Parameters
SYMBOLS = ["btc-usd", "xrp-usd", "ltc-usd", 
           "doge-usd", "sol-usd", 
            "magic-usd",  "wld-usd",
           "fartcoin-usd", "aero-usd", "link-usd","hbar-usd",
            "crv-usd","tao-usd", "avax-usd", "xcn-usd",
           "uni-usd", "mkr-usd", "toshi-usd", "near-usd",
           "bch-usd", "xlm-usd", "moodeng-usd", 
           "dot-usd",  "arb-usd", "icp-usd",
           "qnt-usd",  "ip-usd" , "pnut-usd",  "vet-usd",
            "turbo-usd", "bera-usd", "pol-usd",  "ach-usd",
           "pyth-usd",  
           "axs-usd"]
RSI_PERIOD = 14
ATR_PERIOD = 14
OVERSOLD = 25
OVERBOUGHT = 75
VOLUME_MULTIPLIER = 1.1
TAKE_PROFIT_MULTIPLIER = 1.5
STOP_LOSS_MULTIPLIER = 0.5
TRADE_SIZE = 10000

# Fetch Historical Data
def fetch_historical_data(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity=3600"
    response = requests.get(url)
    if response.status_code == 200:
        columns = ["time", "low", "high", "open", "close", "volume"]
        data = pd.DataFrame(response.json(), columns=columns)
        data = data.sort_values(by="time").astype(float)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        return data
    else:
        raise Exception(f"Failed to fetch data for {symbol}: {response.status_code}")

# Calculate Indicators
def calculate_indicators(data):
    data['rsi'] = RSIIndicator(close=data['close'], window=RSI_PERIOD).rsi()
    atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=ATR_PERIOD)
    data['atr'] = atr_indicator.average_true_range()
    data['avg_volume'] = data['volume'].rolling(window=RSI_PERIOD).mean()
    return data

# Analyze Signals
def analyze_signals(data):
    signals = []
    for i in range(len(data)):
        row = data.iloc[i]
        close_price = row['close']
        atr = row['atr']
        rsi = row['rsi']
        volume = row['volume']
        avg_volume = row['avg_volume']

        long_signal = rsi < OVERSOLD and volume > VOLUME_MULTIPLIER * avg_volume
        short_signal = rsi > OVERBOUGHT and volume > VOLUME_MULTIPLIER * avg_volume

        if long_signal:
            tp = close_price + (atr * TAKE_PROFIT_MULTIPLIER)
            sl = close_price - (atr * STOP_LOSS_MULTIPLIER)
            signals.append({
                'time': row['time'],
                'signal': 'LONG',
                'entry': close_price,
                'take_profit': tp,
                'stop_loss': sl
            })
        elif short_signal:
            tp = close_price - (atr * TAKE_PROFIT_MULTIPLIER)
            sl = close_price + (atr * STOP_LOSS_MULTIPLIER)
            signals.append({
                'time': row['time'],
                'signal': 'SHORT',
                'entry': close_price,
                'take_profit': tp,
                'stop_loss': sl
            })
    return signals

# Simulate Trades
def simulate_trades(data, signals):
    total_profit_loss = 0
    results = []

    for signal in signals:
        trade_data = data[data['time'] > signal['time']]
        if trade_data.empty:
            break

        for _, row in trade_data.iterrows():
            if signal['signal'] == 'LONG':
                if row['high'] >= signal['take_profit']:
                    profit = (signal['take_profit'] - signal['entry']) / signal['entry'] * TRADE_SIZE
                    total_profit_loss += profit
                    results.append({'time': signal['time'], 'signal': 'LONG', 'result': 'TP', 'profit_loss': profit})
                    break
                elif row['low'] <= signal['stop_loss']:
                    loss = (signal['entry'] - signal['stop_loss']) / signal['entry'] * TRADE_SIZE
                    total_profit_loss -= loss
                    results.append({'time': signal['time'], 'signal': 'LONG', 'result': 'SL', 'profit_loss': -loss})
                    break
            elif signal['signal'] == 'SHORT':
                if row['low'] <= signal['take_profit']:
                    profit = (signal['entry'] - signal['take_profit']) / signal['entry'] * TRADE_SIZE
                    total_profit_loss += profit
                    results.append({'time': signal['time'], 'signal': 'SHORT', 'result': 'TP', 'profit_loss': profit})
                    break
                elif row['high'] >= signal['stop_loss']:
                    loss = (signal['stop_loss'] - signal['entry']) / signal['entry'] * TRADE_SIZE
                    total_profit_loss -= loss
                    results.append({'time': signal['time'], 'signal': 'SHORT', 'result': 'SL', 'profit_loss': -loss})
                    break

    return results, total_profit_loss

# Backtest
def backtest():
    all_results = {}
    total_profit_loss = 0

    for symbol in SYMBOLS:
        try:
            print(f"Processing {symbol}...")
            data = fetch_historical_data(symbol)
            data = calculate_indicators(data)
            signals = analyze_signals(data)
            trade_results, symbol_profit_loss = simulate_trades(data, signals)
            all_results[symbol] = {
                'trades': trade_results,
                'profit_loss': symbol_profit_loss
            }
            total_profit_loss += symbol_profit_loss
        except Exception as e:
            print(f"Error with {symbol}: {str(e)}")
    
    return all_results, total_profit_loss

# Run Backtest
if __name__ == "__main__":
    all_results, total_profit_loss = backtest()
    for symbol, results in all_results.items():
        print(f"\n--- {symbol} ---")
        for trade in results['trades']:
            print(f"Time: {trade['time']} | Signal: {trade['signal']} | Result: {trade['result']} | Profit/Loss: ${trade['profit_loss']:.2f}")
        print(f"Total Profit/Loss for {symbol}: ${results['profit_loss']:.2f}")
    print(f"\nTotal Profit/Loss across all symbols: ${total_profit_loss:.2f}")
