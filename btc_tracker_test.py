import requests
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# Configuration Parameters
SYMBOLS = ["btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd",
           "doge-usd", "sol-usd", "wif-usd", "ondo-usd",
           "sei-usd", "magic-usd", "ape-usd", "jasmy-usd", "wld-usd",
           "syrup-usd","fartcoin-usd", "aero-usd", "link-usd","hbar-usd",
           "aave-usd", "fet-usd", "crv-usd","tao-usd", "avax-usd", "xcn-usd",
           "uni-usd", "mkr-usd", "toshi-usd", "near-usd", "algo-usd", "trump-usd",
           "bch-usd", "inj-usd", "pepe-usd", "xlm-usd", "moodeng-usd", "bonk-usd",
           "dot-usd", "popcat-usd", "arb-usd", "icp-usd",
           "qnt-usd", "tia-usd", "ip-usd" , "pnut-usd", "apt-usd", "vet-usd",
           "ena-usd", "turbo-usd", "bera-usd", "pol-usd", "mask-usd", "ach-usd",
           "pyth-usd", "sand-usd", "morpho-usd", "mana-usd", "velo-usd", "coti-usd",
           "axs-usd"]
RSI_PERIOD = 14
ATR_PERIOD = 14
OVERSOLD = 25
OVERBOUGHT = 75
VOLUME_MULTIPLIER = 1.1
TAKE_PROFIT_MULTIPLIER = 5
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
            signals.append({'entry': close_price, 'take_profit': tp, 'stop_loss': sl, 'signal': 'LONG'})
        elif short_signal:
            tp = close_price - (atr * TAKE_PROFIT_MULTIPLIER)
            sl = close_price + (atr * STOP_LOSS_MULTIPLIER)
            signals.append({'entry': close_price, 'take_profit': tp, 'stop_loss': sl, 'signal': 'SHORT'})
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
    results = {}
    total_profit_loss = 0

    for symbol in SYMBOLS:
        try:
            data = fetch_historical_data(symbol)
            data = calculate_indicators(data)
            signals = analyze_signals(data)
            symbol_profit_loss = simulate_trades(data, signals)
            results[symbol] = symbol_profit_loss
            total_profit_loss += symbol_profit_loss
        except Exception as e:
            results[symbol] = f"Error: {str(e)}"

    return results, total_profit_loss

# Run Backtest
if __name__ == "__main__":
    results, total_profit_loss = backtest()
    for symbol, profit_loss in results.items():
        print(f"{symbol}: Total Profit/Loss = ${profit_loss:.2f}" if isinstance(profit_loss, (float, int)) else f"{symbol}: {profit_loss}")
    print(f"\nOverall Total Profit/Loss: ${total_profit_loss:.2f}")
