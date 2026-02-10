import requests
import pandas as pd
from ta.volatility import AverageTrueRange

# Configuration Parameters
SYMBOL = "BTC-USD"
TRADE_SIZE = 1000  # Each trade is $1,000
ATR_LENGTH = 14
RISK_REWARD_RATIO = 2.0

# Fetch Historical Data
def fetch_historical_data():
    url = f"https://api.exchange.coinbase.com/products/{SYMBOL}/candles?granularity=900"  # 15-minute intervals
    response = requests.get(url)
    if response.status_code == 200:
        columns = ["time", "low", "high", "open", "close", "volume"]
        data = pd.DataFrame(response.json(), columns=columns)
        data = data.sort_values(by="time").astype(float)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        return data
    else:
        raise Exception(f"Failed to fetch data for {SYMBOL}: {response.status_code}")

# Calculate ATR
def calculate_atr(data):
    atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=ATR_LENGTH)
    data['atr'] = atr_indicator.average_true_range()
    return data

# Simulate Trades
def simulate_trades(data):
    results = []
    total_profit_loss = 0

    for i in range(1, len(data)):
        row = data.iloc[i]
        prev_row = data.iloc[i - 1]
        atr = prev_row['atr']
        close_price = prev_row['close']

        # Example Buy Logic: If low is below a certain level
        long_signal = row['close'] < close_price - (0.01 * close_price)
        short_signal = row['close'] > close_price + (0.01 * close_price)

        if long_signal:
            tp1 = close_price + (atr * (RISK_REWARD_RATIO - 0.5))
            tp2 = close_price + (atr * RISK_REWARD_RATIO)
            sl = close_price - (atr * 0.5)
            result = evaluate_trade(data[i:], 'LONG', close_price, tp1, tp2, sl)
            results.append(result)
            total_profit_loss += result['profit_loss']

        elif short_signal:
            tp1 = close_price - (atr * (RISK_REWARD_RATIO - 0.5))
            tp2 = close_price - (atr * RISK_REWARD_RATIO)
            sl = close_price + (atr * 0.5)
            result = evaluate_trade(data[i:], 'SHORT', close_price, tp1, tp2, sl)
            results.append(result)
            total_profit_loss += result['profit_loss']

    return results, total_profit_loss

# Evaluate Individual Trade
def evaluate_trade(trade_data, signal_type, entry, tp1, tp2, sl):
    for _, row in trade_data.iterrows():
        if signal_type == 'LONG':
            if row['high'] >= tp1:
                profit = (tp1 - entry) / entry * TRADE_SIZE
                return {'signal': 'LONG', 'result': 'TP1', 'profit_loss': profit}
            elif row['low'] <= sl:
                loss = (entry - sl) / entry * TRADE_SIZE
                return {'signal': 'LONG', 'result': 'SL', 'profit_loss': -loss}
        elif signal_type == 'SHORT':
            if row['low'] <= tp1:
                profit = (entry - tp1) / entry * TRADE_SIZE
                return {'signal': 'SHORT', 'result': 'TP1', 'profit_loss': profit}
            elif row['high'] >= sl:
                loss = (sl - entry) / entry * TRADE_SIZE
                return {'signal': 'SHORT', 'result': 'SL', 'profit_loss': -loss}

    return {'signal': signal_type, 'result': 'NO RESULT', 'profit_loss': 0}

# Backtest
def backtest():
    data = fetch_historical_data()
    data = calculate_atr(data)
    results, total_profit_loss = simulate_trades(data)
    return results, total_profit_loss

# Run Backtest
if __name__ == "__main__":
    trade_results, total_profit_loss = backtest()
    for trade in trade_results:
        print(f"Signal: {trade['signal']} | Result: {trade['result']} | Profit/Loss: ${trade['profit_loss']:.2f}")
    print(f"Total Profit/Loss: ${total_profit_loss:.2f}")
