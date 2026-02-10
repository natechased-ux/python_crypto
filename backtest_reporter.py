import requests
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import pytz
import schedule
import time

# --- Config ---
COINS = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'WIF-USD', 'XCN-USD',
         'ONDO-USD', 'ENA-USD', 'WLD-USD', 'SEI-USD', 'DOGE-USD', 'SUI-USD']
TELEGRAM_TOKEN = '8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw'
CHAT_ID = '7967738614'
PST = pytz.timezone("America/Los_Angeles")
WEIGHT_FILE = 'weights.json'
LOG_FILE = 'forecast_log.csv'

# --- Weight Management ---
def load_weights():
    if os.path.exists(WEIGHT_FILE):
        with open(WEIGHT_FILE, 'r') as f:
            return json.load(f)
    return {
        'ema_crossover': 1.3, 'stoch_cross': 1.1, 'macd_momentum': 0.9,
        'volatility_compression': 0.6, 'support_resistance': 0.7,
        'price_action': 0.5, 'daily_trend': 0.6, 'relative_strength': 0.8,
        'inside_bar': 0.4, 'bull_flag': 0.5, 'fvg': 0.45
    }

def save_weights(weights):
    with open(WEIGHT_FILE, 'w') as f:
        json.dump(weights, f)

WEIGHTS = load_weights()

# --- Data Fetching ---
def fetch_candles(symbol, granularity='HOUR', limit=200):
    granularity_map = {
        'HOUR': 3600,
        'DAY': 86400
    }
    gran = granularity_map[granularity]
    end = datetime.utcnow()
    start = end - timedelta(seconds=gran * limit)
    url = (
        f"https://api.exchange.coinbase.com/products/{symbol}/candles"
        f"?start={start.isoformat()}&end={end.isoformat()}&granularity={gran}"
    )
    r = requests.get(url)
    if r.status_code != 200:
        raise Exception(f"Failed to fetch {symbol}: {r.status_code} {r.text}")
    data = r.json()
    df = pd.DataFrame(data, columns=['time','low','high','open','close','volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('time').reset_index(drop=True)
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    return df

# --- Indicators ---
def ema_crossover_score(df):
    ema_short = df['close'].ewm(span=12).mean()
    ema_long = df['close'].ewm(span=26).mean()
    return float((ema_short.iloc[-1] > ema_long.iloc[-1]) * 1.0)

def stoch_cross_score(df):
    low14 = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    k = 100 * ((df['close'] - low14) / (high14 - low14))
    d = k.rolling(3).mean()
    return float((k.iloc[-2] < d.iloc[-2]) and (k.iloc[-1] > d.iloc[-1]))

def macd_momentum_score(df):
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return float((macd.iloc[-1] > signal.iloc[-1]) * 1.0)

def volatility_compression_score(df):
    atr = df['high'].rolling(14).max() - df['low'].rolling(14).min()
    current_atr = atr.iloc[-1]
    mean_atr = atr.mean()
    return float(current_atr < mean_atr)

def support_resistance_score(df):
    recent = df['close'].tail(20)
    sr_level = recent.mean()
    price = df['close'].iloc[-1]
    return float(abs(price - sr_level) < 0.01 * price)

def price_action_score(df):
    return float(df['close'].iloc[-1] > df['open'].iloc[-1])

def daily_trend_score(df_daily):
    return float(df_daily['close'].iloc[-1] > df_daily['close'].iloc[0])

def inside_bar_score(df):
    return float((df['high'].iloc[-1] < df['high'].iloc[-2]) and (df['low'].iloc[-1] > df['low'].iloc[-2]))

def bull_flag_score(df):
    up = df['close'].pct_change().rolling(5).mean().iloc[-1]
    down = df['close'].pct_change().rolling(5).mean().iloc[-3]
    return float((up > 0.01) and (down < -0.005))

def fvg_score(df):
    gaps = df['high'].shift(1) < df['low']
    return float(gaps.iloc[-1])

# --- Probability Clamping ---
def clamp_probability(value):
    if not np.isfinite(value):
        return 0.5
    return max(0.0, min(1.0, value))

# --- Telegram Alerting ---
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

# --- Forecast Logging ---
def log_forecast(symbol, bull_prob, bear_prob):
    timestamp = datetime.now(PST).strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a') as f:
        f.write(f"{timestamp},{symbol},{bull_prob:.2f},{bear_prob:.2f}\n")

# --- Forecast Computation ---
def compute_forecast(symbol):
    try:
        df_hourly = fetch_candles(symbol, granularity='HOUR')
        df_daily = fetch_candles(symbol, granularity='DAY', limit=30)

        scores = {
            'ema_crossover': ema_crossover_score(df_hourly),
            'stoch_cross': stoch_cross_score(df_hourly),
            'macd_momentum': macd_momentum_score(df_hourly),
            'volatility_compression': volatility_compression_score(df_hourly),
            'support_resistance': support_resistance_score(df_hourly),
            'price_action': price_action_score(df_hourly),
            'daily_trend': daily_trend_score(df_daily),
            'inside_bar': inside_bar_score(df_hourly),
            'bull_flag': bull_flag_score(df_hourly),
            'fvg': fvg_score(df_hourly)
        }

        weighted_sum = sum(WEIGHTS.get(k, 0) * v for k, v in scores.items())
        total_weight = sum(WEIGHTS.get(k, 0) for k in scores)
        bull_prob = clamp_probability(weighted_sum / total_weight)
        bear_prob = clamp_probability(1.0 - bull_prob)

        price = df_hourly['close'].iloc[-1]
        prediction = price * (1 + (bull_prob - bear_prob) * 0.02)

        log_forecast(symbol, bull_prob, bear_prob)

        trend = "Bullish" if scores['daily_trend'] > 0 else "Bearish"
        decimals = 4 if price < 1 else 2
        send_telegram_message(
            f"{symbol} Trend: {trend}\n"
            f"Bull: {bull_prob:.2f}, Bear: {bear_prob:.2f}\n"
            f"Price: ${price:.{decimals}f} → Prediction: ${prediction:.{decimals}f}"
        )

    except Exception as e:
        print(f"Forecast failed for {symbol}: {e}")

# --- Scheduled Forecasting ---
def forecast_all():
    for symbol in COINS:
        compute_forecast(symbol)

schedule.every(4).hours.do(forecast_all)

if __name__ == "__main__":
    print("Starting Adaptive Crypto Forecaster...")
    forecast_all()
    while True:
        schedule.run_pending()
        time.sleep(60)
