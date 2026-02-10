import websocket, json, time
import pandas as pd
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# === CONFIGURATION ===
PRODUCT_IDS = [   "BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "DOGE-USD", "ADA-USD", "XLM-USD", "SUI-USD",
    "BCH-USD", "LINK-USD", "HBAR-USD", "LTC-USD", "AVAX-USD", "SHIB-USD", "UNI-USD", "DOT-USD",
    "CRO-USD",  "AAVE-USD", "NEAR-USD", "ETC-USD", "ONDO-USD",
    "APT-USD", "ICP-USD", "ALGO-USD", "ARB-USD", "VET-USD",
    "ATOM-USD", "POL-USD", "BONK-USD", "RENDER-USD",
    "FET-USD", "SEI-USD", "FLR-USD", "FIL-USD", "QNT-USD", "LSETH-USD", "INJ-USD",
    "CRV-USD", "STX-USD", "TIA-USD"]  # Coins to track

CANDLE_INTERVAL = 300   # 5 min candles for entries
HTF_INTERVAL = 3600     # 1 hour candles for confirmation
BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"

USE_6H_FILTER = False   # set False to disable 6h trend filter


# === STRATEGY FILTER SETTINGS ===
RSI_LONG_MIN     = 60     # min RSI for long
RSI_SHORT_MAX    = 40     # max RSI for short
VOLUME_MULT      = 1.5    # volume > this × 20-bar average
BREAKOUT_LOOKBACK = 50    # bars to look back for breakout
MIN_ATR_PCT      = 0.002  # min ATR as % of price (0.002 = 0.2%)
EMA_SLOPE_BARS   = 5      # bars to measure EMA slope
MIN_EMA_SLOPE_PCT = 0.001
EMA_GAP_MIN_PCT = 0.005     # EMA gap must be at least 0.5% of price
ADX_MIN = 20                # min ADX for trend strength
MIN_RR = 1.5                # min reward:risk ratio
SESSION_FILTER = True # min EMA slope as % of price (0.001 = 0.1%)



#TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
#CHAT_ID = "7967738614"

#personal chat id:"7967738614"
#channel chat id:
#"-4916911067"
# === DATA STORAGE ===
ohlcv_data_5m = {coin: pd.DataFrame(columns=["time","open","high","low","close","volume"]) for coin in PRODUCT_IDS}
ohlcv_data_1h = {coin: pd.DataFrame(columns=["time","open","high","low","close","volume"]) for coin in PRODUCT_IDS}
ohlcv_data_6h = {coin: pd.DataFrame(columns=["time","open","high","low","close","volume"]) for coin in PRODUCT_IDS}
current_candle_6h = {coin: None for coin in PRODUCT_IDS}

current_candle_5m = {coin: None for coin in PRODUCT_IDS}
current_candle_1h = {coin: None for coin in PRODUCT_IDS}

# === TECHNICAL INDICATORS ===
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr(high, low, close, period=14):
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_val = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_val)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(window=period).mean()

def in_active_session(dt):
    """Return True if time is during London or New York session."""
    hour = dt.hour
    return (7 <= hour < 17) or (12 <= hour < 22)  # London & NY approx UTC


def format_price(price):
    if price >= 100:
        return f"{price:.2f}"      # e.g., BTC
    elif price >= 1:
        return f"{price:.4f}"      # e.g., LTC, SOL
    elif price >= 0.01:
        return f"{price:.6f}"      # e.g., DOGE
    elif price >= 0.0001:
        return f"{price:.8f}"      # e.g., SHIB
    else:
        return f"{price:.10f}"     # ultra-low priced tokens


# === TELEGRAM ALERT ===
def send_alert(symbol, signal, entry, tp, sl):
    msg = (
        f"🚨 {signal} {symbol}\n"
        f"Entry: {format_price(entry)}\n"
        f"TP: {format_price(tp)}\n"
        f"SL: {format_price(sl)}\n"
        f"Reason: Multi-timeframe Trend + Momentum + Volume + Breakout"
    )
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
        )
    except Exception as e:
        print(f"Telegram send error: {e}")


# === STRATEGY CHECK ===
def trend_direction(df):
    if len(df) < 55:
        return None
    df['ema_fast'] = ema(df['close'], 21)
    df['ema_slow'] = ema(df['close'], 55)
    last = df.iloc[-1]
    if last['ema_fast'] > last['ema_slow']:
        return "BULL"
    elif last['ema_fast'] < last['ema_slow']:
        return "BEAR"
    return None

def generate_signal(df_5m, df_1h, df_6h=None):
    if len(df_5m) < 55 or len(df_1h) < 55:
        return None
    if df_6h is not None and len(df_6h) < 55:
        return None

    # Trend agreement
    trend_5m = trend_direction(df_5m)
    trend_1h = trend_direction(df_1h)
    if df_6h is not None and USE_6H_FILTER:
        trend_6h = trend_direction(df_6h)
        if not (trend_5m == trend_1h == trend_6h):
            return None
    else:
        if trend_5m != trend_1h:
            return None

    # [rest of your strict filters here...]


    # Only trade in active sessions
    if SESSION_FILTER and not in_active_session(df_5m.iloc[-1]['time'].to_pydatetime()):
        return None

    # Indicators on 5m
    df_5m['ema_fast'] = ema(df_5m['close'], 21)
    df_5m['ema_slow'] = ema(df_5m['close'], 55)
    df_5m['rsi'] = rsi(df_5m['close'], 14)
    df_5m['atr'] = atr(df_5m['high'], df_5m['low'], df_5m['close'], 14)
    df_5m['vol_ma'] = df_5m['volume'].rolling(20).mean()
    df_5m['adx'] = adx(df_5m['high'], df_5m['low'], df_5m['close'], 14)

    last = df_5m.iloc[-1]

    # ATR filter
    if last['atr'] < last['close'] * MIN_ATR_PCT:
        return None

    # EMA gap filter
    ema_gap_pct = abs(last['ema_fast'] - last['ema_slow']) / last['close']
    if ema_gap_pct < EMA_GAP_MIN_PCT:
        return None

    # ADX filter
    if last['adx'] < ADX_MIN:
        return None

    # Breakout filter
    prev_high = df_5m['high'].iloc[-BREAKOUT_LOOKBACK:-1].max()
    prev_low = df_5m['low'].iloc[-BREAKOUT_LOOKBACK:-1].min()

    # Long condition
    long_cond = (
        trend_5m == "BULL" and
        last['rsi'] > RSI_LONG_MIN and
        last['volume'] > VOLUME_MULT * last['vol_ma'] and
        last['close'] > prev_high
    )

    # Short condition
    short_cond = (
        trend_5m == "BEAR" and
        last['rsi'] < RSI_SHORT_MAX and
        last['volume'] > VOLUME_MULT * last['vol_ma'] and
        last['close'] < prev_low
    )

    if long_cond:
        entry = last['close']
        tp = entry + 1.5 * last['atr']
        sl = entry - 1.0 * last['atr']
        if (tp - entry) / (entry - sl) < MIN_RR:
            return None
        return "LONG", entry, tp, sl

    elif short_cond:
        entry = last['close']
        tp = entry - 1.5 * last['atr']
        sl = entry + 1.0 * last['atr']
        if (entry - tp) / (sl - entry) < MIN_RR:
            return None
        return "SHORT", entry, tp, sl

    return None



# === CANDLE BUILDER ===
def update_candle(symbol, price, size, trade_time, tf_data, current_candle, interval):
    if current_candle[symbol] is None:
        candle_start = trade_time - timedelta(seconds=trade_time.second % interval,
                                              microseconds=trade_time.microsecond)
        current_candle[symbol] = {
            "start": candle_start,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": size
        }
        return

    candle = current_candle[symbol]
    if trade_time >= candle["start"] + timedelta(seconds=interval):
        tf_data[symbol] = pd.concat(
            [tf_data[symbol],
             pd.DataFrame([[candle["start"], candle["open"], candle["high"],
                            candle["low"], candle["close"], candle["volume"]]],
                          columns=["time","open","high","low","close","volume"])],
            ignore_index=True
        )

        # Only check for signals on 5m candle close
        if interval == CANDLE_INTERVAL:
            signal = generate_signal(ohlcv_data_5m[symbol], ohlcv_data_1h[symbol], ohlcv_data_6h[symbol] if USE_6H_FILTER else None)

            if signal:
                s_type, entry, tp, sl = signal
                send_alert(symbol, s_type, entry, tp, sl)

        # Start new candle
        candle_start = trade_time - timedelta(seconds=trade_time.second % interval,
                                              microseconds=trade_time.microsecond)
        current_candle[symbol] = {
            "start": candle_start,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": size
        }
    else:
        candle["high"] = max(candle["high"], price)
        candle["low"] = min(candle["low"], price)
        candle["close"] = price
        candle["volume"] += size

# === WEBSOCKET CALLBACKS ===
def on_message(ws, message):
    data = json.loads(message)
    if data['type'] == 'ticker' and 'price' in data:
        symbol = data['product_id']
        price = float(data['price'])
        size = float(data.get('last_size', 0) or 0)
        trade_time = datetime.fromisoformat(data['time'].replace('Z', '+00:00'))

        update_candle(symbol, price, size, trade_time, ohlcv_data_5m, current_candle_5m, CANDLE_INTERVAL)
        update_candle(symbol, price, size, trade_time, ohlcv_data_1h, current_candle_1h, HTF_INTERVAL)
        update_candle(symbol, price, size, trade_time, ohlcv_data_6h, current_candle_6h, 21600)


def on_open(ws):
    ws.send(json.dumps({"type": "subscribe", "channels": [{"name": "ticker", "product_ids": PRODUCT_IDS}]}))

def on_error(ws, error):
    print(f"ERROR: {error}")

def on_close(ws):
    print("WebSocket closed, reconnecting...")
    time.sleep(5)
    start_ws()
    
def fetch_historical(product_id, granularity, limit=60):
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {"granularity": granularity}
    r = requests.get(url, params=params)
    if r.status_code != 200:
        print(f"Error fetching {product_id} history: {r.text}")
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])
    data = r.json()
    df = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df[["time","open","high","low","close","volume"]]

def preload_history():
    print("Preloading historical data...")
    for coin in PRODUCT_IDS:
        # 5m history
        df_5m = fetch_historical(coin, CANDLE_INTERVAL)
        ohlcv_data_5m[coin] = df_5m.copy()
        # 1h history
        df_1h = fetch_historical(coin, HTF_INTERVAL)
        ohlcv_data_1h[coin] = df_1h.copy()
        # 6h history
        if USE_6H_FILTER:
            df_6h = fetch_historical(coin, 21600)
            ohlcv_data_6h[coin] = df_6h.copy()

    print("Historical preload complete.")
    

def start_ws():
    ws = websocket.WebSocketApp(
        "wss://ws-feed.exchange.coinbase.com",
        on_message=on_message,
        on_open=on_open,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

if __name__ == "__main__":
    preload_history()
    start_ws()
