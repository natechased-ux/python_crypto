import os
import pytz
import torch
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from telegram import Bot

# ==== CONFIG ====
COINS = ["BTC-USD", "ETH-USD", "XRP-USD"]  # Add all coins you trained
SEQ_LEN = 60
CONF_THRESHOLD = 0.70
MODELS_ENTRY_PATH = "models_entry_1h"
MODELS_TP_SL_PATH = "models_tp_sl_1h"
CHAT_ID = "7967738614"
BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"

#TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
#CHAT_ID = "-4916911067"

#personal chat id:"7967738614"
#channel chat id:
#"-4916911067"

FEATURES = [
    'low', 'high', 'open', 'close', 'volume',
    'rsi', 'adx', 'plus_di', 'minus_di', 'macd_diff',
    'ema10', 'ema10_slope', 'ema20', 'ema20_slope',
    'ema50', 'ema50_slope', 'ema200', 'ema200_slope',
    'atr', 'vol_change', 'body_wick_ratio', 'above_ema200',
    'fib_long_0', 'fib_long_236', 'fib_long_382', 'fib_long_5', 'fib_long_618', 'fib_long_786', 'fib_long_1',
    'fib_med_0', 'fib_med_236', 'fib_med_382', 'fib_med_5', 'fib_med_618', 'fib_med_786', 'fib_med_1',
    'fib_short_0', 'fib_short_236', 'fib_short_382', 'fib_short_5', 'fib_short_618', 'fib_short_786', 'fib_short_1',
    'btc_close', 'btc_return', 'coin_return', 'rel_strength', 'btc_dominance'
]

# ==== TELEGRAM ====
bot = Bot(token=BOT_TOKEN)
last_sent_signals = set()

# ==== MODEL DEFINITIONS ====
class LSTMEntryModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

class LSTMTpSlModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# ==== DATA HELPERS ====
def fetch_coingecko_btcd():
    """Fetch latest BTC dominance from Coingecko"""
    url = "https://api.coingecko.com/api/v3/global"
    try:
        r = requests.get(url, timeout=10).json()
        return r["data"]["market_cap_percentage"]["btc"] / 100
    except:
        return np.nan

def fetch_candles(symbol, granularity=3600, limit=200):
    """Fetch candles from Coinbase Pro API"""
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None
    data = pd.DataFrame(r.json(), columns=["time", "low", "high", "open", "close", "volume"])
    data["time"] = pd.to_datetime(data["time"], unit="s", utc=True)
    return data.sort_values("time")

import pandas as pd
import numpy as np
import requests
import pytz
from datetime import datetime, timedelta
import ta  # Technical Analysis library

# =========================
# 📌 LIVE BTC & BTC.D FETCH
# =========================
def fetch_btc_close(granularity=3600, lookback_hours=720):
    """Fetch recent BTC close prices from Coinbase."""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=lookback_hours)
    url = f"https://api.exchange.coinbase.com/products/BTC-USD/candles"
    params = {
        "granularity": granularity,
        "start": start_time.isoformat(),
        "end": end_time.isoformat()
    }
    r = requests.get(url)
    if r.status_code != 200:
        print(f"❌ Failed to fetch BTC: {r.text}")
        return None
    data = pd.DataFrame(r.json(), columns=["time", "low", "high", "open", "close", "volume"])
    data["time"] = pd.to_datetime(data["time"], unit="s", utc=True)
    data.sort_values("time", inplace=True)
    data["btc_return"] = data["close"].pct_change()
    return data[["time", "close", "btc_return"]].rename(columns={"close": "btc_close"})

import pandas as pd
import requests
from datetime import datetime, timezone

def fetch_btcd_data():
    """
    Fetch current BTC Dominance % from CoinGecko API
    and return as a DataFrame with columns ['time', 'btc_dominance'].
    """
    try:
        url = "https://api.coingecko.com/api/v3/global"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        # Extract BTC dominance
        btc_dominance = data.get("data", {}).get("market_cap_percentage", {}).get("btc", None)

        if btc_dominance is None:
            print("⚠️ BTC Dominance not found in API response")
            return pd.DataFrame(columns=["time", "btc_dominance"])

        # Make DataFrame with current UTC timestamp
        now_utc = datetime.now(timezone.utc)
        df = pd.DataFrame([{
            "time": now_utc,
            "btc_dominance": btc_dominance
        }])

        return df

    except Exception as e:
        print(f"❌ Error fetching BTC dominance: {e}")
        return pd.DataFrame(columns=["time", "btc_dominance"])


# =========================
# 📌 TECHNICAL FEATURES
# =========================
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical features for trading model."""
    df = df.copy()

    # RSI, ADX, MACD
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()
    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd_diff"] = macd.macd_diff()

    # EMAs + slopes
    for length in [10, 20, 50, 200]:
        ema = ta.trend.EMAIndicator(df["close"], window=length).ema_indicator()
        df[f"ema{length}"] = ema
        df[f"ema{length}_slope"] = ema.pct_change()

    # ATR
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

    # Volume change
    df["vol_change"] = df["volume"].pct_change()

    # Candle body/wick ratio
    body = (df["close"] - df["open"]).abs()
    wick = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_wick_ratio"] = body / wick

    # Above EMA200
    df["above_ema200"] = (df["close"] > df["ema200"]).astype(int)

    return df

# =========================
# 📌 FIBONACCI LEVELS
# =========================
def calculate_fib_levels(df: pd.DataFrame, lookback: int, prefix: str) -> pd.DataFrame:
    """Calculate Fibonacci retracement levels."""
    df = df.copy()
    fib_cols = []
    for i in range(len(df)):
        if i < lookback:
            for level in ["0", "236", "382", "5", "618", "786", "1"]:
                df.loc[i, f"{prefix}_{level}"] = np.nan
            continue
        window = df.iloc[i-lookback:i]
        high = window["high"].max()
        low = window["low"].min()
        diff = high - low
        levels = {
            "0": high,
            "236": high - 0.236 * diff,
            "382": high - 0.382 * diff,
            "5": high - 0.5 * diff,
            "618": high - 0.618 * diff,
            "786": high - 0.786 * diff,
            "1": low
        }
        for k, v in levels.items():
            df.loc[i, f"{prefix}_{k}"] = v
            fib_cols.append(f"{prefix}_{k}")
    return df

# =========================
# 📌 FINAL MERGE FUNCTION
# =========================
def prepare_coin_features(symbol, btc_df, btcd_df):
    """
    Prepares feature DataFrame for a given coin.
    Merges with BTC close/returns and BTC dominance history.
    """

    # Fetch coin OHLCV
    df = fetch_candles(symbol, granularity=3600, limit=200)
    if df is None or df.empty:
        return None

    # Technical indicators
    df = calculate_features(df)

    # Fibonacci levels
    df = calculate_fib_levels(df, 720, "fib_long")
    df = calculate_fib_levels(df, 360, "fib_med")
    df = calculate_fib_levels(df, 168, "fib_short")

    # Merge with BTC close & returns
    df = df.merge(btc_df, on="time", how="left")

    # Coin returns & relative strength
    df["coin_return"] = df["close"].pct_change()
    df["rel_strength"] = df["coin_return"] - df["btc_return"]

    # Merge with BTC dominance history
    btcd_df = btcd_df.rename(columns={"value": "btc_dominance"})  # if not already renamed
    df = df.merge(btcd_df, on="time", how="left")

    # Fill forward for any missing BTC dominance values
    df["btc_dominance"].fillna(method="ffill", inplace=True)

    df.dropna(inplace=True)
    return df



# ==== SIGNAL CHECK ====
def check_ai_signals():
    print(f"\n🔍 Checking AI signals at {datetime.utcnow().isoformat()} UTC")
    all_signals = []
    market_outputs = []

    # === Fetch BTC data once ===
    btc_df = fetch_candles("BTC-USD", granularity=3600)
    btc_df = calculate_features(btc_df)
    btc_df['btc_return'] = btc_df['close'].pct_change()
    btc_df = btc_df[['time', 'close', 'btc_return']].rename(columns={'close': 'btc_close'})

    # === Fetch BTC Dominance once ===
    btcd_df = fetch_btcd_data()  # <- Make sure you have a function for this
    #btcd_df = btcd_df.rename(columns={'value': 'btc_dominance'})  # if needed

    for symbol in COINS:
        print(f"\n📌 Processing {symbol.upper()} ...")

        # === Build full feature set for this coin ===
        df = prepare_coin_features(symbol, btc_df, btcd_df)
        if df is None or len(df) < SEQ_LEN + 1:
            print(f"❌ Not enough data for {symbol}")
            continue

        # === Drop incomplete candle ===
        last_candle_time = df["time"].iloc[-1]
        if last_candle_time.tzinfo is None:
            last_candle_time = last_candle_time.tz_localize(pytz.UTC)

        utc_now = datetime.utcnow().replace(tzinfo=pytz.UTC)
        if last_candle_time > utc_now - timedelta(hours=1):  # 1h candles for Charlie
            df_closed = df.iloc[:-1]
            print(f"⏳ Dropping incomplete candle ({last_candle_time})")
        else:
            df_closed = df
            print(f"✅ Using last closed candle ({last_candle_time})")

        if len(df_closed) < SEQ_LEN:
            print(f"❌ Not enough closed candles for {symbol}")
            continue

        # === ENTRY MODEL ===
        entry_state_dict = load_entry_model_for_coin(symbol)
        entry_scaler = global_scaler  # or per-coin scaler if you saved one

        entry_model = LSTMEntryModel(input_dim=len(FEATURES))
        entry_model.load_state_dict(entry_state_dict)
        entry_model.eval()

        seq_entry = entry_scaler.transform(df_closed[FEATURES].tail(SEQ_LEN))
        seq_entry = torch.tensor(seq_entry, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            probs = torch.softmax(entry_model(seq_entry), dim=1).numpy()[0]
            short_prob, no_prob, long_prob = probs

        last_price = df_closed["close"].iloc[-1]
        candle_time_str = df_closed["time"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")

        print(f"=== {symbol.upper()} ===")
        print(f"Last closed candle: {df_closed['time'].iloc[-1]} | Close: {last_price:.6f}")
        print(f"Probs → Short: {short_prob:.2%}, No Trade: {no_prob:.2%}, Long: {long_prob:.2%}")

        market_outputs.append({
            'symbol': symbol,
            'long_prob': float(long_prob),
            'short_prob': float(short_prob),
            'no_prob': float(no_prob)
        })

        # === Trade decision ===
        signal_triggered = False
        if long_prob > CONF_THRESHOLD:
            direction = "LONG"
            conf = long_prob
            signal_triggered = True
        elif short_prob > CONF_THRESHOLD:
            direction = "SHORT"
            conf = short_prob
            signal_triggered = True

        if not signal_triggered:
            continue

        # === TP/SL MODEL ===
        tp_sl_state_dict = load_tp_sl_model_for_coin(symbol)
        tp_sl_model_instance = LSTMTpSlModel(input_dim=len(FEATURES))
        tp_sl_model_instance.load_state_dict(tp_sl_state_dict)
        tp_sl_model_instance.eval()

        seq_tp_sl = tp_sl_scaler.transform(df_closed[FEATURES].tail(SEQ_LEN))
        seq_tp_sl = torch.tensor(seq_tp_sl, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            tp_pct, sl_pct = tp_sl_model_instance(seq_tp_sl).numpy()[0]

        if direction == "LONG":
            tp_price = last_price * (1 + abs(tp_pct))
            sl_price = last_price * (1 - abs(sl_pct))
        else:
            tp_price = last_price * (1 - abs(tp_pct))
            sl_price = last_price * (1 + abs(sl_pct))

        all_signals.append({
            'symbol': symbol,
            'direction': direction,
            'conf': conf,
            'entry': last_price,
            'tp': tp_price,
            'sl': sl_price,
            'candle_time': candle_time_str
        })

    # === Bias calculation & alerts ===
    #bias_dir, long_sum, short_sum, no_sum = calculate_market_bias_from_all_scores(market_outputs)

    bias_msg = (
        f"📊 Total Market: {bias_dir}\n"
        f"LONG total: {long_sum:.2f} | SHORT total: {short_sum:.2f} | NO TRADE total: {no_sum:.2f}\n"
        f"Based on {len(market_outputs)} coins."
    )
    bot.send_message(chat_id=CHAT_ID, text=bias_msg, parse_mode="Markdown")

    final_signals = filter_trades_by_bias(all_signals, top_n=5, high_conf_cutoff=0.90)

    for sig in final_signals:
        signal_id = (sig['symbol'], sig['direction'], sig['candle_time'])
        if signal_id in last_sent_signals:
            continue
        last_sent_signals.add(signal_id)
        send_alert(sig['symbol'], sig['direction'], sig['conf'],
                   sig['entry'], sig['tp'], sig['sl'], sig['candle_time'])


# ==== MAIN ====
if __name__ == "__main__":
    check_ai_signals()
    while True:
        now = datetime.utcnow().replace(tzinfo=pytz.UTC)
        if now.minute >= 3:  # run a few minutes after the hour
            check_ai_signals()
            time.sleep(3600 - (now.minute * 60 + now.second) + 180)
        else:
            time.sleep(30)
    
