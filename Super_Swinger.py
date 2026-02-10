import pandas as pd
import numpy as np
import requests
import time
import torch
import torch.nn as nn
import telegram
import pytz
from datetime import datetime, timedelta
from ta.trend import ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import schedule
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")  # Ignore harmless Telegram warning

# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "-4916911067"

#personal chat id:"7967738614"
#channel chat id:
#"-4916911067"
COINS = [
    "btc-usd", "eth-usd", "xrp-usd", "ltc-usd", "ada-usd", "doge-usd",
    "sol-usd", "wif-usd", "ondo-usd", "sei-usd", "magic-usd", "ape-usd",
    "jasmy-usd", "wld-usd", "syrup-usd", "fartcoin-usd", "aero-usd",
    "link-usd", "hbar-usd", "aave-usd", "fet-usd", "crv-usd", "tao-usd",
    "avax-usd", "xcn-usd", "uni-usd", "mkr-usd", "toshi-usd", "near-usd",
    "algo-usd", "trump-usd", "bch-usd", "inj-usd", "pepe-usd", "xlm-usd",
    "moodeng-usd", "bonk-usd", "dot-usd", "popcat-usd", "arb-usd", "icp-usd",
    "qnt-usd", "tia-usd", "ip-usd", "pnut-usd", "apt-usd", "ena-usd", "turbo-usd",
    "bera-usd", "pol-usd", "mask-usd", "pyth-usd", "sand-usd", "morpho-usd",
    "mana-usd", "coti-usd", "c98-usd", "axs-usd"
]

bot = telegram.Bot(token=TELEGRAM_TOKEN)
BASE_URL = "https://api.exchange.coinbase.com"
GRANULARITY = 21600  # 6-hour candles
SEQ_LEN = 60
CONF_THRESHOLD = 0.80
MAX_SL_ATR_MULT = 2.0
MIN_RR = 1.2  # Minimum Risk-Reward ratio
SAFETY_DELAY_MIN = 3  # Wait 3 min after close

FEATURES = [
    'rsi', 'adx', 'plus_di', 'minus_di', 'macd_diff',
    'ema10', 'ema20', 'ema50', 'ema200',
    'ema10_slope', 'ema20_slope', 'ema50_slope', 'ema200_slope',
    'atr', 'vol_change', 'body_wick_ratio', 'above_ema200'
]

# === MODEL CLASSES ===
class LSTMTrader(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(h[:, -1, :])

class LSTM_TP_SL(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(h[:, -1, :])

# === ALLOW SCALER LOADING IN PYTORCH 2.6 ===
torch.serialization.add_safe_globals([StandardScaler])

# === LOAD ENTRY MODEL ===
entry_ckpt = torch.load("lstm_crypto_model_3candle.pt", map_location="cpu", weights_only=False)
entry_scaler = entry_ckpt["scaler"]
entry_model = LSTMTrader(input_size=len(FEATURES))
entry_model.load_state_dict(entry_ckpt["model_state_dict"])
entry_model.eval()
print("✅ Entry model loaded")

# === LOAD TP/SL MODEL ===
tp_sl_ckpt = torch.load("lstm_tp_sl_model.pt", map_location="cpu", weights_only=False)
tp_sl_scaler = tp_sl_ckpt["scaler"]
tp_sl_model = LSTM_TP_SL(input_size=len(FEATURES))
tp_sl_model.load_state_dict(tp_sl_ckpt["model_state_dict"])
tp_sl_model.eval()
print("✅ TP/SL model loaded")

# === FETCH CANDLES ===
def fetch_candles(symbol, lookback_days=60):
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    df = pd.DataFrame()

    while start < end:
        chunk_end = min(start + timedelta(hours=(GRANULARITY / 3600) * 300), end)
        params = {
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "granularity": GRANULARITY
        }
        res = requests.get(f"{BASE_URL}/products/{symbol}/candles", params=params)
        time.sleep(0.25)
        try:
            data = res.json()
            if isinstance(data, list) and data:
                chunk = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
                df = pd.concat([df, chunk], ignore_index=True)
        except:
            pass
        start = chunk_end

    if df.empty:
        return None
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.drop_duplicates(subset="time", inplace=True)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# === FEATURE CALCULATION ===
def calculate_features(df):
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    adx = ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()
    macd = MACD(df["close"])
    df["macd_diff"] = macd.macd_diff()
    for p in [10, 20, 50, 200]:
        df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean()
        df[f"ema{p}_slope"] = df[f"ema{p}"].diff()
    atr = AverageTrueRange(df["high"], df["low"], df["close"])
    df["atr"] = atr.average_true_range()
    df["vol_change"] = df["volume"].pct_change()
    df["body_wick_ratio"] = np.where(
        (df["high"] - df["low"]) != 0,
        abs(df["close"] - df["open"]) / (df["high"] - df["low"]),
        0
    )
    df["above_ema200"] = (df["close"] > df["ema200"]).astype(int)
    return df

from datetime import datetime, timedelta
import pytz

# === AI SIGNAL CHECK ===
from datetime import datetime, timedelta
import pytz

# === AI SIGNAL CHECK ===
from datetime import datetime, timedelta
import pytz

def check_ai_signals():
    print(f"\n🔍 Checking AI signals at {datetime.utcnow().isoformat()} UTC")

    for symbol in COINS:
        df = fetch_candles(symbol, lookback_days=60)
        if df is None or len(df) < SEQ_LEN + 1:
            print(f"{symbol.upper()} → Not enough data")
            continue

        df = calculate_features(df)
        df.dropna(inplace=True)

        # ✅ Correct forming vs closed candle detection
        last_candle_time = df["time"].iloc[-1]
        if last_candle_time.tzinfo is None:
            last_candle_time = last_candle_time.tz_localize(pytz.UTC)

        now = datetime.utcnow().replace(tzinfo=pytz.UTC)
        expected_close = now.replace(minute=0, second=0, microsecond=0)
        expected_close = expected_close - timedelta(hours=now.hour % 6)

        if last_candle_time == expected_close:
            df_closed = df
            print(f"✅ {symbol.upper()} → Keeping last closed candle ({last_candle_time})")
        else:
            df_closed = df.iloc[:-1]
            print(f"⏳ {symbol.upper()} → Dropping last candle ({last_candle_time}, still forming)")

        # ENTRY PREDICTION
        seq_entry = entry_scaler.transform(df_closed[FEATURES].tail(SEQ_LEN))
        seq_entry = torch.tensor(seq_entry, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(entry_model(seq_entry), dim=1).numpy()[0]
            short_prob, no_prob, long_prob = probs

        last_price = df_closed["close"].iloc[-1]
        atr_val = df_closed["atr"].iloc[-1]

        # Debug logging
        print(f"\n=== {symbol.upper()} ===")
        print(f"Last closed candle: {df_closed['time'].iloc[-1]} | Close: {last_price:.6f}")
        print(f"Probs → Short: {short_prob:.2%}, No Trade: {no_prob:.2%}, Long: {long_prob:.2%}")

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
            print("→ No trade: confidence too low")
            continue

        # TP/SL PREDICTION
        seq_tp_sl = tp_sl_scaler.transform(df_closed[FEATURES].tail(SEQ_LEN))
        seq_tp_sl = torch.tensor(seq_tp_sl, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            tp_pct, sl_pct = tp_sl_model(seq_tp_sl).numpy()[0]

        if direction == "LONG":
            tp_price = last_price * (1 + abs(tp_pct))
            sl_price = last_price * (1 - abs(sl_pct))
        else:
            tp_price = last_price * (1 - abs(tp_pct))
            sl_price = last_price * (1 + abs(sl_pct))

        # Safety filters
        sl_atr_mult = abs(last_price - sl_price) / atr_val
        rr_ratio = abs(tp_price - last_price) / abs(last_price - sl_price)

        print(f"ATR: {atr_val:.6f} | SL ATR multiple: {sl_atr_mult:.2f} | RR: {rr_ratio:.2f}")

        if sl_atr_mult > MAX_SL_ATR_MULT:
            print("→ Trade skipped: SL too wide")
            continue
        if rr_ratio < MIN_RR:
            print("→ Trade skipped: RR too low")
            continue

        # If it passes all filters → send alert
        print("✅ Trade triggered → Sending alert")
        send_alert(symbol, direction, conf, last_price, tp_price, sl_price)



# === ALERTING ===
import time

def send_alert(symbol, side, conf, entry, tp, sl):
    # Confidence color
    if conf >= 0.90:
        conf_color = "🟢"
    elif conf >= 0.80:
        conf_color = "🟡"
    elif conf >= 0.70:
        conf_color = "🔴"
    else:
        conf_color = "⚪"

    # Price formatting
    def fmt_price(p):
        if p >= 1000:
            return f"{p:,.2f}"
        elif p >= 1:
            return f"{p:,.4f}"
        elif p >= 0.1:
            return f"{p:,.6f}"
        else:
            return f"{p:,.10f}"

    entry_fmt = fmt_price(entry)
    tp_fmt = fmt_price(tp)
    sl_fmt = fmt_price(sl)

    tp_pct = ((tp - entry) / entry) * 100
    sl_pct = ((sl - entry) / entry) * 100

    msg = (
        f"🤖 *AI SIGNAL*: {side} on `{symbol.upper()}`\n"
        f"Confidence: {conf_color} *{conf:.2%}*\n"
        f"Entry: `{entry_fmt}`\n"
        f"TP: `{tp_fmt}` (*{tp_pct:+.2f}%*)\n"
        f"SL: `{sl_fmt}` (*{sl_pct:+.2f}%*)"
    )

    try:
        bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
        print(f"✅ Sent alert for {symbol.upper()}")
        time.sleep(1.5)  # 🛡 Anti-flood: wait 1.5 seconds between messages
    except Exception as e:
        print(f"❌ Telegram error: {e}")




def get_next_6h_close_utc():
    from datetime import datetime, timedelta
    import pytz
    now = datetime.utcnow().replace(tzinfo=pytz.UTC)
    current_hour = now.hour
    next_hour = ((current_hour // 6) + 1) * 6
    if next_hour >= 24:
        next_hour -= 24
        next_day = now.date() + timedelta(days=1)
    else:
        next_day = now.date()
    return datetime(
        year=next_day.year,
        month=next_day.month,
        day=next_day.day,
        hour=next_hour,
        tzinfo=pytz.UTC
    )

print("\n🚀 Running immediate check on last closed 6-hour candle...")
check_ai_signals()

# === Forever loop aligned to 6-hour closes ===
while True:
    now = datetime.utcnow().replace(tzinfo=pytz.UTC)
    next_close = get_next_6h_close_utc()

    # Calculate seconds until next 6-hour close + safety delay
    wait_seconds = (next_close - now).total_seconds() + (SAFETY_DELAY_MIN * 60)

    if wait_seconds < 0:
        # If we somehow passed the close time, skip to the next one
        next_close = get_next_6h_close_utc()
        wait_seconds = (next_close - now).total_seconds() + (SAFETY_DELAY_MIN * 60)

    print(f"\n⏳ Waiting until {next_close} UTC + {SAFETY_DELAY_MIN} min delay "
          f"({wait_seconds/60:.1f} min) to run next check...")
    time.sleep(wait_seconds)

    print(f"\n🔍 Running check for 6-hour close at {next_close} UTC...")
    check_ai_signals()
