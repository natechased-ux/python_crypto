import pandas as pd
import numpy as np
import requests
import time
import os
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
# Tracks last alerts sent to avoid duplicates
last_sent_signals = set()

# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"

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

class LSTMTpSlModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)  # TP%, SL%


    

# === ALLOW SCALER LOADING IN PYTORCH 2.6 ===
torch.serialization.add_safe_globals([StandardScaler])

# === LOAD ENTRY MODEL ===
class LSTMEntryModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)

# === LOAD GLOBAL SCALER ONCE ===
global_ckpt = torch.load("models/global_model.pt", map_location="cpu", weights_only=False)
global_scaler = global_ckpt["scaler"]

# === HELPER: LOAD PER-COIN OR GLOBAL MODEL ===
def load_entry_model_for_coin(symbol):
    coin_name = symbol.upper().replace("-", "_")
    coin_model_path = os.path.join("models", f"{coin_name}_model.pt")
    global_model_path = os.path.join("models", "global_model.pt")

    if os.path.exists(coin_model_path):
        print(f"📌 Loading fine-tuned entry model for {coin_name}")
        ckpt = torch.load(coin_model_path, map_location="cpu", weights_only=False)
    else:
        print(f"📌 No fine-tuned model for {coin_name}, using global model")
        ckpt = torch.load(global_model_path, map_location="cpu", weights_only=False)

    # If it's a raw state_dict (fine-tuned per-coin), wrap it
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt  # already a state_dict

    return state_dict

def load_tp_sl_model_for_coin(symbol):
    coin_name = symbol.upper().replace("-", "_")
    coin_model_path = os.path.join("models", f"{coin_name}_tp_sl_model.pt")
    global_model_path = os.path.join("models", "tp_sl_model.pt")

    if os.path.exists(coin_model_path):
        print(f"📌 Loading fine-tuned TP/SL model for {coin_name}")
        ckpt = torch.load(coin_model_path, map_location="cpu", weights_only=False)
    else:
        print(f"📌 No fine-tuned TP/SL model for {coin_name}, using global TP/SL model")
        ckpt = torch.load(global_model_path, map_location="cpu", weights_only=False)

    # Handle both state_dict and raw dict cases
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    return state_dict




# === LOAD TP/SL MODEL ===
# We still need the scaler for TP/SL
tp_sl_ckpt = torch.load("models/tp_sl_model.pt", map_location="cpu", weights_only=False)
tp_sl_scaler = tp_sl_ckpt["scaler"]


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

def calculate_market_bias_from_all_scores(all_coin_outputs):
    """
    all_coin_outputs: list of dicts with keys 'symbol', 'long_prob', 'short_prob', 'no_prob'
    Returns: (bias_direction, long_total, short_total, no_total)
    """

    long_total = sum(c['long_prob'] for c in all_coin_outputs)
    short_total = sum(c['short_prob'] for c in all_coin_outputs)
    no_total = sum(c['no_prob'] for c in all_coin_outputs)

    if long_total >= short_total and long_total >= no_total:
        bias = 'LONG'
    elif short_total >= long_total and short_total >= no_total:
        bias = 'SHORT'
    else:
        bias = 'NO TRADE'

    return bias, long_total, short_total, no_total


def check_ai_signals():
    print(f"\n🔍 Checking AI signals at {datetime.utcnow().isoformat()} UTC")
    all_signals = []
    market_outputs = []

    for symbol in COINS:

        # === Load entry model for coin ===
        entry_state_dict = load_entry_model_for_coin(symbol)
        entry_scaler = global_scaler  # always use global scaler

        entry_model = LSTMEntryModel(input_dim=len(FEATURES))
        entry_model.load_state_dict(entry_state_dict)
        entry_model.eval()

        # === Fetch data ===
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

        # === ENTRY PREDICTION ===
        seq_entry = entry_scaler.transform(df_closed[FEATURES].tail(SEQ_LEN))
        seq_entry = torch.tensor(seq_entry, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(entry_model(seq_entry), dim=1).numpy()[0]
            short_prob, no_prob, long_prob = probs

        # Force last_price to be from the exact last official closed candle
        last_price = df_closed["close"].iloc[-1]
        candle_time_str = last_candle_time
    
        atr_val = df_closed["atr"].iloc[-1]

        # Debug logging
        print(f"\n=== {symbol.upper()} ===")
        print(f"Last closed candle: {candle_time_str} | Close: {last_price:.6f}")
        print(f"Probs → Short: {short_prob:.2%}, No Trade: {no_prob:.2%}, Long: {long_prob:.2%}")

        # Store for market bias calc
        market_outputs.append({
            'symbol': symbol,
            'long_prob': float(long_prob),
            'short_prob': float(short_prob),
            'no_prob': float(no_prob)
        })

        # Trade decision
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

        # === TP/SL PREDICTION ===
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

        # Safety checks
        sl_atr_mult = abs(last_price - sl_price) / atr_val
        rr_ratio = abs(tp_price - last_price) / abs(last_price - sl_price)

        print(f"ATR: {atr_val:.6f} | SL ATR multiple: {sl_atr_mult:.2f} | RR: {rr_ratio:.2f}")

        if sl_atr_mult > MAX_SL_ATR_MULT:
            print("→ Trade skipped: SL too wide")
            continue
        if rr_ratio < MIN_RR:
            print("→ Trade skipped: RR too low")
            continue

        # Save final trade
        all_signals.append({
            'symbol': symbol,
            'direction': direction,
            'conf': conf,
            'entry': last_price,
            'tp': tp_price,
            'sl': sl_price,
            'candle_time': candle_time_str
        })

    # === Calculate and send bias ===
    bias_dir, long_sum, short_sum, no_sum = calculate_market_bias_from_all_scores(market_outputs)
    bias_msg = (
        f"📊 Total Market: {bias_dir}\n"
        f"LONG total: {long_sum:.2f} | SHORT total: {short_sum:.2f} | NO TRADE total: {no_sum:.2f}\n"
        f"Based on {len(market_outputs)} coins."
    )
    bot.send_message(chat_id=CHAT_ID, text=bias_msg, parse_mode="Markdown")

    # === Filter by bias & send alerts ===
    final_signals = filter_trades_by_bias(all_signals, top_n=5, high_conf_cutoff=0.90)

    for sig in final_signals:
        signal_id = (sig['symbol'], sig['direction'], sig['candle_time'])
        if signal_id in last_sent_signals:
            print(f"🔁 Skipping duplicate alert for {sig['symbol']} {sig['direction']} ({sig['candle_time']})")
            continue

        last_sent_signals.add(signal_id)
        send_alert(sig['symbol'], sig['direction'], sig['conf'],
                   sig['entry'], sig['tp'], sig['sl'], sig['candle_time'])



def filter_trades_by_bias(all_signals, top_n=10, high_conf_cutoff=0.90, min_bias_ratio=0.60, bias_conf_min=0.70):
    """
    Filters trades:
    - Only count trades >= bias_conf_min toward bias calculation
    - Apply bias filter only if dominance >= min_bias_ratio
    - Always keep >= high_conf_cutoff trades regardless of bias
    - Sends Telegram message about bias before sending trades
    """

    # Only count high-conviction trades toward bias
    bias_pool = [s for s in all_signals if s['conf'] >= bias_conf_min]
    longs = sum(1 for s in bias_pool if s['direction'] == 'LONG')
    shorts = sum(1 for s in bias_pool if s['direction'] == 'SHORT')
    total = longs + shorts

    # Avoid divide-by-zero
    if total == 0:
        msg = "📊 *No trades meet bias confidence threshold*.\nShowing only ≥90% confidence trades."
        bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
        return [s for s in all_signals if s['conf'] >= high_conf_cutoff]

    long_ratio = longs / total
    short_ratio = shorts / total

    # Determine bias
    if long_ratio >= min_bias_ratio:
        dominant_bias = 'LONG'
    elif short_ratio >= min_bias_ratio:
        dominant_bias = 'SHORT'
    else:
        dominant_bias = None

    if dominant_bias:
        agreement_pct = max(long_ratio, short_ratio) * 100
        msg = f"📊 *High Prob Trades:* {dominant_bias} ({agreement_pct:.0f}% agreement)\n"
        msg += f"{longs} LONG vs {shorts} SHORT (≥{bias_conf_min*100:.0f}% confidence signals)"
        bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")

        # Keep dominant direction top N
        dominant_trades = sorted(
            [s for s in all_signals if s['direction'] == dominant_bias],
            key=lambda x: x['conf'],
            reverse=True
        )[:top_n]

        # Keep strong countertrend trades
        high_conf_outliers = [
            s for s in all_signals
            if s['direction'] != dominant_bias and s['conf'] >= high_conf_cutoff
        ]

        final_trades = dominant_trades + high_conf_outliers

    else:
        msg = f"📊 *No Clear Market Bias*\n For High Prob Trades\n ({longs} LONG vs {shorts} SHORT)\nShowing only ≥90% confidence trades."
        bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
        final_trades = [s for s in all_signals if s['conf'] >= high_conf_cutoff]

    print(f"✅ Keeping {len(final_trades)} trades after filtering")
    return final_trades



# === ALERTING ===
import time

def send_alert(symbol, side, conf, entry, tp, sl, candle_time_str):
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
        f"Based on last closed candle: `{candle_time_str}`\n"
        f"Confidence: {conf_color} *{conf:.2%}*\n"
        f"Entry: `{entry_fmt}`\n"
        f"TP: `{tp_fmt}` (*{tp_pct:+.2f}%*)\n"
        f"SL: `{sl_fmt}` (*{sl_pct:+.2f}%*)"
    )

    try:
        bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
        print(f"✅ Sent alert for {symbol.upper()}")
        time.sleep(1.5)  # 🛡 Anti-flood
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
