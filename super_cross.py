#!/usr/bin/env python3

import requests
import pandas as pd
import time
from datetime import datetime, timedelta

# ==============================
# CONFIG
# ==============================

TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"

COINBASE_API = "https://api.exchange.coinbase.com"
POLL_INTERVAL = 60  # run every 1 minute

QUOTE = "USD"

COOLDOWN_HOURS = 2
last_alert = {}   # {"symbol-long": datetime, "symbol-short": datetime}


# ==============================
# HELPERS
# ==============================

# ==============================
# WHITELISTS
# ==============================

LONG_WHITELIST = [
    "WMTX-USD","CLANKER-USD","NOICE-USD","AWE-USD","FORT-USD","TIME-USD",
    "BNKR-USD","DBR-USD","ZEN-USD","CFG-USD","YB-USD","BLZ-USD","DASH-USD",
    "WLFI-USD","MET-USD","ORCA-USD","RARI-USD","BICO-USD","BOOMER-USD",
    "BOME-USD","GRAPE-USD","NTRN-USD","SNEK-USD","AERO-USD","ORD-USD",
    "FLR-USD","AEVO-USD","NIBI-USD","SAGA-USD","CREAM-USD","JUP-USD",
    "STRK-USD","RAIN-USD","EUL-USD","GALA-USD","GLM-USD","FET-USD",
    "TRB-USD","SEI-USD","WLD-USD","ASTR-USD","ENJ-USD"
]

SHORT_WHITELIST = [
    "MON-USD","RLS-USD","GTC-USD","ARPA-USD","ALCX-USD","BONK-USD","BADGER-USD",
    "ENS-USD","API3-USD","AERO-USD","PEOPLE-USD","ORDI-USD","ORCA-USD",
    "FORT-USD","PYTH-USD","DYDX-USD","CRPT-USD","ETHW-USD","EIGEN-USD",
    "TNSR-USD","STRAX-USD","ALT-USD","RENDER-USD","ANCHOR-USD","MOG-USD",
    "TIA-USD","SAGA-USD","AEVO-USD","BICO-USD","FLR-USD","FET-USD",
    "ORBS-USD","AR-USD","ZETA-USD","MOBILE-USD","DASH-USD","POND-USD",
    "CREAM-USD","TRB-USD","SNX-USD","OP-USD","RON-USD","WLD-USD",
    "LRC-USD","SPELL-USD","GLMR-USD","AKT-USD","PYR-USD","DENT-USD",
    "NTRN-USD","BLZ-USD","AMB-USD","BETA-USD","ZK-USD","GALA-USD",
    "MASK-USD","RNDR-USD","ANKR-USD","EGLD-USD","IMX-USD","CKB-USD",
    "MOVR-USD","UNI-USD","GLM-USD","AVAX-USD","BADGER-USD","WAXL-USD",
    "DYP-USD","MKR-USD","GRT-USD","RAIN-USD","APT-USD","INJ-USD",
    "CRV-USD","ETC-USD","ID-USD","BAT-USD","COMP-USD","YFII-USD",
    "USDD-USD"
]


def telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"})
    except:
        pass


def coinbase_get(path, params=None):
    url = COINBASE_API + path
    return requests.get(url, params=params, timeout=10).json()


def fetch_candles(symbol, granularity):
    raw = coinbase_get(f"/products/{symbol}/candles", params={"granularity": granularity})
    if not isinstance(raw, list): return None
    df = pd.DataFrame(raw, columns=["time","low","high","open","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").reset_index(drop=True)
    return df


def resample_4h(df):
    df = df.set_index("time")
    out = pd.DataFrame()
    out["open"] = df["open"].resample("4H").first()
    out["high"] = df["high"].resample("4H").max()
    out["low"] = df["low"].resample("4H").min()
    out["close"] = df["close"].resample("4H").last()
    out["volume"] = df["volume"].resample("4H").sum()
    out.dropna(inplace=True)
    return out.reset_index()


def RSI(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def can_alert(key):
    now = datetime.utcnow()
    if key not in last_alert:
        return True
    return now - last_alert[key] >= timedelta(hours=COOLDOWN_HOURS)


def update_alert(key):
    last_alert[key] = datetime.utcnow()


# ==============================
# SIGNAL CONDITIONS
# ==============================

def long_condition(df_coin_15m, df_btc_15m, df_btc_4h):
    df = df_coin_15m
    if len(df) < 100:
        return False

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    # -------------------------------
    # 1. RSI Oversold → Recovery
    # -------------------------------
    recent = df["rsi"].iloc[-5:]
    dipped_oversold = (recent < 30).any()

    # recovery but not overbought
    rsi_recovery = (curr["rsi"] > 30) and (curr["rsi"] < 50)

    if not (dipped_oversold and rsi_recovery):
        return False

    # -------------------------------
    # 2. Price reclaims MA25
    # -------------------------------
    if curr["close"] <= curr.get("ma25", curr["ma7"]):  # fallback safety
        return False

    # -------------------------------
    # 3. Volume > SMA(volume, 20)
    # -------------------------------
    vol_sma20 = df["volume"].rolling(20).mean().iloc[-1]
    if pd.isna(vol_sma20) or curr["volume"] <= vol_sma20:
        return False

    # -------------------------------
    # 4. Trend filter: EMA25 > EMA50 > EMA75
    # -------------------------------
    if not ("ema25" in df.columns and "ema50" in df.columns and "ema75" in df.columns):
        return False
    if not (curr["ema25"] > curr["ema50"] > curr["ema75"]):
        return False

    # -------------------------------
    # 5. BTC macro confirmation
    # -------------------------------
    if df_btc_4h["rsi"].iloc[-1] <= 50:
        return False

    return True



def short_condition(df_coin_15m, df_btc_15m, df_btc_4h):
    # MA cross down on coin (15m)
    df = df_coin_15m
    prev = df.iloc[-2]
    curr = df.iloc[-1]

    ma7_prev, ma7_curr = prev["ma7"], curr["ma7"]
    ma99_prev, ma99_curr = prev["ma99"], curr["ma99"]

    cross_down = ma7_prev >= ma99_prev and ma7_curr < ma99_curr

    # RSI < 50
    rsi_coin = df["rsi"].iloc[-1] < 50
    rsi_btc_15m = df_btc_15m["rsi"].iloc[-1] < 50
    rsi_btc_4h = df_btc_4h["rsi"].iloc[-1] < 50

    return cross_down and rsi_coin and rsi_btc_15m and rsi_btc_4h


# ==============================
# MAIN LOOP
# ==============================

def run():
    # Get product list
    products = coinbase_get("/products")
    symbols = []
    for p in products:
        sym = p["id"]
        if p.get("quote_currency") != QUOTE:
            continue

        # Only keep coins that appear in at least one whitelist
        if sym in LONG_WHITELIST or sym in SHORT_WHITELIST:
            symbols.append(sym)

    while True:
        try:
            # Fetch BTC 15m + 4h indicators
            btc15 = fetch_candles("BTC-USD", 900)
            btc1h = fetch_candles("BTC-USD", 3600)

            if btc15 is None or btc1h is None:
                time.sleep(POLL_INTERVAL)
                continue

            # Compute BTC indicators
            btc15["rsi"] = RSI(btc15)
            btc1h["rsi"] = RSI(btc1h)
            btc4h = resample_4h(btc1h)
            btc4h["rsi"] = RSI(btc4h)

            # Loop each coin
            for symbol in symbols:
                df15 = fetch_candles(symbol, 900)
                if df15 is None or len(df15) < 100:
                    continue

                # Compute required indicators
                df15["rsi"] = RSI(df15)
                df15["ma7"]  = df15["close"].rolling(7).mean()
                df15["ma25"] = df15["close"].rolling(25).mean()
                df15["ma99"] = df15["close"].rolling(99).mean()

                df15["ema25"] = df15["close"].ewm(span=25, adjust=False).mean()
                df15["ema50"] = df15["close"].ewm(span=50, adjust=False).mean()
                df15["ema75"] = df15["close"].ewm(span=75, adjust=False).mean()

                df15["vol_sma20"] = df15["volume"].rolling(20).mean()

                # Skip broken data
                if df15["ma99"].iloc[-1] == 0:
                    continue

                # ================================
                # LONG CONDITION + WHITELIST CHECK
                # ================================
                if symbol in LONG_WHITELIST:
                    if long_condition(df15, btc15, btc4h):
                        key = f"{symbol}-long"
                        if can_alert(key):
                            telegram(
                                f"🟢 LONG SIGNAL\n{symbol}\nPrice: {df15['close'].iloc[-1]}"
                            )
                            update_alert(key)

                # =================================
                # SHORT CONDITION + WHITELIST CHECK
                # =================================
                if symbol in SHORT_WHITELIST:
                    if short_condition(df15, btc15, btc4h):
                        key = f"{symbol}-short"
                        if can_alert(key):
                            telegram(
                                f"🔴 SHORT SIGNAL\n{symbol}\nPrice: {df15['close'].iloc[-1]}"
                            )
                            update_alert(key)

                time.sleep(0.2)

        except Exception as e:
            print("Error:", e)

        time.sleep(POLL_INTERVAL)




if __name__ == "__main__":
    print("🚀 Starting 15m strategy with BTC confirmation…")
    run()
