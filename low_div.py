#!/usr/bin/env python3
import ccxt
import pandas as pd
import numpy as np
import time
import requests

# ============================================================
# CONFIGURATION
# ============================================================

# Use ccxt.coinbase() for Coinbase Exchange.
# If you get auth/timeframe issues, try ccxt.coinbasepro() instead.
EXChANGE = ccxt.coinbase()

SCAN_INTERVAL = 60 * 30   # seconds between full scans (30 minutes)
STOCh_ThREShOLD = 10.0    # 1D Stoch RSI must be below this
LOOKBACK_DIV = 20         # candles to look back for divergence

TELEGRAM_BOT_TOKEN = "8177096945:AAhTg5nxVTA6hcPkAL4MLAAhPsZ1at7Ywmw"
TELEGRAM_ChAT_ID = "7967738614"

# Exclude stablecoin bases
STABLE_BASES = {"USDC", "USDT", "DAI", "PYUSD", "USDP", "GUSD", "USDS"}


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(msg: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            data={"chat_id": TELEGRAM_ChAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        print("Telegram error:", e)


# ============================================================
# INDICATORS
# ============================================================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.ewm(span=period, adjust=False).mean()
    roll_down = down.ewm(span=period, adjust=False).mean()

    rs = roll_up / roll_down
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def stoch_rsi(series: pd.Series, period: int = 14, smoothK: int = 3, smoothD: int = 3):
    """
    Returns (%K, %D) Stoch RSI in [0, 100].
    """
    delta = series.diff()
    rsi_up = delta.clip(lower=0)
    rsi_down = -delta.clip(upper=0)

    ema_up = rsi_up.ewm(com=period, adjust=False).mean()
    ema_down = rsi_down.ewm(com=period, adjust=False).mean()
    rsi_val = 100 * ema_up / (ema_up + ema_down)

    min_rsi = rsi_val.rolling(period).min()
    max_rsi = rsi_val.rolling(period).max()

    stoch = (rsi_val - min_rsi) / (max_rsi - min_rsi)
    k = stoch.rolling(smoothK).mean() * 100
    d = k.rolling(smoothD).mean()
    return k, d


# ============================================================
# DIVERGENCE DETECTION (closed candles, “active” divergence)
# ============================================================

def find_recent_bullish_divergence(df: pd.DataFrame) -> bool:
    """
    Detect bullish RSI divergence using *closed* candles within the last LOOKBACK_DIV candles.

    Conditions:
      - Two recent swing lows exist
      - Price makes a lower low (p2 < p1)
      - RSI makes a higher low (r2 > r1)
      - Current price has NOT clearly broken below the second swing low
    """
    if len(df) < LOOKBACK_DIV:
        return False

    recent = df.tail(LOOKBACK_DIV).copy()

    # Swing lows: middle bar lower than neighbors
    recent["swing_low"] = recent["close"].rolling(3).apply(
        lambda x: float(x[1] < x[0] and x[1] < x[2]) if len(x) == 3 else 0,
        raw=False
    )

    swings = recent[recent["swing_low"] == 1.0]
    if len(swings) < 2:
        return False

    # Last two swing lows
    swings_tail = swings.tail(2)
    p1, p2 = swings_tail["close"].iloc[0], swings_tail["close"].iloc[1]
    r1, r2 = swings_tail["rsi"].iloc[0], swings_tail["rsi"].iloc[1]

    price_lower_low = p2 < p1
    rsi_higher_low = r2 > r1

    if not (price_lower_low and rsi_higher_low):
        return False

    # Divergence still "active" if current price hasn't broken well below p2
    current_price = recent["close"].iloc[-1]
    # Allow a little wiggle room (2% below p2 invalidates it)
    if current_price < p2 * 0.98:
        return False

    return True


# ============================================================
# MARKET DATA hELPERS
# ============================================================

def fetch_ohlcv_safe(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """
    Safe OhLCV fetch: always returns a DataFrame (possibly empty).
    Columns: time, open, high, low, close, volume
    """
    try:
        data = EXChANGE.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        print("FETCh ERROR:", symbol, timeframe, e)
        return pd.DataFrame()

    if not data or len(data) < 10:
        print(symbol, timeframe, "returned too little data:", len(data))
        return pd.DataFrame()

    try:
        df = pd.DataFrame(
            data,
            columns=["time", "open", "high", "low", "close", "volume"]
        )
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df = df.sort_values("time").reset_index(drop=True)
        return df
    except Exception as e:
        print("PARSE ERROR:", symbol, timeframe, e)
        return pd.DataFrame()


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Create 4h candles using 1h Coinbase data.
    """
    if df_1h.empty:
        return df_1h

    df = df_1h.copy()
    df = df.set_index("time")
    df = df.sort_index()

    # Ensure hourly frequency so resample is clean
    df = df.asfreq("1h", method="pad")

    out = pd.DataFrame()
    out["open"] = df["open"].resample("4h").first()
    out["high"] = df["high"].resample("4h").max()
    out["low"] = df["low"].resample("4h").min()
    out["close"] = df["close"].resample("4h").last()
    out["volume"] = df["volume"].resample("4h").sum()

    out = out.dropna().reset_index()
    return out


# ============================================================
# CORE SCAN LOGIC
# ============================================================

def check_coin(symbol):
    # 1h data (for resampling)
    df_1h = fetch_ohlcv_safe(symbol, "1h", limit=500)
    if df_1h.empty or len(df_1h) < 120:  # 120 hours = 30 four-hour candles
        return None

    # Build clean 4h candles
    df_4h = resample_to_4h(df_1h)
    if df_4h.empty or len(df_4h) < 40:
        return None

    # Indicators
    df_4h["rsi"] = rsi(df_4h["close"])
    k4h, _ = stoch_rsi(df_4h["close"])

    if k4h.empty or pd.isna(k4h.iloc[-1]):
        return None

    # Oversold condition
    if k4h.iloc[-1] >= STOCh_ThREShOLD:
        return None

    # 4h bullish divergence detection
    div_4h = find_recent_bullish_divergence(df_4h)
    if not div_4h:
        return None

    # Prepare alert
    msg = f"📌 *{symbol} qualifies (hH only)*\n"
    msg += f"• 4h Stoch RSI = {k4h.iloc[-1]:.2f}\n"
    msg += "🟩 hH Bullish RSI Divergence (active)"

    print(symbol, "MEETS CONDITIONS")
    return msg


# ============================================================
# MAIN LOOP
# ============================================================

def build_symbol_list():
    markets = EXChANGE.load_markets()
    symbols = []

    for m in markets.values():
        if not m.get("active", False):
            continue
        if not m.get("spot", False):
            continue
        if m.get("quote") != "USD":
            continue

        base = m.get("base", "").upper()
        if base in STABLE_BASES:
            continue

        symbol = m.get("symbol")
        if symbol and "/" in symbol:   # drop routing IDs
            symbols.append(symbol)

    symbols = sorted(set(symbols))
    print("Scanning symbols:", len(symbols))
    return symbols



def main():
    symbols = build_symbol_list()
    send_telegram("🔍 *Coinbase Scanner Started* — 1D Stoch RSI < 10 + hH/1D Bullish RSI Divergence")

    while True:
        for symbol in symbols:
            try:
                result = check_coin(symbol)
                if result:
                    send_telegram(result)
            except Exception as e:
                # Never let one coin kill the loop
                print("Error on", symbol, "->", repr(e))
                continue

        print("Scan cycle complete. Sleeping...")
        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
