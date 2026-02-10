import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
from openai import OpenAI

# -----------------------------
# 1. Coinbase FREE API Endpoints
# -----------------------------
PUBLIC_API = "https://api.exchange.coinbase.com"

print("Script started!")

import numpy as np
import pandas as pd

from datetime import datetime
import json
import math


ALL_COINS = [
    "MTL-USD", "INV-USD", "NEAR-USD", "HFT-USD", "ZETACHAIN-USD",
    "PRIME-USD", "FLOKI-USD", "DOGE-USD", "SUKU-USD", "GAL-USD",
    "QI-USD", "ALCX-USD", "SOL-USD", "MLN-USD", "MASK-USD",
    "NKN-USD", "ABT-USD", "WCFG-USD", "RLY-USD", "IOTX-USD",
    "COVAL-USD", "C98-USD", "ACS-USD", "TURBO-USD",
    "MSOL-USD", "ZETA-USD", "ORN-USD", "B3-USD", "CLV-USD",
    "ALICE-USD", "LOKA-USD", "KNC-USD", "FLOW-USD", "VARA-USD",
    "WELL-USD", "CORECHAIN-USD", "BCH-USD", "ANKR-USD",
    "PRQ-USD", "BAND-USD", "RGT-USD", "SPELL-USD",
    "ORCA-USD", "ATH-USD", "TRB-USD", "FOX-USD", "IDEX-USD",
    "ZEN-USD", "FLR-USD", "WAMPL-USD", "BIT-USD", "JTO-USD",
    "REP-USD", "GLM-USD", "TRUMP-USD", "LRC-USD", "JUP-USD",
    "MOG-USD", "LQTY-USD", "XTZ-USD", "RED-USD", "AURORA-USD",
    "UMA-USD", "SKL-USD", "FIL-USD", "PYTH-USD", "SEAM-USD",
    "PAXG-USD", "SUI-USD", "UNFI-USD", "T-USD", "GTC-USD",
    "VELO-USD", "MKR-USD", "RAI-USD", "BLUR-USD", "ACH-USD",
    "API3-USD", "ETH-USD", "FIS-USD", "PNG-USD", "ADA-USD",
    "MONA-USD", "ZK-USD", "YFI-USD", "POWR-USD", "JASMY-USD",
    "TVK-USD", "PENDLE-USD", "OCEAN-USD", "FX-USD", "POL-USD",
    "SUPER-USD", "WAXL-USD", "MATH-USD", "MOBILE-USD", "SHDW-USD",
    "YFII-USD", "LRDS-USD", "RPL-USD", "TRAC-USD", "LCX-USD",
    "CHZ-USD", "VTHO-USD", "TNSR-USD", "SEI-USD", "SAND-USD",
    "AGLD-USD", "GRT-USD", "FIDA-USD", "UPI-USD",
    "BNT-USD", "HNT-USD", "UNI-USD", "STORJ-USD", "VET-USD",
    "LTC-USD", "AST-USD", "AVAX-USD", "ARKM-USD", "HOPR-USD",
    "VOXEL-USD", "MANA-USD", "RARI-USD", "SHPING-USD", "PLA-USD",
     "POPCAT-USD", "FET-USD", "SWFTC-USD", "IP-USD",
    "UST-USD", "1INCH-USD", "DIA-USD", 
    "ICP-USD", "MNDE-USD", "MORPHO-USD", "SYRUP-USD", "HONEY-USD",
     "RBN-USD", "A8-USD", "ZEC-USD", "RLC-USD",
    "SXT-USD", "ILV-USD", "AKT-USD", "RNDR-USD", "ONDO-USD",
    "DEGEN-USD", "EGLD-USD", "SD-USD", "LOOM-USD", "PEPE-USD",
    "ACX-USD", "PYUSD-USD", "GHST-USD", "CRO-USD", "PROMPT-USD",
    "GYEN-USD", "XLM-USD", "LDO-USD", "KRL-USD", "ALEPH-USD",
    "OGN-USD", "PENGU-USD", "XYO-USD", "OXT-USD", "SYLO-USD",
    "WLD-USD", "OMG-USD", "BAL-USD", "PRO-USD", "SYN-USD",
    "EIGEN-USD", "GMT-USD", "GFI-USD", "RENDER-USD", "MINA-USD",
    "MAGIC-USD", "DYP-USD", "BICO-USD", "ZORA-USD", "COW-USD",
    "CELR-USD", "PIRATE-USD", "WLUNA-USD", "NEST-USD", "REZ-USD",
     "DESO-USD", "TAO-USD", "ME-USD", "RONIN-USD",
    "NCT-USD", "INJ-USD", "NU-USD", "OP-USD", "KARRAT-USD",
    "ALEO-USD", "INDEX-USD", "VGX-USD", "KERNEL-USD", "FAI-USD",
    "PUNDIX-USD", "AIOZ-USD", "AERO-USD", "ELA-USD",
    "ALGO-USD", "IO-USD", "APT-USD", "CRV-USD", "MCO2-USD",
    "ERN-USD", "AUCTION-USD", "PRCL-USD", "VVV-USD", "BOBA-USD",
    "DREP-USD", "AXL-USD", "HBAR-USD", "MPL-USD", "STRK-USD",
    "REQ-USD", "BERA-USD", "AERGO-USD", "GODS-USD", "GALA-USD",
    "ROSE-USD",  "KSM-USD", "TRU-USD", "STG-USD",
    "AAVE-USD", "EDGE-USD", "LINK-USD", "TIA-USD", "RARE-USD",
    "WIF-USD", "BTC-USD", "ENJ-USD", "FORTH-USD", 
    "DOT-USD", "AUDIO-USD", "DNT-USD", "CTSI-USD", "00-USD",
    "RSR-USD", "GIGA-USD", "QNT-USD", "DOGINME-USD", "ATOM-USD",
    "DRIFT-USD", "KAVA-USD", "SPA-USD", "BAT-USD", "POLS-USD",
    "MATIC-USD", "SHIB-USD", "COTI-USD", "RAD-USD", "LSETH-USD",
    "IMX-USD", "GNO-USD", "L3-USD"
]



def calc_rsi(series, period=14):
    series = series.astype(float)
    if len(series) < period + 1:
        return None
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    last = rsi.iloc[-1]
    return float(last) if not np.isnan(last) else None


def calc_macd(series, fast=12, slow=26, signal=9):
    series = series.astype(float)
    if len(series) < slow + signal:
        return None, None, None
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    hist = macd - signal_line
    return float(macd.iloc[-1]), float(signal_line.iloc[-1]), float(hist.iloc[-1])


def calc_bbands(series, length=20, num_std=2.0):
    series = series.astype(float)
    if len(series) < length:
        return None, None, None, None
    ma = series.rolling(length).mean()
    std = series.rolling(length).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    width = (upper - lower) / ma
    return float(upper.iloc[-1]), float(lower.iloc[-1]), float(ma.iloc[-1]), float(width.iloc[-1])


def find_recent_swings(df, lookback=30, left=2, right=2):
    """
    Simple recent swing high/low detector on last `lookback` candles.
    """
    if len(df) < left + right + 1:
        return None, None
    sub = df.tail(lookback).reset_index(drop=True)
    highs = sub["high"].values
    lows = sub["low"].values
    swing_high = None
    swing_low = None
    for i in range(left, len(sub) - right):
        window_high = highs[i-left:i+right+1]
        window_low = lows[i-left:i+right+1]
        if highs[i] == window_high.max():
            swing_high = float(highs[i])
        if lows[i] == window_low.min():
            swing_low = float(lows[i])
    return swing_high, swing_low



def get_candles(product_id="BTC-USD", granularity=3600, limit=100):
    """
    Returns OHLCV candles from the free Coinbase API.
    """
    url = f"{PUBLIC_API}/products/{product_id}/candles"
    params = {"granularity": granularity}

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Error fetching candles for {product_id}: {e}")
        return None

    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").reset_index(drop=True)
    return df.tail(limit)


def get_orderbook(product_id="BTC-USD", level=2):
    url = f"{PUBLIC_API}/products/{product_id}/book?level={level}"
    try:
        return requests.get(url, timeout=10).json()
    except Exception as e:
        print(f"Orderbook error for {product_id}: {e}")
        return None


def get_ticker(product_id="BTC-USD"):
    url = f"{PUBLIC_API}/products/{product_id}/ticker"
    try:
        return requests.get(url, timeout=10).json()
    except Exception as e:
        print(f"Ticker error for {product_id}: {e}")
        return None


# -----------------------------
# 2. Feature Extraction
# -----------------------------
def compute_features(df_1h, df_1d):
    """
    Enhanced feature computation for a single symbol.
    df_1h: 1-hour candles DataFrame with columns [time, low, high, open, close, volume]
    df_1d: 1-day candles DataFrame with same columns
    """
    # Basic sanity checks
    if df_1h is None or df_1d is None:
        return None
    if len(df_1h) < 200 or len(df_1d) < 60:
        return None

    df_1h = df_1h.copy()
    df_1d = df_1d.copy()

    # Ensure numeric
    for col in ["open", "high", "low", "close", "volume"]:
        df_1h[col] = df_1h[col].astype(float)
        df_1d[col] = df_1d[col].astype(float)

    price = float(df_1h["close"].iloc[-1])

    # ==== Trend (multi-timeframe) ====
    # 1h EMAs
    df_1h["ema20_1h"] = df_1h["close"].ewm(span=20).mean()
    df_1h["ema50_1h"] = df_1h["close"].ewm(span=50).mean()
    df_1h["ema200_1h"] = df_1h["close"].ewm(span=200).mean()

    # 1d EMAs
    df_1d["ema50_1d"] = df_1d["close"].ewm(span=50).mean()
    df_1d["ema200_1d"] = df_1d["close"].ewm(span=200).mean()

    def trend_state(close, ema50, ema200):
        above50 = close > ema50
        above200 = close > ema200
        if above50 and above200:
            return "bull"
        if (not above50) and (not above200):
            return "bear"
        return "mixed"

    trend_state_1h = trend_state(
        df_1h["close"].iloc[-1],
        df_1h["ema50_1h"].iloc[-1],
        df_1h["ema200_1h"].iloc[-1],
    )
    trend_state_1d = trend_state(
        df_1d["close"].iloc[-1],
        df_1d["ema50_1d"].iloc[-1],
        df_1d["ema200_1d"].iloc[-1],
    )

    trend_strength_1h = float(df_1h["ema50_1h"].iloc[-1] / df_1h["ema200_1h"].iloc[-1])

    # Trend maturity: bars since 1h trend_state last changed
    df_1h["trend_state_1h"] = [
        trend_state(c, e50, e200)
        for c, e50, e200 in zip(df_1h["close"], df_1h["ema50_1h"], df_1h["ema200_1h"])
    ]
    last_state = df_1h["trend_state_1h"].iloc[-1]
    bars_in_trend_1h = 0
    for s in reversed(df_1h["trend_state_1h"].tolist()):
        if s == last_state:
            bars_in_trend_1h += 1
        else:
            break

    # ==== Volatility ====
    # True range & ATR(14)
    df_1h["prev_close"] = df_1h["close"].shift(1)
    def tr(row):
        high, low, prev = row["high"], row["low"], row["prev_close"]
        a = high - low
        b = abs(high - prev) if pd.notna(prev) else 0
        c = abs(low - prev) if pd.notna(prev) else 0
        return max(a, b, c)
    df_1h["true_range"] = df_1h.apply(tr, axis=1)
    df_1h["atr_1h"] = df_1h["true_range"].rolling(14).mean()

    atr_1h = float(df_1h["atr_1h"].iloc[-1])
    atr_pct_1h = float(atr_1h / price * 100) if price else None

    # Realized volatility over last 24h
    if len(df_1h) > 24:
        rets_1h = df_1h["close"].pct_change().iloc[-24:]
        realized_vol_24h_pct = float(rets_1h.std() * np.sqrt(24) * 100)
    else:
        realized_vol_24h_pct = None

    # Bollinger bands (1h, 20-period)
    bb_upper, bb_lower, bb_mid, bb_width = calc_bbands(
        df_1h["close"], length=20, num_std=2.0
    )
    # Simple volatility regime from BB width
    if bb_width is not None:
        if bb_width < 0.02:
            vol_regime = "compression"
        elif bb_width > 0.06:
            vol_regime = "expansion"
        else:
            vol_regime = "normal"
    else:
        vol_regime = "normal"

    # ==== Momentum ====
    if len(df_1h) > 24:
        change_24h_pct = float(
            (df_1h["close"].iloc[-1] - df_1h["close"].iloc[-25])
            / df_1h["close"].iloc[-25]
            * 100
        )
    else:
        change_24h_pct = 0.0

    if len(df_1h) > 24 * 7:
        change_7d_pct = float(
            (df_1h["close"].iloc[-1] - df_1h["close"].iloc[-24 * 7])
            / df_1h["close"].iloc[-24 * 7]
            * 100
        )
    else:
        change_7d_pct = 0.0

    rsi_14_1h = calc_rsi(df_1h["close"], period=14)
    macd_val, macd_signal, macd_hist = calc_macd(df_1h["close"])

    # ==== Volume features ====
    if len(df_1h) > 48:
        vol_24h = float(df_1h["volume"].iloc[-24:].sum())
        vol_prev_24h = float(df_1h["volume"].iloc[-48:-24].sum())
        vol_ratio_24h = float(vol_24h / vol_prev_24h) if vol_prev_24h > 0 else None
    else:
        vol_24h = float(df_1h["volume"].iloc[-24:].sum())
        vol_prev_24h = None
        vol_ratio_24h = None

    # ==== Candle structure / impulse vs correction ====
    last_n = df_1h.tail(20).copy()
    bodies = (last_n["close"] - last_n["open"]).abs()
    ranges = (last_n["high"] - last_n["low"]).replace(0, np.nan)
    body_ratio = (bodies / ranges).fillna(0.0)
    bullish = (last_n["close"] > last_n["open"]).astype(int)
    bearish = (last_n["close"] < last_n["open"]).astype(int)
    bullish_count_10 = int(bullish.tail(10).sum())
    bearish_count_10 = int(bearish.tail(10).sum())
    avg_body_ratio_10 = float(body_ratio.tail(10).mean())

    # ==== Market structure swings ====
    swing_high, swing_low = find_recent_swings(df_1h, lookback=40, left=2, right=2)
    dist_to_swing_high_pct = None
    dist_to_swing_low_pct = None
    if swing_high is not None:
        dist_to_swing_high_pct = float((swing_high - price) / price * 100)
    if swing_low is not None:
        dist_to_swing_low_pct = float((price - swing_low) / price * 100)

    # ==== Composite meta-score ====
    score = 0.0

    # Trend contribution
    if trend_state_1h == "bull":
        score += 2.0
    elif trend_state_1h == "bear":
        score -= 2.0

    if trend_state_1d == "bull":
        score += 2.0
    elif trend_state_1d == "bear":
        score -= 2.0

    # Momentum contribution
    score += change_24h_pct / 10.0     # 10% → +1
    score += change_7d_pct / 50.0      # 50% → +1

    # Volume contribution
    if vol_ratio_24h is not None:
        score += (vol_ratio_24h - 1.0)  # >1 means rising volume

    # Volatility regime tweak
    if vol_regime == "compression":
        score += 0.5    # compression often precedes strong moves
    elif vol_regime == "expansion":
        score -= 0.5    # might be extended

    # RSI extremes tweak
    if rsi_14_1h is not None:
        if rsi_14_1h > 75:
            score -= 1.0
        elif rsi_14_1h < 30:
            score += 1.0

    # Trend maturity preference
    if bars_in_trend_1h > 80:
        score -= 1.0
    elif 10 < bars_in_trend_1h < 60:
        score += 0.5

    # Clip and scale to 0–100
    raw_score = max(min(score, 15.0), -15.0)
    scaled_score = 50.0 + (raw_score / 15.0) * 50.0  # -15→0, 0→50, +15→100

    features = {
        "current_price": price,
        "trend": {
            "trend_state_1h": trend_state_1h,
            "trend_state_1d": trend_state_1d,
            "trend_strength_1h": trend_strength_1h,
            "bars_in_trend_1h": int(bars_in_trend_1h),
            "above_50ema_1h": bool(df_1h["close"].iloc[-1] > df_1h["ema50_1h"].iloc[-1]),
            "above_200ema_1h": bool(df_1h["close"].iloc[-1] > df_1h["ema200_1h"].iloc[-1]),
            "above_50ema_1d": bool(df_1d["close"].iloc[-1] > df_1d["ema50_1d"].iloc[-1]),
            "above_200ema_1d": bool(df_1d["close"].iloc[-1] > df_1d["ema200_1d"].iloc[-1]),
        },
        "volatility": {
            "atr_1h": atr_1h,
            "atr_pct_1h": atr_pct_1h,
            "realized_vol_24h_pct": realized_vol_24h_pct,
            "bb_width_1h": bb_width,
            "vol_regime_1h": vol_regime,
        },
        "momentum": {
            "change_24h_pct": change_24h_pct,
            "change_7d_pct": change_7d_pct,
            "rsi_14_1h": rsi_14_1h,
            "macd_1h": macd_val,
            "macd_signal_1h": macd_signal,
            "macd_hist_1h": macd_hist,
        },
        "volume": {
            "volume_24h": vol_24h,
            "volume_prev_24h": vol_prev_24h,
            "volume_ratio_24h": vol_ratio_24h,
        },
        "structure": {
            "bullish_count_10": bullish_count_10,
            "bearish_count_10": bearish_count_10,
            "avg_body_ratio_10": avg_body_ratio_10,
            "recent_swing_high": swing_high,
            "recent_swing_low": swing_low,
            "dist_to_swing_high_pct": dist_to_swing_high_pct,
            "dist_to_swing_low_pct": dist_to_swing_low_pct,
        },
        "composite": {
            "raw_score": raw_score,
            "scaled_score": scaled_score,
        },
    }

    return features



# -----------------------------
# 3. Build Each Coin's Object
# -----------------------------
def build_coin_object(symbol="BTC-USD"):
    # 1h & 1d candles
    df_1h = get_candles(symbol, granularity=3600, limit=400)
    df_1d = get_candles(symbol, granularity=86400, limit=120)

    features = compute_features(df_1h, df_1d)

    # If feature extraction failed → return a safe, consistent structure
    if features is None:
        return {
            "symbol": symbol,
            "error": "Insufficient data for features",
            "price": None,
            "trend": None,
            "volatility": None,
            "momentum": None,
            "volume": None,
            "structure": None,
            "composite": None,
            "orderbook": None,
            "ticker": None,
        }

    # === SAFE ORDERBOOK EXTRACTION ===
    orderbook = get_orderbook(symbol)
    best_bid = None
    best_ask = None

    if isinstance(orderbook, dict):
        try:
            if "bids" in orderbook and orderbook["bids"]:
                best_bid = float(orderbook["bids"][0][0])
            if "asks" in orderbook and orderbook["asks"]:
                best_ask = float(orderbook["asks"][0][0])
        except Exception:
            best_bid = None
            best_ask = None

    # === SAFE TICKER EXTRACTION ===
    ticker = get_ticker(symbol)
    last_price = None
    bid = None
    ask = None

    if isinstance(ticker, dict):
        try:
            last_price = float(ticker.get("price")) if ticker.get("price") else None
        except:
            last_price = None
        try:
            bid = float(ticker.get("bid")) if ticker.get("bid") else None
        except:
            bid = None
        try:
            ask = float(ticker.get("ask")) if ticker.get("ask") else None
        except:
            ask = None

    # === RETURN CONSISTENT OBJECT WITH NEW FEATURE GROUPS ===
    return {
        "symbol": symbol,
        "price": features["current_price"],
        "trend": features["trend"],                  # <--- updated
        "volatility": features["volatility"],        # <--- updated
        "momentum": features["momentum"],            # <--- updated
        "volume": features["volume"],                # <--- updated
        "structure": features["structure"],          # <--- updated
        "composite": features["composite"],          # <--- updated (contains scaled_score)

        "orderbook": {
            "best_bid": best_bid,
            "best_ask": best_ask
        },

        "ticker": {
            "last_price": last_price,
            "bid": bid,
            "ask": ask
        }
    }





# -----------------------------
# 4. Build Market Snapshot
# -----------------------------
def build_market_snapshot(coins = [
    "MTL-USD", "INV-USD", "NEAR-USD", "HFT-USD", "ZETACHAIN-USD",
    "PRIME-USD", "FLOKI-USD", "DOGE-USD", "SUKU-USD", "GAL-USD",
    "QI-USD", "ALCX-USD", "SOL-USD", "MLN-USD", "MASK-USD",
    "NKN-USD", "ABT-USD", "WCFG-USD", "RLY-USD", "IOTX-USD",
    "COVAL-USD", "C98-USD", "ACS-USD", "TURBO-USD",
    "MSOL-USD", "ZETA-USD", "ORN-USD", "B3-USD", "CLV-USD",
    "ALICE-USD", "LOKA-USD", "KNC-USD", "FLOW-USD", "VARA-USD",
    "WELL-USD", "CORECHAIN-USD", "BCH-USD", "ANKR-USD",
    "PRQ-USD", "BAND-USD", "RGT-USD", "SPELL-USD",
    "ORCA-USD", "ATH-USD", "TRB-USD", "FOX-USD", "IDEX-USD",
    "ZEN-USD", "FLR-USD", "WAMPL-USD", "BIT-USD", "JTO-USD",
    "REP-USD", "GLM-USD", "TRUMP-USD", "LRC-USD", "JUP-USD",
    "MOG-USD", "LQTY-USD", "XTZ-USD", "RED-USD", "AURORA-USD",
    "UMA-USD", "SKL-USD", "FIL-USD", "PYTH-USD", "SEAM-USD",
    "PAXG-USD", "SUI-USD", "UNFI-USD", "T-USD", "GTC-USD",
    "VELO-USD", "MKR-USD", "RAI-USD", "BLUR-USD", "ACH-USD",
    "API3-USD", "ETH-USD", "FIS-USD", "PNG-USD", "ADA-USD",
    "MONA-USD", "ZK-USD", "YFI-USD", "POWR-USD", "JASMY-USD",
    "TVK-USD", "PENDLE-USD", "OCEAN-USD", "FX-USD", "POL-USD",
    "SUPER-USD", "WAXL-USD", "MATH-USD", "MOBILE-USD", "SHDW-USD",
    "YFII-USD", "LRDS-USD", "RPL-USD", "TRAC-USD", "LCX-USD",
    "CHZ-USD", "VTHO-USD", "TNSR-USD", "SEI-USD", "SAND-USD",
    "AGLD-USD", "GRT-USD", "FIDA-USD", "UPI-USD",
    "BNT-USD", "HNT-USD", "UNI-USD", "STORJ-USD", "VET-USD",
    "LTC-USD", "AST-USD", "AVAX-USD", "ARKM-USD", "HOPR-USD",
    "VOXEL-USD", "MANA-USD", "RARI-USD", "SHPING-USD", "PLA-USD",
     "POPCAT-USD", "FET-USD", "SWFTC-USD", "IP-USD",
    "UST-USD", "1INCH-USD", "DIA-USD", 
    "ICP-USD", "MNDE-USD", "MORPHO-USD", "SYRUP-USD", "HONEY-USD",
     "RBN-USD", "A8-USD", "ZEC-USD", "RLC-USD",
    "SXT-USD", "ILV-USD", "AKT-USD", "RNDR-USD", "ONDO-USD",
    "DEGEN-USD", "EGLD-USD", "SD-USD", "LOOM-USD", "PEPE-USD",
    "ACX-USD", "PYUSD-USD", "GHST-USD", "CRO-USD", "PROMPT-USD",
    "GYEN-USD", "XLM-USD", "LDO-USD", "KRL-USD", "ALEPH-USD",
    "OGN-USD", "PENGU-USD", "XYO-USD", "OXT-USD", "SYLO-USD",
    "WLD-USD", "OMG-USD", "BAL-USD", "PRO-USD", "SYN-USD",
    "EIGEN-USD", "GMT-USD", "GFI-USD", "RENDER-USD", "MINA-USD",
    "MAGIC-USD", "DYP-USD", "BICO-USD", "ZORA-USD", "COW-USD",
    "CELR-USD", "PIRATE-USD", "WLUNA-USD", "NEST-USD", "REZ-USD",
     "DESO-USD", "TAO-USD", "ME-USD", "RONIN-USD",
    "NCT-USD", "INJ-USD", "NU-USD", "OP-USD", "KARRAT-USD",
    "ALEO-USD", "INDEX-USD", "VGX-USD", "KERNEL-USD", "FAI-USD",
    "PUNDIX-USD", "AIOZ-USD", "AERO-USD", "ELA-USD", 
    "ALGO-USD", "IO-USD", "APT-USD", "CRV-USD", "MCO2-USD",
    "ERN-USD", "AUCTION-USD", "PRCL-USD", "VVV-USD", "BOBA-USD",
    "DREP-USD", "AXL-USD", "HBAR-USD", "MPL-USD", "STRK-USD",
    "REQ-USD", "BERA-USD", "AERGO-USD", "GODS-USD", "GALA-USD",
    "ROSE-USD",  "KSM-USD", "TRU-USD", "STG-USD",
    "AAVE-USD", "EDGE-USD", "LINK-USD", "TIA-USD", "RARE-USD",
    "WIF-USD", "BTC-USD", "ENJ-USD", "FORTH-USD", 
    "DOT-USD", "AUDIO-USD", "DNT-USD", "CTSI-USD", "00-USD",
    "RSR-USD", "GIGA-USD", "QNT-USD", "DOGINME-USD", "ATOM-USD",
    "DRIFT-USD", "KAVA-USD", "SPA-USD", "BAT-USD", "POLS-USD",
    "MATIC-USD", "SHIB-USD", "COTI-USD", "RAD-USD", "LSETH-USD",
    "IMX-USD", "GNO-USD", "L3-USD"
]):
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "coins": [build_coin_object(c) for c in coins]
    }


def build_full_snapshot(coins):
    """
    Stage 1: Build a full market snapshot for all coins.
    No GPT here, just Coinbase + features.
    """
    snap = {
        "timestamp": datetime.utcnow().isoformat(),
        "coins": []
    }

    for symbol in coins:
        try:
            coin_obj = build_coin_object(symbol)
        except Exception as e:
            coin_obj = {"symbol": symbol, "error": f"exception: {e}"}
        snap["coins"].append(coin_obj)

    return snap


def filter_candidates(full_snapshot,
                      min_volume_24h=50_000,   # filter obvious dead markets
                      min_atr_pct=0.3,         # filter ultra-flat markets
                      min_composite=25.0,      # filter total garbage
                      max_candidates=18):
    """
    Stage 1 filtering – NOT a precise ranking, just a broad quality gate.
    Keeps coins that are 'interesting enough' for GPT to look at.
    """
    candidates = []

    for coin in full_snapshot.get("coins", []):
        if coin.get("error"):
            continue

        vol = coin.get("volume", {}).get("volume_24h")
        atr_pct = coin.get("volatility", {}).get("atr_pct_1h")
        comp = coin.get("composite", {}).get("scaled_score")

        if vol is None or atr_pct is None or comp is None:
            continue

        # Very basic "is this tradeable?" checks
        if vol < min_volume_24h:
            continue
        if atr_pct < min_atr_pct:
            continue
        if comp < min_composite:
            continue

        candidates.append(coin)

    # Optional: sort by composite score just so GPT sees the stronger ones first
    candidates.sort(
        key=lambda c: c.get("composite", {}).get("scaled_score", 0.0),
        reverse=True
    )

    # Keep up to N candidates – broad set, not just 1-2
    return candidates[:max_candidates]


def build_reduced_snapshot(full_snapshot, candidate_coins):
    """
    Build the smaller snapshot we actually send to GPT.
    Same shape as full snapshot, just fewer coins.
    """
    return {
        "timestamp": full_snapshot["timestamp"],
        "coins": candidate_coins
    }


# -----------------------------
# 5. ChatGPT Prompt
# -----------------------------
SYSTEM_PROMPT = "You are an elite crypto trading analyst specializing in long/short setup selection using structured market feature data."


USER_PROMPT = """
IMPORTANT — OUTPUT RULES:
- Your response MUST be valid JSON.
- The FIRST character of your response MUST be '{'.
- Do NOT include markdown, code blocks, commentary, or any text before or after the JSON.
- If you cannot produce valid JSON, return {}.


You are an elite crypto market analyst operating at a quant-discretionary hybrid level.
You receive a reduced list of candidate coins that are already filtered to be "interesting enough".
Your job is to:
- Evaluate both long and short potential on EACH candidate,
- Consider the entire feature set jointly (trend, volatility, structure, momentum, volume, composite score),
- Select high-conviction setups only.

FEATURES YOU SEE (per coin):
- trend: multi-timeframe trend states, strength, and maturity (bars_in_trend_1h)
- volatility: ATR, ATR%, realized volatility, volatility regime (compression/normal/expansion)
- momentum: 24h/7d % change, RSI, MACD and histogram
- volume: 24h volume and 24h vs previous 24h ratio
- structure: bullish/bearish candle counts, body-to-wick ratio, recent swing highs/lows and distance
- composite: raw_score and scaled_score (0–100), which summarize signal quality

REASONING RULES (use these internally, then only output JSON):
- Long setups:
    - Prefer coins with bullish 1h and 1d trend, healthy trend_strength_1h,
      reasonable bars_in_trend_1h (not extremely extended),
      positive or recovering momentum, supportive structure near swing lows or demand.
    - Volatility regime: compression with rising volume or early expansion is ideal.
    - RSI: mid-range (35–65) is healthier than extreme.
- Short setups:
    - Prefer coins with bearish 1h and 1d trend or obvious exhaustion after a long uptrend,
      weakening momentum, structure near prior highs/supply, and stretched volatility.
    - Be conservative with shorts if most candidates are strongly bullish.
- Composite.scaled_score is a hint of overall quality, but do NOT blindly follow it.
  Use it as one factor among many. A coin with slightly lower score can still be a top setup if structure, volatility, and momentum align beautifully.

- Avoid setups where:
    - volume_24h is low,
    - volatility regime is chaotic and structure unclear,
    - RSI is extremely overbought/oversold AND structure does not clearly support the trade,
    - there is no clean and logical place for stop placement.
    - If any coin is missing price, volume, candles, or features, you MUST exclude it entirely and NOT produce a setup for it.Do NOT infer or guess prices or entries.




TASK:
1. For all candidates, reason internally about:
   - long_strength (0–100)
   - short_strength (0–100)
   - overall_setup_quality (A+, A, B, avoid)

2. Select ONLY:
   - the top 3 long setups, and
   - the top 3 short setups,
   IF they are at least quality B.
   If fewer valid setups exist on either side, return fewer.
   If there are no good setups at all, return empty lists.

3. For each returned setup, output:
   - symbol
   - setup_quality: "A+", "A", "B" (do NOT return "avoid" in final list)
   - bias: "long" or "short"
   - entry_zone: a price RANGE [low, high], not just a single price:
       LONG:
         - Prefer an entry range BELOW current price near support, swing lows, EMAs,
           or volatility pockets, unless price is exactly at ideal support.
       SHORT:
         - Prefer an entry range ABOVE current price near resistance, prior highs,
           or breakdown retests, unless price is exactly at ideal resistance.
   - stop_loss:
       LONG: below a sensible invalidation level (below local structure, volatility cluster, or swing low).
       SHORT: above a sensible invalidation level (above local structure, swing high, or liquidity pocket).
   - take_profits: 2–3 targets that make sense:
       Use prior structure levels, recent extremes, or volatility-based projections.
   - rationale:
       A concise explanation referencing:
         - trend (1h & 1d),
         - volatility regime,
         - momentum (24h/7d, RSI, MACD),
         - structure (swings, support/resistance),
         - and any notable composite score context.

OUTPUT FORMAT:
Return ONLY valid JSON in this exact structure:

{
  "long_setups": [
     {
        "symbol": "...",
        "setup_quality": "...",
        "bias": "long",
        "entry_zone": [low, high],
        "stop_loss": ...,
        "take_profits": [...],
        "rationale": "..."
     }
  ],
  "short_setups": [
     {
        "symbol": "...",
        "setup_quality": "...",
        "bias": "short",
        "entry_zone": [low, high],
        "stop_loss": ...,
        "take_profits": [...],
        "rationale": "..."
     }
  ]
}

Market snapshot (candidates only):
<<SNAPSHOT>>
"""







from openai import OpenAI
import json

def ask_chatgpt_stage2(reduced_snapshot, model_name="gpt-5.1"):
    client = OpenAI()

    snapshot_json = json.dumps(reduced_snapshot, default=str)

    # Replace placeholder safely (no .format!)
    prompt = USER_PROMPT.replace("<<SNAPSHOT>>", snapshot_json)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"error": "JSON decode failed", "raw": content}



if __name__ == "__main__":
    print("Building full market snapshot (Stage 1)...")
    full_snapshot = build_full_snapshot(ALL_COINS)

    print(f"Total coins in snapshot: {len(full_snapshot['coins'])}")

    candidates = filter_candidates(full_snapshot)
    print(f"Candidate coins after Stage 1 filter: {len(candidates)}")
    print("Candidate symbols:", [c["symbol"] for c in candidates])

    if not candidates:
        print("No valid candidates found after filtering. Exiting.")
    else:
        reduced_snapshot = build_reduced_snapshot(full_snapshot, candidates)
        print("\nSending reduced snapshot to GPT (Stage 2)...\n")

        analysis = ask_chatgpt_stage2(reduced_snapshot, model_name="gpt-5.1")

        print("\nAI Result:")
        print(json.dumps(analysis, indent=2))



