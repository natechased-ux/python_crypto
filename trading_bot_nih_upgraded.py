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
    "PRQ-USD", "BAND-USD", "RGT-USD", "SPELL-USD", "OSMO-USD",
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
    "AGLD-USD", "GRT-USD", "ASM-USD", "FIDA-USD", "UPI-USD",
    "BNT-USD", "HNT-USD", "UNI-USD", "STORJ-USD", "VET-USD",
    "LTC-USD", "AST-USD", "AVAX-USD", "ARKM-USD", "HOPR-USD",
    "VOXEL-USD", "MANA-USD", "RARI-USD", "SHPING-USD", "PLA-USD",
    "SNT-USD", "POPCAT-USD", "FET-USD", "SWFTC-USD", "IP-USD",
    "UST-USD", "1INCH-USD", "DIA-USD", "DEXT-USD", "KEEP-USD",
    "ICP-USD", "MNDE-USD", "MORPHO-USD", "SYRUP-USD", "HONEY-USD",
    "AVT-USD", "RBN-USD", "A8-USD", "ZEC-USD", "RLC-USD",
    "SXT-USD", "ILV-USD", "AKT-USD", "RNDR-USD", "ONDO-USD",
    "DEGEN-USD", "EGLD-USD", "SD-USD", "LOOM-USD", "PEPE-USD",
    "ACX-USD", "PYUSD-USD", "GHST-USD", "CRO-USD", "PROMPT-USD",
    "GYEN-USD", "XLM-USD", "LDO-USD", "KRL-USD", "ALEPH-USD",
    "OGN-USD", "PENGU-USD", "XYO-USD", "OXT-USD", "SYLO-USD",
    "WLD-USD", "OMG-USD", "BAL-USD", "PRO-USD", "SYN-USD",
    "EIGEN-USD", "GMT-USD", "GFI-USD", "RENDER-USD", "MINA-USD",
    "MAGIC-USD", "DYP-USD", "BICO-USD", "ZORA-USD", "COW-USD",
    "CELR-USD", "PIRATE-USD", "WLUNA-USD", "NEST-USD", "REZ-USD",
    "OMNI-USD", "DESO-USD", "TAO-USD", "ME-USD", "RONIN-USD",
    "NCT-USD", "INJ-USD", "NU-USD", "OP-USD", "KARRAT-USD",
    "ALEO-USD", "INDEX-USD", "VGX-USD", "KERNEL-USD", "FAI-USD",
    "PUNDIX-USD", "AIOZ-USD", "AERO-USD", "ELA-USD", "PYR-USD",
    "ALGO-USD", "IO-USD", "APT-USD", "CRV-USD", "MCO2-USD",
    "ERN-USD", "AUCTION-USD", "PRCL-USD", "VVV-USD", "BOBA-USD",
    "DREP-USD", "AXL-USD", "HBAR-USD", "MPL-USD", "STRK-USD",
    "REQ-USD", "BERA-USD", "AERGO-USD", "GODS-USD", "GALA-USD",
    "ROSE-USD", "MANTLE-USD", "KSM-USD", "TRU-USD", "STG-USD",
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
    Enhanced feature computation for a single symbol + Nihilius bottom-detection signals.

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
        df_1h[col] = pd.to_numeric(df_1h[col], errors="coerce")
        df_1d[col] = pd.to_numeric(df_1d[col], errors="coerce")

    df_1h = df_1h.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    df_1d = df_1d.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    if len(df_1h) < 200 or len(df_1d) < 60:
        return None

    price = float(df_1h["close"].iloc[-1])
    if not np.isfinite(price) or price <= 0:
        return None

    # ===========================
    # EMAs (1h and 1d)
    # ===========================
    df_1h["ema50_1h"] = df_1h["close"].ewm(span=50, adjust=False).mean()
    df_1h["ema200_1h"] = df_1h["close"].ewm(span=200, adjust=False).mean()

    df_1d["ema50_1d"] = df_1d["close"].ewm(span=50, adjust=False).mean()
    df_1d["ema200_1d"] = df_1d["close"].ewm(span=200, adjust=False).mean()

    def trend_state(close, ema50, ema200):
        if close > ema50 and ema50 > ema200:
            return "bull"
        if close < ema50 and ema50 < ema200:
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

    # Trend maturity label (for GPT)
    if bars_in_trend_1h >= 120:
        trend_maturity = "late"
    elif bars_in_trend_1h >= 60:
        trend_maturity = "mid"
    else:
        trend_maturity = "early"

    # ===========================
    # Volatility: ATR (1h) + realized vol proxy (24h)
    # ===========================
    high = df_1h["high"]
    low = df_1h["low"]
    close = df_1h["close"]

    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_1h = float(tr.rolling(14).mean().iloc[-1])
    atr_pct_1h = float(atr_1h / price * 100) if atr_1h and price else None

    # "Realized volatility" proxy: std of hourly returns over last 24 bars
    returns_1h = close.pct_change()
    realized_vol_24h = float(returns_1h.tail(24).std() * np.sqrt(24)) if len(returns_1h) >= 30 else None
    realized_vol_24h_pct = float(realized_vol_24h * 100) if realized_vol_24h is not None else None

    # Volatility regime: compare last 24h range to prior 7d range average
    rng_1h = (high - low).abs()
    recent_rng = float(rng_1h.tail(24).mean())
    prior_rng = float(rng_1h.tail(24 * 7).mean()) if len(rng_1h) >= 24 * 7 else recent_rng
    if prior_rng <= 0:
        vol_regime = "normal"
    else:
        ratio = recent_rng / prior_rng
        if ratio < 0.75:
            vol_regime = "compression"
        elif ratio > 1.25:
            vol_regime = "expansion"
        else:
            vol_regime = "normal"

    # ===========================
    # Momentum: RSI & MACD (1h), plus 24h/7d % change
    # ===========================
    def rsi(series, period=14):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.rolling(period).mean()
        ma_down = down.rolling(period).mean()
        rs = ma_up / ma_down.replace(0, np.nan)
        out = 100 - (100 / (1 + rs))
        return out

    rsi_14_1h = float(rsi(close, 14).iloc[-1]) if len(close) >= 20 else None

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    macd_val = float(macd.iloc[-1])
    macd_signal_val = float(macd_signal.iloc[-1])
    macd_hist_val = float(macd_hist.iloc[-1])

    # 24h and 7d momentum (approx)
    chg_24h = float((close.iloc[-1] / close.iloc[-25] - 1) * 100) if len(close) >= 25 else None
    chg_7d = float((close.iloc[-1] / close.iloc[-(24 * 7 + 1)] - 1) * 100) if len(close) >= (24 * 7 + 1) else None

    # ===========================
    # Volume: 24h volume & 24h vs prev 24h ratio
    # ===========================
    vol_24h = float(df_1h["volume"].tail(24).sum())
    prev_vol_24h = float(df_1h["volume"].tail(48).head(24).sum()) if len(df_1h) >= 48 else vol_24h
    vol_ratio_24h = float(vol_24h / prev_vol_24h) if prev_vol_24h and prev_vol_24h > 0 else None

    # ===========================
    # Structure: simple candle stats + swing highs/lows
    # ===========================
    # last 10 candles bullish count
    last10 = df_1h.tail(10)
    bullish_count_10 = int((last10["close"] > last10["open"]).sum())
    bearish_count_10 = int((last10["close"] < last10["open"]).sum())

    # body-to-wick ratio
    body = (last10["close"] - last10["open"]).abs()
    wick = (last10["high"] - last10[["close", "open"]].max(axis=1)).abs() + (last10[["close", "open"]].min(axis=1) - last10["low"]).abs()
    body_to_wick = float((body.sum() / wick.sum())) if wick.sum() and wick.sum() > 0 else None

    # swing high/low (last 50)
    look = df_1h.tail(50)
    swing_high = float(look["high"].max())
    swing_low = float(look["low"].min())
    dist_to_swing_high_pct = float((swing_high - price) / price * 100) if price else None
    dist_to_swing_low_pct = float((price - swing_low) / price * 100) if price else None

    # ===========================
    # Composite (0-100) - lightweight hint only
    # ===========================
    raw_score = 0.0
    # trend alignment
    if trend_state_1h == "bull":
        raw_score += 1.5
    elif trend_state_1h == "bear":
        raw_score -= 1.0

    if trend_state_1d == "bull":
        raw_score += 2.0
    elif trend_state_1d == "bear":
        raw_score -= 1.5

    # vol regime
    if vol_regime == "compression":
        raw_score += 1.0
    elif vol_regime == "expansion":
        raw_score += 0.2

    # momentum
    if rsi_14_1h is not None:
        if 40 <= rsi_14_1h <= 60:
            raw_score += 1.0
        elif rsi_14_1h < 30 or rsi_14_1h > 75:
            raw_score -= 0.5

    if macd_hist_val > 0:
        raw_score += 0.6
    else:
        raw_score -= 0.3

    if vol_ratio_24h is not None and vol_ratio_24h > 1.1:
        raw_score += 0.6

    # scale
    scaled_score = float(max(0.0, min(100.0, (raw_score + 5) * 10)))

    # =====================================================
    # NIHILIUS 1–7 UPGRADES
    # =====================================================

    # Lookback window for Nihilius pattern detection (about 10 days of 1h)
    lb = min(240, len(df_1h))
    recent = df_1h.tail(lb).reset_index(drop=True)

    # (1) Early vs Confirmed bottoms
    # Confirmed: reclaim above ema50 + improving momentum
    confirmed_break = bool(
        price > float(df_1h["ema50_1h"].iloc[-1])
        and macd_hist_val > 0
        and rsi_14_1h is not None
        and rsi_14_1h >= 40
    )
    early_bottom = bool(
        (vol_regime == "compression")
        and (dist_to_swing_low_pct is not None and dist_to_swing_low_pct <= 5.0)
        and (rsi_14_1h is not None and 28 <= rsi_14_1h <= 45)
        and (not confirmed_break)
    )
    phase = "confirmed" if confirmed_break else "early" if early_bottom else "none"

    # (2) Trendline break strength
    # Fit a simple trendline to recent highs (last 60 bars)
    hi_win = min(60, len(recent))
    highs = recent["high"].tail(hi_win).astype(float).values
    x = np.arange(len(highs), dtype=float)
    if len(highs) >= 3:
        slope, intercept = np.polyfit(x, highs, 1)
        trendline_now = float(slope * (len(highs) - 1) + intercept)
    else:
        trendline_now = float(highs[-1]) if len(highs) else price

    break_strength_pct = float((price - trendline_now) / trendline_now * 100) if trendline_now > 0 else 0.0
    strong_break = bool(break_strength_pct > 1.5)
    weak_break = bool(0.0 < break_strength_pct <= 1.5)

    # (3) Base touches (horizontal base)
    base_window = min(60, len(recent))
    base = recent.tail(base_window)
    base_level = float(base["low"].mean())
    tol = 0.01  # 1%
    base_touches = int(((base["low"] - base_level).abs() / base_level < tol).sum()) if base_level > 0 else 0

    if base_touches >= 6:
        base_quality = "excellent"
    elif base_touches >= 4:
        base_quality = "good"
    elif base_touches >= 3:
        base_quality = "weak"
    else:
        base_quality = "poor"

    # (4) Precomputed entry zones
    base_low = float(base["low"].min())
    base_high = float(base["low"].max())
    # Trendline retest zone +/- 1%
    tr_low = float(trendline_now * 0.99)
    tr_high = float(trendline_now * 1.01)

    entry_zones = {
        "base": [base_low, base_high],
        "trendline_retest": [min(tr_low, tr_high), max(tr_low, tr_high)],
    }

    # (5) Structured context signals (labels)
    volatility_state = "compressed" if vol_regime == "compression" else "expanded" if vol_regime == "expansion" else "normal"
    breakout_state = "confirmed_break" if confirmed_break else "strong_break" if strong_break else "weak_break" if weak_break else "no_break"
    structure_state = "base" if base_quality in ("good", "excellent") else "unclear"

    context = {
        "trend_maturity": trend_maturity,
        "volatility_state": volatility_state,
        "structure_state": structure_state,
        "breakout_state": breakout_state,
    }

    # (7) Time-in-base (bars hugging the base)
    # Count last 60 bars where low is within 1% of base_level
    bars_in_base = int(((base["low"] - base_level).abs() / base_level < tol).sum()) if base_level > 0 else 0

    # (6) Refuse trades if invalid / poor-quality context
    # We'll compute a boolean "is_tradeable" that Stage 1 & GPT both use.
    is_tradeable = bool(
        vol_24h > 50_000
        and atr_pct_1h is not None and atr_pct_1h >= 0.25
        and base_quality in ("good", "excellent")
        and volatility_state in ("compressed", "normal")
    )

    # Nihilius score (0-100): weighted, but not overly strict
    # downtrend must exist historically (prolonged downtrend), then base+compression+break adds conviction
    pct_drop = float((recent["close"].iloc[0] - price) / recent["close"].iloc[0] * 100) if recent["close"].iloc[0] > 0 else 0.0
    prolonged_downtrend = bool(pct_drop >= 30.0 and trend_state_1d in ("bear", "mixed"))

    score_raw = 0.0
    if prolonged_downtrend:
        score_raw += 2.5
    if volatility_state == "compressed":
        score_raw += 2.0
    if base_quality in ("good", "excellent"):
        score_raw += 2.5
    if momentum_shift or confirmed_break:
        score_raw += 1.5
    if strong_break:
        score_raw += 1.5
    if vol_ratio_24h is not None and vol_ratio_24h >= 1.2:
        score_raw += 1.0
    if bars_in_base >= 20:
        score_raw += 1.0

    # cap and scale
    score_raw = max(0.0, min(12.0, score_raw))
    nihilius_score = float(score_raw / 12.0 * 100.0)
    nihilius_candidate = bool(nihilius_score >= 60.0 and is_tradeable)

    features = {
        "current_price": price,
        "trend": {
            "trend_state_1h": trend_state_1h,
            "trend_state_1d": trend_state_1d,
            "trend_strength_1h": trend_strength_1h,
            "bars_in_trend_1h": bars_in_trend_1h,
            "trend_maturity": trend_maturity,
        },
        "volatility": {
            "atr_1h": atr_1h,
            "atr_pct_1h": atr_pct_1h,
            "realized_vol_24h_pct": realized_vol_24h_pct,
            "vol_regime": vol_regime,
        },
        "momentum": {
            "chg_24h_pct": chg_24h,
            "chg_7d_pct": chg_7d,
            "rsi_14_1h": rsi_14_1h,
            "macd": macd_val,
            "macd_signal": macd_signal_val,
            "macd_hist": macd_hist_val,
        },
        "volume": {
            "volume_24h": vol_24h,
            "volume_ratio_24h": vol_ratio_24h,
        },
        "structure": {
            "bullish_count_10": bullish_count_10,
            "bearish_count_10": bearish_count_10,
            "body_to_wick": body_to_wick,
            "swing_high": swing_high,
            "swing_low": swing_low,
            "dist_to_swing_high_pct": dist_to_swing_high_pct,
            "dist_to_swing_low_pct": dist_to_swing_low_pct,
        },
        "composite": {
            "raw_score": raw_score,
            "scaled_score": scaled_score,
        },
        "nihilius": {
            "score": nihilius_score,
            "is_candidate": nihilius_candidate,
            "phase": phase,
            "prolonged_downtrend": prolonged_downtrend,
            "pct_drop_lookback": pct_drop,
            "base_touches": base_touches,
            "base_quality": base_quality,
            "bars_in_base": bars_in_base,
            "volatility_state": volatility_state,
            "trendline_now": trendline_now,
            "break_strength_pct": break_strength_pct,
            "strong_break": strong_break,
            "entry_zones": entry_zones,
            "context": context,
            "is_tradeable": is_tradeable,
        },
    }

    return features


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
            "nihilius": None,
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
    "PRQ-USD", "BAND-USD", "RGT-USD", "SPELL-USD", "OSMO-USD",
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
    "AGLD-USD", "GRT-USD", "ASM-USD", "FIDA-USD", "UPI-USD",
    "BNT-USD", "HNT-USD", "UNI-USD", "STORJ-USD", "VET-USD",
    "LTC-USD", "AST-USD", "AVAX-USD", "ARKM-USD", "HOPR-USD",
    "VOXEL-USD", "MANA-USD", "RARI-USD", "SHPING-USD", "PLA-USD",
    "SNT-USD", "POPCAT-USD", "FET-USD", "SWFTC-USD", "IP-USD",
    "UST-USD", "1INCH-USD", "DIA-USD", "DEXT-USD", "KEEP-USD",
    "ICP-USD", "MNDE-USD", "MORPHO-USD", "SYRUP-USD", "HONEY-USD",
    "AVT-USD", "RBN-USD", "A8-USD", "ZEC-USD", "RLC-USD",
    "SXT-USD", "ILV-USD", "AKT-USD", "RNDR-USD", "ONDO-USD",
    "DEGEN-USD", "EGLD-USD", "SD-USD", "LOOM-USD", "PEPE-USD",
    "ACX-USD", "PYUSD-USD", "GHST-USD", "CRO-USD", "PROMPT-USD",
    "GYEN-USD", "XLM-USD", "LDO-USD", "KRL-USD", "ALEPH-USD",
    "OGN-USD", "PENGU-USD", "XYO-USD", "OXT-USD", "SYLO-USD",
    "WLD-USD", "OMG-USD", "BAL-USD", "PRO-USD", "SYN-USD",
    "EIGEN-USD", "GMT-USD", "GFI-USD", "RENDER-USD", "MINA-USD",
    "MAGIC-USD", "DYP-USD", "BICO-USD", "ZORA-USD", "COW-USD",
    "CELR-USD", "PIRATE-USD", "WLUNA-USD", "NEST-USD", "REZ-USD",
    "OMNI-USD", "DESO-USD", "TAO-USD", "ME-USD", "RONIN-USD",
    "NCT-USD", "INJ-USD", "NU-USD", "OP-USD", "KARRAT-USD",
    "ALEO-USD", "INDEX-USD", "VGX-USD", "KERNEL-USD", "FAI-USD",
    "PUNDIX-USD", "AIOZ-USD", "AERO-USD", "ELA-USD", "PYR-USD",
    "ALGO-USD", "IO-USD", "APT-USD", "CRV-USD", "MCO2-USD",
    "ERN-USD", "AUCTION-USD", "PRCL-USD", "VVV-USD", "BOBA-USD",
    "DREP-USD", "AXL-USD", "HBAR-USD", "MPL-USD", "STRK-USD",
    "REQ-USD", "BERA-USD", "AERGO-USD", "GODS-USD", "GALA-USD",
    "ROSE-USD", "MANTLE-USD", "KSM-USD", "TRU-USD", "STG-USD",
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
                      min_volume_24h=50_000,
                      max_candidates=18):
    """
    NIHILIUS Stage-1 filter (broad-but-focused):
    - Keep USD/USDT markets already collected
    - Remove broken/incomplete coins
    - Require nihilius.is_candidate True (score>=60 + tradeable context)
    - Sort by nihilius.score (descending)
    """
    candidates = []

    for coin in full_snapshot.get("coins", []):
        if coin.get("error"):
            continue

        vol = coin.get("volume", {}).get("volume_24h")
        nih = coin.get("nihilius", {}) or {}
        score = nih.get("score", 0.0)
        is_cand = nih.get("is_candidate", False)

        if vol is None or vol < min_volume_24h:
            continue
        if not is_cand:
            continue
        # extra guard: base quality must be at least good
        if nih.get("base_quality") not in ("good", "excellent"):
            continue

        candidates.append(coin)

    candidates.sort(
        key=lambda c: (
            c.get("nihilius", {}).get("score", 0.0),
            c.get("composite", {}).get("scaled_score", 0.0)
        ),
        reverse=True
    )

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
SYSTEM_PROMPT = (
    "You are an elite crypto bottom-finding analyst specialized in Nihilius-style reversal setups "
    "after prolonged downtrends. You are conservative and prefer clean structure and good risk/reward."
)

USER_PROMPT = """
IMPORTANT — OUTPUT RULES:
- Your response MUST be valid JSON.
- The FIRST character of your response MUST be '{'.
- Do NOT include markdown, code blocks, commentary, or any text before or after the JSON.
- If you cannot produce valid JSON, return {}.

You are scanning for **Nihilius-style bottoming long setups** (NO SHORTS).

A Nihilius setup typically has:
- A prolonged downtrend (multi-day) that appears exhausted
- Volatility compression into a tight coil / wedge
- A horizontal base / accumulation with multiple base touches
- Early reclaim or confirmed breakout (trendline break + higher low)
- Preferably a volume uptick as price breaks/reclaims

You will receive candidate coins that already include a `nihilius` block:
- nihilius.score (0–100)
- nihilius.phase: "early" or "confirmed"
- nihilius.base_quality: poor/weak/good/excellent
- nihilius.base_touches, nihilius.bars_in_base
- nihilius.break_strength_pct and nihilius.strong_break
- nihilius.entry_zones: { base: [low, high], trendline_retest: [low, high] }
- nihilius.context: labels for trend_maturity, volatility_state, structure_state, breakout_state

STRICT RULES:
- Return ONLY long setups.
- DO NOT invent prices or entry zones: use snapshot values (current_price, entry_zones, swings).
- Only consider coins where nihilius.is_candidate == true AND nihilius.score >= 60.
- Prefer base_quality "good" or "excellent".
- Prefer phase "confirmed" unless an "early" setup has exceptional asymmetry and a very clean base.
- If no setups meet quality, return empty list.

(Refuse trades) If the setup is structurally correct but there is no clean stop or the R:R is unclear,
SKIP the coin. It's better to output fewer trades than weak ones.

TASK:
1) Rank all candidate coins by **bottoming long potential**, strongest first (use the whole snapshot jointly).
2) Select ONLY the top **3** long setups (quality B or better). If fewer exist, return fewer.
3) For each selected setup output:
   - symbol
   - setup_quality: "A+", "A", or "B"
   - bias: "long"
   - entry_zone: [low, high] picked from either:
       - nihilius.entry_zones.base, OR
       - nihilius.entry_zones.trendline_retest
     Unless current price is exactly at the best zone.
   - stop_loss: below base low or below a clear invalidation level (swing_low / base shelf).
   - take_profits: 2–3 targets based on prior structure (swing highs), volatility multiples (ATR), or mean reversion.
   - rationale: concise and specific referencing:
       - downtrend exhaustion,
       - volatility compression,
       - base touches / time in base,
       - breakout strength / phase,
       - and why the chosen entry zone makes sense.

Return ONLY valid JSON in this exact format:

{
  "nihilius_setups": [
     {
        "symbol": "...",
        "setup_quality": "...",
        "bias": "long",
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



