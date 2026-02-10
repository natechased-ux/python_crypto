import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
from openai import OpenAI
import json
import math

# =====================================================
# CONFIG
# =====================================================
PUBLIC_API = "https://api.exchange.coinbase.com"

SYSTEM_PROMPT = (
    "You are an elite crypto bottom-finding analyst specializing in "
    "Nihilius-style reversal setups after prolonged downtrends."
)

print("NIHILIUS BOT STARTED")

# =====================================================
# COIN UNIVERSE (USD / USDT ONLY, NO STABLE BASES)
# =====================================================
ALL_COINS = [
      "NEAR-USD", "HFT-USD", "ZETACHAIN-USD",
     "FLOKI-USD", "DOGE-USD", "SUKU-USD", "GAL-USD",
    "QI-USD", "SOL-USD", "MLN-USD", "MASK-USD",
     "ABT-USD", "WCFG-USD", "RLY-USD", "IOTX-USD",
    "COVAL-USD", "C98-USD", "TURBO-USD",
     "ZETA-USD", "ORN-USD", "B3-USD", "CLV-USD",
    "ALICE-USD", "LOKA-USD", "FLOW-USD", "VARA-USD",
    "WELL-USD",  "BCH-USD", "ANKR-USD",
    "PRQ-USD", "BAND-USD", "RGT-USD", "SPELL-USD", "OSMO-USD",
    "ORCA-USD", "ATH-USD", "TRB-USD", "FOX-USD",
    "ZEN-USD", "FLR-USD", "WAMPL-USD", "BIT-USD", "JTO-USD",
     "GLM-USD", "TRUMP-USD", "LRC-USD", "JUP-USD",
    "MOG-USD", "LQTY-USD", "XTZ-USD", "RED-USD", "AURORA-USD",
    "UMA-USD", "SKL-USD", "FIL-USD", "PYTH-USD", "SEAM-USD",
     "SUI-USD", "UNFI-USD", "T-USD", "GTC-USD",
      "RAI-USD", "BLUR-USD", "ACH-USD",
    "API3-USD",  "FIS-USD", "ADA-USD",
    "MONA-USD", "ZK-USD", "YFI-USD", "JASMY-USD",
    "TVK-USD", "PENDLE-USD", "OCEAN-USD", "FX-USD", "POL-USD",
    "SUPER-USD", "MOBILE-USD", "SHDW-USD",
    "YFII-USD", "LRDS-USD", "RPL-USD", "LCX-USD",
    "CHZ-USD", "VTHO-USD", "TNSR-USD", "SEI-USD", "SAND-USD",
    "AGLD-USD", "GRT-USD", "ASM-USD", "FIDA-USD", "UPI-USD",
    "BNT-USD", "HNT-USD", "UNI-USD", "VET-USD",
    "LTC-USD", "AST-USD", "AVAX-USD", "ARKM-USD",
    "VOXEL-USD", "MANA-USD", "SHPING-USD", "PLA-USD",
    "SNT-USD", "POPCAT-USD", "FET-USD", "IP-USD",
    "UST-USD", "1INCH-USD", "DIA-USD", "DEXT-USD", 
    "ICP-USD", "MORPHO-USD", "SYRUP-USD", "HONEY-USD",
     "RBN-USD", "ZEC-USD", "RLC-USD",
    "SXT-USD", "ILV-USD", "AKT-USD", "RNDR-USD", "ONDO-USD",
    "DEGEN-USD", "EGLD-USD", "SD-USD",  "PEPE-USD",
    "ACX-USD", "PYUSD-USD", "GHST-USD", "CRO-USD", "PROMPT-USD",
    "GYEN-USD", "XLM-USD", "LDO-USD",
    "OGN-USD", "PENGU-USD", "XYO-USD", "OXT-USD", "SYLO-USD",
    "WLD-USD", "OMG-USD", "BAL-USD", "SYN-USD",
    "EIGEN-USD", "GMT-USD", "GFI-USD", "RENDER-USD", "MINA-USD",
    "MAGIC-USD", "DYP-USD", "ZORA-USD", "COW-USD",
    "CELR-USD", "PIRATE-USD", "WLUNA-USD", "NEST-USD", "REZ-USD",
     "DESO-USD", "TAO-USD", "ME-USD", "RONIN-USD",
    "NCT-USD", "INJ-USD", "NU-USD", "OP-USD", "KARRAT-USD",
    "ALEO-USD",  "VGX-USD", "KERNEL-USD", 
    "PUNDIX-USD", "AIOZ-USD", "AERO-USD", "ELA-USD", "PYR-USD",
    "ALGO-USD", "IO-USD", "APT-USD", "CRV-USD", "MCO2-USD",
    "ERN-USD", "AUCTION-USD", "PRCL-USD", "VVV-USD", "BOBA-USD",
    "DREP-USD", "AXL-USD", "HBAR-USD", "STRK-USD",
     "BERA-USD", "AERGO-USD", "GODS-USD", "GALA-USD",
    "ROSE-USD", "KSM-USD", "TRU-USD", "STG-USD",
    "AAVE-USD", "EDGE-USD", "LINK-USD", "TIA-USD", "RARE-USD",
    "WIF-USD", "ENJ-USD", "FORTH-USD", 
    "DOT-USD", "AUDIO-USD", "DNT-USD", "CTSI-USD",
    "RSR-USD", "GIGA-USD", "QNT-USD", "DOGINME-USD", "ATOM-USD",
    "DRIFT-USD", "KAVA-USD", "SPA-USD", "BAT-USD", "POLS-USD",
    "MATIC-USD", "SHIB-USD", "COTI-USD", "RAD-USD",
    "IMX-USD",  "L3-USD"
]
# =====================================================
# INDICATORS
# =====================================================
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


# =====================================================
# DATA FETCH
# =====================================================
def get_candles(product_id, granularity, limit):
    url = f"{PUBLIC_API}/products/{product_id}/candles"
    try:
        r = requests.get(url, params={"granularity": granularity}, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None

    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").reset_index(drop=True)
    return df.tail(limit)


def get_ticker(product_id):
    try:
        return requests.get(
            f"{PUBLIC_API}/products/{product_id}/ticker", timeout=10
        ).json()
    except Exception:
        return None


# =====================================================
# NIHILIUS FEATURE ENGINE (UPGRADED)
# =====================================================
def compute_features(df_6h, df_1d):
    if df_6h is None or df_1d is None:
        return None
    if len(df_6h) < 150 or len(df_1d) < 60:
        return None

    price = float(df_6h["close"].iloc[-1])

    df_6h["ema25"] = df_6h["close"].ewm(span=25).mean()
    df_6h["ema50"] = df_6h["close"].ewm(span=50).mean()
    df_6h["ema200"] = df_6h["close"].ewm(span=200).mean()

    rsi = calc_rsi(df_6h["close"])
    macd, signal, hist = calc_macd(df_6h["close"])

    # -------------------------------------------------
    # 1. Downtrend maturity
    # -------------------------------------------------
    ema_slope = (df_6h["ema50"].iloc[-1] - df_6h["ema50"].iloc[-30]) / price
    prolonged_downtrend = (
        df_6h["ema50"].iloc[-1] < df_6h["ema200"].iloc[-1]
        and ema_slope < -0.002
    )

    # -------------------------------------------------
    # 2. Volatility compression
    # -------------------------------------------------
    ranges = (df_6h["high"] - df_6h["low"]) / price * 100
    early_vol = ranges.iloc[: len(ranges)//2].mean()
    late_vol = ranges.iloc[len(ranges)//2 :].mean()
    compression = late_vol < early_vol * 0.8

    # -------------------------------------------------
    # 3. Flat base + base touches
    # -------------------------------------------------
    recent_lows = df_6h["low"].tail(25)
    base_level = recent_lows.mean()
    base_range_pct = (recent_lows.max() - recent_lows.min()) / price * 100
    base_is_flat = base_range_pct < 1.5

    base_touches = int(
        sum(abs(l - base_level) / base_level < 0.01 for l in recent_lows)
    )

    if base_touches >= 5:
        base_quality = "excellent"
    elif base_touches >= 3:
        base_quality = "good"
    else:
        base_quality = "weak"

    # -------------------------------------------------
    # 4. Momentum exhaustion / shift
    # -------------------------------------------------
    momentum_shift = (
        rsi is not None and 30 < rsi < 50 and hist is not None and hist > -0.05
    )

    # -------------------------------------------------
    # 5. Trendline break strength
    # -------------------------------------------------
    highs = df_6h["high"].tail(60).values
    x = np.arange(len(highs))
    slope, intercept = np.polyfit(x, highs, 1)
    trendline_now = slope * (len(highs) - 1) + intercept

    break_strength_pct = (price - trendline_now) / trendline_now * 100
    strong_break = break_strength_pct > 1.5

    # -------------------------------------------------
    # 6. Early vs Confirmed phase
    # -------------------------------------------------
    confirmed_break = price > df_6h["ema25"].iloc[-1] or momentum_shift
    early_bottom = base_is_flat and compression and not confirmed_break

    if confirmed_break:
        phase = "confirmed"
    elif early_bottom:
        phase = "early"
    else:
        phase = "none"

    # -------------------------------------------------
    # 7. Time in base
    # -------------------------------------------------
    bars_in_base = int(
        sum(abs(df_6h["low"].iloc[-i] - base_level) / base_level < 0.01
            for i in range(1, 40))
    )

    # -------------------------------------------------
    # NIHILIUS SCORE (still used as a gate, not a decider)
    # -------------------------------------------------
    score = 0
    if prolonged_downtrend:
        score += 30
    if compression:
        score += 20
    if base_is_flat:
        score += 20
    if momentum_shift:
        score += 30
    if strong_break:
        score += 30

    score = min(100, score)

    is_tradeable = (
        score >= 60
        and base_quality != "weak"
        and phase != "none"
    )

    return {
        "price": price,
        "nihilius": {
            "score": score,
            "is_candidate": score >= 60,
            "is_tradeable": is_tradeable,
            "phase": phase,
            "base_quality": base_quality,
            "base_touches": base_touches,
            "bars_in_base": bars_in_base,
            "break_strength_pct": break_strength_pct,
            "strong_break": strong_break,
            "context": {
                "trend_maturity": "late" if prolonged_downtrend else "early",
                "volatility_state": "compressed" if compression else "expanded",
                "structure": "base" if base_is_flat else "none",
            },
            "entry_zones": {
                "base": [
                    float(recent_lows.min()),
                    float(recent_lows.max())
                ],
                "trendline_retest": [
                    float(trendline_now * 0.99),
                    float(trendline_now * 1.01)
                ]
            }
        }
    }


# =====================================================
# SNAPSHOT BUILD
# =====================================================
def build_snapshot():
    coins = []

    for symbol in ALL_COINS:
        df_6h = get_candles(symbol, 21600, 300)
        df_1d = get_candles(symbol, 86400, 120)
        ticker = get_ticker(symbol)

        if df_6h is None or df_1d is None or ticker is None:
            continue

        features = compute_features(df_6h, df_1d)
        if features is None:
            continue

        if not features["nihilius"]["is_tradeable"]:
            continue

        coins.append({
            "symbol": symbol,
            "price": features["price"],
            "nihilius": features["nihilius"]
        })

        time.sleep(0.25)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "coins": coins
    }


# =====================================================
# GPT CALL
# =====================================================
USER_PROMPT = """
You are a Nihilius-style bottom-finding trader.

ONLY return LONG setups.
ONLY return coins with nihilius.is_tradeable == true.

Prefer confirmed bottoms over early ones unless asymmetry is exceptional.
We are looking for coins that are coming out of a prolonged downtrend, with the possibility of explosive upside.

Keep the top 3 candidates in your opinion.

For each setup give:
- symbol
- setup_quality (A+, A, B)
- entry_zone (prefer precomputed zones)
- stop_loss (below base)
- take_profits (early expansion targets)
- rationale

Return ONLY valid JSON.

Snapshot:
<<SNAPSHOT>>
"""


def ask_gpt(snapshot):
    client = OpenAI()
    prompt = USER_PROMPT.replace(
        "<<SNAPSHOT>>",
        json.dumps(snapshot, default=str)
    )

    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.15,
    )
    return json.loads(response.choices[0].message.content)


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    snapshot = build_snapshot()

    if not snapshot["coins"]:
        print("No Nihilius candidates found.")
        exit()

    print(f"Found {len(snapshot['coins'])} Nihilius candidates")

    result = ask_gpt(snapshot)
    print(json.dumps(result, indent=2))
