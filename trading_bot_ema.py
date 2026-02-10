import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
from openai import OpenAI
import json
import math

from collections import defaultdict

# =====================================================
# CONFIG
# =====================================================
PUBLIC_API = "https://api.exchange.coinbase.com"

SYSTEM_PROMPT = (
    "You are an elite crypto trend trader specializing in EMA-supported bull trends."

)

print("FLAG BOT STARTED")

# =====================================================
# COIN UNIVERSE (USD / USDT ONLY, NO STABLE BASES)
# =====================================================
ALL_COINS = [
    "MTL-USD", "INV-USD", "NEAR-USD", "HFT-USD", "ZETACHAIN-USD",
    "PRIME-USD", "FLOKI-USD", "DOGE-USD", "SUKU-USD", "GAL-USD",
    "QI-USD", "ALCX-USD", "SOL-USD", "MLN-USD", "MASK-USD",
    "NKN-USD", "ABT-USD", "WCFG-USD", "RLY-USD", "IOTX-USD",
    "COVAL-USD", "C98-USD", "ACS-USD", "TURBO-USD",
     "ZETA-USD", "ORN-USD", "B3-USD", "CLV-USD",
    "ALICE-USD", "LOKA-USD", "KNC-USD", "FLOW-USD", "VARA-USD",
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
    "MONA-USD", "ZK-USD", "YFI-USD", "POWR-USD", "JASMY-USD",
    "TVK-USD", "PENDLE-USD", "OCEAN-USD", "FX-USD", "POL-USD",
    "SUPER-USD", "MOBILE-USD", "SHDW-USD",
    "YFII-USD", "LRDS-USD", "RPL-USD", "TRAC-USD", "LCX-USD",
    "CHZ-USD", "VTHO-USD", "TNSR-USD", "SEI-USD", "SAND-USD",
    "AGLD-USD", "GRT-USD", "ASM-USD", "FIDA-USD", "UPI-USD",
    "BNT-USD", "HNT-USD", "UNI-USD", "VET-USD",
    "LTC-USD", "AST-USD", "AVAX-USD", "ARKM-USD",
    "VOXEL-USD", "MANA-USD", "RARI-USD", "SHPING-USD", "PLA-USD",
    "SNT-USD", "POPCAT-USD", "FET-USD", "SWFTC-USD", "IP-USD",
    "UST-USD", "1INCH-USD", "DIA-USD", "DEXT-USD", 
    "ICP-USD", "MNDE-USD", "MORPHO-USD", "SYRUP-USD", "HONEY-USD",
     "RBN-USD", "A8-USD", "ZEC-USD", "RLC-USD",
    "SXT-USD", "ILV-USD", "AKT-USD", "RNDR-USD", "ONDO-USD",
    "DEGEN-USD", "EGLD-USD", "SD-USD",  "PEPE-USD",
    "ACX-USD", "PYUSD-USD", "GHST-USD", "CRO-USD", "PROMPT-USD",
    "GYEN-USD", "XLM-USD", "LDO-USD",
    "OGN-USD", "PENGU-USD", "XYO-USD", "OXT-USD", "SYLO-USD",
    "WLD-USD", "OMG-USD", "BAL-USD", "PRO-USD", "SYN-USD",
    "EIGEN-USD", "GMT-USD", "GFI-USD", "RENDER-USD", "MINA-USD",
    "MAGIC-USD", "DYP-USD", "ZORA-USD", "COW-USD",
    "CELR-USD", "PIRATE-USD", "WLUNA-USD", "NEST-USD", "REZ-USD",
    "OMNI-USD", "DESO-USD", "TAO-USD", "ME-USD", "RONIN-USD",
    "NCT-USD", "INJ-USD", "NU-USD", "OP-USD", "KARRAT-USD",
    "ALEO-USD",  "VGX-USD", "KERNEL-USD", "FAI-USD",
    "PUNDIX-USD", "AIOZ-USD", "AERO-USD", "ELA-USD", "PYR-USD",
    "ALGO-USD", "IO-USD", "APT-USD", "CRV-USD", "MCO2-USD",
    "ERN-USD", "AUCTION-USD", "PRCL-USD", "VVV-USD", "BOBA-USD",
    "DREP-USD", "AXL-USD", "HBAR-USD", "STRK-USD",
    "REQ-USD", "BERA-USD", "AERGO-USD", "GODS-USD", "GALA-USD",
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

def resample_to_4h(df_1h):
    if df_1h is None or len(df_1h) < 4:
        return None

    df = df_1h.copy()
    df = df.set_index("time")

    df_4h = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna().reset_index()

    return df_4h

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# =====================================================
# DATA FETCH
# =====================================================
def get_6h_candles_chunked(product_id, total_candles=300):
    granularity = 21600  # 6H
    max_per_call = 300

    all_rows = []
    now = int(time.time())

    chunks = (total_candles // max_per_call) + 1

    for i in range(chunks):
        end = now - (i * max_per_call * granularity)
        start = end - (max_per_call * granularity)

        params = {
            "granularity": granularity,
            "start": start,
            "end": end
        }

        try:
            r = requests.get(
                f"{PUBLIC_API}/products/{product_id}/candles",
                params=params,
                timeout=10
            )
            r.raise_for_status()
            data = r.json()
        except Exception:
            break

        if not data:
            break

        all_rows.extend(data)
        time.sleep(0.25)

    if not all_rows:
        return None

    df = pd.DataFrame(
        all_rows,
        columns=["time", "low", "high", "open", "close", "volume"]
    )

    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    return df.tail(total_candles)



# =====================================================
# EMA STACK FEATURE ENGINE (4H)
# =====================================================
def compute_features(df_4h, df_1d=None):
    if df_4h is None or len(df_4h) < 220:
        return None

    close = df_4h["close"].astype(float)
    price = float(close.iloc[-1])

    # ===== RSI =====
    rsi14 = rsi(close, 14)
    rsi_now = rsi14.iloc[-1]
    rsi_prev = rsi14.iloc[-5]


    # ===== EMAs =====
    ema25 = ema(close, 25)
    ema50 = ema(close, 50)
    ema75 = ema(close, 75)
    ema150 = ema(close, 150)
    ema200 = ema(close, 200)

    e25 = ema25.iloc[-1]
    e50 = ema50.iloc[-1]
    e75 = ema75.iloc[-1]
    e150 = ema150.iloc[-1]
    e200 = ema200.iloc[-1]

    # ===== STACK ORDER =====
    stacked = e25 > e50 #> e75 > e150 > e200
    if not stacked:
        return None

    # ===== SLOPES (TREND HEALTH) =====
    slope_ok = (
        ema25.iloc[-1] > ema25.iloc[-10] and
        ema50.iloc[-1] > ema50.iloc[-10] and
        ema75.iloc[-1] > ema75.iloc[-10]
    )
   # if not slope_ok:
   #     return None

    # ===== PRICE LOCATION =====
    #if price <= e25:
     #   return None

    # ===== EXTENSION FILTER =====
    extension_pct = (price - e25) / e25 * 100
   # if extension_pct > 12:
    #    return None

    # ===== EMA SPREADS =====
    ema_spread_pct = {
        "25_50": round((e25 - e50) / e50 * 100, 2),
        "50_75": round((e50 - e75) / e75 * 100, 2),
        "75_150": round((e75 - e150) / e150 * 100, 2),
    }

    # ===== DISTANCE TO EMAS =====
    distance_to_emas = {
        "ema25": round((price - e25) / e25 * 100, 2),
        "ema50": round((price - e50) / e50 * 100, 2),
        "ema75": round((price - e75) / e75 * 100, 2),
    }

    # ===== BARS STACKED (MATURITY) =====
    bars_stacked = sum(
        ema25.iloc[-i] > ema50.iloc[-i] > ema75.iloc[-i] > ema150.iloc[-i]
        for i in range(1, 30)
    )

    # ===== DAILY CONTEXT =====

    # ===== VOLATILITY REGIME =====
    range_20 = (
        df_4h["high"].iloc[-20:].max()
        - df_4h["low"].iloc[-20:].min()
    ) / price * 100

    return {
        "price": round(price, 6),
        "ema_stack": {
            "ema25": round(e25, 6),
            "ema50": round(e50, 6),
            "ema75": round(e75, 6),
            "ema150": round(e150, 6),
            "ema200": round(e200, 6),
            "extension_pct": round(extension_pct, 2),
            "ema_spread_pct": ema_spread_pct,
            "distance_to_emas": distance_to_emas,
            "bars_stacked": bars_stacked,
            "range_20": round(range_20, 2),
            "is_stacked": True,
            "rsi": {"rsi14": round(rsi_now, 2),"rsi_trend": "rising" if rsi_now > rsi_prev else "falling"}

        }
    }

# =====================================================
# SNAPSHOT BUILD
# =====================================================
def build_snapshot():
    coins = []
    fails = defaultdict(int)

    for symbol in ALL_COINS:
        df_6h = get_6h_candles_chunked(symbol, total_candles=300)

        if df_6h is None or len(df_6h) < 220:
            print(symbol, "missing data",
                 "6h:", "None" if df_6h is None else len(df_6h))
            fails["failed_data"] += 1
            continue
        else:
            print(symbol, "6h:", len(df_6h))


        features = compute_features(df_6h)
        if features is None:
            fails["failed_features"] += 1
            continue

        coins.append({
            "symbol": symbol,
            "price": features["price"],
            "ema_stack": features["ema_stack"]
        })

        time.sleep(0.2)

    print("\nFAILURE SUMMARY")
    print("-" * 30)
    for k, v in sorted(fails.items()):
        print(f"{k}: {v}")

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "coins": coins
    }

# =====================================================
# GPT CALL
# =====================================================
USER_PROMPT = """
You are an elite crypto trend trader.

You are given coins that already meet ALL of the following:
- 6H EMA stack (25 > 50)
- Rising EMAs
- Price above EMA25
- Not overextended

Your task:
1. Your strategy is to find bullish coins and ideal entry points for Long positions.
2. Rank ALL coins from MOST bullish to LEAST bullish.
3. Select the TOP 7 best long setups only.

For each setup provide:
- symbol
- setup_quality (A+, A, B)
- ideal_entry (EMA pullback zone or continuation)
- stop_loss (EMA-based)
- take_profits (trend continuation targets)
- rationale (trend strength, EMA structure, RSI, maturity)

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
        print("No EMA stack candidates found.")
        exit()

    print(f"\nFound {len(snapshot['coins'])} EMA stack candidates")

    result = ask_gpt(snapshot)
    print(json.dumps(result, indent=2))
