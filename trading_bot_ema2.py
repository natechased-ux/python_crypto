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
    "1INCH-USD", "ABT-USD", "ACH-USD", "ACX-USD", "ADA-USD", "AERGO-USD",
    "AERO-USD", "AGLD-USD", "AIOZ-USD", "AKT-USD", "ALCX-USD", "ALEO-USD",
    "ALEPH-USD", "ALGO-USD", "ALICE-USD", "ANKR-USD", "APE-USD", "API3-USD",
    "APT-USD", "ARPA-USD", "ARKM-USD", "ASM-USD", "AST-USD", "ATH-USD",
    "ATOM-USD", "AUDIO-USD", "AUCTION-USD", "AVAX-USD", "AXL-USD", "AXS-USD",
    "BAL-USD", "BAND-USD", "BAT-USD", "BEAM-USD", "BERA-USD", "BIGTIME-USD",
    "BIT-USD", "BLAST-USD", "BLUR-USD", "BNB-USD", "BOBA-USD", "BCH-USD",
    "BIRB-USD", "BNT-USD", "BTRST-USD", "CAKE-USD", "CELR-USD", "CHZ-USD",
    "CLV-USD", "COMP-USD", "COOKIE-USD", "COTI-USD", "COVAL-USD", "COW-USD",
    "CRO-USD", "CRV-USD", "CTX-USD", "CTSI-USD", "CVX-USD", "DASH-USD",
    "DEGEN-USD", "DESO-USD", "DEXT-USD", "DIA-USD", "DIMO-USD", "DNT-USD",
    "DOGE-USD", "DOGINME-USD", "DOT-USD", "DRIFT-USD", "DREP-USD", "DYP-USD",
    "EDGE-USD", "EGLD-USD", "EIGEN-USD", "ELA-USD", "ENA-USD", "ENJ-USD",
    "ERN-USD", "ETC-USD", "EUL-USD", "FAI-USD", "FARM-USD", "FARTCOIN-USD",
    "FET-USD", "FIDA-USD", "FIL-USD", "FLOW-USD", "FLR-USD", "FLUID-USD",
    "FLOKI-USD", "FORTH-USD", "FX-USD", "G-USD", "GALA-USD", "GAL-USD",
    "GFI-USD", "GIGA-USD", "GLM-USD", "GNO-USD", "GODS-USD", "GMT-USD",
    "GRT-USD", "GYEN-USD", "HBAR-USD", "HFT-USD", "HNT-USD", "HONEY-USD",
    "HOPR-USD", "HYPER-USD", "ICP-USD", "IDEX-USD", "ILV-USD", "IMU-USD",
    "IMX-USD", "INJ-USD", "IO-USD", "IP-USD", "IOTX-USD", "JASMY-USD",
    "JITOSOL-USD", "JTO-USD", "JUP-USD", "JUPITER-USD", "KARRAT-USD",
    "KAVA-USD", "KEYCAT-USD", "KERNEL-USD", "KSM-USD", "KTA-USD",
    "LCX-USD", "LDO-USD", "LINEA-USD", "LINK-USD", "LQTY-USD", "LRC-USD",
    "LTC-USD", "MAGIC-USD", "MAMO-USD", "MANA-USD", "MANTLE-USD", "MASK-USD",
    "MATH-USD", "ME-USD", "MET-USD", "METIS-USD", "MINA-USD", "MLN-USD",
    "MOBILE-USD", "MON-USD", "MONA-USD", "MORPHO-USD", "MSOL-USD",
    "MOG-USD", "NEAR-USD", "NEON-USD", "NEST-USD", "NCT-USD", "NU-USD",
    "OGN-USD", "OMG-USD", "OMNI-USD", "ONDO-USD", "OP-USD", "ORCA-USD",
    "OSMO-USD", "OXT-USD", "PEPE-USD", "PENDLE-USD", "PENGU-USD",
    "PERP-USD", "PLA-USD", "PLUME-USD", "POL-USD", "POLS-USD", "POND-USD",
    "POPCAT-USD", "PRCL-USD", "PRIME-USD", "PRO-USD", "PROMPT-USD",
    "PRQ-USD", "PYR-USD", "PYTH-USD", "PYUSD-USD", "QI-USD", "QNT-USD",
    "RAD-USD", "RAI-USD", "RARE-USD", "RARI-USD", "RAY-USD", "RED-USD",
    "RENDER-USD", "REQ-USD", "REZ-USD", "RGT-USD", "RLC-USD", "RLY-USD",
    "RNDR-USD", "RONIN-USD", "ROSE-USD", "RPL-USD", "RSR-USD", "SAFE-USD",
    "SAND-USD", "SD-USD", "SEAM-USD", "SEI-USD", "SHDW-USD", "SHIB-USD",
    "SKL-USD", "SKR-USD", "SOL-USD", "SPA-USD", "SPELL-USD", "SPK-USD",
    "STG-USD", "STORJ-USD", "STRK-USD", "SUPER-USD", "SUKU-USD", "SUI-USD",
    "SYN-USD", "SYLO-USD", "SYRUP-USD", "SXT-USD", "TAO-USD", "T-USD",
    "TIA-USD", "TNSR-USD", "TOWNS-USD", "TREE-USD", "TRB-USD", "TRU-USD",
    "TRUMP-USD", "TROLL-USD", "TVK-USD", "UMA-USD", "UNI-USD", "UNFI-USD",
    "UPI-USD", "USDS-USD", "UST-USD", "VARA-USD", "VELO-USD", "VET-USD",
    "VGX-USD", "VOXEL-USD", "VVV-USD", "W-USD", "WAMPL-USD", "WCFG-USD",
    "WCT-USD", "WELL-USD", "WIF-USD", "WLFI-USD", "WLD-USD", "WLUNA-USD",
    "XAN-USD", "XLM-USD", "XRP-USD", "XTZ-USD", "XYO-USD", "YFI-USD",
    "YFII-USD", "ZEN-USD", "ZEC-USD", "ZETA-USD", "ZETACHAIN-USD", "ZK-USD",
    "ZORA-USD", "ZRO-USD"
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

def atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period, adjust=False).mean()


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

    atr14 = atr(df_4h, 14)
    atr_now = atr14.iloc[-1]
    atr_prev = atr14.iloc[-5]
    vol_ema = df_4h["volume"].ewm(span=20, adjust=False).mean()
    range_pct = (df_4h["high"].rolling(10).max() -df_4h["low"].rolling(10).min()) / close


    # ===== EMAs =====
    ema10 = ema(close, 10)
    ema20 = ema(close, 20)
    ema25 = ema(close, 25)
    ema50 = ema(close, 50)
    ema75 = ema(close, 75)
    ema150 = ema(close, 150)
    ema200 = ema(close, 200)

    e10 = ema10.iloc[-1]
    e20 = ema20.iloc[-1]
    e25 = ema25.iloc[-1]
    e50 = ema50.iloc[-1]
    e75 = ema75.iloc[-1]
    e150 = ema150.iloc[-1]
    e200 = ema200.iloc[-1]

    # ===== STACK ORDER =====
    stacked = e25 > e50 > e75 > e150 > e200
    #if not stacked:
     #   return None

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
    ema_spread = abs(ema10 - ema20) / close
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
            "rsi": {"rsi14": round(rsi_now, 2),"rsi_trend": "rising" if rsi_now > rsi_prev else "falling"},
            "atr": {"atr14": round(atr_now, 6),"atr_trend": "contracting" if atr_now < atr_prev else "expanding"},
            "volume": {"vol_vs_avg": round(df_4h["volume"].iloc[-1] / vol_ema.iloc[-1], 2),"volume_trend": "rising" if vol_ema.iloc[-1] > vol_ema.iloc[-5] else "flat"},
            "ema_compression": {"spread_pct": round(ema_spread.iloc[-1] * 100, 2),"state": "tight" if ema_spread.iloc[-1] < ema_spread.iloc[-5] else "expanding"},
            "price_range": {"range_pct": round(range_pct.iloc[-1] * 100, 2),"state": "tight" if range_pct.iloc[-1] < range_pct.iloc[-5] else "wide"}
            

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
You are an elite crypto momentum and breakout trader.

Your task:
1. Identify coins that are BUILDING BUYING PRESSURE and are likely approaching an upside expansion.
2. Focus on early-to-mid trend continuation setups — NOT late-stage or exhaustion moves.
3. Rank ALL coins from MOST bullish to LEAST bullish.
4. Select ONLY the TOP 3 best long setups.

Buying pressure is defined as a combination of:
- Bullish EMA structure with tightening or stable EMA spacing
- Volatility contraction (ATR compression or tight price range)
- Quiet or rising volume WITHOUT large downside candles (accumulation)
- RSI holding between 50–65 and rising (not overbought)
- Price respecting EMA support on pullbacks

For each selected setup, provide:
- symbol
- setup_quality (A+, A, or B)
- ideal_entry (EMA pullback, range low, or continuation trigger, and give price)
- stop_loss (EMA-based or structure-based, give price)
- take_profits (measured move or trend continuation targets, give price)
- rationale (EMA structure, buying pressure, volatility state, RSI behavior, volume behavior)

Important rules:
- Penalize coins that appear to be in a second or late impulse unless clear re-accumulation is present.
- Favor coins with visible compression or coiling over coins already expanding.
- Avoid overbought RSI (>70) unless volume and volatility suggest continuation.

Return ONLY valid JSON.
Do NOT include explanations outside the JSON.

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
