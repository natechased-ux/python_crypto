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
    "You are an elite crypto trend trader specializing in EMA-supported RSI, and price action bearish signals."

)


print("FLAG BOT STARTED")

# =====================================================
# COIN UNIVERSE (USD / USDT ONLY, NO STABLE BASES)
# =====================================================
ALL_COINS = [
      "NEAR-USD", "HFT-USD", "ZETACHAIN-USD",
    "PRIME-USD", "FLOKI-USD", "DOGE-USD", "SUKU-USD", "GAL-USD",
    "QI-USD", "SOL-USD", "MLN-USD", "MASK-USD",
     "ABT-USD", "WCFG-USD", "RLY-USD", "IOTX-USD",
    "COVAL-USD", "C98-USD", "ACS-USD", "TURBO-USD",
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
    "ICP-USD", "MNDE-USD", "MORPHO-USD", "SYRUP-USD", "HONEY-USD",
     "RBN-USD", "ZEC-USD", "RLC-USD",
    "SXT-USD", "ILV-USD", "AKT-USD", "RNDR-USD", "ONDO-USD",
    "DEGEN-USD", "EGLD-USD", "SD-USD",  "PEPE-USD",
    "ACX-USD", "PYUSD-USD", "GHST-USD", "CRO-USD", "PROMPT-USD",
    "GYEN-USD", "XLM-USD", "LDO-USD",
    "OGN-USD", "PENGU-USD", "XYO-USD", "OXT-USD", "SYLO-USD",
    "WLD-USD", "OMG-USD", "BAL-USD", "PRO-USD", "SYN-USD",
    "EIGEN-USD", "GMT-USD", "GFI-USD", "RENDER-USD", "MINA-USD",
    "MAGIC-USD", "DYP-USD", "ZORA-USD", "COW-USD",
    "CELR-USD", "PIRATE-USD", "WLUNA-USD", "NEST-USD", "REZ-USD",
     "DESO-USD", "TAO-USD", "ME-USD", "RONIN-USD",
    "NCT-USD", "INJ-USD", "NU-USD", "OP-USD", "KARRAT-USD",
    "ALEO-USD",  "VGX-USD", "KERNEL-USD", "FAI-USD",
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
def ema(values, length):
    k = 2 / (length + 1)
    ema_vals = [values[0]]
    for price in values[1:]:
        ema_vals.append(price * k + ema_vals[-1] * (1 - k))
    return ema_vals

def rsi(values, length=14):
    gains, losses = [], []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0))
        losses.append(abs(min(diff, 0)))

    avg_gain = sum(gains[:length]) / length
    avg_loss = sum(losses[:length]) / length

    rsis = [50]
    for i in range(length, len(gains)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsis.append(100 - (100 / (1 + rs)))
    return [50] * (len(values) - len(rsis)) + rsis

# =====================================================
# DATA
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
# SNAPSHOT BUILDER (BEARISH FILTER)
# =====================================================
def build_snapshot():

    coins = []

    for symbol in ALL_COINS:
        try:
            closes, volumes = get_6h_candles_chunked(symbol)
            if len(closes) < 200:
                continue

            ema25 = ema(closes, 25)
            ema50 = ema(closes, 50)
            ema200 = ema(closes, 200)
            rsi14 = rsi(closes)

            price = closes[-1]

            # === BEARISH BASELINE CONDITIONS ===
            if not (
                ema25[-1] < ema50[-1] and
                ema50[-1] < ema200[-1] and
                price < ema25[-1] and
                rsi14[-1] > 55
            ):
                continue

            coins.append({
                "symbol": symbol,
                "price": price,
                "ema25": ema25[-1],
                "ema50": ema50[-1],
                "ema200": ema200[-1],
                "rsi": rsi14[-1],
                "volume": volumes[-1],
            })

        except Exception:
            continue

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "coins": coins[:TOP_N],
    }

# =====================================================
# GPT PROMPTS (SHORT BIAS)
# =====================================================
SYSTEM_PROMPT = """
You are an elite crypto derivatives trader specializing in short-selling,
distribution phases, and market tops.

You aggressively filter out weak or unclear setups.
You prioritize asymmetric downside opportunities.
"""

USER_PROMPT = """
You are an elite crypto short-selling strategist.

You are given coins that already meet ALL of the following baseline conditions:
- 4H EMA stack is bearish (EMA25 < EMA50 < EMA200)
- Price is below EMA25
- Trend context is neutral-to-bearish, not deeply oversold
- RSI is elevated enough to allow downside expansion
- Bonus points for Bearish RSI divergence.

Your task:
1. Identify coins that are DISTRIBUTING or LOSING BUYING PRESSURE.
2. Focus on early-to-mid bearish continuation or topping structures.
3. Rank ALL coins from MOST bearish to LEAST bearish.
4. Select ONLY the TOP 3 best SHORT setups.

Bearish pressure is defined as:
- Bearish EMA structure with price failing to reclaim EMAs
- Rising or elevated RSI rolling over (55–70)
- Volatility compression after an impulse (bear flags, wedges, diamonds)
- Signs of exhaustion: long upper wicks, failed breakouts, decreasing upside volume
- Price acceptance below key EMAs or VWAP-like levels

For each selected setup, provide:
- symbol
- setup_quality (A+, A, or B)
- ideal_entry (EMA rejection, range high, or breakdown trigger, give price)
- stop_loss (above structure or EMA, give price)
- take_profits (measured move, prior lows, or liquidity pools, give price)
- rationale (EMA structure, momentum loss, RSI behavior, volume behavior)

Important rules:
- Penalize coins already extremely oversold.
- Favor compression and failed breakout structures.
- Avoid random chop with no clear trend bias.

Return ONLY valid JSON.
Do NOT include explanations outside the JSON.

Snapshot:
<<SNAPSHOT>>
"""

# =====================================================
# GPT CALL
# =====================================================
def ask_gpt(snapshot):
    client = OpenAI(api_key=OPENAI_API_KEY)

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
        print("No bearish EMA candidates found.")
        exit()

    print(f"\nFound {len(snapshot['coins'])} bearish EMA candidates")

    result = ask_gpt(snapshot)
    print(json.dumps(result, indent=2))
