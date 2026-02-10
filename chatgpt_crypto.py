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


def get_candles(product_id="BTC-USD", granularity=3600, limit=100):
    """
    Returns OHLCV candles from the free Coinbase API.
    Granularity is in seconds (e.g., 3600 = 1 hour).
    """
    url = f"{PUBLIC_API}/products/{product_id}/candles"
    params = {"granularity": granularity}

    r = requests.get(url, params=params)
    data = r.json()

    # Coinbase returns newest → oldest, so reverse
    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").reset_index(drop=True)
    return df.tail(limit)


def get_orderbook(product_id="BTC-USD", level=2):
    """
    Level 2 order book snapshot.
    """
    url = f"{PUBLIC_API}/products/{product_id}/book?level={level}"
    return requests.get(url).json()


def get_ticker(product_id="BTC-USD"):
    """
    Real-time best bid/ask and price.
    """
    url = f"{PUBLIC_API}/products/{product_id}/ticker"
    return requests.get(url).json()


# -----------------------------
# 2. Feature Extraction
# -----------------------------
def compute_features(df):
    df = df.copy()

    # Trend indicators
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()

    # ATR volatility
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()

    # Momentum: 24h change
    if len(df) > 24:
        change_24h = (df["close"].iloc[-1] - df["close"].iloc[-25]) / df["close"].iloc[-25] * 100
    else:
        change_24h = 0

    features = {
        "current_price": float(df["close"].iloc[-1]),
        "trend": {
            "above_50ema": bool(df["close"].iloc[-1] > df["ema50"].iloc[-1]),
            "above_200ema": bool(df["close"].iloc[-1] > df["ema200"].iloc[-1]),
            "trend_strength": float(df["ema50"].iloc[-1] / df["ema200"].iloc[-1])
        },
        "volatility": {
            "atr_pct": float(df["atr"].iloc[-1] / df["close"].iloc[-1] * 100)
        },
        "momentum": {
            "change_24h_pct": float(change_24h)
        }
    }

    return features


# -----------------------------
# 3. Build Coin Data Object
# -----------------------------
def build_coin_object(symbol="BTC-USD"):
    df = get_candles(symbol)
    orderbook = get_orderbook(symbol)
    ticker = get_ticker(symbol)
    features = compute_features(df)

    coin = {
        "symbol": symbol,
        "price": features["current_price"],
        "trend": features["trend"],
        "volatility": features["volatility"],
        "momentum": features["momentum"],
        "orderbook": {
            "best_bid": float(orderbook["bids"][0][0]),
            "best_ask": float(orderbook["asks"][0][0]),
            "bid_size": float(orderbook["bids"][0][1]),
            "ask_size": float(orderbook["asks"][0][1])
        },
        "ticker": {
            "last_price": float(ticker["price"]),
            "bid": float(ticker["bid"]),
            "ask": float(ticker["ask"])
        }
    }
    return coin


# -----------------------------
# 4. Build Full Market Snapshot
# -----------------------------
def build_market_snapshot(coins=["BTC-USD", "ETH-USD", "SOL-USD"]):
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "coins": [build_coin_object(c) for c in coins]
    }


# -----------------------------
# 5. ChatGPT Prompt Builder
# -----------------------------
SYSTEM_PROMPT = """
You are a professional crypto trading analyst.
Your job is to evaluate coins using trend, momentum, volatility, and order book behavior.
Only select high-quality setups. Avoid unclear or low-liquidity situations.
Output ONLY JSON.
"""

USER_PROMPT_TEMPLATE = """
Analyze the following market snapshot:

1. Rank the coins from strongest long setup to weakest.
2. Select only the TOP 2 setups if quality is A or B+.
3. For each setup, provide:
   - Setup quality (A+, A, B, C, avoid)
   - Bias (long / avoid)
   - Ideal entry zone (price range)
   - Stop-loss level (structure-based)
   - 2–3 take profit levels
   - Rationale (2–3 sentences)

Return JSON only.

Market snapshot:
```json
{snapshot}
"""
