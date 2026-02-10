# Hybrid Smart Crypto Alert Bot
# Combines breakout, mean-reversion, whale flow, structure + confirmation
# Uses Coinbase API + Telegram

import os
import time
import math
import json as _json
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from collections import deque

import numpy as np
import pandas as pd
import requests
import websocket
import telegram

from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange

# === CONFIGURATION ===
TELEGRAM_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
CHAT_ID = "7967738614"
bot = telegram.Bot(token=TELEGRAM_TOKEN)


COINS =  [
    "btc-usd","eth-usd","xrp-usd","ltc-usd","ada-usd","doge-usd",
    "sol-usd","wif-usd","ondo-usd","sei-usd","magic-usd","ape-usd",
    "jasmy-usd","wld-usd","syrup-usd","fartcoin-usd","aero-usd",
    "link-usd","hbar-usd","aave-usd","fet-usd","crv-usd","tao-usd",
    "avax-usd","xcn-usd","uni-usd","mkr-usd","toshi-usd","near-usd",
    "algo-usd","trump-usd","bch-usd","inj-usd","pepe-usd","xlm-usd",
    "moodeng-usd","bonk-usd","dot-usd","popcat-usd","arb-usd","icp-usd",
    "qnt-usd","tia-usd","ip-usd","pnut-usd","apt-usd","ena-usd","turbo-usd",
    "bera-usd","pol-usd","mask-usd","pyth-usd","sand-usd","morpho-usd",
    "mana-usd","c98-usd","axs-usd"
]

BASE_URL = "https://api.exchange.coinbase.com"

GRAN_15M = 900
GRAN_6H = 21600
MIN_SCORE = 0
ENABLE_TIME_FILTER = False
SLEEP_HOURS_UTC = range(0, 6)

COOLDOWN_MINUTES = 30
cooldowns = {}
whale_trades = {}

# === Fetch Candle Data ===
def fetch_candles(symbol, granularity, lookback_days=30):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    df = pd.DataFrame()
    while start < end:
        chunk_end = min(start + timedelta(hours=300), end)
        params = {
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "granularity": granularity,
        }
        url = f"{BASE_URL}/products/{symbol}/candles"
        try:
            res = requests.get(url, params=params)
            data = res.json()
            if isinstance(data, list):
                chunk = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
                df = pd.concat([df, chunk], ignore_index=True)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
        start = chunk_end
        time.sleep(0.2)
    if df.empty:
        return None
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)
    return df

# === Indicators ===
def add_indicators(df):
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    df["adx"] = ADXIndicator(df["high"], df["low"], df["close"]).adx()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
    df["macd_hist"] = MACD(df["close"]).macd_diff()
    for p in [10, 20, 50, 200]:
        df[f"ema{p}"] = EMAIndicator(df["close"], window=p).ema_indicator()
    return df

# === Whale Watcher ===
class WhaleWatcher:
    def __init__(self, symbols, min_usd=100000, near_pct=0.5, max_age_sec=600):
        self.symbols = symbols
        self.min_usd = min_usd
        self.near_pct = near_pct / 100
        self.max_age = max_age_sec
        self._deques = {s: deque(maxlen=5000) for s in symbols}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            ws = None
            try:
                ws = websocket.create_connection("wss://ws-feed.exchange.coinbase.com", timeout=30)
                sub = {
                    "type": "subscribe",
                    "channels": [{"name": "matches", "product_ids": self.symbols}]
                }
                ws.send(_json.dumps(sub))
                print("[WhaleWatcher] WebSocket connected")
                while True:
                    try:
                        msg = ws.recv()
                        if not msg:
                            continue
                        data = _json.loads(msg)
                        if data.get("type") != "match":
                            continue
                        product = data.get("product_id")
                        price = float(data.get("price", 0))
                        size = float(data.get("size", 0))
                        side = data.get("side")
                        ts = datetime.fromisoformat(data["time"].replace("Z", "+00:00"))
                        notional = price * size
                        if notional >= self.min_usd and product in self._deques:
                            self._deques[product].append({
                                "ts": ts, "price": price, "side": side, "usd": notional
                            })
                    except Exception as recv_err:
                        print(f"[WhaleWatcher] recv error: {recv_err}")
                        break  # force reconnect
            except Exception as conn_err:
                print(f"[WhaleWatcher] connection error: {conn_err}")
            finally:
                try:
                    if ws:
                        ws.close()
                except:
                    pass
                print("[WhaleWatcher] reconnecting in 5s...")
                time.sleep(5)


    def is_aligned(self, symbol, side, ref_price):
        dq = self._deques.get(symbol)
        if not dq:
            return False
        now = datetime.now(timezone.utc)
        for t in reversed(dq):
            if (now - t["ts"]).total_seconds() > self.max_age:
                break
            if abs(t["price"] - ref_price) / ref_price > self.near_pct:
                continue
            if side == "long" and t["side"] == "buy":
                return True
            if side == "short" and t["side"] == "sell":
                return True
        return False

whale_watcher = WhaleWatcher([s.upper() for s in COINS])

# === Confluence Score ===
def detect_fvg(df):
    if len(df) < 3: return False, False
    c2 = df.iloc[-3]; c0 = df.iloc[-1]
    return c0["low"] > c2["high"], c0["high"] < c2["low"]

def compute_fib_levels(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    if len(df) < 50:
        return None
    highs = df["high"].iloc[-50:]
    lows = df["low"].iloc[-50:]
    swing_high = highs.max()
    swing_low = lows.min()
    diff = swing_high - swing_low
    return {
        "0.382": swing_high - 0.382 * diff,
        "0.5": swing_high - 0.5 * diff,
        "0.618": swing_high - 0.618 * diff,
        "0.786": swing_high - 0.786 * diff,
    } if diff > 0 else None


def fib_confluence(entry_price: float, side: str, fib_levels: Dict[str, float], tolerance: float = 0.005) -> Tuple[bool, str]:
    for level_name, level_price in fib_levels.items():
        pct_diff = abs(entry_price - level_price) / entry_price
        if pct_diff <= tolerance:
            return True, level_name
    return False, ""


def compute_confluence_score(df: pd.DataFrame, side: str, fvg_up: bool, fvg_dn: bool, whale_ok: bool,
                              fib_nearby: bool) -> Tuple[int, List[str]]:
    score = 0
    notes = []
    last = df.iloc[-2]
    if (side == "long" and last["close"] > last["ema200"]) or (side == "short" and last["close"] < last["ema200"]):
        score += 1; notes.append("Trend✅")
    if side == "long" and last["ema10"] > last["ema20"] > last["ema50"] > last["ema200"]:
        score += 1; notes.append("EMA✅")
    if side == "short" and last["ema10"] < last["ema20"] < last["ema50"] < last["ema200"]:
        score += 1; notes.append("EMA✅")
    if whale_ok:
        score += 1; notes.append("Whale✅")
    if (side == "long" and fvg_up) or (side == "short" and fvg_dn):
        score += 1; notes.append("FVG✅")
    if fib_nearby:
        score += 1; notes.append("Fib✅")
    return score, notes

# === Signal Logic ===
def is_bollinger_reversal(df, side):
    close = df["close"]
    avg = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = avg + 2 * std
    lower = avg - 2 * std
    last = close.iloc[-2]
    return (side == "long" and last < lower.iloc[-2]) or (side == "short" and last > upper.iloc[-2])

def detect_structure_signal(df):
    last = df.iloc[-2]
    if last["adx"] >= 25:
        if last["macd_hist"] > 0 and last["rsi"] > 60 and last["close"] > last["ema50"]:
            return {"side": "long", "strategy": "breakout", "entry": last["close"], "atr": last["atr"]}
        elif last["macd_hist"] < 0 and last["rsi"] < 40 and last["close"] < last["ema50"]:
            return {"side": "short", "strategy": "breakout", "entry": last["close"], "atr": last["atr"]}
    elif last["adx"] <= 20:
        if last["rsi"] < 30 and is_bollinger_reversal(df, "long"):
            return {"side": "long", "strategy": "mean-reversion", "entry": last["close"], "atr": last["atr"]}
        elif last["rsi"] > 70 and is_bollinger_reversal(df, "short"):
            return {"side": "short", "strategy": "mean-reversion", "entry": last["close"], "atr": last["atr"]}
    return None

def confirm_stoch_rsi(df, side):
    stoch = StochRSIIndicator(df["close"]).stochrsi_k().dropna() * 100
    if len(stoch) < 3: return False
    k1, k0 = stoch.iloc[-1], stoch.iloc[-2]
    return (side == "long" and k1 > k0 and k1 < 40) or (side == "short" and k1 < k0 and k1 > 60)

# === Risk, Logging, Alerts ===
def is_on_cooldown(symbol, side):
    key = (symbol, side)
    now = datetime.now(timezone.utc)
    if key in cooldowns:
        return (now - cooldowns[key]).total_seconds() / 60 < COOLDOWN_MINUTES
    return False

def set_cooldown(symbol, side):
    cooldowns[(symbol, side)] = datetime.now(timezone.utc)

def compute_tp_sl(entry, atr, side, score):
    tp_mult = 2.0 if score == 4 else 1.5 if score == 3 else 1.2
    sl_mult = 1.0
    tp = entry + tp_mult * atr if side == "long" else entry - tp_mult * atr
    sl = entry - sl_mult * atr if side == "long" else entry + sl_mult * atr
    return round(tp, 6), round(sl, 6)

def log_trade(symbol, side, entry, tp, sl, strategy, score):
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "side": side,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "strategy": strategy,
        "score": score
    }
    df = pd.DataFrame([row])
    df.to_csv("trade_log.csv", mode="a", header=not os.path.exists("trade_log.csv"), index=False)

def send_alert(symbol: str, side: str, entry: float, tp: float, sl: float, strategy: str,
               score: int, notes: List[str], fib_tp: Optional[float] = None, fib_label: str = ""):
    emoji = "📈" if side == "long" else "📉"
    msg = (
        f"{emoji} *{strategy.upper()} {side.upper()}* on `{symbol}`\n"
        f"Entry: `{entry:.4f}`\nTP: `{tp:.4f}`\nSL: `{sl:.4f}`"
    )
    if fib_tp:
        msg += f"\nTP2 (Fib {fib_label}): `{fib_tp:.4f}`"
    msg += f"\nScore: `{score}` {' '.join(notes)}"
    msg += f"\nTime: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
    bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")



# === Main Loop ===
def run_scan():
    print(f"🔍 Scan @ {datetime.now(timezone.utc).isoformat()}")
    for symbol in COINS:
        try:
            df_6h = fetch_candles(symbol, GRAN_6H, 7)
            df_15m = fetch_candles(symbol, GRAN_15M, 3)
            if df_6h is None or df_15m is None:
                continue
            df_6h = add_indicators(df_6h)
            df_15m = add_indicators(df_15m)
            sig = detect_structure_signal(df_6h)
            if not sig:
                continue
            if is_on_cooldown(symbol, sig["side"]):
                continue
            if not confirm_stoch_rsi(df_15m, sig["side"]):
                continue
            fvg_up, fvg_dn = detect_fvg(df_6h)
            whale_ok = whale_watcher.is_aligned(symbol.upper(), sig["side"], sig["entry"])
            fib_data = compute_fib_levels(df_6h)
            fib_near, fib_label = fib_confluence(sig["entry"], sig["side"], fib_data) if fib_data else (False, "")
            fib_tp = fib_data[fib_label] if fib_near and fib_label in fib_data else None
            score, notes = compute_confluence_score(df_6h, sig["side"], fvg_up, fvg_dn, whale_ok, fib_near)
            if score < MIN_SCORE:
                continue
            if ENABLE_TIME_FILTER and datetime.now(timezone.utc).hour in SLEEP_HOURS_UTC and score < 4:
                continue
            tp, sl = compute_tp_sl(sig["entry"], sig["atr"], sig["side"], score)
            send_alert(symbol, sig["side"], sig["entry"], tp, sl, sig["strategy"], score, notes, fib_tp, fib_label)
            log_trade(symbol, sig["side"], sig["entry"], tp, sl, sig["strategy"], score)
            set_cooldown(symbol, sig["side"])
        except Exception as e:
            print(f"[{symbol}] error: {e}")

print("🚀 Hybrid Alert Bot Running")
while True:
    run_scan()
    time.sleep(300)
