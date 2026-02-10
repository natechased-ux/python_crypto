#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weekend Trap - Live Alert Bot (Core 16, Coinbase 6H)
- Monitors 6H candles for weekend fakeout & re-entry conditions.
- Sends Telegram alerts with per-coin TP (R-multiple) and SL (extension-based).
- Runs in a loop (default: every 5 minutes), evaluates only on 6H candle closes.
- Weekend-focused: scans Sat/Sun fakeouts and enters at next 6H open (often late Sun/Mon UTC).

Dependencies: pandas, requests, python-dateutil
"""

import os
import time
import json
import math
import signal
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparser

# ==== USER CONFIG ====
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"

# Core 16 coin list (Coinbase product IDs)
COINS = [
    "SOL-USD","ADA-USD","ARB-USD","ALGO-USD","WIF-USD","BONK-USD",
    "INJ-USD","AXS-USD","C98-USD","NEAR-USD","MKR-USD","TIA-USD",
    "XRP-USD","ETH-USD","DOT-USD","AVAX-USD"
]

# Path to per-coin TP rules (csv with columns: symbol,tp_r)
TP_RULES_CSV = "per_coin_tp_rules_core.csv"

# SL sizing multiplier on weekend extension (same as backtest best)
SL_MULT = 0.25

# Loop interval (seconds)
SLEEP_SECONDS = 300  # 5 minutes

# ===== END USER CONFIG =====

CB_BASE = "https://api.exchange.coinbase.com"
STATE_PATH = "weekend_trap_state.json"
LOG_PATH = "live_signals.csv"

def isoformat(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat()

def fetch_candles(product_id: str, start: datetime, end: datetime, granularity: int = 21600, max_per_req: int = 300, pause_sec: float = 0.35, max_retries: int = 5) -> pd.DataFrame:
    rows = []
    chunk_seconds = granularity * max_per_req
    t0 = start.astimezone(timezone.utc)
    t1 = end.astimezone(timezone.utc)
    if t1 <= t0:
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])

    while t0 < t1:
        t2 = min(t0 + timedelta(seconds=chunk_seconds), t1)
        params = {"granularity": granularity, "start": isoformat(t0), "end": isoformat(t2)}
        url = f"{CB_BASE}/products/{product_id}/candles"
        for attempt in range(max_retries):
            try:
                r = requests.get(url, params=params, headers={"User-Agent":"weekend-trap-live/1.0"}, timeout=20)
                if r.status_code == 429:
                    time.sleep(pause_sec * (2 ** attempt)); continue
                r.raise_for_status()
                rows.extend(r.json())
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[{product_id}] ERROR fetching {t0}→{t2}: {e}")
                else:
                    time.sleep(pause_sec * (2 ** attempt))
        time.sleep(pause_sec)
        t0 = t2

    if not rows:
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])

    df = pd.DataFrame(rows, columns=["epoch","low","high","open","close","volume"])
    df = df.sort_values("epoch").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df = df[["time","open","high","low","close","volume"]].astype({
        "open": float, "high": float, "low": float, "close": float, "volume": float
    })
    df = df[~df["time"].duplicated()].reset_index(drop=True)
    return df

def load_tp_rules(csv_path: str) -> dict:
    if not os.path.exists(csv_path):
        print(f"[WARN] TP rules file not found: {csv_path}. Using default 1.0R for all.")
        return {}
    df = pd.read_csv(csv_path)
    rules = {}
    for _, row in df.iterrows():
        sym = str(row["symbol"]).upper()
        try:
            rules[sym] = float(row["tp_r"])
        except:
            continue
    return rules

def tg_send(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print("[TG ERROR]", e)

def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_state(state: dict):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)

def log_signal(row: dict):
    exists = os.path.exists(LOG_PATH)
    df = pd.DataFrame([row])
    if exists:
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)

def last_6h_close(dt: datetime) -> datetime:
    # Coinbase 6H buckets at 00,06,12,18 UTC
    dt = dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    hour = dt.hour
    bucket = hour - (hour % 6)
    return dt.replace(hour=bucket)

def compute_weekend_trap(df6h: pd.DataFrame, symbol: str, tp_r: float):
    """
    Evaluate latest close for a valid weekend fakeout + re-entry.
    Return an alert dict or None.
    """
    if df6h.empty or len(df6h) < 40:
        return None

    d = df6h.copy()
    d["dow"] = d["time"].dt.weekday     # Monday=0 ... Sunday=6
    d["date"] = d["time"].dt.date

    now = datetime.now(timezone.utc)
    last_close_time = d["time"].iloc[-1]
    # We only act on candle CLOSE times
    if (now - last_close_time).total_seconds() < 60:
        # safety: just closed within last minute, still fine, but we'll proceed
        pass

    # Identify last Friday
    fridays = d.loc[d["dow"] == 4, "date"].unique()
    if len(fridays) == 0:
        return None
    fri_date = fridays[-1]
    fri_df = d[d["date"] == fri_date]
    if fri_df.empty:
        return None

    fri_high = fri_df["high"].max()
    fri_low  = fri_df["low"].min()

    # Weekend dates
    sat_date = fri_date + timedelta(days=1)
    sun_date = fri_date + timedelta(days=2)
    wknd_df = d[d["date"].isin([sat_date, sun_date])].reset_index(drop=True)
    if wknd_df.empty:
        return None

    # Scan for the most recent re-entry signal that would trigger an entry at the next 6H open
    # We'll check both upper and lower fakeouts
    for i in range(len(wknd_df) - 1):
        row = wknd_df.iloc[i]
        for w in range(1, 3):  # allow up to 2-candle re-entry window (as tuned)
            if i + w >= len(wknd_df):
                break
            nxt = wknd_df.iloc[i + w]

            # Upper fakeout -> re-entry short
            if row["close"] > fri_high and nxt["close"] <= fri_high:
                ext_high = max(wknd_df.loc[i:i+w, "high"])
                extension = max(0.0, ext_high - fri_high)
                if extension <= 0: break

                # Entry is next candle after re-entry
                nxt_time = nxt["time"]
                # Find the next candle in the full dataframe
                pos = d.index[d["time"] == nxt_time]
                if len(pos) == 0 or pos[0] + 1 >= len(d):
                    break
                entry_row = d.iloc[pos[0] + 1]
                entry_time = entry_row["time"]
                entry = float(entry_row["open"])

                sl = entry + SL_MULT * extension
                risk = max(1e-12, sl - entry)
                tp = entry - tp_r * risk

                return {
                    "symbol": symbol, "side": "SHORT",
                    "entry_time": entry_time.isoformat(),
                    "entry": entry, "sl": sl, "tp": tp,
                    "friday_high": fri_high, "friday_low": fri_low,
                    "extension": extension, "reentry_wait": w,
                    "generated_at": datetime.now(timezone.utc).isoformat()
                }

            # Lower fakeout -> re-entry long
            if row["close"] < fri_low and nxt["close"] >= fri_low:
                ext_low = min(wknd_df.loc[i:i+w, "low"])
                extension = max(0.0, fri_low - ext_low)
                if extension <= 0: break

                nxt_time = nxt["time"]
                pos = d.index[d["time"] == nxt_time]
                if len(pos) == 0 or pos[0] + 1 >= len(d):
                    break
                entry_row = d.iloc[pos[0] + 1]
                entry_time = entry_row["time"]
                entry = float(entry_row["open"])

                sl = entry - SL_MULT * extension
                risk = max(1e-12, entry - sl)
                tp = entry + tp_r * risk

                return {
                    "symbol": symbol, "side": "LONG",
                    "entry_time": entry_time.isoformat(),
                    "entry": entry, "sl": sl, "tp": tp,
                    "friday_high": fri_high, "friday_low": fri_low,
                    "extension": extension, "reentry_wait": w,
                    "generated_at": datetime.now(timezone.utc).isoformat()
                }
    return None

def format_price(v: float) -> str:
    if v >= 1000: return f"{v:,.2f}"
    if v >= 1:    return f"{v:,.2f}"
    return f"{v:.6f}"

def send_alert(signal: dict, tp_r: float):
    sym = signal["symbol"]
    side = signal["side"]
    entry = signal["entry"]
    sl = signal["sl"]
    tp = signal["tp"]
    etime = signal["entry_time"]
    ext = signal["extension"]

    text = (
        f"📣 <b>Weekend Trap</b> — {sym}\n"
        f"Side: <b>{side}</b>\n"
        f"Entry (next 6H open): <code>{format_price(entry)}</code>\n"
        f"TP ({tp_r:.1f}R): <code>{format_price(tp)}</code>\n"
        f"SL: <code>{format_price(sl)}</code>\n"
        f"Fri High/Low: <code>{format_price(signal['friday_high'])}</code> / <code>{format_price(signal['friday_low'])}</code>\n"
        f"Extension: <code>{format_price(ext)}</code>  |  Re-entry wait: {signal['reentry_wait']} candle(s)\n"
        f"Entry Candle Time (UTC): <code>{etime}</code>\n"
        f"Generated: <code>{signal['generated_at']}</code>"
    )
    tg_send(text)

def main_loop():
    print("Starting Weekend Trap Live Bot (Core 16)...")
    tp_rules = load_tp_rules(TP_RULES_CSV)
    state = load_state()
    last_checked_close = state.get("last_checked_close")  # ISO string

    def handle_exit(signum, frame):
        print("Shutting down...")
        save_state(state)
        raise SystemExit

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    while True:
        try:
            now = datetime.now(timezone.utc)
            close_ts = last_6h_close(now)
            close_iso = close_ts.isoformat()

            # Only process once per 6H close
            if last_checked_close == close_iso:
                time.sleep(SLEEP_SECONDS)
                continue

            # Weekend-only scan (Sat/Sun UTC) but allow entries that fall into Monday morning UTC
            dow = close_ts.weekday()  # Mon=0 ... Sun=6
            if dow not in (5, 6, 0):  # Sat=5, Sun=6, Mon=0
                # Not in weekend window; skip but keep loop alive
                last_checked_close = close_iso
                state["last_checked_close"] = last_checked_close
                save_state(state)
                time.sleep(SLEEP_SECONDS)
                continue

            # Look back 14 days to ensure we have the last two Fridays + weekends
            end_dt = close_ts
            start_dt = end_dt - timedelta(days=14)

            print(f"[{close_iso}] Checking signals...")
            for sym in COINS:
                tp_r = tp_rules.get(sym, 1.5)  # default sensible
                df = fetch_candles(sym, start_dt, end_dt, granularity=21600)
                if df.empty or len(df) < 20:
                    print(f"  {sym}: insufficient data.")
                    continue

                sig = compute_weekend_trap(df, sym, tp_r)
                if sig is None:
                    continue

                # Deduplicate by symbol+entry_time
                key = f"{sym}|{sig['entry_time']}"
                already = state.get("sent", [])
                if key in already:
                    continue

                # Send alert
                send_alert(sig, tp_r)

                # Log
                row = {
                    "symbol": sym,
                    "side": sig["side"],
                    "entry_time": sig["entry_time"],
                    "entry": sig["entry"],
                    "tp": sig["tp"],
                    "sl": sig["sl"],
                    "tp_r": tp_r,
                    "generated_at": sig["generated_at"]
                }
                log_signal(row)

                # Update state
                already.append(key)
                # keep last 500 keys
                state["sent"] = already[-500:]

            # Mark this close as processed
            last_checked_close = close_iso
            state["last_checked_close"] = last_checked_close
            save_state(state)

            # Sleep until next iteration
            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print("[LOOP ERROR]", e)
            time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main_loop()
