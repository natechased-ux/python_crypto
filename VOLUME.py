#!/usr/bin/env python3
"""
Coinbase Relative Hourly Volume Scanner (Hard-Coded Version)

Relative Hourly Volume =
    (current UTC hour volume so far)
    ---------------------------------
    (average hourly volume over last N FULL hours)

Alerts sent via Telegram when RelVol > THRESHOLD
"""

import time
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Optional

# =====================================================
# 🔧 CONFIG — EDIT THESE
# =====================================================

COINS = [
    "1INCH-USD", "ACH-USD", "ADA-USD", "AERGO-USD", "AERO-USD",
    "AKT-USD", "ALEO-USD", "ALGO-USD", "AMP-USD", "ANKR-USD", "APE-USD", "API3-USD",
    "APR-USD", "APT-USD", "ARB-USD", "ASM-USD", "ASTER-USD", "ATH-USD", "ATOM-USD",
    "AUCTION-USD", "AURORA-USD", "AVAX-USD", "AVNT-USD", "AXL-USD", "AXS-USD",
    "BAL-USD", "BAT-USD", "BERA-USD", "BCH-USD", "BIRB-USD", "BNB-USD", "BNKR-USD",
    "BONK-USD", "BREV-USD", "BTC-USD", "CBETH-USD", "CHZ-USD", "CLANKER-USD",
    "C98-USD", "COMP-USD", "CRV-USD", "CRO-USD", "CTSI-USD", "CVX-USD", "DASH-USD",
    "DEXT-USD", "DOGE-USD", "DOT-USD", "EDGE-USD", "EIGEN-USD", "ELSA-USD", "ENA-USD",
    "ENS-USD", "ETC-USD", "ETH-USD", "ETHFI-USD", "FARTCOIN-USD", "FET-USD",
    "FIGHT-USD", "FIL-USD", "FLR-USD", "FLUID-USD", "GFI-USD", "GHST-USD", "GIGA-USD",
    "GLM-USD", "GRT-USD", "HBAR-USD", "HFT-USD", "HNT-USD", "IMU-USD", "IMX-USD",
    "INJ-USD", "INX-USD", "IP-USD", "IOTX-USD", "IRYS-USD", "JASMY-USD", "JITOSOL-USD",
    "JTO-USD", "JUPITER-USD", "KAITO-USD", "KERNEL-USD", "KITE-USD", "KMNO-USD",
    "KSM-USD", "KTA-USD", "LCX-USD", "LDO-USD", "LIGHTER-USD", "LINK-USD", "LPT-USD",
    "LRC-USD", "LTC-USD", "MAGIC-USD", "MANA-USD", "MANTLE-USD", "MATH-USD",
    "MET-USD", "MINA-USD", "MOG-USD", "MON-USD", "MOODENG-USD", "MORPHO-USD",
    "NEAR-USD", "NKN-USD", "NOICE-USD", "NMR-USD", "OGN-USD", "ONDO-USD", "OP-USD",
    "ORCA-USD", "PAXG-USD", "PEPE-USD", "PENDLE-USD", "PENGU-USD", "PERP-USD",
    "PIRATE-USD", "PLUME-USD", "POPCAT-USD", "POL-USD", "PRIME-USD", "PROMPT-USD",
    "PROVE-USD", "PUMP-USD", "PYTH-USD", "QNT-USD", "RED-USD", "RENDER-USD",
    "REZ-USD", "RLS-USD", "ROSE-USD", "RSR-USD", "SAFE-USD", "SAND-USD",
    "SAPIEN-USD", "SEI-USD", "SENT-USD", "SHIB-USD", "SKL-USD", "SKR-USD", "SKY-USD",
    "SNX-USD", "SOL-USD", "SPK-USD", "SPX-USD", "STX-USD", "STRK-USD", "SUPER-USD",
    "SWFTC-USD", "SYRUP-USD", "TAO-USD", "THQ-USD", "TIA-USD", "TON-USD",
    "TOSHI-USD", "TRAC-USD", "TREE-USD", "TRB-USD", "TRUMP-USD", "TRUST-USD",
    "TROLL-USD", "TURBO-USD", "UNI-USD", "USELESS-USD", "USD1-USD", "VARA-USD",
    "VET-USD", "VOXEL-USD", "VVV-USD", "W-USD", "WET-USD", "WIF-USD", "WLD-USD",
    "WLFI-USD", "XCN-USD", "XLM-USD", "XPL-USD", "XRP-USD", "XTZ-USD", "XYO-USD",
    "YFI-USD", "ZAMA-USD", "ZEC-USD", "ZEN-USD", "ZK-USD", "ZKC-USD", "ZKP-USD",
    "ZORA-USD", "ZRO-USD", "ZRX-USD"
]


TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"

REL_VOL_THRESHOLD = 15.0
AVG_HOURS = 24
POLL_SECONDS = 60
ALERT_COOLDOWN_MINUTES = 20

# =====================================================

COINBASE_API = "https://api.exchange.coinbase.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "relvol-scanner/1.0"})


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def get_candles(product: str, start: datetime, end: datetime, granularity: int):
    params = {
        "start": iso_z(start),
        "end": iso_z(end),
        "granularity": granularity,
    }
    r = SESSION.get(
        f"{COINBASE_API}/products/{product}/candles",
        params=params,
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def compute_relvol(product: str, now: datetime) -> Tuple[Optional[float], float, float]:
    now = now.astimezone(timezone.utc)
    hour_start = now.replace(minute=0, second=0, microsecond=0)

    # --- current hour volume (live) ---
    minute_candles = get_candles(product, hour_start, now, 60)
    current_hour_vol = sum(float(c[5]) for c in minute_candles)

    # --- average hourly volume (past N full hours) ---
    end = hour_start
    start = end - timedelta(hours=AVG_HOURS)
    hourly_candles = get_candles(product, start, end, 3600)

    vols = [float(c[5]) for c in hourly_candles if float(c[5]) > 0]
    if len(vols) < max(6, AVG_HOURS // 4):
        return None, 0.0, 0.0

    avg_hour_vol = sum(vols) / len(vols)
    relvol = current_hour_vol / avg_hour_vol if avg_hour_vol > 0 else None

    return relvol, current_hour_vol, avg_hour_vol


def send_telegram(msg: str, token: str, chat_id: str, timeout: int = 15, retries: int = 6) -> None:
    """
    Robust Telegram sender:
    - retries transient network errors (WinError 10054, timeouts, etc.)
    - uses a fresh request each attempt (no shared keep-alive issues)
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": msg,
        "disable_web_page_preview": True,
    }

    last_err = None
    for attempt in range(retries):
        try:
            # Use a *fresh* connection; avoids stale keep-alive resets
            r = requests.post(
                url,
                data=payload,
                timeout=timeout,
                headers={"Connection": "close"},
            )
            r.raise_for_status()

            # Telegram sometimes returns {"ok":false,...} with 200; check it
            data = r.json()
            if not data.get("ok", False):
                raise RuntimeError(f"Telegram API responded ok=false: {data}")

            return  # success

        except Exception as e:
            last_err = e
            # exponential backoff with a little jitter
            sleep_s = min(2 ** attempt, 30) + (0.1 * attempt)
            time.sleep(sleep_s)

    raise RuntimeError(f"Telegram send failed after {retries} retries: {last_err}")


def main():
    print(f"🚀 Scanning {len(COINS)} coins for RelVol > {REL_VOL_THRESHOLD}")
    last_alert: Dict[str, datetime] = {}

    while True:
        now = datetime.now(timezone.utc)

        for coin in COINS:
            if coin in last_alert:
                if now - last_alert[coin] < timedelta(minutes=ALERT_COOLDOWN_MINUTES):
                    continue

            try:
                relvol, cur_vol, avg_vol = compute_relvol(coin, now)
            except Exception as e:
                print(f"[{coin}] error: {e}")
                continue

            if relvol is None:
                continue

            if relvol > REL_VOL_THRESHOLD:
                msg = (
                    f"🚨 VOLUME SPIKE DETECTED\n\n"
                    f"Coin: {coin}\n"
                    f"Relative Hourly Volume: {relvol:.2f}\n"
                    f"Current Hour Vol: {cur_vol:,.4f}\n"
                    f"Avg Hour Vol ({AVG_HOURS}h): {avg_vol:,.4f}\n"
                    f"UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}"
                )

                send_telegram(msg, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                last_alert[coin] = now
                print(f"[ALERT] {coin} relvol={relvol:.2f}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
