import time
import math
import random
import threading
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Dict, Optional

import requests


# =========================
# CONFIG (EDIT THESE)
# =========================

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

# Scanner cadence
SCAN_EVERY_SECONDS = 900  # scan once per minute

# Coinbase REST
BASE_URL = "https://api.exchange.coinbase.com"

# Candle granularity: 15m = 900
GRANULARITY_SECONDS = 900

# Value Area settings
VALUE_AREA_PCT = 0.70          # 70% value area
PRICE_BIN_PCT = 0.001          # 0.10% bins (0.001 = 0.1%)
MIN_CANDLES_REQUIRED = 40      # avoid garbage if week just started

# Alert spam protection
ALERT_COOLDOWN_SECONDS = 30 * 60  # 30 min per symbol per side


# =========================
# HTTP + Rate limiting
# =========================

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "vwap-value-scanner/1.0"})


class RateLimiter:
    """Simple shared limiter: caps global request rate across threads."""
    def __init__(self, rps: float):
        self.min_interval = 1.0 / rps
        self.lock = threading.Lock()
        self.next_ok = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.time()
            if now < self.next_ok:
                time.sleep(self.next_ok - now)
            self.next_ok = max(self.next_ok, time.time()) + self.min_interval


# Stay safely under Coinbase public limits
REST_LIMITER = RateLimiter(rps=8.0)


def rest_get_json(url: str, params=None, timeout=20, max_retries=8):
    last_err = None
    for attempt in range(max_retries):
        try:
            REST_LIMITER.wait()
            r = SESSION.get(url, params=params, timeout=timeout)

            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait_s = float(ra) if ra else min(2 ** attempt, 30) + random.random()
                time.sleep(wait_s)
                continue

            r.raise_for_status()
            return r.json()

        except requests.RequestException as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30) + random.random())

    raise RuntimeError(f"REST failed after retries: {last_err}")


def send_telegram(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TELEGRAM] missing token/chat_id; printing instead:\n", text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    for attempt in range(6):
        try:
            r = SESSION.post(url, data=payload, timeout=15)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait_s = float(ra) if ra else min(2 ** attempt, 30) + random.random()
                time.sleep(wait_s)
                continue
            r.raise_for_status()
            return
        except requests.RequestException:
            time.sleep(min(2 ** attempt, 30) + random.random())


# =========================
# Time helpers
# =========================

def week_start_utc(dt: datetime) -> datetime:
    """Return Monday 00:00 UTC for dt."""
    dt = dt.astimezone(timezone.utc)
    # Monday = 0
    days_since_monday = dt.weekday()
    start = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc) - timedelta(days=days_since_monday)
    return start.replace(hour=0, minute=0, second=0, microsecond=0)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# =========================
# Market calc
# =========================

def fetch_week_candles(product_id: str, start: datetime, end: datetime) -> List[Tuple[float, float, float, float, float, int]]:
    """
    Coinbase candles endpoint:
    returns [time, low, high, open, close, volume]
    """
    url = f"{BASE_URL}/products/{product_id}/candles"
    params = {
        "start": iso(start),
        "end": iso(end),
        "granularity": GRANULARITY_SECONDS,
    }
    data = rest_get_json(url, params=params)
    # API returns newest-first; sort ascending by time
    data.sort(key=lambda x: x[0])
    return [(c[0], c[1], c[2], c[3], c[4], c[5]) for c in data]


def weekly_vwap(candles) -> Optional[float]:
    """
    Approx VWAP using typical price and candle volume.
    typical = (H+L+C)/3
    """
    pv = 0.0
    vv = 0.0
    for _, low, high, _open, close, vol in candles:
        if vol is None:
            continue
        tp = (high + low + close) / 3.0
        pv += tp * vol
        vv += vol
    if vv <= 0:
        return None
    return pv / vv


def value_area_vah_val(candles, bin_pct: float = PRICE_BIN_PCT, value_area_pct: float = VALUE_AREA_PCT
                       ) -> Optional[Tuple[float, float, float]]:
    """
    Approx VAH/POC/VAL by building a volume-at-price histogram using typical price.

    Steps:
      1) Bin each candle's volume into a price bucket based on typical price.
      2) POC = bucket with max volume.
      3) Expand up/down from POC, adding next-highest adjacent buckets,
         until we include `value_area_pct` of total volume.
    """
    vols: Dict[int, float] = {}
    total_vol = 0.0

    tps = []
    for _, low, high, _open, close, vol in candles:
        if not vol or vol <= 0:
            continue
        tp = (high + low + close) / 3.0
        tps.append(tp)
        total_vol += vol

        # bin index in log space so bins scale with price
        # bucket width ~ bin_pct (e.g., 0.1%)
        idx = int(round(math.log(tp) / math.log(1 + bin_pct)))
        vols[idx] = vols.get(idx, 0.0) + vol

    if total_vol <= 0 or not vols:
        return None

    # Find POC
    poc_idx = max(vols.items(), key=lambda kv: kv[1])[0]
    poc_price = (1 + bin_pct) ** poc_idx

    # Expand to cover value_area_pct of volume
    target = total_vol * value_area_pct
    included = set([poc_idx])
    included_vol = vols[poc_idx]

    low_idx = poc_idx
    high_idx = poc_idx

    # Expand outward by choosing the next adjacent side with more volume
    while included_vol < target:
        left = low_idx - 1
        right = high_idx + 1
        left_vol = vols.get(left, 0.0)
        right_vol = vols.get(right, 0.0)

        if left_vol == 0.0 and right_vol == 0.0:
            break

        if right_vol >= left_vol:
            included.add(right)
            included_vol += right_vol
            high_idx = right
        else:
            included.add(left)
            included_vol += left_vol
            low_idx = left

    val_price = (1 + bin_pct) ** low_idx
    vah_price = (1 + bin_pct) ** high_idx
    return (val_price, poc_price, vah_price)


# =========================
# Scanner logic
# =========================

_last_alert: Dict[Tuple[str, str], float] = {}  # (product, side) -> last_ts


def should_alert(product: str, side: str) -> bool:
    now = time.time()
    key = (product, side)
    last = _last_alert.get(key, 0.0)
    if now - last >= ALERT_COOLDOWN_SECONDS:
        _last_alert[key] = now
        return True
    return False


def scan_symbol(product_id: str) -> Optional[str]:
    now = datetime.now(timezone.utc)
    start = week_start_utc(now)
    end = now

    candles = fetch_week_candles(product_id, start, end)
    if len(candles) < MIN_CANDLES_REQUIRED:
        return None

    vwap = weekly_vwap(candles)
    va = value_area_vah_val(candles)
    if vwap is None or va is None:
        return None

    val, poc, vah = va
    last_close = candles[-1][4]

    # Conditions
    long_ok = (last_close > vwap) and (last_close > vah)
    short_ok = (last_close < vwap) and (last_close < val)

    if long_ok and should_alert(product_id, "LONG"):
        return (
            f"🟢 VWAP+VALUE LONG SETUP\n\n"
            f"Coin: {product_id}\n"
            f"Price: {last_close:.6g}\n"
            f"Weekly VWAP: {vwap:.6g}\n"
            f"VAH: {vah:.6g}\n"
            f"POC: {poc:.6g}\n"
            f"VAL: {val:.6g}\n"
            f"UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

    if short_ok and should_alert(product_id, "SHORT"):
        return (
            f"🔴 VWAP+VALUE SHORT SETUP\n\n"
            f"Coin: {product_id}\n"
            f"Price: {last_close:.6g}\n"
            f"Weekly VWAP: {vwap:.6g}\n"
            f"VAH: {vah:.6g}\n"
            f"POC: {poc:.6g}\n"
            f"VAL: {val:.6g}\n"
            f"UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

    return None


def main() -> None:
    print("Starting weekly VWAP + value area scanner...")
    print(f"Coins: {len(COINS)} | Granularity: {GRANULARITY_SECONDS}s | Scan every {SCAN_EVERY_SECONDS}s")

    while True:
        t0 = time.time()
        for product in COINS:
            try:
                msg = scan_symbol(product)
                if msg:
                    print(msg)
                    send_telegram(msg)
            except Exception as e:
                print(f"[SCAN] {product}: error: {e}")

        dt = time.time() - t0
        sleep_for = max(0.0, SCAN_EVERY_SECONDS - dt)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
