from dataclasses import dataclass
from typing import Any, Dict
from datetime import datetime, timezone
import time
import requests
import pandas as pd

TIMEFRAME_TO_SECONDS = {
    "15m": 900,
    "1H": 3600,
    "6H": 21600,
    "1D": 86400,
}

CB_BASE = "https://api.exchange.coinbase.com"

@dataclass
class ExchangeClient:
    cfg: Dict[str, Any]

    def _granularity(self, timeframe: str) -> int:
        if timeframe not in TIMEFRAME_TO_SECONDS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return TIMEFRAME_TO_SECONDS[timeframe]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
        """
        Robust OHLCV fetch with exponential backoff on 429/5xx and network errors.
        """
        gran = self._granularity(timeframe)
        params = {"granularity": gran}
        url = f"{CB_BASE}/products/{symbol.upper()}/candles"

        max_attempts = 6        # ~1 + 2 + 4 + 8 + 16 + 32s worst-case backoff
        base_delay = 1.0
        last_err = None

        for attempt in range(max_attempts):
            try:
                r = requests.get(url, params=params, timeout=15)
                if r.status_code == 200:
                    data = r.json() or []
                    rows = []
                    for row in data[:limit]:
                        t, low, high, opn, close, vol = row
                        rows.append({
                            "timestamp": datetime.fromtimestamp(t, tz=timezone.utc),
                            "open": float(opn),
                            "high": float(high),
                            "low": float(low),
                            "close": float(close),
                            "volume": float(vol),
                        })
                    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
                    return df

                # Explicit 429 handling: rate limit
                if r.status_code == 429:
                    delay = base_delay * (2 ** attempt)
                    jitter = (attempt % 3) * 0.1  # tiny jitter (0–0.2–0.3s)
                    time.sleep(delay + jitter)
                    last_err = RuntimeError(f"429 rate limit for {symbol} {timeframe}: {r.text}")
                    continue

                # Transient 5xx: back off and retry
                if 500 <= r.status_code < 600:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    last_err = RuntimeError(f"{r.status_code} server error for {symbol} {timeframe}: {r.text}")
                    continue

                # Other errors: raise
                r.raise_for_status()

            except requests.RequestException as e:
                # Network hiccup — back off and retry
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                last_err = e

        # Exhausted attempts
        raise RuntimeError(f"Failed to fetch candles for {symbol} {timeframe} after retries: {last_err}")

    def fetch_orderbook(self, symbol: str, depth: int = 200) -> Dict[str, Any]:
        level = 2 if depth and depth > 1 else 1
        url = f"{CB_BASE}/products/{symbol.upper()}/book"
        params = {"level": level}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def fetch_last_price(self, symbol: str) -> float:
        url = f"{CB_BASE}/products/{symbol.upper()}/ticker"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        j = r.json()
        return float(j.get("price"))
