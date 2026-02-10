from typing import Dict, Any, List
import yaml
import pandas as pd
from .data import ExchangeClient
from .indicators import ema, atr, adx, macd, rsi, stoch_rsi, bb_width

def _load_universe(cfg: Dict[str, Any]) -> List[str]:
    path = cfg['universe']['include']
    with open(path, 'r') as f:
        coins = yaml.safe_load(f) or []
    if not isinstance(coins, list):
        coins = []
    excl = set(cfg['universe'].get('exclude', []))
    return [c for c in coins if c not in excl]

def build_features(ex: ExchangeClient, cfg: Dict[str, Any], tfs: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Build indicator features per timeframe.
    Any fetch failure (incl. 429) for a (symbol, timeframe) is logged and skipped,
    but the rest of the scan continues.
    """
    symbols = _load_universe(cfg)
    out: Dict[str, Dict[str, pd.DataFrame]] = {tf: {} for tf in tfs}

    for tf in tfs:
        for sym in symbols:
            try:
                df = ex.fetch_ohlcv(sym, tf, limit=300)
            except Exception as e:
                print(f"[WARN] fetch failed {sym} {tf}: {e}")
                out[tf][sym] = pd.DataFrame()
                continue

            if df.empty:
                out[tf][sym] = df
                continue

            # EMAs
            df[f'ema10_{tf}'] = ema(df['close'], 10)
            df[f'ema20_{tf}'] = ema(df['close'], 20)
            df[f'ema50_{tf}'] = ema(df['close'], 50)
            df[f'ema200_{tf}'] = ema(df['close'], 200)

            # DI/ADX on higher frames
            if tf in ['6H', '1D', '1H']:
                adx_v, pdi, mdi = adx(df, 14)
                df[f'adx_{tf}'] = adx_v
                df[f'pdi_{tf}'] = pdi
                df[f'mdi_{tf}'] = mdi

            # MACD hist + RSI
            _, _, m_hist = macd(df['close'])
            df[f'macd_hist_{tf}'] = m_hist
            df[f'rsi_{tf}'] = rsi(df['close'], 14)

            # Stoch RSI on 15m/1H
            if tf in ['15m', '1H']:
                sr = stoch_rsi(df['close'], 14, 14)
                df[f'stochk_{tf}'] = sr['k']
                df[f'stochd_{tf}'] = sr['d']

            # ATR + BB width on 1H/6H
            if tf in ['1H', '6H']:
                df[f'atr_{tf}'] = atr(df, 14)
                df[f'bb_width_{tf}'] = bb_width(df['close'], 20, 2.0)

            out[tf][sym] = df

    return out
