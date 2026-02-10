
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

def _recent_swing(df: pd.DataFrame, lookback: int = 120) -> Tuple[Optional[float], Optional[float]]:
    if len(df) < 5:
        return None, None
    recent = df.tail(min(lookback, len(df)))
    return float(recent['low'].min()), float(recent['high'].max())

def _fib_zone_touch(price: float, swing_low: float, swing_high: float, zone=(0.618, 0.66), tol_pct=0.25) -> bool:
    if swing_low is None or swing_high is None or swing_high <= swing_low:
        return False
    z1 = swing_low + (swing_high - swing_low) * zone[0]
    z2 = swing_low + (swing_high - swing_low) * zone[1]
    low, high = (min(z1, z2), max(z1, z2))
    tol = price * (tol_pct / 100.0)
    return (low - tol) <= price <= (high + tol)

def _stoch_confirm_15m(df15: pd.DataFrame, side: str, klt=40, kgt=60) -> bool:
    if 'stochk_15m' not in df15 or 'stochd_15m' not in df15:
        return False
    k = df15['stochk_15m'].tail(3).values
    d = df15['stochd_15m'].tail(3).values
    if len(k) < 3 or np.isnan(k).any() or np.isnan(d).any():
        return False
    prev_cross_long = k[-3] <= d[-3] or k[-2] <= d[-2]
    prev_cross_short = k[-3] >= d[-3] or k[-2] >= d[-2]
    now_long = (k[-1] > d[-1]) and (k[-1] < klt)
    now_short = (k[-1] < d[-1]) and (k[-1] > kgt)
    if side == 'long':
        return prev_cross_long and now_long
    else:
        return prev_cross_short and now_short

def _volume_pctile_ok(df1h: pd.DataFrame, min_pctile: int = 70) -> Tuple[bool, float]:
    vol = df1h['volume'].tail(120)
    if len(vol) < 20:
        return False, 0.0
    pctile = (vol.rank(pct=True).iloc[-1]) * 100.0
    return pctile >= min_pctile, float(pctile)

def _ema_stack_ok(df1h: pd.DataFrame, side: str) -> bool:
    e10 = df1h['ema10_1H'].iloc[-1]
    e20 = df1h['ema20_1H'].iloc[-1]
    e50 = df1h['ema50_1H'].iloc[-1]
    e200 = df1h['ema200_1H'].iloc[-1]
    if side == 'long':
        return e10 > e20 > e50 > e200
    else:
        return e10 < e20 < e50 < e200

def _ema_targets(df1h: pd.DataFrame, price: float) -> Tuple[Optional[float], Optional[float]]:
    e10 = df1h['ema10_1H'].iloc[-1]
    e20 = df1h['ema20_1H'].iloc[-1]
    e50 = df1h['ema50_1H'].iloc[-1]
    e200 = df1h['ema200_1H'].iloc[-1]
    emas = sorted([e10, e20, e50, e200])
    below = max([e for e in [e10,e20,e50,e200] if e <= price], default=None)
    above = min([e for e in [e10,e20,e50,e200] if e >= price], default=None)
    return below, above

class EntryEngine:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def check_trend_entry(self, df1h: pd.DataFrame, df15: pd.DataFrame, bias: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(bias, dict) or bias.get('bias') not in ('long','short'):
            return None
        side = bias['bias']
        if df1h is None or df1h.empty or df15 is None or df15.empty:
            return None
        price = float(df1h['close'].iloc[-1])

        swing_low, swing_high = _recent_swing(df1h, lookback=120)
        zone = tuple(self.cfg['entry']['fib_golden_zone'])
        tol = float(self.cfg['entry']['fib_tolerance_pct'])
        in_fib = _fib_zone_touch(price, swing_low, swing_high, zone=zone, tol_pct=tol)

        klt = int(self.cfg['entry']['stoch_rsi_kd_long']['k_lt'])
        kgt = int(self.cfg['entry']['stoch_rsi_kd_short']['k_gt'])
        stoch_ok = _stoch_confirm_15m(df15, side, klt=klt, kgt=kgt)

        vol_ok, vol_pct = _volume_pctile_ok(df1h, min_pctile=int(self.cfg['entry']['volume_pctile_min']))

        ema_stack = _ema_stack_ok(df1h, side)
        ema_below, ema_above = _ema_targets(df1h, price)

        if not (in_fib and stoch_ok and vol_ok):
            return None

        ctx = {
            'side': side,
            'entry_price': price,
            'swings': {'low': swing_low, 'high': swing_high},
            'ema_below': ema_below,
            'ema_above': ema_above,
            'volume_pctile': vol_pct,
            'bias_alignment': True,
            'fib_or_fvg_touch': True,
            'stoch_confirmation': True,
            'volume_ok': True,
            'ema_stack': bool(ema_stack),
            # 'cluster_support' filled later (clusters service)
        }
        return ctx

    def check_mean_reversion(self, df1h: pd.DataFrame, df15: pd.DataFrame) -> Optional[Dict[str, Any]]:
        return None
