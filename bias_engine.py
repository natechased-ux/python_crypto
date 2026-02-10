
from typing import Dict, Any

class BiasEngine:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def compute(self, feats_6h: Dict[str, Any], feats_1d: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out = {}
        adx_min = self.cfg['bias']['adx_min']
        daily_ema_filter = self.cfg['bias'].get('daily_ema200_filter', True)
        for sym, df6 in feats_6h.items():
            if df6 is None or df6.empty:
                out[sym] = {"bias": "neutral", "reasons": {"empty": True}}
                continue
            df1d = feats_1d.get(sym)
            price = df6['close'].iloc[-1]
            adx_val = df6['adx_6H'].iloc[-1] if 'adx_6H' in df6 else None
            pdi = df6['pdi_6H'].iloc[-1] if 'pdi_6H' in df6 else None
            mdi = df6['mdi_6H'].iloc[-1] if 'mdi_6H' in df6 else None

            ok_daily = True
            if daily_ema_filter and df1d is not None and not df1d.empty and 'ema200_1D' in df1d:
                ema200 = df1d['ema200_1D'].iloc[-1]
                ok_daily = price > ema200

            if adx_val is not None and pdi is not None and mdi is not None and adx_val >= adx_min:
                if pdi > mdi and ok_daily:
                    out[sym] = {"bias": "long", "reasons": {"adx": float(adx_val), "+DI>-DI": True, "daily_ok": ok_daily}}
                elif mdi > pdi and (not ok_daily):
                    out[sym] = {"bias": "short", "reasons": {"adx": float(adx_val), "-DI>+DI": True, "daily_ok": ok_daily}}
                else:
                    out[sym] = {"bias": "neutral", "reasons": {"adx": float(adx_val), "daily_ok": ok_daily}}
            else:
                out[sym] = {"bias": "neutral", "reasons": {"adx": float(adx_val) if adx_val is not None else None}}
        return out
