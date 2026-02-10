#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autotune_from_logs.py — offline tuner for overshoot/confluence gates and TP/SL (ATR)
------------------------------------------------------------------------------------
Reads live_flow_levels_log.csv (5-min samples) and proposes:
- GATE_OVERSHOOT_MIN (per-coin, optional global fallback)
- GATE_LONG_CONF_MIN (1/2/3)
- TP1_ATR and SL_ATR (per-coin or global)
Writes tuning_rules.json (hot-reloadable by your main script).

Method (fast, conservative approximation):
- Treat a trade as "TP hit" if ret_pct first crosses +TP1% before -SL% within 120 min.
- If neither hit, use final ret_pct at 120 min as outcome for this parameter set.
- Evaluate grid of candidates and maximize mean return (or median for robustness).
- Require minimum samples per coin (e.g., 50 trades) else fall back to global.

Run:
  python autotune_from_logs.py --csv live_flow_levels_log.csv --out tuning_rules.json \
      --min-samples 50 --win-metric mean --coins ALL
"""
import os, json, argparse
import pandas as pd
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='live_flow_levels_log.csv')
    ap.add_argument('--out', default='tuning_rules.json')
    ap.add_argument('--min-samples', type=int, default=50)
    ap.add_argument('--win-metric', choices=['mean','median'], default='mean')
    ap.add_argument('--coins', default='ALL', help='Comma list of symbols or ALL')
    return ap.parse_args()

# Candidate grids (adjust as needed)
OVERSHOOT_GRID = [1.25, 1.5, 2.0, 3.0]
CONF_LONG_GRID = [1, 2, 3]
TP1_ATR_GRID   = [0.08, 0.10, 0.12, 0.15, 0.20]
SL_ATR_GRID    = [0.25, 0.30, 0.35, 0.40]

# Distance gate fixed per prior analysis
DIST_MIN_ATR = 0.10
DIST_MAX_ATR = 0.50

LOOKBACKS = [60,300,900,1800,3600]

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    # Basic types
    for c in ['entry','atr_1h','sample_px','ret_pct','mae_pct','mfe_pct']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ['timestamp_utc','sample_ts']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce', utc=True)
    # Trade id and elapsed
    df['trade_id'] = df['symbol'].astype(str)+'|'+df['level'].astype(str)+'|'+df['timestamp_utc'].astype(str)
    df = df.sort_values(['trade_id','sample_ts'])
    df['elapsed_min'] = (df['sample_ts'] - df['timestamp_utc']).dt.total_seconds()/60.0
    return df

def parse_meta_cell(s: str):
    if not isinstance(s, str):
        return {}
    i = s.find('{'); j = s.rfind('}')
    if i != -1 and j != -1 and j > i:
        s = s[i:j+1]
    try:
        return json.loads(s)
    except Exception:
        return {}

def build_entries(df: pd.DataFrame) -> pd.DataFrame:
    last = df.groupby('trade_id').tail(1).copy().reset_index(drop=True)
    last['meta'] = last['meta_json'].apply(parse_meta_cell) if 'meta_json' in last.columns else [{}]*len(last)
    # derive features
    rows = []
    for i, r in last.iterrows():
        m = r['meta'] if isinstance(r['meta'], dict) else {}
        # confluence
        fb = m.get('flow_breakdown', {}) if isinstance(m, dict) else {}
        best_lb = m.get('best_lb_s', None)
        best_delta = m.get('best_delta_usd', None)
        thr = m.get('threshold_usd', None)
        if isinstance(best_lb, (int,float)) and str(int(best_lb)) in fb:
            bd = fb[str(int(best_lb))]
            sgn = 1 if float(bd.get('delta', 0)) > 0 else (-1 if float(bd.get('delta', 0)) < 0 else 0)
        else:
            sgn = 0
        conf_long = 0
        for lb in (900,1800,3600):
            rec = fb.get(str(lb), {})
            if isinstance(rec, dict) and 'delta' in rec:
                if sgn != 0 and float(rec['delta']) * sgn > 0:
                    conf_long += 1
        # overshoot
        overshoot = None
        try:
            overshoot = abs(float(best_delta)) / float(thr)
        except Exception:
            overshoot = None
        # distance to triggered level mid (ATR units)
        dist_atr = None
        dist_map = m.get('dist_atr', {}) if isinstance(m, dict) else {}
        if isinstance(dist_map, dict):
            # use the level name in this row if present, else fallback to any single entry
            lvl = r['level']
            key = f"{lvl}_mid" if isinstance(lvl,str) and lvl.startswith('FIB_') else lvl
            try:
                dist_atr = abs(float(dist_map.get(key, np.nan)))
            except Exception:
                dist_atr = None
        rows.append({
            'trade_id': r['trade_id'], 'symbol': r['symbol'], 'side': r['side'], 'level': r['level'],
            'conf_long': conf_long, 'overshoot': overshoot, 'dist_atr': dist_atr
        })
    ent = pd.DataFrame(rows)
    return ent

def simulate_trade(tr_df: pd.DataFrame, side: str, entry: float, atr_1h: float, tp1_atr: float, sl_atr: float) -> float:
    """Return realized % change under simple TP1/SL model within 120m.
    - If ret_pct crosses +tp1 first → realize +tp1 (in % of entry).
    - If ret_pct crosses -sl first  → realize -sl.
    - Else use final ret_pct at last sample.
    """
    if tr_df.empty or not np.isfinite(entry) or not np.isfinite(atr_1h) or atr_1h <= 0:
        return 0.0
    # percent thresholds from ATR
    tp1_pct = (tp1_atr * atr_1h / entry) * 100.0
    sl_pct  = (sl_atr  * atr_1h / entry) * 100.0
    # side-aware ret
    ret = tr_df['ret_pct'].to_numpy()  # already side-adjusted in your logger
    # first index where ret >= tp1_pct
    hit_tp = np.argmax(ret >= tp1_pct) if np.any(ret >= tp1_pct) else -1
    # first index where ret <= -sl_pct
    hit_sl = np.argmax(ret <= -sl_pct) if np.any(ret <= -sl_pct) else -1
    if hit_tp == -1 and hit_sl == -1:
        return float(ret[-1])
    if hit_tp != -1 and (hit_sl == -1 or hit_tp < hit_sl):
        return float(tp1_pct)
    else:
        return float(-sl_pct)

def evaluate(df: pd.DataFrame, entries: pd.DataFrame, coins: list[str], min_samples: int, win_metric: str):
    # Merge raw rows needed for per-trade simulation
    keep_cols = ['trade_id','symbol','side','entry','atr_1h','ret_pct','timestamp_utc','sample_ts']
    dfr = df[keep_cols].dropna().copy()
    # Ensure sorted by time
    dfr = dfr.sort_values(['trade_id','sample_ts'])

    # candidate combos
    best = {
        'global': {'score': -1e9, 'params': None, 'n': 0}
    }
    per_coin = {}

    # helper to compute score for a subset
    def score_for(trade_ids, tp1_atr, sl_atr):
        vals = []
        for tid in trade_ids:
            rows = dfr[dfr['trade_id'] == tid]
            if rows.empty: continue
            entry = float(rows['entry'].iloc[0]); atr = float(rows['atr_1h'].iloc[0]); side = rows['side'].iloc[0]
            vals.append(simulate_trade(rows, side, entry, atr, tp1_atr, sl_atr))
        if not vals: return -1e9
        if win_metric == 'mean':
            return float(np.nanmean(vals))
        else:
            return float(np.nanmedian(vals))

    # Build trade list per coin respecting gates from entries
    def gated_trade_ids(symbol, conf_min, ov_min):
        sub = entries.copy()
        if symbol is not None:
            sub = sub[sub['symbol'] == symbol]
        m = pd.Series(True, index=sub.index)
        m &= sub['conf_long'] >= conf_min
        m &= sub['overshoot'] >= ov_min
        m &= (sub['dist_atr'] >= DIST_MIN_ATR) & (sub['dist_atr'] <= DIST_MAX_ATR)
        return list(sub.loc[m, 'trade_id'])

    # Global search (all coins together)
    all_tids_cache = {}
    for conf in CONF_LONG_GRID:
        for ov in OVERSHOOT_GRID:
            key = (conf, ov)
            all_tids_cache[key] = gated_trade_ids(None, conf, ov)
            if len(all_tids_cache[key]) < min_samples:
                continue
            for tp1 in TP1_ATR_GRID:
                for sl in SL_ATR_GRID:
                    s = score_for(all_tids_cache[key], tp1, sl)
                    if s > best['global']['score']:
                        best['global'] = {'score': s, 'params': {'conf_long_min': conf, 'overshoot_min': ov, 'tp1_atr': tp1, 'sl_atr': sl}, 'n': len(all_tids_cache[key])}

    # Per-coin search
    for symbol in (coins or sorted(entries['symbol'].unique())):
        best_sym = {'score': -1e9, 'params': None, 'n': 0}
        for conf in CONF_LONG_GRID:
            for ov in OVERSHOOT_GRID:
                tids = gated_trade_ids(symbol, conf, ov)
                if len(tids) < min_samples:
                    continue
                for tp1 in TP1_ATR_GRID:
                    for sl in SL_ATR_GRID:
                        s = score_for(tids, tp1, sl)
                        if s > best_sym['score']:
                            best_sym = {'score': s, 'params': {'symbol': symbol, 'conf_long_min': conf, 'overshoot_min': ov, 'tp1_atr': tp1, 'sl_atr': sl}, 'n': len(tids)}
        if best_sym['params'] is not None:
            per_coin[symbol] = best_sym

    return best, per_coin


def main():
    args = parse_args()
    df = load_csv(args.csv)
    entries = build_entries(df)
    coins = None if args.coins == 'ALL' else [s.strip() for s in args.coins.split(',') if s.strip()]
    best_global, best_per = evaluate(df, entries, coins, args.min_samples, args.win_metric)

    out = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'global': best_global['global'],
        'per_coin': {k: v for k, v in best_per.items()}
    }
    # Write tuning rules
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}. Global: {best_global['global']}")

if __name__ == '__main__':
    from datetime import datetime
    main()
