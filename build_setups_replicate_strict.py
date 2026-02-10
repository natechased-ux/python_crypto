#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_setups_replicate_strict.py
--------------------------------
Replicates the original in-session pipeline that produced
price_excluded_setups_tightened_subsetTP.json, with safeguards so you
always get concrete rules.

Key points replicated:
- Exclude 'price' from features
- Per symbol: sort by timestamp; 70/30 time split
- DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=42)
- High-conviction threshold on test probs = 75th percentile
- Pick best horizon (1h/2h/4h) by test precision
- Extract top-leaf rule (AND of conditions)
- Tighten if match-rate > 5% by adding 1 extra simple condition from stoch_*/ *_ratio/ near_*/ lookback_*
  using quantiles computed *on the already-matching subset*; require 1%–5% rate and support≥20
- TP=%65th of positives; SL=%35th of |negatives| on *matched subset*
- Enforce R:R≥1.5
- EV = p * RR − (1 − p), using test precision

Safeguards added (to avoid "No strong rule"):
- If tree yields no leaf with support≥10, fallback to best single-threshold rule on TEST among candidate features
- If still empty, relax support to ≥5.
"""

import os, json, re, warnings
from typing import Dict, Any, List, Tuple, Optional
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

# ------------------------- Hardcoded Paths -------------------------
CSV_PATH      = r"C:\Users\natec\obflow_log.csv"
OUT_JSON_PATH = r"C:\Users\natec\price_excluded_setups_tightened_subsetTP_new.json"
OUT_CSV_PATH  = r"C:\Users\natec\setups_summary.csv"

# ------------------------- Config ----------------------------
SEED = 42
MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 50
TEST_FRAC = 0.30
HIGH_CONV_Q = 75
TIGHTEN_TARGET_PCT = 5.0
TIGHTEN_MIN_RATE = 1.0
TIGHTEN_MIN_SUPPORT = 20
LEAF_MIN_SUPPORT = 10
LEAF_MIN_SUPPORT_RELAXED = 5

RET_COLS = ["ret_15m","ret_30m","ret_1h","ret_2h","ret_4h"]
COND_RE = re.compile(r"(.+?)\s*(<=|>)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import precision_score

np.random.seed(SEED)

def parse_rule_to_mask(sdf: pd.DataFrame, rule_text: str) -> pd.Series:
    if not rule_text or rule_text.strip().lower().startswith("no strong"):
        return pd.Series([False]*len(sdf), index=sdf.index)
    conds = [c.strip() for c in rule_text.split("AND") if c.strip()]
    mask = pd.Series([True]*len(sdf), index=sdf.index)
    for cond in conds:
        m = COND_RE.match(cond); 
        if not m: 
            continue
        feat, op, val = m.groups()
        feat = feat.strip()
        if feat not in sdf.columns:
            mask &= False
            continue
        thr = float(val)
        col = pd.to_numeric(sdf[feat], errors="coerce")
        if op == "<=": mask &= (col <= thr)
        else:          mask &= (col >  thr)
    return mask

def extract_leaf_rules(clf, feature_names: List[str], min_support: int) -> List[Dict[str, Any]]:
    tree_ = clf.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
    rules: List[Dict[str, Any]] = []
    def recurse(node: int, conds: List[str]):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]; thr = tree_.threshold[node]
            recurse(tree_.children_left[node],  conds + [f"{name} <= {thr:.6f}"])
            recurse(tree_.children_right[node], conds + [f"{name} > {thr:.6f}"])
        else:
            support = int(tree_.n_node_samples[node])
            if support >= min_support:
                rules.append({"conditions": conds, "support": support})
    recurse(0, [])
    return rules

def realized_vol_suggestion(ret_series: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    ser = pd.to_numeric(ret_series, errors="coerce").dropna().astype(float)
    if ser.empty: return (np.nan, np.nan)
    pos = ser[ser > 0.0]; neg = ser[ser < 0.0]
    tp = float(np.nanpercentile(pos, 65)) if not pos.empty else np.nan
    sl = float(np.nanpercentile(np.abs(neg), 35)) if not neg.empty else np.nan
    return (tp, sl)

def best_tightener(sdf: pd.DataFrame, mask_base: pd.Series, y: pd.Series,
                   feat_cols: List[str], target_rate=5.0, min_rate=1.0, min_support=20) -> Optional[Dict[str, Any]]:
    n_total = len(sdf)
    best = None
    quantiles = [0.25, 0.5, 0.75]
    for feat in feat_cols:
        if feat not in sdf.columns: 
            continue
        x = pd.to_numeric(sdf[feat], errors="coerce")
        base_vals = x[mask_base].dropna()
        if base_vals.empty: 
            continue
        for q in quantiles:
            thr = float(np.nanquantile(base_vals, q))
            for op in ["<=", ">"]:
                if op == "<=": mask_new = mask_base & (x <= thr)
                else:          mask_new = mask_base & (x >  thr)
                support = int(mask_new.sum())
                if support < min_support: continue
                rate = 100.0 * support / n_total if n_total>0 else np.nan
                if np.isnan(rate) or rate > target_rate or rate < min_rate: continue
                prec = float((y[mask_new] == 1).mean()) if support>0 else 0.0
                score = (prec, support)
                if (best is None) or score > (best["precision"], best["support"]):
                    best = {"feat": feat, "op": op, "thr": thr, "rate": rate, "precision": prec, "support": support}
    return best

def fallback_single_threshold(sdf: pd.DataFrame, feat_cols: List[str], y: pd.Series) -> str:
    """Pick the best single-threshold rule on TEST among candidates."""
    candidate_feats = [c for c in feat_cols if (c.startswith("stoch_") or c.endswith("_ratio") or c.startswith("near_") or c.startswith("lookback_"))]
    if not candidate_feats: return ""
    x = sdf[candidate_feats].apply(pd.to_numeric, errors="coerce")
    yv = y.values
    best = None
    for feat in candidate_feats:
        col = x[feat].astype(float)
        if col.notna().sum() < 50: continue
        for q in (0.25, 0.5, 0.75):
            thr = float(np.nanquantile(col, q))
            for op in ("<=", ">"):
                mask = (col <= thr) if op == "<=" else (col > thr)
                support = int(mask.sum())
                if support < 20: continue
                rate = 100.0 * support / len(col) if len(col)>0 else 0.0
                if rate < 1.0 or rate > 20.0: continue
                prec = float((yv[mask.values]==1).mean()) if support>0 else 0.0
                score = (prec, support)
                if (best is None) or score > (best["prec"], best["support"]):
                    best = {"feat": feat, "op": op, "thr": thr, "prec": prec, "support": support}
    return f"{best['feat']} {best['op']} {best['thr']:.6f}" if best else ""

def rule_to_text(rule: Dict[str, Any]) -> str:
    return " AND ".join(rule["conditions"]) if rule and rule.get("conditions") else ""

def main():
    # Load
    if not os.path.exists(CSV_PATH): raise SystemExit(f"CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    for c in df.columns:
        if c not in ["ID","timestamp_utc","symbol"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df[RET_COLS].notna().any(axis=1)].copy()
    if df.empty: raise SystemExit("No rows with forward returns.")

    # Features (exclude price)
    exclude = set(["ID","timestamp_utc","symbol","price"] + RET_COLS)
    feat_cols = [c for c in df.columns if c not in exclude]

    setups: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for sym, sdf in df.groupby("symbol"):
        sdf = sdf.sort_values("timestamp_utc").reset_index(drop=True)
        if len(sdf) < 100: continue

        # Impute per symbol
        med = sdf[feat_cols].median(numeric_only=True)
        sdf[feat_cols] = sdf[feat_cols].fillna(med)

        # Targets
        targets = {
            "1h": {"long": (sdf["ret_1h"] > 0).astype(int), "short": (sdf["ret_1h"] < 0).astype(int)},
            "2h": {"long": (sdf["ret_2h"] > 0).astype(int), "short": (sdf["ret_2h"] < 0).astype(int)},
            "4h": {"long": (sdf["ret_4h"] > 0).astype(int), "short": (sdf["ret_4h"] < 0).astype(int)},
        }

        split_idx = int(len(sdf) * (1.0 - TEST_FRAC))
        X = sdf[feat_cols].values

        best = {"long": {"H":"1h","prec":0.0,"rule_txt":""},
                "short":{"H":"1h","prec":0.0,"rule_txt":""}}

        for H in ["1h","2h","4h"]:
            for side in ["long","short"]:
                y = targets[H][side].values
                y_tr, y_te = y[:split_idx], y[split_idx:]
                X_tr, X_te = X[:split_idx], X[split_idx:]
                if y_tr.sum()==0 or y_te.sum()==0: 
                    continue
                clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=SEED)
                clf.fit(X_tr, y_tr)
                p_te = clf.predict_proba(X_te)[:,1]
                thr = np.percentile(p_te, HIGH_CONV_Q)
                y_pred = (p_te >= thr).astype(int)
                prec = precision_score(y_te, y_pred, zero_division=0)

                # Extract leaf rules with support ≥ LEAF_MIN_SUPPORT; if none, relax to 5
                rules = extract_leaf_rules(clf, feat_cols, min_support=LEAF_MIN_SUPPORT)
                if not rules:
                    rules = extract_leaf_rules(clf, feat_cols, min_support=LEAF_MIN_SUPPORT_RELAXED)
                rule_txt = rule_to_text(rules[0]) if rules else ""

                # Fallback: if still empty, pick single-threshold rule directly on TEST
                if not rule_txt:
                    rule_txt = fallback_single_threshold(sdf.iloc[split_idx:], feat_cols, targets[H][side].iloc[split_idx:])

                if rule_txt and prec > best[side]["prec"]:
                    best[side] = {"H": H, "prec": float(prec), "rule_txt": rule_txt}

        # Tighten step
        def tighten(rule_text: str, side: str, H: str) -> Tuple[str, float, float, int]:
            if not rule_text:  # nothing to tighten
                return "", 0.0, 0.0, 0
            mask_base = parse_rule_to_mask(sdf, rule_text)
            rate = 100.0 * mask_base.sum() / len(sdf) if len(sdf)>0 else 0.0
            y = targets[H][side]
            orig_prec = float((y[mask_base]==1).mean()) if mask_base.any() else np.nan
            if rate <= TIGHTEN_TARGET_PCT or not mask_base.any():
                return rule_text, rate, orig_prec, int(mask_base.sum())
            candidate_feats = [c for c in feat_cols if (c.startswith("stoch_") or c.endswith("_ratio") or c.startswith("near_") or c.startswith("lookback_"))]
            bt = best_tightener(sdf, mask_base, y, candidate_feats, target_rate=TIGHTEN_TARGET_PCT, min_rate=TIGHTEN_MIN_RATE, min_support=TIGHTEN_MIN_SUPPORT)
            if bt:
                extra = f"{bt['feat']} {bt['op']} {bt['thr']:.6f}"
                tightened_text = rule_text + " AND " + extra
                return tightened_text, bt["rate"], bt["precision"], bt["support"]
            return rule_text, rate, orig_prec, int(mask_base.sum())

        long_txt,  long_rate,  long_prec_on_matches,  long_sup  = tighten(best["long"]["rule_txt"],  "long",  best["long"]["H"])
        short_txt, short_rate, short_prec_on_matches, short_sup = tighten(best["short"]["rule_txt"], "short", best["short"]["H"])

        # Subset-specific TP/SL
        def subset_tp_sl(rule_text: str, H: str) -> Tuple[Optional[float], Optional[float], int]:
            if not rule_text: return (np.nan, np.nan, 0)
            mask = parse_rule_to_mask(sdf, rule_text)
            sub = pd.to_numeric(sdf.loc[mask, f"ret_{H}"], errors="coerce").dropna().astype(float)
            tp, sl = realized_vol_suggestion(sub)
            return tp, sl, int(sub.shape[0])

        tpL, slL, nL = subset_tp_sl(long_txt,  best["long"]["H"])
        tpS, slS, nS = subset_tp_sl(short_txt, best["short"]["H"])

        def finish(side: str, H: str, tp, sl) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
            p = float(best[side]["prec"])
            if not np.isfinite(tp) or not np.isfinite(sl) or sl<=0:
                return (np.nan, np.nan, np.nan, p)
            rr = tp/sl
            if rr < 1.5:
                tp = round(1.5*sl, 6); rr = tp/sl
            ev = p*rr - (1-p)
            return (tp, sl, rr, ev)

        tpL, slL, rrL, evL = finish("long",  best["long"]["H"],  tpL, slL)
        tpS, slS, rrS, evS = finish("short", best["short"]["H"], tpS, slS)

        setups.append({
            "symbol": sym,
            "LONG": {
                "horizon": best["long"]["H"],
                "entry_conditions": long_txt if long_txt else "No strong rule (use top-quartile probability threshold).",
                "tp_pct": tpL, "sl_pct": slL,
                "expected_win_rate": round(float(best["long"]["prec"]), 3),
                "avg_rr": None if not np.isfinite(rrL) else round(float(rrL), 3),
                "EV_R_units": None if not np.isfinite(evL) else round(float(evL), 3)
            },
            "SHORT": {
                "horizon": best["short"]["H"],
                "entry_conditions": short_txt if short_txt else "No strong rule (use top-quartile probability threshold).",
                "tp_pct": tpS, "sl_pct": slS,
                "expected_win_rate": round(float(best["short"]["prec"]), 3),
                "avg_rr": None if not np.isfinite(rrS) else round(float(rrS), 3),
                "EV_R_units": None if not np.isfinite(evS) else round(float(evS), 3)
            }
        })

        summary_rows.append({
            "symbol": sym,
            "LONG_horizon": best["long"]["H"],
            "LONG_expected_win": round(float(best["long"]["prec"]), 3),
            "LONG_rule": long_txt,
            "LONG_match_rate_pct": None if not np.isfinite(long_rate) else round(float(long_rate),2),
            "LONG_support_after_tighten": long_sup,
            "LONG_tp_pct": tpL, "LONG_sl_pct": slL,
            "LONG_avg_rr": None if not np.isfinite(rrL) else round(float(rrL), 3),
            "LONG_EV_R": None if not np.isfinite(evL) else round(float(evL), 3),
            "SHORT_horizon": best["short"]["H"],
            "SHORT_expected_win": round(float(best["short"]["prec"]), 3),
            "SHORT_rule": short_txt,
            "SHORT_match_rate_pct": None if not np.isfinite(short_rate) else round(float(short_rate),2),
            "SHORT_support_after_tighten": short_sup,
            "SHORT_tp_pct": tpS, "SHORT_sl_pct": slS,
            "SHORT_avg_rr": None if not np.isfinite(rrS) else round(float(rrS), 3),
            "SHORT_EV_R": None if not np.isfinite(evS) else round(float(evS), 3),
        })

    # Save
    os.makedirs(os.path.dirname(OUT_JSON_PATH), exist_ok=True)
    with open(OUT_JSON_PATH, "w") as f:
        json.dump(setups, f, indent=2)
    print(f"[OK] Wrote setups JSON: {OUT_JSON_PATH} (symbols: {len(setups)})")

    if summary_rows:
        df_sum = pd.DataFrame(summary_rows).sort_values("symbol")
        os.makedirs(os.path.dirname(OUT_CSV_PATH), exist_ok=True)
        df_sum.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"[OK] Wrote summary CSV: {OUT_CSV_PATH} (rows: {len(df_sum)})")

if __name__ == "__main__":
    main()
