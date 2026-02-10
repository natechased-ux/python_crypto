#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
import os
import re
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.tree import DecisionTreeClassifier, _tree
    from sklearn.metrics import precision_score, accuracy_score
    SKLEARN_OK = True
except Exception as e:
    SKLEARN_OK = False
    DecisionTreeClassifier = None
    _tree = None
    precision_score = None
    accuracy_score = None

# ------------------------- CLI -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Rebuild per-symbol tightened rules + TP/SL + EV from obflow_log.csv")
    ap.add_argument("--csv", required=True, default=r"C:\Users\natec\obflow_log.csv")
    ap.add_argument("--out_json", default=r"C:\Users\natec\price_excluded_setups_tightened_subsetTP_new.json",
                    help="Output JSON file path for alert engine")
    ap.add_argument("--out_csv", default=r"C:\Users\natec\setups_summary.csv",
                    help="Output CSV summary path")
    ap.add_argument("--min_rows_per_symbol", type=int, default=400,
                    help="Minimum rows per symbol to include (fallback to median if none pass)")
    ap.add_argument("--tighten_target_pct", type=float, default=5.0,
                    help="Max %% of rows allowed to match a rule after tightening")
    ap.add_argument("--tighten_min_support", type=int, default=20,
                    help="Minimum number of matching rows to accept a tightened rule")
    return ap.parse_args()

# ---------------------- Utilities ----------------------
COND_RE = re.compile(r"(.+?)\s*(<=|>)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

def parse_rule_to_mask(sdf: pd.DataFrame, rule_text: str) -> pd.Series:
    """Convert rule text ('feat <= thr AND feat > thr') to a boolean mask over sdf."""
    if not rule_text or rule_text.strip().lower().startswith("no strong"):
        return pd.Series([False]*len(sdf), index=sdf.index)
    conds = [c.strip() for c in rule_text.split("AND") if c.strip()]
    mask = pd.Series([True]*len(sdf), index=sdf.index)
    for cond in conds:
        m = COND_RE.match(cond)
        if not m:
            continue
        feat, op, val = m.groups()
        feat = feat.strip()
        if feat not in sdf.columns:
            mask &= False
            continue
        try:
            thr = float(val)
        except Exception:
            mask &= False
            continue
        if op == "<=":
            mask &= (pd.to_numeric(sdf[feat], errors="coerce") <= thr)
        else:
            mask &= (pd.to_numeric(sdf[feat], errors="coerce") >  thr)
    return mask

def extract_rules_from_tree(clf, feature_names: List[str]) -> List[Dict[str, Any]]:
    """Translate a fitted DecisionTreeClassifier into human-readable rules with (support, prob)."""
    rules: List[Dict[str, Any]] = []
    if clf is None:
        return rules
    tree_ = clf.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    def recurse(node: int, conds: List[str]):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            thr = tree_.threshold[node]
            recurse(tree_.children_left[node],  conds + [f"{name} <= {thr:.6f}"])
            recurse(tree_.children_right[node], conds + [f"{name} > {thr:.6f}"])
        else:
            value = tree_.value[node][0]  # [neg, pos]
            total = int(value.sum())
            pos = float(value[1]) if len(value) >= 2 else float(value.max())
            prob = (pos / total) if total > 0 else np.nan
            rules.append({"conditions": conds, "n": total, "prob": float(prob)})
    recurse(0, [])
    rules = [r for r in rules if r["n"] >= 10]
    rules.sort(key=lambda r: (r["prob"], r["n"]), reverse=True)
    return rules[:10]

def realized_vol_suggestion(ret_series: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """Return (tp_pct, sl_pct) from the subset return distribution: TP=P65(pos), SL=P35(|neg|)."""
    ret_series = pd.to_numeric(ret_series, errors="coerce").dropna().astype(float)
    if ret_series.empty:
        return (np.nan, np.nan)
    pos = ret_series[ret_series > 0.0]
    neg = ret_series[ret_series < 0.0]
    tp = float(np.nanpercentile(pos, 65)) if not pos.empty else np.nan
    sl = float(np.nanpercentile(np.abs(neg), 35)) if not neg.empty else np.nan
    return (tp, sl)

def best_extra_filter(sdf: pd.DataFrame, base_mask: pd.Series, y: pd.Series,
                      feat_cols: List[str],
                      target_rate: float = 5.0,
                      min_support: int = 20) -> Optional[Dict[str, Any]]:
    """Find one extra simple condition to reduce match rate <= target_rate and maximize precision."""
    n_total = len(sdf)
    if n_total == 0 or not base_mask.any():
        return None
    # Candidate features: stoch_*, *_ratio, near_*, lookback_*
    cands = [c for c in feat_cols if (
        c.startswith("stoch_") or c.endswith("_ratio") or c.startswith("near_") or c.startswith("lookback_")
    )]
    quantiles = [0.25, 0.5, 0.75]
    best = None
    for feat in cands:
        if feat not in sdf.columns:
            continue
        x = pd.to_numeric(sdf[feat], errors="coerce")
        base_vals = x[base_mask].dropna()
        if base_vals.empty:
            continue
        for q in quantiles:
            thr = float(np.nanquantile(base_vals, q))
            for op in ("<=", ">"):
                if op == "<=":
                    m_new = base_mask & (x <= thr)
                else:
                    m_new = base_mask & (x > thr)
                support = int(m_new.sum())
                if support < min_support:
                    continue
                rate = 100.0 * support / n_total
                if rate <= target_rate + 1e-9:
                    prec = float((y[m_new] == 1).mean()) if support > 0 else 0.0
                    # Slight preference for rates near 60-100% of target to avoid too tiny cohorts
                    desired_min = max(1.0, target_rate * 0.6)
                    rate_penalty = abs(max(desired_min, 0.0) - rate)
                    score = prec - 0.001 * rate_penalty
                    if best is None or score > best["score"]:
                        best = {"feat": feat, "op": op, "thr": thr, "rate": rate, "precision": prec, "score": score, "support": support}
    return best

# ---------------------- Main Logic ----------------------
def main():
    args = parse_args()
    if not SKLEARN_OK:
        raise SystemExit("scikit-learn is required. Install with: pip install scikit-learn")

    # Load data
    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")
    df = pd.read_csv(args.csv, encoding="utf-8-sig")

    # Parse time and numerics
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    for c in df.columns:
        if c not in ["ID","timestamp_utc","symbol"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep rows with any forward returns
    ret_cols = ["ret_15m","ret_30m","ret_1h","ret_2h","ret_4h"]
    df = df[df[ret_cols].notna().any(axis=1)].copy()

    # Eligibility by symbol count
    counts = df["symbol"].value_counts()
    eligible = counts[counts >= args.min_rows_per_symbol].index.tolist()
    if not eligible:
        fallback = max(50, int(df["symbol"].value_counts().median()))
        eligible = df["symbol"].value_counts()[df["symbol"].value_counts() >= fallback].index.tolist()

    # Feature columns (drop identifiers, targets, and price)
    exclude = set(["ID","timestamp_utc","symbol","price"] + ret_cols)
    feat_cols = [c for c in df.columns if c not in exclude]

    setups_out: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for sym in sorted(eligible):
        sdf = df[df["symbol"]==sym].sort_values("timestamp_utc").copy()
        if len(sdf) < 100:
            continue

        # Impute per symbol
        med = sdf[feat_cols].median(numeric_only=True)
        sdf[feat_cols] = sdf[feat_cols].fillna(med)

        # Targets
        targets = {
            "1h": {"long": (sdf["ret_1h"] > 0).astype(int), "short": (sdf["ret_1h"] < 0).astype(int)},
            "2h": {"long": (sdf["ret_2h"] > 0).astype(int), "short": (sdf["ret_2h"] < 0).astype(int)},
            "4h": {"long": (sdf["ret_4h"] > 0).astype(int), "short": (sdf["ret_4h"] < 0).astype(int)},
        }

        # Time split
        split = int(len(sdf)*0.7)
        X = sdf[feat_cols].values

        best: Dict[str, Dict[str, Any]] = {"long": {"H": "1h", "prec": 0.0, "rule": None},
                                           "short":{"H": "1h", "prec": 0.0, "rule": None}}

        for H in ["1h","2h","4h"]:
            for side in ["long","short"]:
                y = targets[H][side].values
                y_tr, y_te = y[:split], y[split:]
                X_tr, X_te = X[:split], X[split:]
                if y_tr.sum()==0 or y_te.sum()==0:
                    continue
                clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=42)
                clf.fit(X_tr, y_tr)
                p_te = clf.predict_proba(X_te)[:,1]
                thr = np.percentile(p_te, 75)  # high-conviction
                y_pred = (p_te >= thr).astype(int)
                prec = precision_score(y_te, y_pred, zero_division=0)
                rules = extract_rules_from_tree(clf, feat_cols)
                top_rule = rules[0] if rules else None
                if top_rule and prec > best[side]["prec"]:
                    best[side] = {"H": H, "prec": float(prec), "rule": top_rule}

        # Build initial rule texts
        def rule_to_text(rule):
            return " AND ".join(rule["conditions"]) if rule else "No strong rule (use top-quartile probability threshold)."

        long_rule_text  = rule_to_text(best["long"]["rule"])
        short_rule_text = rule_to_text(best["short"]["rule"])

        # Tighten if match rate > target
        def tighten(rule_text: str, side: str, H: str) -> Tuple[str, float, float, int]:
            mask_base = parse_rule_to_mask(sdf, rule_text)
            rate = 100.0 * mask_base.sum() / len(sdf) if len(sdf)>0 else 0.0
            y = targets[H][side]
            orig_prec = float((y[mask_base]==1).mean()) if mask_base.any() else np.nan
            if rate <= args.tighten_target_pct or not mask_base.any():
                return rule_text, rate, orig_prec, int(mask_base.sum())
            # search extra filter
            best_f = best_extra_filter(sdf, mask_base, y, feat_cols,
                                       target_rate=args.tighten_target_pct,
                                       min_support=args.tighten_min_support)
            if best_f:
                extra = f"{best_f['feat']} {best_f['op']} {best_f['thr']:.6f}"
                new_text = (rule_text + " AND " + extra) if not rule_text.startswith("No strong") else rule_text
                mask_new = parse_rule_to_mask(sdf, new_text)
                new_rate = 100.0 * mask_new.sum() / len(sdf) if len(sdf)>0 else 0.0
                new_prec = float((y[mask_new]==1).mean()) if mask_new.any() else np.nan
                return new_text, new_rate, new_prec, int(mask_new.sum())
            return rule_text, rate, orig_prec, int(mask_base.sum())

        long_text_tight, long_rate, long_prec_on_matches, long_support = tighten(long_rule_text, "long", best["long"]["H"])
        short_text_tight, short_rate, short_prec_on_matches, short_support = tighten(short_rule_text, "short", best["short"]["H"])

        # Subset-specific TP/SL from matched rows
        def subset_tp_sl(rule_text: str, H: str) -> Tuple[Optional[float], Optional[float], int]:
            mask = parse_rule_to_mask(sdf, rule_text)
            ret_col = f"ret_{H}"
            sub = pd.to_numeric(sdf.loc[mask, ret_col], errors="coerce").dropna().astype(float)
            tp, sl = realized_vol_suggestion(sub)
            return tp, sl, int(sub.shape[0])

        tpL, slL, nL = subset_tp_sl(long_text_tight, best["long"]["H"])
        tpS, slS, nS = subset_tp_sl(short_text_tight, best["short"]["H"])

        def enforce_rr(tp, sl):
            if tp is None or sl is None or not np.isfinite(tp) or not np.isfinite(sl) or sl <= 0:
                return (np.nan, np.nan, np.nan)
            rr = tp / sl
            if rr < 1.5:
                tp = round(1.5 * sl, 6)
                rr = tp / sl
            return (tp, sl, rr)

        tpL, slL, rrL = enforce_rr(tpL, slL)
        tpS, slS, rrS = enforce_rr(tpS, slS)

        pL = float(best["long"]["prec"])
        pS = float(best["short"]["prec"])
        evL = (pL * rrL - (1 - pL)) if (np.isfinite(rrL)) else np.nan
        evS = (pS * rrS - (1 - pS)) if (np.isfinite(rrS)) else np.nan

        # Store outputs
        setups_out.append({
            "symbol": sym,
            "LONG": {
                "horizon": best["long"]["H"],
                "entry_conditions": long_text_tight,
                "tp_pct": tpL, "sl_pct": slL,
                "expected_win_rate": round(pL, 3),
                "avg_rr": None if not np.isfinite(rrL) else round(float(rrL), 3),
                "EV_R_units": None if not np.isfinite(evL) else round(float(evL), 3)
            },
            "SHORT": {
                "horizon": best["short"]["H"],
                "entry_conditions": short_text_tight,
                "tp_pct": tpS, "sl_pct": slS,
                "expected_win_rate": round(pS, 3),
                "avg_rr": None if not np.isfinite(rrS) else round(float(rrS), 3),
                "EV_R_units": None if not np.isfinite(evS) else round(float(evS), 3)
            }
        })

        summary_rows.append({
            "symbol": sym,
            "LONG_horizon": best["long"]["H"],
            "LONG_expected_win": round(pL, 3),
            "LONG_rule": long_text_tight,
            "LONG_support_for_TP_SL": nL,
            "LONG_tp_pct": tpL, "LONG_sl_pct": slL,
            "LONG_avg_rr": None if not np.isfinite(rrL) else round(float(rrL), 3),
            "LONG_EV_R": None if not np.isfinite(evL) else round(float(evL), 3),
            "SHORT_horizon": best["short"]["H"],
            "SHORT_expected_win": round(pS, 3),
            "SHORT_rule": short_text_tight,
            "SHORT_support_for_TP_SL": nS,
            "SHORT_tp_pct": tpS, "SHORT_sl_pct": slS,
            "SHORT_avg_rr": None if not np.isfinite(rrS) else round(float(rrS), 3),
            "SHORT_EV_R": None if not np.isfinite(evS) else round(float(evS), 3),
        })

    # Save artifacts
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(setups_out, f, indent=2)
    print(f"[OK] Wrote setups JSON: {args.out_json} (symbols: {len(setups_out)})")

    if summary_rows:
        df_sum = pd.DataFrame(summary_rows).sort_values("symbol")
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        df_sum.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Wrote summary CSV: {args.out_csv} (rows: {len(df_sum)})")
    else:
        print("[WARN] No summary rows created – check symbol counts or data quality.")

if __name__ == "__main__":
    main()
