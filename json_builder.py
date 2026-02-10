#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, re, warnings
from typing import Dict, Any, List, Tuple, Optional
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

# ------------------------- Paths -------------------------
CSV_PATH      = r"C:\Users\natec\obflow_log.csv"
OUT_JSON_PATH = r"C:\Users\natec\price_excluded_setups_tightened_subsetTP_new.json"
OUT_CSV_PATH  = r"C:\Users\natec\setups_summary.csv"

# ------------------------- Config ------------------------
SEED = 42
TEST_FRAC = 0.30

# GLOBAL (loose)
MAX_DEPTH_GLOBAL = 2
MIN_SAMPLES_LEAF_GLOBAL = 50
LEAF_MIN_SUPPORT_GLOBAL = 20
LEAF_MIN_SUPPORT_RELAXED_GLOBAL = 10
STRICT_RATE_CAP_GLOBAL = 12.0
RELAXED_RATE_CEILINGS_GLOBAL = [18.0, 25.0]
MIN_RATE = 1.0

# COIN (strict, precision-first)
MAX_COIN_ATOMS = 4
LEAF_MIN_SUPPORT_COIN = 10
STRICT_RATE_CAP_COIN = 4.0
RELAXED_RATE_COIN = [6.0, 8.0]
MIN_PRECISION_GAIN = 0.02  # +2pp per added atom

# Quantile snapping
SNAP_Q_GRID = [i/100 for i in range(5, 100, 5)]

# Recency cap
PER_SYMBOL_MAX_ROWS = 2500

# TP/SL logic
TP_POS_PCTL = 65
SL_NEG_PCTL = 35
MIN_RR = 1.5

RET_COLS = ["ret_15m","ret_30m","ret_1h","ret_2h","ret_4h"]
COND_RE = re.compile(r"(.+?)\s*(<=|>)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

np.random.seed(SEED)

# ------------------------- Helpers -----------------------
def parse_rule_to_mask(sdf: pd.DataFrame, rule_text: str) -> pd.Series:
    if not rule_text or str(rule_text).strip().lower().startswith("no strong"):
        return pd.Series([False]*len(sdf), index=sdf.index)
    conds = [c.strip() for c in str(rule_text).split("AND") if c.strip()]
    mask = pd.Series([True]*len(sdf), index=sdf.index)
    for cond in conds:
        m = COND_RE.match(cond)
        if not m: 
            mask &= False
            continue
        feat, op, val = m.groups()
        feat = feat.strip()
        if feat not in sdf.columns: 
            mask &= False; continue
        thr = float(val)
        col = pd.to_numeric(sdf[feat], errors="coerce")
        mask &= (col <= thr) if op == "<=" else (col > thr)
    return mask

def extract_leaf_rules(clf, feature_names: List[str], min_support: int) -> List[str]:
    tree_ = clf.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
    rules: List[str] = []
    def recurse(node: int, conds: List[str]):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]; thr = tree_.threshold[node]
            recurse(tree_.children_left[node],  conds + [f"{name} <= {thr:.6f}"])
            recurse(tree_.children_right[node], conds + [f"{name} > {thr:.6f}"])
        else:
            if int(tree_.n_node_samples[node]) >= min_support:
                rules.append(" AND ".join(conds))
    recurse(0, [])
    return rules


# --- Add this helper once near the top (after imports) ---
def _dedupe_and_impute(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Drop duplicate-named columns, then median-impute only existing cols using dict alignment."""
    # 1) de-duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # 2) only impute columns that actually exist
    present = [c for c in cols if c in df.columns]
    if present:
        med = df[present].median(numeric_only=True).to_dict()
        # dict-based fillna aligns by column name; avoids length/key mismatches
        df[present] = df[present].fillna(value=med)
    return df



def snap_to_quantiles(rule_text: str, train_df: pd.DataFrame) -> str:
    if not rule_text: return rule_text
    out = []
    for cond in rule_text.split("AND"):
        cond = cond.strip()
        m = COND_RE.match(cond)
        if not m: continue
        feat, op, val = m.groups()
        if feat not in train_df.columns:
            out.append(cond); continue
        col = pd.to_numeric(train_df[feat], errors="coerce").dropna().astype(float)
        if col.empty:
            out.append(cond); continue
        qs = np.nanquantile(col, SNAP_Q_GRID)
        thr = float(val)
        if len(qs):
            thr = float(qs[np.abs(qs - thr).argmin()])
        out.append(f"{feat} {op} {thr:.6f}")
    return " AND ".join(out) if out else rule_text

def realized_tp_sl(ret_series: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    ser = pd.to_numeric(ret_series, errors="coerce").dropna().astype(float)
    if ser.empty: return (np.nan, np.nan)
    pos = ser[ser > 0.0]; neg = ser[ser < 0.0]
    tp = float(np.nanpercentile(pos, TP_POS_PCTL)) if not pos.empty else np.nan
    sl = float(np.nanpercentile(np.abs(neg), SL_NEG_PCTL)) if not neg.empty else np.nan
    return (tp, sl)

def precision_rate_support(test_df: pd.DataFrame, y_test: pd.Series, rule_text: str) -> Tuple[float,float,int]:
    m = parse_rule_to_mask(test_df, rule_text)
    sup = int(m.sum())
    rate = 100.0 * sup / len(test_df) if len(test_df)>0 else 0.0
    prec = float((y_test[m] == 1).mean()) if sup>0 else 0.0
    return prec, rate, sup

# ------------------------- Main ------------------------
def main():
    if not os.path.exists(CSV_PATH): raise SystemExit(f"CSV not found: {CSV_PATH}")
    df0 = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    if "timestamp_utc" in df0.columns:
        df0["timestamp_utc"] = pd.to_datetime(df0["timestamp_utc"], utc=True, errors="coerce")
    for c in df0.columns:
        if c not in ["ID","timestamp_utc","symbol"]:
            df0[c] = pd.to_numeric(df0[c], errors="coerce")

    df0 = df0[df0[RET_COLS].notna().any(axis=1)].copy()

    # Recency cap per symbol
    frames = []
    for sym, sdf in df0.groupby("symbol"):
        frames.append(sdf.sort_values("timestamp_utc").tail(PER_SYMBOL_MAX_ROWS))
    df = pd.concat(frames, axis=0).sort_values("timestamp_utc").reset_index(drop=True)

    # Feature groups
    exclude = set(["ID","timestamp_utc","symbol","price"] + RET_COLS)
    all_feats = [c for c in df.columns if c not in exclude]

    global_feats = sorted(list(set(
        [f for f in all_feats if f.endswith("_cum")] +
        [f for f in all_feats if f.startswith("near_")] +
        [f for f in all_feats if f.startswith("in_fib_")] +
        [f for f in all_feats if f.startswith("stoch_")]
    )))

    # coin features: *_ratio + non-cum lookbacks
    ratio_feats = sorted([f for f in all_feats if f.endswith("_ratio")])
    nonratio_feats = sorted([f for f in all_feats
                             if f.startswith(("lookback_5m","lookback_15m","lookback_30min","lookback_1hr","lookback_4hr"))
                             and not f.endswith("_cum")])
    coin_feats = ratio_feats + nonratio_feats

    # after you set global_feats / coin_feats:
    global_feats = sorted(set(global_feats))
    coin_feats   = sorted(set(coin_feats))


    split = int(len(df) * (1.0 - TEST_FRAC))
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()

    def mk_labels(d):
        return {
            "1h": {"long": (d["ret_1h"] > 0).astype(int), "short": (d["ret_1h"] < 0).astype(int)},
            "2h": {"long": (d["ret_2h"] > 0).astype(int), "short": (d["ret_2h"] < 0).astype(int)},
            "4h": {"long": (d["ret_4h"] > 0).astype(int), "short": (d["ret_4h"] < 0).astype(int)},
        }
    labels = mk_labels(df)

    # ---------- Stage A: GLOBAL (loose) ----------
    GLOBAL = {"LONG": {"horizon": "1h", "rule": "", "precision": 0.0},
              "SHORT":{"horizon": "1h", "rule": "", "precision": 0.0}}

    for side in ["long","short"]:
        best = {"H":"1h","txt":"","prec":0.0}
        for H in ["1h","2h","4h"]:
            y_tr = labels[H][side].iloc[:split].values
            y_te = labels[H][side].iloc[split:].values
            if y_tr.sum()==0 or y_te.sum()==0: continue

            clf = DecisionTreeClassifier(max_depth=MAX_DEPTH_GLOBAL,
                                         min_samples_leaf=MIN_SAMPLES_LEAF_GLOBAL,
                                         random_state=SEED)
            clf.fit(train_df[global_feats].values, y_tr)

            leaves = extract_leaf_rules(clf, global_feats, LEAF_MIN_SUPPORT_GLOBAL)
            if not leaves:
                leaves = extract_leaf_rules(clf, global_feats, LEAF_MIN_SUPPORT_RELAXED_GLOBAL)
            snapped = [snap_to_quantiles(t, train_df) for t in leaves]

            cand = []
            for txt in snapped:
                prec, rate, sup = precision_rate_support(test_df, y_te, txt)
                if sup < LEAF_MIN_SUPPORT_RELAXED_GLOBAL: continue
                cand.append({"txt": txt, "prec": prec, "rate": rate, "sup": sup})

            chosen = None
            pool = [c for c in cand if c["rate"] <= STRICT_RATE_CAP_GLOBAL]
            if pool:
                pool.sort(key=lambda d: (d["prec"], d["sup"]), reverse=True)
                chosen = pool[0]
            else:
                for cap in RELAXED_RATE_CEILINGS_GLOBAL:
                    pool = [c for c in cand if c["rate"] <= cap]
                    if not pool: continue
                    pool.sort(key=lambda d: (d["prec"], d["sup"]), reverse=True)
                    chosen = pool[0]; break
            if chosen and chosen["prec"] > best["prec"]:
                best = {"H": H, "txt": chosen["txt"], "prec": chosen["prec"]}

        GLOBAL[side.upper()] = {"horizon": best["H"], "rule": best["txt"], "precision": best["prec"]}

    # ---------- Stage B: PER-COIN (strict, multi-atom; ratio handled as 3 families) ----------
    out_symbols, summaries = [], []

    for sym, sdf in df.groupby("symbol"):
        sdf = sdf.loc[:, ~sdf.columns.duplicated()].copy()
        # impute coin feats
        
        present_cols = [c for c in coin_feats if c in sdf.columns]
        if present_cols:
            med = sdf[present_cols].median(numeric_only=True).to_dict()
            sdf.loc[:, present_cols] = sdf.loc[:, present_cols].fillna(value=med)

        rec = {"symbol": sym}
        summ = {"symbol": sym}

        for side_key in ["LONG","SHORT"]:
            H = GLOBAL[side_key]["horizon"]
            gtxt = GLOBAL[side_key]["rule"]
            if not gtxt:
                rec[side_key] = {"horizon": H, "entry_conditions": "No strong rule (use top-quartile probability threshold).",
                                 "tp_pct": np.nan, "sl_pct": np.nan,
                                 "expected_win_rate": np.nan, "avg_rr": np.nan, "EV_R_units": np.nan}
                summ[f"{side_key}_horizon"] = H
                summ[f"{side_key}_rule"] = "No strong rule (use top-quartile probability threshold)."
                continue

            m_g = parse_rule_to_mask(sdf, gtxt)
            S = sdf[m_g].copy()
            if len(S) < LEAF_MIN_SUPPORT_COIN:
                final_txt = gtxt
                split_c = int(len(sdf) * (1.0 - TEST_FRAC))
                y_coin_test = ((sdf[f"ret_{H}"].iloc[split_c:] > 0) if side_key=="LONG"
                               else (sdf[f"ret_{H}"].iloc[split_c:] < 0)).astype(int)
                p_final = float((y_coin_test[parse_rule_to_mask(sdf.iloc[split_c:], gtxt)] == 1).mean()) if split_c < len(sdf) else np.nan
            else:
                split_s = int(len(S) * (1.0 - TEST_FRAC))
                S_train, S_test = S.iloc[:split_s], S.iloc[split_s:]
                y_test = ((S_test[f"ret_{H}"] > 0) if side_key=="LONG"
                          else (S_test[f"ret_{H}"] < 0)).astype(int)

                current = gtxt
                base_mask = parse_rule_to_mask(S_test, current)
                p_current = float((y_test[base_mask] == 1).mean()) if base_mask.any() else 0.0
                added = 0

                def try_candidates(feature_list, cap) -> bool:
                    nonlocal current, base_mask, p_current, added
                    best = None
                    for feat in feature_list:
                        if feat not in S_test.columns: continue
                        x = pd.to_numeric(S_test[feat], errors="coerce")
                        if x.notna().sum() < LEAF_MIN_SUPPORT_COIN: continue

                        # --- Ratio families ---
                        if feat.endswith("_ratio"):
                            # alignment: ratio > +thr_pos, divergence: ratio < -thr_neg, magnitude: abs(ratio) > thr_abs
                            pos_thr = float(np.nanquantile(x, 0.70))  # stronger than market
                            neg_thr = float(np.nanquantile(x, 0.30))  # weaker than market
                            abs_thr = float(np.nanquantile(np.abs(x), 0.70))

                            # alignment (>)
                            for op, thr in ((">", pos_thr), ("<=", neg_thr)):
                                m = base_mask & ((x <= thr) if op=="<=" else (x > thr))
                                sup = int(m.sum()); 
                                if sup < LEAF_MIN_SUPPORT_COIN: continue
                                rate = 100.0 * sup / len(S_test)
                                if rate > cap or rate < MIN_RATE: continue
                                p_new = float((y_test[m] == 1).mean()) if sup>0 else 0.0
                                gain = p_new - p_current
                                if gain >= MIN_PRECISION_GAIN:
                                    # small bonus for magnitude/ratio structures can be added if desired
                                    score = (p_new, sup)
                                    if (best is None) or (score > best["score"]):
                                        best = {"feat": feat, "op": op, "thr": thr, "p_new": p_new, "sup": sup, "rate": rate, "score": score}
                            # magnitude (abs > thr_abs): implement as (x > thr_abs) OR (x < -thr_abs)
                            # We approximate with two checks and pick better
                            for op, thr in ((">", abs_thr), ("<=", -abs_thr)):
                                m = base_mask & ((x <= thr) if op=="<=" else (x > thr))
                                sup = int(m.sum()); 
                                if sup < LEAF_MIN_SUPPORT_COIN: continue
                                rate = 100.0 * sup / len(S_test)
                                if rate > cap or rate < MIN_RATE: continue
                                p_new = float((y_test[m] == 1).mean()) if sup>0 else 0.0
                                gain = p_new - p_current
                                if gain >= MIN_PRECISION_GAIN:
                                    score = (p_new, sup)
                                    if (best is None) or (score > best["score"]):
                                        best = {"feat": feat, "op": op, "thr": thr, "p_new": p_new, "sup": sup, "rate": rate, "score": score}

                        else:
                            # --- Non-ratio families (standard) ---
                            for q in (0.25, 0.5, 0.75):
                                thr = float(np.nanquantile(x, q))
                                for op in ("<=", ">"):
                                    m = base_mask & ((x <= thr) if op=="<=" else (x > thr))
                                    sup = int(m.sum()); 
                                    if sup < LEAF_MIN_SUPPORT_COIN: continue
                                    rate = 100.0 * sup / len(S_test)
                                    if rate > cap or rate < MIN_RATE: continue
                                    p_new = float((y_test[m] == 1).mean()) if sup>0 else 0.0
                                    gain = p_new - p_current
                                    if gain >= MIN_PRECISION_GAIN:
                                        score = (p_new, sup)
                                        if (best is None) or (score > best["score"]):
                                            best = {"feat": feat, "op": op, "thr": thr, "p_new": p_new, "sup": sup, "rate": rate, "score": score}

                    if best:
                        atom = f"{best['feat']} {best['op']} {best['thr']:.6f}"
                        current = f"{current} AND {atom}"
                        base_mask = parse_rule_to_mask(S_test, current)
                        p_current = best["p_new"]
                        added += 1
                        return True
                    return False

                # Pass 1: ratio-first strict
                while added < MAX_COIN_ATOMS:
                    if not try_candidates(ratio_feats, STRICT_RATE_CAP_COIN):
                        # relaxed → then tighten-back via another iteration at strict
                        before = current
                        for cap in RELAXED_RATE_COIN:
                            if try_candidates(ratio_feats, cap):
                                break
                        if current == before:
                            break
                    if added >= MAX_COIN_ATOMS: break

                # Pass 2: non-ratio
                while added < MAX_COIN_ATOMS:
                    if not try_candidates(nonratio_feats, STRICT_RATE_CAP_COIN):
                        before = current
                        for cap in RELAXED_RATE_COIN:
                            if try_candidates(nonratio_feats, cap):
                                break
                        if current == before:
                            break
                    if added >= MAX_COIN_ATOMS: break

                final_txt = current

                # pooled coin-level test precision
                split_c = int(len(sdf) * (1.0 - TEST_FRAC))
                m_coin = parse_rule_to_mask(sdf.iloc[split_c:], final_txt)
                y_coin_test = ((sdf[f"ret_{H}"].iloc[split_c:] > 0) if side_key=="LONG"
                               else (sdf[f"ret_{H}"].iloc[split_c:] < 0)).astype(int)
                p_final = float((y_coin_test[m_coin] == 1).mean()) if m_coin.any() else p_current

            # TP/SL on final matched subset in sdf
            m_all = parse_rule_to_mask(sdf, final_txt)
            sub = pd.to_numeric(sdf.loc[m_all, f"ret_{H}"], errors="coerce").dropna().astype(float)
            tp, sl = realized_tp_sl(sub)
            if np.isfinite(tp) and np.isfinite(sl) and sl>0:
                rr = tp/sl
                if rr < MIN_RR: tp = round(MIN_RR * sl, 6); rr = tp/sl
                ev = p_final*rr - (1 - p_final)
            else:
                rr = np.nan; ev = np.nan

            rec[side_key] = {
                "horizon": H,
                "entry_conditions": final_txt,
                "tp_pct": tp, "sl_pct": sl,
                "expected_win_rate": None if np.isnan(p_final) else round(float(p_final), 3),
                "avg_rr": None if not np.isfinite(rr) else round(float(rr), 3),
                "EV_R_units": None if not np.isfinite(ev) else round(float(ev), 3)
            }
            summ[f"{side_key}_horizon"] = H
            summ[f"{side_key}_rule"] = final_txt

        out_symbols.append(rec)
        summaries.append(summ)

    # Compose final JSON with a clear _GLOBAL header AND a symbols array
    payload = {
        "_GLOBAL": {
            "LONG":  {"horizon": GLOBAL["LONG"]["horizon"],  "rule": GLOBAL["LONG"]["rule"]},
            "SHORT": {"horizon": GLOBAL["SHORT"]["horizon"], "rule": GLOBAL["SHORT"]["rule"]}
        },
        "symbols": out_symbols
    }

    # Save
    os.makedirs(os.path.dirname(OUT_JSON_PATH), exist_ok=True)
    with open(OUT_JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Wrote setups JSON: {OUT_JSON_PATH} (symbols: {len(out_symbols)})")

    if summaries:
        df_sum = pd.DataFrame(summaries).sort_values("symbol")
        os.makedirs(os.path.dirname(OUT_CSV_PATH), exist_ok=True)
        df_sum.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"[OK] Wrote summary CSV: {OUT_CSV_PATH} (rows: {len(df_sum)})")

if __name__ == "__main__":
    main()
