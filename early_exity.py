#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, re, os
import pandas as pd
import numpy as np

CSV_PATH  = r"C:\Users\natec\obflow_log.csv"
GATES_JSON = r"C:\Users\natec\price_excluded_setups_tightened_subsetTP.json"

COND_RE = re.compile(r"(.+?)\s*(<=|>)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

def parse_rule_to_mask(sdf: pd.DataFrame, rule_text: str) -> pd.Series:
    if not rule_text or str(rule_text).strip().lower().startswith("no strong"):
        return pd.Series([False]*len(sdf), index=sdf.index)
    conds = [c.strip() for c in str(rule_text).split("AND") if c.strip()]
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
        thr = float(val)
        col = pd.to_numeric(sdf[feat], errors="coerce")
        mask &= (col <= thr) if op == "<=" else (col > thr)
    return mask

def summarize_for_gate(sdf: pd.DataFrame, side: str, H: str):
    retH = f"ret_{H}"
    need = ["ret_15m","ret_30m", retH]
    d = sdf.dropna(subset=need).copy()
    if d.empty: return None
    if side.upper()=="LONG":
        win = (d[retH] > 0).astype(int)
        early15_fav = (d["ret_15m"] > 0)
        early30_fav = (d["ret_30m"] > 0)
    else:
        win = (d[retH] < 0).astype(int)
        early15_fav = (d["ret_15m"] < 0)
        early30_fav = (d["ret_30m"] < 0)
    base = float(win.mean()); n = len(d)
    winners = d[win==1]; losers = d[win==0]

    p15_win = float((early15_fav[win==1]).mean()) if len(winners)>0 else np.nan
    p30_win = float((early30_fav[win==1]).mean()) if len(winners)>0 else np.nan
    p15_los = float((early15_fav[win==0]).mean()) if len(losers)>0 else np.nan
    p30_los = float((early30_fav[win==0]).mean()) if len(losers)>0 else np.nan

    keep15 = early15_fav
    retain15 = float(keep15.mean())
    win_keep15 = float(win[keep15].mean()) if keep15.any() else np.nan

    keep30 = early30_fav
    retain30 = float(keep30.mean())
    win_keep30 = float(win[keep30].mean()) if keep30.any() else np.nan

    adverse15 = ~early15_fav
    cut_hits_losers = float((win[adverse15]==0).mean()) if adverse15.any() else np.nan

    return {
        "n_alert_rows": int(n),
        "base_winrate": round(base,3),
        "P(15m favorable | winner)": None if np.isnan(p15_win) else round(p15_win,3),
        "P(30m favorable | winner)": None if np.isnan(p30_win) else round(p30_win,3),
        "P(15m favorable | loser) ": None if np.isnan(p15_los) else round(p15_los,3),
        "P(30m favorable | loser) ": None if np.isnan(p30_los) else round(p30_los,3),
        "Require 15m fav: retain": None if np.isnan(retain15) else round(retain15,3),
        "Require 15m fav: winrate": None if np.isnan(win_keep15) else round(win_keep15,3),
        "Require 30m fav: retain": None if np.isnan(retain30) else round(retain30,3),
        "Require 30m fav: winrate": None if np.isnan(win_keep30) else round(win_keep30,3),
        "Cut@15m adverse catches losers": None if np.isnan(cut_hits_losers) else round(cut_hits_losers,3),
    }

def main():
    assert os.path.exists(CSV_PATH), f"Missing {CSV_PATH}"
    assert os.path.exists(GATES_JSON), f"Missing {GATES_JSON}"

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    for c in df.columns:
        if c not in ["ID","timestamp_utc","symbol"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    with open(GATES_JSON, "r") as f:
        setups = json.load(f)

    rows = []
    for rec in setups:
        sym = rec["symbol"]
        sym_df = df[df["symbol"]==sym].copy()
        for side in ["LONG","SHORT"]:
            rule = rec[side].get("entry_conditions","")
            H = rec[side].get("horizon","1h")
            m = parse_rule_to_mask(sym_df, rule)
            sdf = sym_df.loc[m].copy()
            if sdf.empty: 
                continue
            out = summarize_for_gate(sdf, side, H)
            if out:
                rows.append({"symbol": sym, "side": side, "horizon": H, **out})

    gate_df = pd.DataFrame(rows).sort_values(["symbol","side"])
    print("\n=== ON GATES — Early move vs outcome (per symbol/side) ===")
    print(gate_df.to_string(index=False))

    if not gate_df.empty:
        total = gate_df["n_alert_rows"].sum()
        def wavg(col):
            s = (gate_df[col] * gate_df["n_alert_rows"]).sum(skipna=True)
            return s/total if total>0 else np.nan
        global_sum = {
            "n_alert_rows": int(total),
            "base_winrate": round(wavg("base_winrate"),3),
            "Require 15m fav: retain": round(wavg("Require 15m fav: retain"),3),
            "Require 15m fav: winrate": round(wavg("Require 15m fav: winrate"),3),
            "Require 30m fav: retain": round(wavg("Require 30m fav: retain"),3),
            "Require 30m fav: winrate": round(wavg("Require 30m fav: winrate"),3),
            "Cut@15m adverse catches losers": round(wavg("Cut@15m adverse catches losers"),3),
            "P(15m favorable | winner)": round(wavg("P(15m favorable | winner)"),3),
            "P(30m favorable | winner)": round(wavg("P(30m favorable | winner)"),3),
            "P(15m favorable | loser) ": round(wavg("P(15m favorable | loser) "),3),
            "P(30m favorable | loser) ": round(wavg("P(30m favorable | loser) "),3),
        }
        print("\n=== GLOBAL (weighted by alert rows) — ON GATES ===")
        print(pd.DataFrame([global_sum]).to_string(index=False))

if __name__ == "__main__":
    main()
