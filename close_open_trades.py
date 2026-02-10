#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
import pandas as pd

def infer_outcome(row):
    # Prefer explicit hit flags if present
    tp_flag = int(row.get("hit_tp", 0) or 0) == 1 if "hit_tp" in row else None
    sl_flag = int(row.get("hit_sl", 0) or 0) == 1 if "hit_sl" in row else None
    outcome = str(row.get("outcome","")).upper()

    if outcome in ("TP","SL"):  # already closed
        return outcome

    # MAE/MFE in R (if present)
    mfe = row.get("mfe_r_12h", np.nan)
    mae = row.get("mae_r_12h", np.nan)
    try:
        mfe = float(mfe)
        mae = abs(float(mae))
    except Exception:
        mfe = np.nan; mae = np.nan

    # If flags exist, prefer them
    if tp_flag or sl_flag is not None:
        if tp_flag and not sl_flag: return "TP"
        if sl_flag and not tp_flag: return "SL"

    # Otherwise decide from 12h excursions if available
    if np.isfinite(mfe) or np.isfinite(mae):
        if mfe >= 1.0: return "TP"
        if mae >= 1.0: return "SL"

    # nothing definitive
    return ""

def main():
    ap = argparse.ArgumentParser(description="Finalize open trades in a log CSV using MAE/MFE (or hit flags).")
    ap.add_argument("--in",  required=True, help="Input CSV (e.g., live_trade_log_fib6_maemfe.csv or live_trade_log_fib6.csv)")
    ap.add_argument("--out", default="",   help="Output CSV (default: overwrite input)")
    args = ap.parse_args()

    path_in  = args.in
    path_out = args.out or path_in

    if not os.path.exists(path_in) or os.path.getsize(path_in)==0:
        print(f"[close] file not found or empty: {path_in}", file=sys.stderr); sys.exit(1)

    df = pd.read_csv(path_in, encoding="utf-8-sig")
    if "timestamp_utc" not in df.columns:
        print("[close] missing 'timestamp_utc' column; aborting.", file=sys.stderr)
        print("        Columns:", ", ".join(df.columns))
        sys.exit(1)

    # Normalize time
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    # Compute outcome for rows without explicit outcome
    open_mask = ~df["outcome"].astype(str).str.upper().isin(["TP","SL"]) if "outcome" in df.columns else pd.Series(True, index=df.index)
    if open_mask.sum() == 0:
        print("[close] nothing to close."); sys.exit(0)

    # Ensure hit_tp/hit_sl exist
    for col in ("hit_tp","hit_sl"):
        if col not in df.columns:
            df[col] = 0

    updated = 0
    for i in df[open_mask].index:
        outcome = infer_outcome(df.loc[i])
        if outcome:
            df.at[i, "outcome"] = outcome
            df.at[i, "hit_tp"]  = 1 if outcome=="TP" else df.at[i,"hit_tp"]
            df.at[i, "hit_sl"]  = 1 if outcome=="SL" else df.at[i,"hit_sl"]
            updated += 1

    df.to_csv(path_out, index=False, encoding="utf-8-sig")
    print(f"[close] updated {updated} rows → {path_out}")

if __name__ == "__main__":
    main()
