#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
import pandas as pd

def compute_R(row) -> float:
    """
    R = (TP1 - entry)/|entry - SL| for LONG
      = (entry - TP1)/|entry - SL| for SHORT
    Uses TP1 only. Returns np.nan if anything needed is missing.
    """
    try:
        side = str(row.get("side","")).upper()
        out  = str(row.get("outcome","")).upper()
        entry = float(row.get("entry", np.nan))
        sl    = float(row.get("sl", np.nan))
        tp1   = row.get("tp1", np.nan)

        # If no explicit outcome but hit flags exist, infer outcome
        if (not out or out == "NAN") and ("hit_tp" in row and "hit_sl" in row):
            tp_hit = int(row.get("hit_tp", 0)) == 1
            sl_hit = int(row.get("hit_sl", 0)) == 1
            if tp_hit and not sl_hit: out = "TP"
            elif sl_hit and not tp_hit: out = "SL"

        if side not in {"LONG","SHORT"}: return np.nan
        if not np.isfinite(entry) or not np.isfinite(sl): return np.nan
        risk = abs(entry - sl)
        if risk <= 0 or not np.isfinite(risk): return np.nan

        if out.startswith("TP"):
            tp1 = float(tp1) if np.isfinite(tp1) else np.nan
            if not np.isfinite(tp1): return np.nan
            return (tp1 - entry)/risk if side=="LONG" else (entry - tp1)/risk
        elif out.startswith("SL"):
            return -1.0
    except Exception:
        pass
    return np.nan

def build_alerts_mask(df: pd.DataFrame, alerts_only: bool) -> pd.Series:
    """Keep only rows that actually sent a Telegram alert."""
    mask = pd.Series(True, index=df.index)
    if not alerts_only:
        return mask
    # Preferred: explicit flag we added to the CSV
    if "alert_sent" in df.columns:
        return mask & (pd.to_numeric(df["alert_sent"], errors="coerce") == 1)
    # Fallbacks (if you inspect older logs)
    if "suppressed" in df.columns:   # 0/1 style
        return mask & (pd.to_numeric(df["suppressed"], errors="coerce") != 1)
    if "stop_type" in df.columns:    # keep anything not explicitly LOGONLY
        return mask & (df["stop_type"].astype(str).str.upper() != "LOGONLY")
    # No alert marker available → show all (same as before)
    return mask

def main():
    ap = argparse.ArgumentParser(description="Print the last N CLOSED trades (TP/SL) with R; optional CSV export.")
    ap.add_argument("--log", required=True, help="Path to your trade log CSV (e.g., live_trade_log_fib6.csv)")
    ap.add_argument("-n", "--num", type=int, default=20, help="How many most recent CLOSED trades to show (default: 20)")
    ap.add_argument("--alerts-only", action="store_true", help="Only include trades that actually sent a Telegram alert")
    ap.add_argument("--out", default="", help="Optional CSV to save (e.g., last_closed.csv)")
    args = ap.parse_args()

    if not os.path.exists(args.log) or os.path.getsize(args.log) == 0:
        print(f"[last] log not found or empty: {args.log}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.log, encoding="utf-8-sig")
    if "timestamp_utc" not in df.columns:
        print("[last] missing 'timestamp_utc' in log; found:", ", ".join(df.columns), file=sys.stderr)
        sys.exit(1)

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).copy()

    # Filter to actual alerts if requested
    df = df[build_alerts_mask(df, alerts_only=args.alerts_only)].copy()

    # Compute R and keep only closed rows
    df["R"] = df.apply(compute_R, axis=1)
    closed = df[df["R"].notna()].sort_values("timestamp_utc", ascending=False).head(args.num).copy()

    if closed.empty:
        msg = "no CLOSED trades with enough data to compute R"
        if args.alerts_only: msg += " (after alerts-only filter)"
        print(f"[last] {msg}.")
        sys.exit(0)

    cols_pref = ["timestamp_utc","symbol","side","outcome","R","entry","tp1","sl","zone","mode","alert_sent"]
    cols = [c for c in cols_pref if c in closed.columns]
    view = closed[cols].copy()
    view["timestamp_utc"] = pd.to_datetime(view["timestamp_utc"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    if "R" in view.columns: view["R"] = view["R"].map(lambda x: f"{x:.2f}")

    print(view.to_string(index=False))

    R_vals = pd.to_numeric(closed["R"], errors="coerce")
    wins   = int((R_vals > 0).sum())
    losses = int((R_vals < 0).sum())
    totalR = float(np.nansum(R_vals))
    avgR   = float(np.nanmean(R_vals))
    print(f"\nLast {len(closed)} CLOSED — wins: {wins}, losses: {losses}, total R: {totalR:.2f}, avg R: {avgR:.2f}")

    if args.out:
        view.to_csv(args.out, index=False, encoding="utf-8-sig")
        print("[last] saved:", args.out)

if __name__ == "__main__":
    main()
