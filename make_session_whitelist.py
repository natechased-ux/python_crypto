# make_session_whitelist.py
"""
Generate a (symbol, session) whitelist from session_cycle_results_by_symbol_session.csv.

Example:
py make_session_whitelist.py --results session_cycle_results_by_symbol_session.csv --min_cum_ret 0.0 --min_trades 50 --out whitelist.csv
"""
import argparse
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="Build whitelist of (symbol, session) pairs with positive edge")
    p.add_argument("--results", required=True, help="Path to session_cycle_results_by_symbol_session.csv")
    p.add_argument("--min_cum_ret", type=float, default=0.0, help="Minimum cumulative return threshold to include (e.g., 0.0 for >0%)")
    p.add_argument("--min_trades", type=int, default=1, help="Minimum number of trades in that bucket")
    p.add_argument("--out", default="whitelist.csv", help="Output CSV (columns: symbol,session)")
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.results)
    need_cols = {"symbol","session","trades","cum_ret_pct"}
    missing = need_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in results CSV: {missing}")
    filt = df[(df["trades"] >= args.min_trades) & (df["cum_ret_pct"] >= args.min_cum_ret)]
    wl = filt[["symbol","session"]].drop_duplicates().sort_values(["symbol","session"])
    wl.to_csv(args.out, index=False)
    print(f"Saved whitelist with {len(wl)} entries to {args.out}")

if __name__ == "__main__":
    main()
