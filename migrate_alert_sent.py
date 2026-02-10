#!/usr/bin/env python3
import argparse, os, sys
import pandas as pd

def migrate_file(path: str) -> None:
    if not os.path.exists(path) or os.path.getsize(path)==0:
        print(f"[migrate] file missing/empty: {path}", file=sys.stderr); return
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", engine="python", on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(path, encoding="utf-8-sig", engine="python", error_bad_lines=False)
    if "alert_sent" not in df.columns:
        df["alert_sent"] = 0
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)
    print(f"[migrate] OK → {path}  rows:{len(df)}")

def main():
    ap = argparse.ArgumentParser(description="Add 'alert_sent' column (default 0) to CSV if missing.")
    ap.add_argument("files", nargs="+", help="One or more CSV paths")
    args = ap.parse_args()
    for f in args.files:
        migrate_file(f)

if __name__ == "__main__":
    main()
