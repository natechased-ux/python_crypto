#!/usr/bin/env python3
import json, os, csv, math
from datetime import datetime, timezone, timedelta

PARAMS = "mr_params.json"
LOG = "mr_live_alerts.csv"

def load_params():
    if not os.path.exists(PARAMS):
        print("[!] No mr_params.json found. Run training first.")
        return {}
    with open(PARAMS, "r", encoding="utf-8") as f:
        return json.load(f)

def load_alerts():
    if not os.path.exists(LOG):
        print("[!] No mr_live_alerts.csv yet. Run live mode to generate alerts.")
        return []
    out = []
    with open(LOG, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # utc_time,symbol,side,entry,tp,sl,z,ma,atr,ma_window,bb_mult,z_entry,hold_max,atr_tp,atr_sl,mean_exit,granularity
            try:
                row["z"] = float(row["z"])
                row["entry"] = float(row["entry"])
                row["tp"] = float(row["tp"])
                row["sl"] = float(row["sl"])
                row["atr"] = float(row["atr"])
                row["ma"] = float(row["ma"])
                row["ma_window"] = int(row["ma_window"])
                row["z_entry"] = float(row["z_entry"])
                row["granularity"] = int(row["granularity"])
                row["utc_time"] = datetime.fromisoformat(row["utc_time"].replace("Z","+00:00"))
                out.append(row)
            except Exception:
                continue
    return out

def norm_sharpe(s):
    # Map Sharpe-ish to 0..1 smoothly
    return max(0.0, min(1.0, (math.tanh(float(s)/2.0) + 1.0)/2.0))

def main(top=15, recent_hours=48):
    store = load_params()
    alerts = load_alerts()

    # ---- 1) Top coins by training metrics
    rows = []
    for sym, conf in store.items():
        m = conf.get("metrics", {})
        rows.append((
            sym,
            float(m.get("sharpe", 0.0)),
            int(m.get("trades", 0)),
            float(m.get("winrate", 0.0)),
            float(m.get("avg_r", 0.0))
        ))
    rows.sort(key=lambda x: (x[1], x[3], x[2]), reverse=True)

    print("\n=== TOP COINS BY TRAINING METRICS ===")
    print(f"{'SYMBOL':7s} {'SHARPE':>7s} {'TRADES':>7s} {'WINRATE':>9s} {'AVG_R':>7s}")
    for sym, sh, tr, wr, ar in rows[:top]:
        print(f"{sym:7s} {sh:7.2f} {tr:7d} {wr:8.2%} {ar:7.3f}")

    # ---- 2) Top recent alerts by combined score (Sharpe + |z| vs threshold)
    if alerts:
        # Only recent
        cutoff = datetime.now(timezone.utc) - timedelta(hours=recent_hours)
        recent = [a for a in alerts if a["utc_time"] >= cutoff]

        # Build lookup for per-coin trained Sharpe + z_entry (in case CSV lacks)
        trained = {}
        for sym, conf in store.items():
            p = conf.get("params", {})
            m = conf.get("metrics", {})
            trained[sym] = {
                "sharpe": float(m.get("sharpe", 0.0)),
                "trades": int(m.get("trades", 0)),
                "winrate": float(m.get("winrate", 0.0)),
                "z_entry": float(p.get("z_entry", 1.5)),
            }

        ranked = []
        for a in recent:
            sym = a["symbol"]
            t = trained.get(sym, {"sharpe": 0.0, "z_entry": max(1.0, a.get("z_entry", 1.5))})
            sharpe = t["sharpe"]
            z_thr = t["z_entry"] if t["z_entry"] else max(1.0, a.get("z_entry", 1.5))
            z_strength = abs(a["z"]) / max(1e-9, z_thr)
            # Combined score (0..10): heavier weight on trained Sharpe, some on z_strength
            score = 10.0 * (0.7 * norm_sharpe(sharpe) + 0.3 * min(1.0, z_strength / 2.0))
            ranked.append((score, a, sharpe, z_strength))

        ranked.sort(key=lambda x: x[0], reverse=True)

        print(f"\n=== TOP RECENT ALERTS (last {recent_hours}h) ===")
        print(f"{'SCORE':>6s} {'TIME(UTC)':20s} {'SYMBOL':7s} {'SIDE':6s} {'Z':>6s} {'xThr':>6s} {'SHARPE':>7s} {'ENTRY':>10s} {'TP':>10s} {'SL':>10s}")
        for score, a, sh, zs in ranked[:top]:
            print(f"{score:6.1f} {a['utc_time'].strftime('%Y-%m-%d %H:%M'):20s} {a['symbol']:7s} {a['side']:6s} "
                  f"{a['z']:6.2f} {zs:6.2f} {sh:7.2f} {a['entry']:10.6f} {a['tp']:10.6f} {a['sl']:10.6f}")

if __name__ == '__main__':
    main()
