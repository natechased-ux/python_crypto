
import csv, json, os
from typing import Dict, Any

class TradeLogger:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.csv_path = cfg["logging"]["csv_path"]
        self.jsonl_path = cfg["logging"]["jsonl_path"]

    def log_open(self, trade_id, sym, side, entry, sl, tp, ctx, score, risk_frac):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(["OPEN", trade_id, sym, side, entry, sl, tp, score, risk_frac])

        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps({"event":"open","id":trade_id,"symbol":sym,"side":side,"entry":entry,"score":score}) + "\n")

    def log_update(self, trade):
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps({"event":"update","id":trade.id,"status":trade.status}) + "\n")
