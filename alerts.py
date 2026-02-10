
import os
import requests
from typing import Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo

def _decimals(price: float) -> int:
    if price >= 1000: return 2
    if price >= 100: return 3
    if price >= 1: return 4
    if price >= 0.1: return 5
    return 6

class TelegramAlerter:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.token = os.getenv(cfg["alerts"]["telegram"]["bot_token_env"], "")
        self.chat_id = os.getenv(cfg["alerts"]["telegram"]["chat_id_env"], "")
        self.tz = ZoneInfo(cfg.get("timezone","America/Los_Angeles"))

    def _send(self, text: str):
        if not self.token or not self.chat_id:
            print("[ALERT] Missing TG credentials; printing instead:\n" + text)
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        r = requests.post(url, json={"chat_id": self.chat_id, "text": text}, timeout=15)
        if r.status_code != 200:
            print(f"[ALERT ERROR] {r.status_code} {r.text}")

    def send_new_trade(self, trade_id, sym, side, entry, sl, tp, score, ctx, last_6h_close_utc):
        d = _decimals(entry)
        pt = last_6h_close_utc.astimezone(self.tz).strftime("%Y-%m-%d %I:%M %p %Z")
        ema_below = ctx.get('ema_below'); ema_above = ctx.get('ema_above')
        vol_pct = ctx.get('volume_pctile')
        cluster = ctx.get('clusters', {})
        lines = [
            f"⚡ SUPERMODEL SIGNAL",
            f"Pair: {sym.upper()}",
            f"Side: {side.upper()}",
            f"Entry: {entry:.{d}f}",
            f"SL: {sl:.{d}f}  |  TP: {tp:.{d}f}",
            f"Score: {score}",
            f"1H EMA below/above: {'' if ema_below is None else f'{ema_below:.{d}f}'} / {'' if ema_above is None else f'{ema_above:.{d}f}'}",
            f"1H Vol pctile: {vol_pct:.0f}",
            f"6H Close (PT): {pt}",
        ]
        if cluster:
            sup = cluster.get('support_nearby'); res = cluster.get('resistance_nearby')
            sd = cluster.get('support_dist_pct'); rd = cluster.get('resistance_dist_pct')
            lines.append(f"Clusters: support_near={sup} ({sd}%) | resistance_near={res} ({rd}%)")
        self._send("\n".join(lines))

    def send_close(self, trade):
        self._send(f"CLOSE {trade.symbol} outcome={trade.outcome} R={trade.r_multiple:.2f}")

    def send_test(self, text: str = "Supermodel is live (full send) 🚀"):
        self._send(text)
