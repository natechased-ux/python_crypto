# coin_rules.py
# Per-zone, per-coin rules loader with sensible defaults + CSV override.
# CSV schema (headers, case-insensitive):
# coin,style_ema,tp_pct_ema,sl_pct_ema,timebox_h_ema,
# style_fib,tp1_pct_fib,tp1_size_pct_fib,tp2_pct_fib,sl_mode_fib,atr_sl_pct_fib,timebox_h_fib,suppress

from typing import Dict, Any, Optional
import os
import csv

# style defaults
DEFAULTS = {
    "SCALP":   {"tp_pct": 0.55, "sl_pct": 0.30, "timebox_h": 3},
    "REVERSAL":{"tp1_pct": 0.65, "tp1_size_pct": 0.70, "tp2_pct": 1.00, "timebox_h": 12,
                "sl_mode": "zone", "atr_sl_pct": 0.70},
    "EXCLUDE": {"suppress": True}
}

_RULES: Optional[Dict[str, Dict[str, Any]]] = None
_ALL_COINS: Optional[list] = None

def set_all_coins(coins: list):
    """Optional: tell the module your live COINS list (for future heuristics if needed)."""
    global _ALL_COINS
    _ALL_COINS = [c.upper() for c in coins]

def _load_csv_rules(csv_path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.isfile(csv_path):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            coin = (row.get("coin") or "").strip().upper()
            if not coin:
                continue
            r: Dict[str, Any] = {"coin": coin}

            # EMA block
            r["style_ema"] = (row.get("style_ema") or "").strip().upper() or "SCALP"
            # numeric ema params
            for k_csv, k_out in (("tp_pct_ema","tp_pct"), ("sl_pct_ema","sl_pct")):
                v = row.get(k_csv)
                if v not in (None, "", "NA"):
                    try: r[k_out] = float(v)
                    except: pass
            v = row.get("timebox_h_ema")
            if v not in (None, "", "NA"):
                try: r["timebox_h_ema"] = int(float(v))
                except: pass

            # FIB block
            r["style_fib"] = (row.get("style_fib") or "").strip().upper() or "REVERSAL"
            for k_csv, k_out in (("tp1_pct_fib","tp1_pct"), ("tp1_size_pct_fib","tp1_size_pct"),
                                 ("tp2_pct_fib","tp2_pct"), ("atr_sl_pct_fib","atr_sl_pct")):
                v = row.get(k_csv)
                if v not in (None, "", "NA"):
                    try: r[k_out] = float(v)
                    except: pass
            v = row.get("sl_mode_fib")
            if v not in (None, "", "NA"):
                r["sl_mode"] = v.strip().lower()
            v = row.get("timebox_h_fib")
            if v not in (None, "", "NA"):
                try: r["timebox_h_fib"] = int(float(v))
                except: pass

            # optional master suppress
            if row.get("suppress") not in (None, ""):
                r["suppress"] = str(row["suppress"]).strip().lower() in ("1","true","yes","y")

            out[coin] = r
    return out

def load_rules(csv_path: str = None) -> Dict[str, Dict[str, Any]]:
    """Load/refresh per-zone rules from CSV. Returns the internal rule dict."""
    global _RULES
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "coin_rules_zones_full.csv")
    _RULES = _load_csv_rules(csv_path)
    return _RULES

def get_coin_rule(coin: str) -> Dict[str, Any]:
    """Return raw dict for coin (merged defaults are applied by get_styles_for)."""
    global _RULES
    if _RULES is None:
        load_rules()  # load default path
    return dict(_RULES.get(coin.upper(), {}))

def _merge_defaults(style: str, values: Dict[str, Any], zone: str) -> Dict[str, Any]:
    # Start from style defaults, overlay explicit values
    merged = dict(DEFAULTS.get(style, {}))
    # remap timebox keys by zone family
    if zone == "EMA":
        # allow EMA-specific timebox override
        if "timebox_h_ema" in values:
            merged["timebox_h"] = values["timebox_h_ema"]
    else:
        if "timebox_h_fib" in values:
            merged["timebox_h"] = values["timebox_h_fib"]
    # overlay generic keys
    for k, v in values.items():
        if v not in (None, "", "NA"):
            merged[k] = v
    return merged

def get_styles_for(coin: str, zone_source: str):
    """
    Return (style_for_zone, params_for_zone)
    style_for_zone: "SCALP" / "REVERSAL" / "EXCLUDE"
    params_for_zone: dict with merged TP/SL/timebox and stop mode
    """
    r = get_coin_rule(coin)
    zfam = "EMA" if zone_source.upper().startswith("EMA") else "FIB"
    style_key = "style_ema" if zfam == "EMA" else "style_fib"
    style = r.get(style_key, "SCALP" if zfam == "EMA" else "REVERSAL").upper()

    if style == "EXCLUDE":
        return style, {"suppress": True}

    # Collect only relevant keys by family to merge with defaults
    if zfam == "EMA":
        values = {
            "tp_pct": r.get("tp_pct"),
            "sl_pct": r.get("sl_pct"),
            "timebox_h_ema": r.get("timebox_h_ema")
        }
    else:
        values = {
            "tp1_pct": r.get("tp1_pct"),
            "tp1_size_pct": r.get("tp1_size_pct"),
            "tp2_pct": r.get("tp2_pct"),
            "sl_mode": r.get("sl_mode"),
            "atr_sl_pct": r.get("atr_sl_pct"),
            "timebox_h_fib": r.get("timebox_h_fib")
        }

    params = _merge_defaults(style, values, zfam)
    return style, params
