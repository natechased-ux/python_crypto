#!/usr/bin/env python3
"""
TTM Squeeze (LazyBear updated) • 1H • Pro Triggers
- Two-stage alerts: HEADS-UP (pre-release) and ENTRY (release/go)
- ENTRY fires only when smarter gates pass:
    * Composite score ≥ SCORE_MIN
    * Squeeze intensity: normalized BB width in bottom percentile
    * ADX rising over lookback
    * DI spread ≥ threshold
    * Minimum squeeze duration before release
    * Momentum building preferred (LIME/RED)

Keeps:
- Relaxed release window (first/second gray after black)
- ATR-based TP/SL
- Exit-on-color-flip and zero-cross
"""

import os
import time
import json
import math
import requests
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Dict

# =============== User Config (1H) ===============
COINS = [
    "btc-usd","eth-usd","xrp-usd","ltc-usd","ada-usd","doge-usd",
    "sol-usd","wif-usd","ondo-usd","sei-usd","magic-usd","ape-usd",
    "jasmy-usd","wld-usd","syrup-usd","fartcoin-usd","aero-usd",
    "link-usd","hbar-usd","aave-usd","fet-usd","crv-usd","tao-usd",
    "avax-usd","xcn-usd","uni-usd","mkr-usd","toshi-usd","near-usd",
    "algo-usd","trump-usd","bch-usd","inj-usd","pepe-usd","xlm-usd",
    "moodeng-usd","bonk-usd","dot-usd","popcat-usd","arb-usd","icp-usd",
    "qnt-usd","tia-usd","ip-usd","pnut-usd","apt-usd","ena-usd","turbo-usd",
    "bera-usd","pol-usd","mask-usd","pyth-usd","sand-usd","morpho-usd",
    "mana-usd","c98-usd","axs-usd"
]
GRANULARITY = 3600     # 1H
CANDLES_LIMIT = 300

# Squeeze / Momentum
BB_LEN = 20
BB_MULT = 2.0
KC_LEN = 20
KC_MULT = 1.5
USE_TRUE_RANGE = True

# ATR TP/SL
ATR_LEN = 14
TP_MULT = 2.0
SL_MULT = 1.5

# Base gates
RELEASE_WINDOW = 2
ABS_VAL_MIN = 0.5
ALT_VAL_STRONG = 1.2

ADX_LEN = 14
ADX_MIN = 15.0
REQUIRE_DI_ALIGNMENT = False

# ==== Pro trigger gates ====
SCORE_MIN = 70              # require B-tier or better
BBW_LOOKBACK = 200
BBW_PCTL_MAX = 30           # 30th percentile or tighter
ADX_SLOPE_LOOKBACK = 3
ADX_SLOPE_MIN = 2.0         # ADX - ADX[3] >= 2.0
DI_SPREAD_MIN = 5.0
MIN_SQZ_BARS = 4
ALLOW_FADING_IF_SCORE_GE = 85

# HEADS-UP settings
HEADSUP_ADX_RISING = True
HEADSUP_MIN_SQZ = 3
HEADSUP_MSG_COOLDOWN_MIN = 90

# Misc
COOLDOWN_MIN = 25
RATE_LIMIT_PAUSE = 1.0
SAVE_STATE_EACH_LOOP = True
STATE_PATH = "positions_state_1h_pro.json"
DEBUG = False

PT = dt.timezone(dt.timedelta(hours=-8))  # Pacific time, no DST handling

# Telegram
TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"
FALLBACK_BOT_TOKEN = ""
FALLBACK_CHAT_ID = ""


# ================= Helpers =================
def sma(series, length):
    out, q, s = [], [], 0.0
    for v in series:
        q.append(v); s += v
        if len(q) > length: s -= q.pop(0)
        out.append(s/length if len(q) == length else float('nan'))
    return out

def stdev(series, length):
    out, window = [], []
    for v in series:
        window.append(v)
        if len(window) > length: window.pop(0)
        if len(window) == length:
            m = sum(window)/length
            var = sum((x-m)**2 for x in window)/length
            out.append(math.sqrt(var))
        else:
            out.append(float('nan'))
    return out

def highest(series, length):
    out, window = [], []
    for v in series:
        window.append(v)
        if len(window) > length: window.pop(0)
        out.append(max(window) if len(window) == length else float('nan'))
    return out

def lowest(series, length):
    out, window = [], []
    for v in series:
        window.append(v)
        if len(window) > length: window.pop(0)
        out.append(min(window) if len(window) == length else float('nan'))
    return out

def linreg_last(series, length):
    out = []
    xs = list(range(length))
    sum_x = sum(xs); sum_x2 = sum(x*x for x in xs)
    denom = length * sum_x2 - sum_x * sum_x
    window = []
    for v in series:
        window.append(v)
        if len(window) > length: window.pop(0)
        if len(window) < length:
            out.append(float('nan')); continue
        sum_y = sum(window); sum_xy = sum(x*y for x,y in zip(xs, window))
        m = (length*sum_xy - sum_x*sum_y)/denom if denom != 0 else 0.0
        b = (sum_y - m*sum_x)/length
        out.append(m*(length-1) + b)
    return out

def true_range(highs, lows, closes):
    trs, prev_close = [], None
    for h, l, c in zip(highs, lows, closes):
        if prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c
    return trs

def atr_sma(highs, lows, closes, length):
    return sma(true_range(highs, lows, closes), length)

# Wilder-style ADX
def adx_di(highs, lows, closes, length):
    n = len(closes)
    tr = [0.0]*n; plus_dm = [0.0]*n; minus_dm = [0.0]*n
    for i in range(n):
        h, l = highs[i], lows[i]
        if i == 0:
            tr[i] = h - l
            plus_dm[i] = minus_dm[i] = 0.0
        else:
            prev_close = closes[i-1]
            prev_high  = highs[i-1]
            prev_low   = lows[i-1]
            tr[i] = max(h - l, abs(h - prev_close), abs(l - prev_close))
            up_move   = h - prev_high
            down_move = prev_low - l
            plus_dm[i]  = up_move   if (up_move > down_move and up_move > 0) else 0.0
            minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

    def wilder_ema(values, length):
        out = [float('nan')]*n
        alpha = 1.0/length
        acc = None
        for i, v in enumerate(values):
            if i < length:
                if i == length-1:
                    seed = sum(values[:length])/length
                    acc = seed
                    out[i] = seed
            else:
                acc = acc + alpha*(v - acc)
                out[i] = acc
        return out

    tr_s = wilder_ema(tr, length)
    plus_dm_s = wilder_ema(plus_dm, length)
    minus_dm_s = wilder_ema(minus_dm, length)

    plus_di = [100.0*(p/t) if (not math.isnan(p) and not math.isnan(t) and t!=0) else float('nan')
               for p,t in zip(plus_dm_s, tr_s)]
    minus_di = [100.0*(m/t) if (not math.isnan(m) and not math.isnan(t) and t!=0) else float('nan')
                for m,t in zip(minus_dm_s, tr_s)]

    dx = [100.0*abs(p-m)/(p+m) if (not math.isnan(p) and not math.isnan(m) and (p+m)!=0) else float('nan')
          for p,m in zip(plus_di, minus_di)]
    adx_vals = wilder_ema(dx, length)
    return plus_di, minus_di, adx_vals

# Coinbase API
def load_candles(product_id, granularity, limit):
    product_id = product_id.upper()
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {"granularity": granularity}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    data.sort(key=lambda x: x[0])
    return data[-limit:]

# ================= Core Indicator =================
def compute_signals(candles):
    times  = [c[0] for c in candles]
    lows   = [float(c[1]) for c in candles]
    highs  = [float(c[2]) for c in candles]
    closes = [float(c[4]) for c in candles]

    source = closes

    basis = sma(source, BB_LEN)
    devs  = [BB_MULT*d if not math.isnan(d) else float('nan') for d in stdev(source, BB_LEN)]
    upperBB = [b + d if not (math.isnan(b) or math.isnan(d)) else float('nan') for b,d in zip(basis, devs)]
    lowerBB = [b - d if not (math.isnan(b) or math.isnan(d)) else float('nan') for b,d in zip(basis, devs)]

    ma = sma(source, KC_LEN)
    rng = true_range(highs, lows, closes) if USE_TRUE_RANGE else [h-l for h,l in zip(highs, lows)]
    rangema = sma(rng, KC_LEN)
    upperKC = [m + rm*KC_MULT if not (math.isnan(m) or math.isnan(rm)) else float('nan') for m,rm in zip(ma, rangema)]
    lowerKC = [m - rm*KC_MULT if not (math.isnan(m) or math.isnan(rm)) else float('nan') for m,rm in zip(ma, rangema)]

    sqzOn, sqzOff = [], []
    for lb, ub, lkc, ukc in zip(lowerBB, upperBB, lowerKC, upperKC):
        if any(math.isnan(x) for x in (lb, ub, lkc, ukc)):
            sqzOn.append(False); sqzOff.append(False)
        else:
            on  = (lb > lkc) and (ub < ukc)
            off = (lb < lkc) and (ub > ukc)
            sqzOn.append(on); sqzOff.append(off)

    hh = highest(highs, KC_LEN)
    ll = lowest(lows, KC_LEN)
    mid_hl = [(a+b)/2.0 if not (math.isnan(a) or math.isnan(b)) else float('nan') for a,b in zip(hh, ll)]
    sma_close_kc = sma(closes, KC_LEN)
    inner_avg = [(a+b)/2.0 if not (math.isnan(a) or math.isnan(b)) else float('nan') for a,b in zip(mid_hl, sma_close_kc)]
    detrended = [s - ia if not math.isnan(ia) else float('nan') for s, ia in zip(source, inner_avg)]
    detrended_clean = [x if not math.isnan(x) else 0.0 for x in detrended]
    val = linreg_last(detrended_clean, KC_LEN)

    atr_vals = atr_sma(highs, lows, closes, ATR_LEN)
    di_plus, di_minus, adx_vals = adx_di(highs, lows, closes, ADX_LEN)

    bbw = []
    for u, l, b in zip(upperBB, lowerBB, basis):
        if any(math.isnan(x) for x in (u,l,b)) or b == 0:
            bbw.append(float('nan'))
        else:
            bbw.append((u - l) / abs(b))

    return {
        "times": times, "closes": closes,
        "sqzOn": sqzOn, "sqzOff": sqzOff,
        "val": val, "atr": atr_vals,
        "di_plus": di_plus, "di_minus": di_minus, "adx": adx_vals,
        "bbw": bbw
    }

# ================== Telegram ==================
def send_telegram(text):
    token = TELEGRAM_BOT_TOKEN or FALLBACK_BOT_TOKEN
    chat_id = TELEGRAM_CHAT_ID or FALLBACK_CHAT_ID
    if not token or not chat_id:
        print("Telegram not configured. Skipping message."); return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    requests.post(url, data=payload, timeout=20)

# ================= State/Entry/Exit logic omitted for brevity here =================
# (full version continues with scoring, entry gating, HEADS-UP logic, and exit rules)
# ================= Momentum helpers =================
def momentum_state(curr, prev):
    if math.isnan(curr) or math.isnan(prev): return "n/a"
    if curr > 0 and curr > prev: return "LIME"
    if curr > 0 and curr <= prev: return "GREEN"
    if curr < 0 and curr < prev: return "RED"
    if curr < 0 and curr >= prev: return "MAROON"
    return "n/a"

def di_spread(dip, dim):
    if math.isnan(dip) or math.isnan(dim): return float('nan')
    return abs(dip - dim)

def percentile_rank(window_vals, target):
    vals = [v for v in window_vals if not math.isnan(v)]
    if not vals: return float('nan')
    less = sum(1 for v in vals if v <= target)
    return 100.0 * less / len(vals)

# =============== Scoring ===============
VAL_TARGET = 1.2
ADX_TARGET = 25.0
WEIGHT_VAL = 0.45
WEIGHT_ADX = 0.35
WEIGHT_MOM_STATE = 0.10
WEIGHT_FRESH = 0.07
WEIGHT_DI = 0.03
GRADE_A = 0.80
GRADE_B = 0.65
GRADE_C = 0.50

def compute_entry_score(val_now, val_prev, adx_now, dip, dim, strict_release, side):
    val_comp = min(abs(val_now)/VAL_TARGET, 1.0) if not math.isnan(val_now) else 0.0
    adx_comp = min((adx_now or 0.0)/ADX_TARGET, 1.0) if not math.isnan(adx_now) else 0.0
    ms = momentum_state(val_now, val_prev)
    mom_comp = 1.0 if (ms in ("LIME","RED")) else (0.4 if (ms in ("GREEN","MAROON")) else 0.0)
    fresh_comp = 1.0 if strict_release else 0.5
    di_comp = 0.0
    if side == "long" and not math.isnan(dip) and not math.isnan(dim) and dip > dim: di_comp = 1.0
    if side == "short" and not math.isnan(dip) and not math.isnan(dim) and dim > dip: di_comp = 1.0
    score01 = (WEIGHT_VAL*val_comp + WEIGHT_ADX*adx_comp +
               WEIGHT_MOM_STATE*mom_comp + WEIGHT_FRESH*fresh_comp + WEIGHT_DI*di_comp)
    return max(0.0, min(1.0, score01))

def grade_from_score01(score01):
    if score01 >= GRADE_A: return "A", "🟢"
    if score01 >= GRADE_B: return "B", "🟡"
    if score01 >= GRADE_C: return "C", "🟠"
    return "D", "🔴"

# =============== State ===============
@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    entry_time: int
    last_state: str
    is_active: bool = True

@dataclass
class HeadsUpCooldowns:
    last_ts: Dict[str, float] = field(default_factory=dict)
    def ok(self, symbol, minutes): 
        t = self.last_ts.get(symbol, 0.0)
        return (time.time() - t) >= minutes * 60
    def mark(self, symbol): 
        self.last_ts[symbol] = time.time()

@dataclass
class State:
    positions: Dict[str, Position] = field(default_factory=dict)
    headsup: HeadsUpCooldowns = field(default_factory=HeadsUpCooldowns)
    def to_json(self):
        return {
            "positions": {k: vars(v) for k,v in self.positions.items()},
            "headsup": {"last_ts": self.headsup.last_ts},
        }
    @staticmethod
    def from_json(d: Dict) -> 'State':
        st = State()
        if "positions" in d:
            for k,v in d["positions"].items():
                st.positions[k] = Position(**v)
        if "headsup" in d and "last_ts" in d["headsup"]:
            st.headsup.last_ts = d["headsup"]["last_ts"]
        return st

def load_state(path: str) -> State:
    if not os.path.exists(path): return State()
    try:
        with open(path, "r", encoding="utf-8") as f: 
            return State.from_json(json.load(f))
    except Exception:
        return State()

def save_state(path: str, state: State):
    try:
        with open(path, "w", encoding="utf-8") as f: json.dump(state.to_json(), f)
    except Exception: pass

# =============== Release helper ===============
def released_within_window(i, sqzOn, sqzOff, window):
    k = None
    for j in range(i-1, max(i-20, 0), -1):
        if sqzOn[j]:
            k = j; break
    if k is None: return False, 0
    dur = 1
    t = k-1
    while t >= 0 and sqzOn[t]:
        dur += 1; t -= 1
    for t in range(k+1, min(k+1+window, i+1)):
        if t < len(sqzOff) and sqzOff[t]:
            return True, dur
    return False, dur

# =============== Main Loop ===============
def run_once(state):
    for sym in COINS:
        try:
            candles = load_candles(sym, GRANULARITY, CANDLES_LIMIT)
        except Exception as e:
            print(f"[{sym}] fetch error: {e}")
            time.sleep(RATE_LIMIT_PAUSE); continue

        s = compute_signals(candles)
        i = len(s["times"]) - 1
        if i < 2 or math.isnan(s["val"][i]): time.sleep(RATE_LIMIT_PAUSE); continue

        val_now, val_prev = s["val"][i], s["val"][i-1]
        adx_i, dip_i, dim_i = s["adx"][i], s["di_plus"][i], s["di_minus"][i]
        ms = momentum_state(val_now, val_prev)
        di_gap = di_spread(dip_i, dim_i)

        bbw_vals = s["bbw"][max(0, i-BBW_LOOKBACK+1):i+1]
        bbw_pctl = percentile_rank(bbw_vals, s["bbw"][i])

        # HEADS-UP
        if s["sqzOn"][i]:
            dur = 1; t = i-1
            while t >= 0 and s["sqzOn"][t]: dur += 1; t -= 1
            adx_rising = (not math.isnan(adx_i)) and (i-ADX_SLOPE_LOOKBACK >= 0) and (s["adx"][i] - s["adx"][i-ADX_SLOPE_LOOKBACK] >= ADX_SLOPE_MIN)
            mom_building = (ms in ("LIME","RED"))
            if dur >= HEADSUP_MIN_SQZ and (not HEADSUP_ADX_RISING or adx_rising) and mom_building:
                if state.headsup.ok(sym, HEADSUP_MSG_COOLDOWN_MIN):
                    ts_utc = dt.datetime.utcfromtimestamp(s["times"][i]).replace(tzinfo=dt.timezone.utc)
                    ts_pt = ts_utc.astimezone(PT)
                    direction_hint = "📈 bias" if val_now > 0 else "📉 bias" if val_now < 0 else "•"
                    txt = (
                        f"🟦 <b>HEADS-UP • TTM Squeeze 1H</b>\n"
                        f"<b>{sym}</b> • {direction_hint}\n"
                        f"Black squeeze ongoing (dur={dur}) • ADX {'rising' if adx_rising else 'flat'} • {ms}\n"
                        f"BBW pct: {('%.0f'%bbw_pctl) if not math.isnan(bbw_pctl) else 'n/a'} (≤ {BBW_PCTL_MAX} better)\n"
                        f"Time (UTC): {ts_utc.strftime('%Y-%m-%d %H:%M')}   PT: {ts_pt.strftime('%Y-%m-%d %H:%M')}"
                    )
                    send_telegram(txt)
                    state.headsup.mark(sym)

        # Release
        strict_rel_up = s["sqzOff"][i] and s["sqzOn"][i-1] and val_now > 0
        strict_rel_dn = s["sqzOff"][i] and s["sqzOn"][i-1] and val_now < 0
        rel_up_window, dur_up = released_within_window(i, s["sqzOn"], s["sqzOff"], RELEASE_WINDOW)
        rel_dn_window, dur_dn = released_within_window(i, s["sqzOn"], s["sqzOff"], RELEASE_WINDOW)
        is_rel_up   = strict_rel_up or (rel_up_window and val_now > 0)
        is_rel_down = strict_rel_dn or (rel_dn_window and val_now < 0)
        sqz_dur = dur_up if is_rel_up else (dur_dn if is_rel_down else 0)

        # Entry gates
        def base_gates_ok():
    # --- Soft ADX logic ---
    # Treat ADX level as *informative* via scoring; don't hard-block on ADX_MIN.
    # Require ADX *rising* OR very strong momentum to catch early expansions.
            val_strong = abs(val_now) >= ALT_VAL_STRONG

    # ADX rising over the lookback window (e.g., 3 bars)
            adx_rise = (
                not math.isnan(adx_i)
                and i - ADX_SLOPE_LOOKBACK >= 0
                and (s["adx"][i] - s["adx"][i - ADX_SLOPE_LOOKBACK]) >= ADX_SLOPE_MIN
            )

    # Optional tiny floor to avoid dead tape (set ADX_TINY_FLOOR=0.0 to disable)
            ADX_TINY_FLOOR = 12.0
            adx_tiny_ok = (not math.isnan(adx_i)) and (adx_i >= ADX_TINY_FLOOR)

    # Trend clarity, squeeze quality, and minimum coil
            di_ok  = (not math.isnan(di_gap)) and (di_gap >= DI_SPREAD_MIN)
            dur_ok = sqz_dur >= MIN_SQZ_BARS
            bbw_ok = (not math.isnan(bbw_pctl)) and (bbw_pctl <= BBW_PCTL_MAX)

    # Final gate: need (ADX rising OR very strong momentum), plus the quality checks
            return (adx_rise or val_strong) and adx_tiny_ok and di_ok and dur_ok and bbw_ok

        def momentum_preference_ok(score_pct):
            if ms in ("LIME","RED"): return True
            return score_pct >= ALLOW_FADING_IF_SCORE_GE

        # Try entries
        pos = state.positions.get(sym)
        entry_signal = None
        if is_rel_up and val_now > 0 and base_gates_ok(): entry_signal = "long"
        elif is_rel_down and val_now < 0 and base_gates_ok(): entry_signal = "short"

        if entry_signal and (pos is None or not pos.is_active):
            score01 = compute_entry_score(val_now, val_prev, adx_i, dip_i, dim_i, True, entry_signal)
            score_pct = int(round(score01 * 100))
            if score_pct >= SCORE_MIN and momentum_preference_ok(score_pct):
                entry = s["closes"][i]
                atr_i = s["atr"][i]
                tp = sl = float('nan')
                if not math.isnan(atr_i) and atr_i > 0:
                    if entry_signal == "long":
                        tp = entry + TP_MULT * atr_i
                        sl = entry - SL_MULT * atr_i
                    else:
                        tp = entry - TP_MULT * atr_i
                        sl = entry + SL_MULT * atr_i
                grade, badge = grade_from_score01(score01)
                ts_utc = dt.datetime.utcfromtimestamp(s["times"][i]).replace(tzinfo=dt.timezone.utc)
                ts_pt = ts_utc.astimezone(PT)
                direction = "📈 LONG" if entry_signal == "long" else "📉 SHORT"
                header = f"{badge} <b>ENTRY • TTM Squeeze 1H (pro)</b> {badge}"
                txt = (
                    f"{header}\n"
                    f"<b>{sym}</b> • {direction}\n"
                    f"Score: <b>{score_pct}</b>/100  Grade: <b>{grade}</b>\n"
                    f"val: {val_now:.4f} • ADX: {adx_i:.1f} • DI spread: {di_gap:.1f}\n"
                    f"BBW pct: {('%.0f'%bbw_pctl) if not math.isnan(bbw_pctl) else 'n/a'}  •  sqz dur: {sqz_dur}\n"
                    f"Entry: {entry:.2f}  TP: {tp:.2f}  SL: {sl:.2f}\n"
                    f"Time (UTC): {ts_utc.strftime('%Y-%m-%d %H:%M')}   PT: {ts_pt.strftime('%Y-%m-%d %H:%M')}"
                )
                send_telegram(txt)
                state.positions[sym] = Position(sym, entry_signal, entry, s["times"][i], ms, True)

        # EXIT logic
        pos = state.positions.get(sym)
        if pos and pos.is_active:
            exit_reason = None
            if pos.side == "long":
                if pos.last_state == "LIME" and ms == "GREEN":
                    exit_reason = "Momentum flip LIME→GREEN"
                elif val_now < 0:
                    exit_reason = "Momentum crossed below 0"
            else:
                if pos.last_state == "RED" and ms == "MAROON":
                    exit_reason = "Momentum flip RED→MAROON"
                elif val_now > 0:
                    exit_reason = "Momentum crossed above 0"
            if exit_reason:
                ts_utc = dt.datetime.utcfromtimestamp(s["times"][i]).replace(tzinfo=dt.timezone.utc)
                ts_pt = ts_utc.astimezone(PT)
                txt = (
                    f"🛑 <b>EXIT</b>\n"
                    f"<b>{sym}</b> • {pos.side.upper()}\n"
                    f"Reason: {exit_reason}\n"
                    f"val: {val_now:.4f} • ADX: {adx_i:.1f}\n"
                    f"Price now: {s['closes'][i]:.2f}\n"
                    f"Time (UTC): {ts_utc.strftime('%Y-%m-%d %H:%M')}   PT: {ts_pt.strftime('%Y-%m-%d %H:%M')}"
                )
                send_telegram(txt)
                pos.is_active = False
                state.positions[sym] = pos
            else:
                if ms != "n/a":
                    pos.last_state = ms
                    state.positions[sym] = pos

        time.sleep(RATE_LIMIT_PAUSE)

def main():
    state = load_state(STATE_PATH)
    print("Starting TTM Squeeze 1H (pro) Telegram Alert...")
    while True:
        try:
            run_once(state)
            if SAVE_STATE_EACH_LOOP:
                save_state(STATE_PATH, state)
        except KeyboardInterrupt:
            print("Exiting..."); break
        except Exception as e:
            print(f"Loop error: {e}")
        time.sleep(300)

if __name__ == "__main__":
    main()
