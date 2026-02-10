#!/usr/bin/env python3
"""
Hail Mary — Advanced Crypto Day Trading Alert Bot (Coinbase Edition)
-------------------------------------------------------------------
Single-file script with:
- Coinbase via CCXT (OHLCV, tickers)
- Hybrid strategy: RSI divergence (5m/15m) + 9/21 EMA cross + 1.5× vol + 15m 50EMA trend
  and secondary VWAP proximity + S/R pivots with RSI OB/OS
- Robust pagination for Coinbase's ~300-candle per-call limit
- Risk controls & SQLite logging (paper only)
- Telegram alerts with matplotlib chart
- Backtester with summary + per-trade CSV outputs and charts
- Config via YAML; .env minimal loader for secrets

Usage:
  pip install ccxt pandas numpy pyyaml python-telegram-bot==13.15 matplotlib tenacity
  python hail_mary.py --backtest --config config.yaml
  python hail_mary.py --run --config config.yaml
"""
from __future__ import annotations
import os, sys, time, math, json, yaml, sqlite3, logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import itertools, copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

UTC = timezone.utc

# -------------------- utils --------------------
def utcnow() -> datetime:
    return datetime.now(tz=UTC)

def load_env():
    """Minimal .env loader (no python-dotenv dependency)."""
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

# -------------------- exchange client --------------------
try:
    import ccxt
except Exception:
    print("ccxt is required. pip install ccxt")
    raise

class DataClient:
    def __init__(self, cfg: dict):
        ex_cfg = cfg.get("exchange", {})
        ex_id = ex_cfg.get("id", "coinbase")
        if ex_id != "coinbase":
            logging.warning("Overriding exchange id to 'coinbase'.")
            ex_id = "coinbase"
        self.exchange = getattr(ccxt, ex_id)({"enableRateLimit": ex_cfg.get("enableRateLimit", True)})
        self.exchange.load_markets()

    def normalize_symbol(self, sym: str) -> Optional[str]:
        s = sym.replace("-", "/")
        return s if s in self.exchange.markets else None

    def list_available_symbols(self, symbols: List[str]) -> List[str]:
        out = []
        for s in symbols:
            ns = self.normalize_symbol(s)
            if ns: out.append(ns)
            else: logging.info(f"Skipping unsupported symbol on Coinbase: {s}")
        return out

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30), reraise=True,
           retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeError)))
    def fetch_ohlcv(self, symbol: str, timeframe: str, since: Optional[int] = None, limit: int = 300) -> List[List[float]]:
        # Coinbase commonly caps around 300 candles per call
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)

    def fetch_ohlcv_df(self, symbol: str, timeframe: str, lookback_minutes: int) -> pd.DataFrame:
        """Paginate forward in 300-candle chunks until we reach 'now'."""
        ms_per_bar = {"1m":60_000, "5m":300_000, "15m":900_000, "1h":3_600_000, "4h":14_400_000}[timeframe]
        limit = 300
        start_ms = int((utcnow() - timedelta(minutes=lookback_minutes)).timestamp()*1000)
        now_ms = int(utcnow().timestamp()*1000)
        since = start_ms
        all_rows: List[List[float]] = []
        stall = 0
        while since < now_ms:
            try:
                chunk = self.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            except Exception as e:
                logging.warning(f"fetch_ohlcv error tf={timeframe} since={since}: {e}")
                stall += 1; time.sleep(0.2)
                if stall >= 5: break
                continue
            if not chunk:
                stall += 1
                if stall >= 5: break
                since += limit*ms_per_bar
                continue
            if all_rows and chunk[0][0] <= all_rows[-1][0]:
                since = all_rows[-1][0] + ms_per_bar
                stall += 1
                if stall >= 5: break
                continue
            all_rows += chunk
            since = chunk[-1][0] + ms_per_bar
            stall = 0
            time.sleep(0.05)
            if len(chunk) < limit:
                break
        if not all_rows:
            return pd.DataFrame(columns=["open","high","low","close","volume"]).assign(ts=[])
        df = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","volume"]).drop_duplicates(subset=["ts"]) 
        df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("dt").sort_index()

    def fetch_tickers(self) -> Dict[str, dict]:
        return self.exchange.fetch_tickers()

# -------------------- indicators --------------------
class Indicators:
    @staticmethod
    def ema(series: pd.Series, length: int) -> pd.Series:
        return series.ewm(span=length, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, length: int = 14) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(alpha=1/length, adjust=False).mean()
        roll_down = down.ewm(alpha=1/length, adjust=False).mean()
        rs = roll_up / (roll_down + 1e-12)
        return 100 - (100/(1+rs))

    @staticmethod
    def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def vwap(df: pd.DataFrame, window: int = 390) -> pd.Series:
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        vol = df["volume"].fillna(0)
        pv = tp * vol
        cum_pv = pv.rolling(window, min_periods=1).sum()
        cum_vol = vol.rolling(window, min_periods=1).sum()
        return cum_pv / (cum_vol.replace(0, np.nan))

    @staticmethod
    def pivot_points(df: pd.DataFrame, lookback: int = 50) -> Tuple[float, float]:
        recent = df.tail(lookback)
        return float(recent["low"].min()), float(recent["high"].max())

# -------------------- strategy --------------------
@dataclass
class Signal:
    symbol: str
    side: str  # LONG/SHORT
    entry: float
    sl: float
    tp: float
    score: float
    context: Dict[str, float]
    timeframe: str
    ts: datetime

class Strategy:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.s = cfg["strategy"]

    @staticmethod
    def _detect_divergence(df: pd.DataFrame, rsi_len: int) -> Optional[str]:
        close = df["close"]
        rsi = Indicators.rsi(close, rsi_len)
        window = 5
        highs = (close.shift(1) < close) & (close.shift(-1) < close)
        lows  = (close.shift(1) > close) & (close.shift(-1) > close)
        last_highs = close[highs].tail(3)
        last_lows  = close[lows].tail(3)
        if len(last_lows) >= 2:
            p1, p2 = last_lows.iloc[-2], last_lows.iloc[-1]
            r1, r2 = rsi.loc[last_lows.index[-2]], rsi.loc[last_lows.index[-1]]
            if p2 < p1 and r2 > r1 + 0.1: return "bullish"
        if len(last_highs) >= 2:
            p1, p2 = last_highs.iloc[-2], last_highs.iloc[-1]
            r1, r2 = rsi.loc[last_highs.index[-2]], rsi.loc[last_highs.index[-1]]
            if p2 > p1 and r2 < r1 - 0.1: return "bearish"
        return None

    def _ema_cross_volume(self, df_5m: pd.DataFrame, df_15m: pd.DataFrame) -> Optional[str]:
        e9  = Indicators.ema(df_5m["close"], self.s["ema_fast"])
        e21 = Indicators.ema(df_5m["close"], self.s["ema_slow"])
        cross_up = (e9.iloc[-2] < e21.iloc[-2]) and (e9.iloc[-1] > e21.iloc[-1])
        cross_dn = (e9.iloc[-2] > e21.iloc[-2]) and (e9.iloc[-1] < e21.iloc[-1])

        v = df_5m["volume"]
        vol_mult  = float(self.s.get("vol_mult", 1.5))
        vol_window= int(self.s.get("vol_window", 30))
        avgv = v.rolling(vol_window, min_periods=min(10, vol_window)).mean()
        vol_ok = v.iloc[-1] > vol_mult * (avgv.iloc[-1] or 1e-9)

        trend_50 = Indicators.ema(df_15m["close"], self.s["ema_trend"]).iloc[-1]
        price_15 = df_15m["close"].iloc[-1]
        trend_up, trend_dn = price_15 > trend_50, price_15 < trend_50

        if cross_up and vol_ok and trend_up: return "bullish"
        if cross_dn and vol_ok and trend_dn: return "bearish"
        return None


    def _vwap_filter(self, df: pd.DataFrame) -> Optional[str]:
        vwap = Indicators.vwap(df, self.s["vwap_window"]) ; price = df["close"].iloc[-1] ; vw = vwap.iloc[-1]
        proximity = abs(price - vw) / (vw if vw else 1e-9) * 100
        if proximity <= self.s["vwap_proximity_pct"]:
            prev = df["close"].iloc[-2]
            if prev < vw <= price: return "bullish"
            if prev > vw >= price: return "bearish"
        return None

    def _sr_filter(self, df: pd.DataFrame) -> Optional[str]:
        s, r = Indicators.pivot_points(df, self.s["pivot_lookback"]) ; price = df["close"].iloc[-1]
        ob, os = self.s["rsi_ob"], self.s["rsi_os"]
        rsi = Indicators.rsi(df["close"], self.s["rsi_length"]).iloc[-1]
        near_s = abs(price - s) / (s if s else 1e-9) * 100 <= self.s["sr_proximity_pct"]
        near_r = abs(price - r) / (r if r else 1e-9) * 100 <= self.s["sr_proximity_pct"]
        if near_s and rsi < os: return "bullish"
        if near_r and rsi > ob: return "bearish"
        return None

    def build_signal(self, symbol: str, dfs: Dict[str, pd.DataFrame]) -> Optional[Signal]:
        df5, df15 = dfs["5m"], dfs["15m"]
        # components
        div     = self._detect_divergence(df5, self.s["rsi_length"])   # bullish/bearish/None
        ema_vol = self._ema_cross_volume(df5, df15)                      # bullish/bearish/None
        vwap_sig= self._vwap_filter(df5)
        sr_sig  = self._sr_filter(df5)
        votes = [sig for sig in [ema_vol, div, vwap_sig, sr_sig] if sig]
        if not votes: return None
        bull, bear = sum(v=="bullish" for v in votes), sum(v=="bearish" for v in votes)
        if bull == bear:
            primary_dir = ema_vol or div or vwap_sig or sr_sig
        else:
            primary_dir = "bullish" if bull > bear else "bearish"
        # normalized scoring
        primary_norm   = (int(bool(div)) + int(bool(ema_vol))) / 2.0
        secondary_norm = (int(bool(vwap_sig)) + int(bool(sr_sig))) / 2.0
        if div and ema_vol and div != ema_vol:
            primary_norm = max(0.0, primary_norm - 0.25)
        final_score = self.s["primary_weight"]*primary_norm + self.s["secondary_weight"]*secondary_norm
        if final_score < self.s["signal_threshold"]: return None
        side = "LONG" if primary_dir == "bullish" else "SHORT"
        last_close = df5["close"].iloc[-1]
        atr = Indicators.atr(df5, self.s["atr_length"]).iloc[-1]
        sl = last_close - self.s["atr_mult_sl"]*atr if side=="LONG" else last_close + self.s["atr_mult_sl"]*atr
        risk = abs(last_close - sl)
        tp = last_close + self.s["min_rr"]*risk if side=="LONG" else last_close - self.s["min_rr"]*risk
        ctx = {"div":1.0 if div else 0.0, "ema_vol":1.0 if ema_vol else 0.0, "vwap":1.0 if vwap_sig else 0.0,
               "sr":1.0 if sr_sig else 0.0, "atr": float(atr)}
        return Signal(symbol, side, float(last_close), float(sl), float(tp), float(final_score), ctx, "5m", utcnow())

# -------------------- risk, alerts, db --------------------
class TradeDB:
    def __init__(self, path: str = "trades.db"):
        self.path = path
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS trades(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, symbol TEXT, side TEXT,
                entry REAL, sl REAL, tp REAL, size REAL,
                status TEXT, exit_price REAL, pnl REAL
            )"""
        )
        con.commit(); con.close()
    def insert_trade(self, sig: Signal, size: float):
        con = sqlite3.connect(self.path); cur = con.cursor()
        cur.execute("INSERT INTO trades(ts,symbol,side,entry,sl,tp,size,status) VALUES(?,?,?,?,?,?,?,?)",
                    (sig.ts.isoformat(), sig.symbol, sig.side, sig.entry, sig.sl, sig.tp, size, "OPEN"))
        con.commit(); con.close()
    def count_open_positions(self) -> int:
        con = sqlite3.connect(self.path); cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM trades WHERE status='OPEN'")
        n = cur.fetchone()[0]; con.close(); return n
    def get_today_pnl(self) -> float:
        today = datetime.now(UTC).date().isoformat()
        con = sqlite3.connect(self.path); cur = con.cursor()
        cur.execute("SELECT SUM(pnl) FROM trades WHERE ts LIKE ?", (f"{today}%",))
        res = cur.fetchone()[0]; con.close(); return float(res or 0.0)

class RiskManager:
    def __init__(self, cfg: dict, db: TradeDB):
        self.cfg, self.db = cfg, db
        self.equity = cfg["risk"]["account_equity"]
        self.daily_limit = cfg["risk"]["daily_drawdown_limit_pct"]/100 * self.equity
    def position_size(self, entry: float, sl: float) -> float:
        risk_pct = self.cfg["risk"]["risk_per_trade_pct"]/100.0
        risk_amt = self.equity * risk_pct
        per_unit = abs(entry - sl)
        return 0.0 if per_unit <= 0 else max(risk_amt/per_unit, 0.0)
    def can_open_new(self) -> bool:
        if self.db.count_open_positions() >= self.cfg["risk"]["max_concurrent_positions"]:
            logging.info("Max concurrent positions reached"); return False
        if self.db.get_today_pnl() < -self.daily_limit:
            logging.warning("Daily drawdown limit reached"); return False
        return True

class AlertManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.tg_token = os.getenv("TELEGRAM_BOT_TOKEN"); self.tg_chat = os.getenv("TELEGRAM_CHAT_ID")
        self.discord_url = os.getenv("DISCORD_WEBHOOK_URL")
        self._tg_bot = None
        if cfg.get("alerts", {}).get("enable_telegram", False):
            try:
                from telegram import Bot
                self._tg_bot = Bot(self.tg_token) if self.tg_token else None
            except Exception as e:
                logging.error(f"Telegram init failed: {e}")
    def _format_msg(self, sig: Signal, risk_pct: float, chart_path: Optional[str]) -> str:
        sym = sig.symbol.replace("/", "")
        emoji = "🟢" if sig.side == "LONG" else "🔴"
        msg = (f"{emoji} {sym} {sig.side} | Entry: ${sig.entry:,.4f} | SL: ${sig.sl:,.4f} | TP: ${sig.tp:,.4f} | "
               f"Risk: {risk_pct:.2f}% | Score: {sig.score:.2f}\n"
               f"div={sig.context['div']}, ema_vol={sig.context['ema_vol']}, vwap={sig.context['vwap']}, sr={sig.context['sr']}\n"
               f"TF={sig.timeframe} | {sig.ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        if chart_path: msg += f"\nChart: {chart_path}"
        return msg
    def _plot_chart(self, df: pd.DataFrame, sig: Signal, save_dir: str, width: int, height: int) -> Optional[str]:
        try:
            os.makedirs(save_dir, exist_ok=True)
            fig = plt.figure(figsize=(width/100, height/100)); ax = plt.gca()
            df.tail(200)["close"].plot(ax=ax)
            ax.axhline(sig.entry, linestyle='--'); ax.axhline(sig.sl, linestyle=':'); ax.axhline(sig.tp, linestyle=':')
            ax.set_title(f"{sig.symbol} {sig.side} {sig.timeframe} @ {sig.entry:.4f}")
            fp = os.path.join(save_dir, f"{sig.symbol.replace('/', '')}_{int(sig.ts.timestamp())}.png")
            plt.tight_layout(); plt.savefig(fp); plt.close(fig); return fp
        except Exception as e:
            logging.error(f"chart render failed: {e}"); return None
    def send(self, sig: Signal, df_for_chart: pd.DataFrame):
        chart_path = None
        a_cfg = self.cfg.get("alerts", {})
        if a_cfg.get("save_charts", True):
            chart_path = self._plot_chart(df_for_chart, sig, a_cfg.get("charts_dir", "charts"), a_cfg.get("chart_width", 900), a_cfg.get("chart_height", 500))
        msg = self._format_msg(sig, self.cfg["risk"]["risk_per_trade_pct"], chart_path)
        if self._tg_bot and self.tg_chat:
            try:
                self._tg_bot.send_message(chat_id=self.tg_chat, text=msg)
                if chart_path:
                    with open(chart_path, 'rb') as img: self._tg_bot.send_photo(chat_id=self.tg_chat, photo=img)
            except Exception as e:
                logging.error(f"Telegram send failed: {e}")
        if a_cfg.get("enable_discord", False) and self.discord_url:
            try:
                from discord_webhook import DiscordWebhook
                DiscordWebhook(url=self.discord_url, content=msg).execute()
            except Exception as e:
                logging.error(f"Discord send failed: {e}")

# -------------------- news guard --------------------
def within_news_guard(cfg: dict, now_utc: datetime) -> bool:
    ng = cfg.get("filters", {}).get("news_guard", {})
    if not ng.get("enable", False): return False
    events = ng.get("events", [])
    before = int(ng.get("skip_minutes_before", 30)); after = int(ng.get("skip_minutes_after", 30))
    weekday = ["MON","TUE","WED","THU","FRI","SAT","SUN"][now_utc.weekday()]
    for ev in events:
        try: ev_day, ev_hm = ev.split()
        except ValueError: continue
        if ev_day != weekday: continue
        ev_dt = datetime.combine(now_utc.date(), datetime.strptime(ev_hm, "%H:%M").time(), tzinfo=UTC)
        if abs((now_utc - ev_dt).total_seconds())/60.0 <= max(before, after): return True
    return False

# -------------------- backtester --------------------
class Backtester:
    def __init__(self, cfg: dict, data: DataClient, strat: Strategy):
        self.cfg, self.data, self.strat = cfg, data, strat
        self.trades_log: List[Dict[str, object]] = []  # per-trade rows

    def _simulate_trade(self, df: pd.DataFrame, sig: Signal) -> Tuple[float, float, float, str, float]:
        """Return (pnl_points, exit_level, last_price, outcome, r_multiple_gross)."""
        prices = df["close"].to_numpy(); entry = sig.entry; sl, tp = sig.sl, sig.tp
        risk = abs(entry - sl) if abs(entry - sl) > 0 else 1e-9
        for p in prices:
            if sig.side == "LONG":
                if p <= sl: pnl = -abs(entry - sl); return pnl, sl, p, "SL", pnl/risk
                if p >= tp: pnl = abs(tp - entry); return pnl, tp, p, "TP", pnl/risk
            else:
                if p >= sl: pnl = -abs(sl - entry); return pnl, sl, p, "SL", pnl/risk
                if p <= tp: pnl = abs(entry - tp); return pnl, tp, p, "TP", pnl/risk
        last = prices[-1]; pnl = (last - entry) if sig.side=="LONG" else (entry - last)
        return pnl, last, last, "EOD", pnl/risk

    def run_prefetched(self, symbol: str, df5: pd.DataFrame, df15: pd.DataFrame, df1h: pd.DataFrame) -> Dict[str, float]:
        fees = self.cfg["backtest"]["fee_pct"]
        pnl_list: List[float] = []
        equity = 10000.0; peak = equity; max_dd = 0.0; wins = losses = trades = 0
        df15 = df15.reindex(df5.index, method='pad'); df1h = df1h.reindex(df5.index, method='pad')
        if len(df5) < 400:
            return {"symbol": symbol, "trades": 0}

        for i in range(200, len(df5)-100):
            window5  = df5.iloc[:i].copy()
            window15 = df15.iloc[:i].copy()
            window1h = df1h.iloc[:i].copy()
            sig = self.strat.build_signal(symbol, {"5m":window5, "15m":window15, "1h":window1h})
            if not sig: continue
            sig.ts = window5.index[-1].to_pydatetime()
            fwd = df5.iloc[i:i+100]
            sim_pnl, exit_level, last_p, outcome, r_mult = self._simulate_trade(fwd, sig)
            gross = sim_pnl; est_exit = exit_level; cost = fees*sig.entry + fees*est_exit; net = gross - cost
            equity += net; peak = max(peak, equity); max_dd = max(max_dd, (peak - equity))
            pnl_list.append(net); trades += 1; wins += int(net>0); losses += int(net<=0)
            self.trades_log.append({
                "symbol": symbol, "time": sig.ts.isoformat(), "side": sig.side,
                "entry": round(sig.entry,8), "sl": round(sig.sl,8), "tp": round(sig.tp,8),
                "exit": round(est_exit,8), "outcome": outcome, "r_multiple_gross": round(r_mult,4),
                "pnl_net_points": round(net,8), "score": round(sig.score,4),
                "div": sig.context.get("div",0.0), "ema_vol": sig.context.get("ema_vol",0.0),
                "vwap": sig.context.get("vwap",0.0), "sr": sig.context.get("sr",0.0),
                "atr": round(sig.context.get("atr",0.0),8),
            })

        win_rate = (wins/trades*100) if trades else 0.0
        avg_rr_placeholder = float(np.mean([max(-5, min(5, p)) for p in pnl_list]) if trades else 0.0)
        return {"symbol":symbol, "trades":trades, "win_rate":win_rate, "avg_rr":avg_rr_placeholder,
                "max_drawdown":float(max_dd), "final_equity":float(equity)}


    def run(self, symbol: str) -> Dict[str, float]:
        months = self.cfg["backtest"]["months"]
        lookback_minutes = int(months * 30 * 24 * 60)
        df5  = self.data.fetch_ohlcv_df(symbol, "5m",  lookback_minutes)
        if len(df5) < 400: return {"symbol": symbol, "trades": 0}
        df15 = self.data.fetch_ohlcv_df(symbol, "15m", lookback_minutes)
        df1h = self.data.fetch_ohlcv_df(symbol, "1h",  lookback_minutes)
        fees = self.cfg["backtest"]["fee_pct"]
        pnl_list: List[float] = []
        equity = 10000.0; peak = equity; max_dd = 0.0; wins = losses = trades = 0
        df15 = df15.reindex(df5.index, method='pad'); df1h = df1h.reindex(df5.index, method='pad')
        for i in range(200, len(df5)-100):
            window5 = df5.iloc[:i].copy(); window15 = df15.iloc[:i].copy(); window1h = df1h.iloc[:i].copy()
            sig = self.strat.build_signal(symbol, {"5m":window5, "15m":window15, "1h":window1h})
            if not sig: continue
            sig.ts = window5.index[-1].to_pydatetime()
            fwd = df5.iloc[i:i+100]
            sim_pnl, exit_level, last_p, outcome, r_mult = self._simulate_trade(fwd, sig)
            gross = sim_pnl; est_exit = exit_level; cost = fees*sig.entry + fees*est_exit; net = gross - cost
            equity += net; peak = max(peak, equity); max_dd = max(max_dd, (peak - equity))
            pnl_list.append(net); trades += 1; wins += int(net>0); losses += int(net<=0)
            self.trades_log.append({
                "symbol": symbol, "time": sig.ts.isoformat(), "side": sig.side,
                "entry": round(sig.entry,8), "sl": round(sig.sl,8), "tp": round(sig.tp,8),
                "exit": round(est_exit,8), "outcome": outcome, "r_multiple_gross": round(r_mult,4),
                "pnl_net_points": round(net,8), "score": round(sig.score,4),
                "div": sig.context.get("div",0.0), "ema_vol": sig.context.get("ema_vol",0.0),
                "vwap": sig.context.get("vwap",0.0), "sr": sig.context.get("sr",0.0),
                "atr": round(sig.context.get("atr",0.0),8),
            })
        win_rate = (wins/trades*100) if trades else 0.0
        avg_rr_placeholder = float(np.mean([max(-5, min(5, p)) for p in pnl_list]) if trades else 0.0)
        return {"symbol":symbol, "trades":trades, "win_rate":win_rate, "avg_rr":avg_rr_placeholder,
                "max_drawdown":float(max_dd), "final_equity":float(equity)}

# -------------------- engine --------------------
class Engine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.data = DataClient(cfg)
        self.db   = TradeDB()
        self.risk = RiskManager(cfg, self.db)
        self.strat= Strategy(cfg)
        self.alerts = AlertManager(cfg)
        self.symbols = self.data.list_available_symbols(cfg.get("symbols", []))
        self.min_vol = cfg.get("filters", {}).get("min_usd_24h_volume", 0)
        self.poll_seconds = cfg.get("runtime", {}).get("poll_seconds", 30)
        self.use_threads  = cfg.get("runtime", {}).get("use_threads", True)
        self.max_workers  = cfg.get("runtime", {}).get("max_workers", 6)
    def _volume_filter(self, tickers: Dict[str, dict]) -> List[str]:
        passed = []
        for s in self.symbols:
            t = tickers.get(s)
            if not t: continue
            vol_usd = t.get("quoteVolume")
            if not vol_usd:
                base_v = t.get("baseVolume"); last = t.get("last")
                if base_v and last: vol_usd = base_v*last
            if vol_usd and vol_usd >= self.min_vol: passed.append(s)
        return passed or self.symbols
    def _build_signal_for_symbol(self, symbol: str) -> Optional[Tuple[Signal, pd.DataFrame]]:
        df5  = self.data.fetch_ohlcv_df(symbol, "5m", 5*24*60)
        if len(df5) < 200: return None
        df15 = self.data.fetch_ohlcv_df(symbol, "15m", 15*24*60)
        df1h = self.data.fetch_ohlcv_df(symbol, "1h", 60*24*14)
        sig = self.strat.build_signal(symbol, {"5m":df5, "15m":df15, "1h":df1h})
        return (sig, df5) if sig else None
    def once(self):
        if within_news_guard(self.cfg, utcnow()):
            logging.info("News guard window active; skipping scan."); return
        try: tickers = self.data.fetch_tickers()
        except Exception as e: logging.error(f"fetch_tickers failed: {e}"); return
        universe = self._volume_filter(tickers)
        logging.info(f"Scanning {len(universe)} symbols…")
        results: List[Optional[Tuple[Signal, pd.DataFrame]]] = []
        if self.use_threads and len(universe)>1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                for s in universe:
                    results.append(ex.submit(self._build_signal_for_symbol, s).result())
        else:
            for s in universe:
                results.append(self._build_signal_for_symbol(s))
        for r in results:
            if not r: continue
            sig, df = r
            if not self.risk.can_open_new(): break
            size = self.risk.position_size(sig.entry, sig.sl)
            if size <= 0: continue
            self.db.insert_trade(sig, size)
            self.alerts.send(sig, df)
    def loop(self):
        while True:
            self.once(); time.sleep(self.poll_seconds)

# -------------------- cli --------------------
def main():
    import argparse
    load_env()

    parser = argparse.ArgumentParser(description="Hail Mary — Coinbase Advanced Crypto Alert Bot")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--backtest", action="store_true", help="Run backtest over configured symbols")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep (uses config.sweep ranges)")
    parser.add_argument("--run", action="store_true", help="(Placeholder) Live run loop / alerts")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("runtime", {}).get("log_level", "INFO"))

    engine = Engine(cfg)

    # ----------------------- BACKTEST -----------------------
    if args.backtest:
        bt = Backtester(cfg, engine.data, engine.strat)
        rows = []
        for s in engine.symbols:
            try:
                res = bt.run(s)
                logging.info(f"Backtest {s}: {res}")
                rows.append(res)
            except Exception as e:
                logging.error(f"Backtest failed for {s}: {e}")

        if rows:
            summary_df = pd.DataFrame(rows)
            print(summary_df)

            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            summary_path = f"backtest_results_{ts}.csv"
            trades_path  = f"backtest_trades_{ts}.csv"

            try:
                summary_df.to_csv(summary_path, index=False)
                if bt.trades_log:
                    pd.DataFrame(bt.trades_log).to_csv(trades_path, index=False)
                logging.info(f"Saved CSVs: {summary_path} and {trades_path}")
            except Exception as e:
                logging.error(f"Failed to save CSVs: {e}")

            # charts (timestamped)
            try:
                plt.figure()
                summary_df.set_index("symbol")["win_rate"].plot(kind="bar")
                plt.title("Win Rate (%) by Symbol")
                plt.tight_layout()
                plt.savefig(f"bt_winrate_{ts}.png")
                plt.close()

                plt.figure()
                summary_df.set_index("symbol")["max_drawdown"].plot(kind="bar")
                plt.title("Max Drawdown by Symbol")
                plt.tight_layout()
                plt.savefig(f"bt_maxdd_{ts}.png")
                plt.close()
            except Exception:
                pass

        return

    # ------------------------ SWEEP ------------------------
    if args.sweep:
        sweep_cfg = cfg.get("sweep", {})
    # Which symbols to sweep
        symbols = sweep_cfg.get("symbols") or engine.symbols[: sweep_cfg.get("max_symbols", 5)]

    # Prefetch data once per symbol
        data_cache = {}
        months = cfg["backtest"]["months"]
        lookback_minutes = int(months * 30 * 24 * 60)
        for s in symbols:
            try:
                df5  = engine.data.fetch_ohlcv_df(s, "5m",  lookback_minutes)
                df15 = engine.data.fetch_ohlcv_df(s, "15m", lookback_minutes)
                df1h = engine.data.fetch_ohlcv_df(s, "1h",  lookback_minutes)
                data_cache[s] = (df5, df15, df1h)
            except Exception as e:
                logging.error(f"Prefetch failed for {s}: {e}")

    # Build parameter grid
        grid_dict = {
            "ema_fast":           sweep_cfg.get("ema_fast", [8, 9, 12]),
            "ema_slow":           sweep_cfg.get("ema_slow", [20, 21, 26]),
            "vol_mult":           sweep_cfg.get("vol_mult", [1.25, 1.5, 2.0]),
            "vol_window":         sweep_cfg.get("vol_window", [20, 30, 50]),
            "rsi_length":         sweep_cfg.get("rsi_length", [12, 14, 16]),
            "atr_mult_sl":        sweep_cfg.get("atr_mult_sl", [2.0, 2.5, 3.0]),
            "min_rr":             sweep_cfg.get("min_rr", [1.5, 2.0, 2.5]),
            "signal_threshold":   sweep_cfg.get("signal_threshold", [0.55, 0.60, 0.65]),
            "vwap_proximity_pct": sweep_cfg.get("vwap_proximity_pct", [0.5, 0.6, 0.75]),
            "sr_proximity_pct":   sweep_cfg.get("sr_proximity_pct", [0.2, 0.25, 0.3]),
        }
        keys   = list(grid_dict.keys())
        values = [grid_dict[k] for k in keys]
        combos = list(itertools.product(*values))
        logging.info(f"Sweep combinations: {len(combos)} across {len(symbols)} symbols")

        results_rows = []
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        for combo in combos:
            overrides = dict(zip(keys, combo))
            cfg2 = copy.deepcopy(cfg)
            cfg2["strategy"].update(overrides)
            strat2 = Strategy(cfg2)
            bt2 = Backtester(cfg2, engine.data, strat2)
            for s in symbols:
                dfs = data_cache.get(s)
                if not dfs: continue
                df5, df15, df1h = dfs
                try:
                    res = bt2.run_prefetched(s, df5, df15, df1h)
                    results_rows.append({**res, **overrides, "base_symbol": s})
                except Exception as e:
                    logging.error(f"Sweep run failed for {s} with {overrides}: {e}")

        if results_rows:
            sweep_df = pd.DataFrame(results_rows)
            sweep_path = f"backtest_sweep_{ts}.csv"
            sweep_df.to_csv(sweep_path, index=False)
            logging.info(f"Saved sweep CSV: {sweep_path}")

        # Also save a best-per-symbol file for quick takeaways
            best_rows = []
            for s in symbols:
                sub = sweep_df[sweep_df["base_symbol"] == s]
                if not sub.empty:
                    best = sub.sort_values(["final_equity", "win_rate"], ascending=[False, False]).head(1)
                    best_rows.append(best)
            if best_rows:
                best_df = pd.concat(best_rows)
                best_path = f"backtest_sweep_best_{ts}.csv"
                best_df.to_csv(best_path, index=False)
                logging.info(f"Saved best-per-symbol CSV: {best_path}")
        return

    # ------------------------- RUN -------------------------
    if args.run:
        logging.info("Live run mode is not implemented in this minimal script.")
        logging.info("Use --backtest or --sweep, or extend Engine with a live loop and alerts.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
