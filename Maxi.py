#!/usr/bin/env python3
"""
Regime-Adaptive Crypto Alert System (Python + Telegram)

Core ideas
- Classify market regime (Trending vs. Ranging) using ADX and Bollinger Band Width (BBW)
- Apply a Breakout strategy in Trending regimes; Mean-Reversion in Ranging regimes
- Use ATR-based TP/SL sizing (SL=1.5x ATR, TP=2.5x ATR)
- Non-async Telegram alerts with cooldown per symbol
- Coinbase Exchange public API for candles (no auth)

Notes
- Default timeframe: 1H (granularity=3600). You can switch to 15m (900) in CONFIG.
- Keep dependencies light: requests, pandas, numpy.
"""

from __future__ import annotations
import os, time, math, json, signal, logging, requests
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Optional

# -------------------------------
# Configuration
# -------------------------------
CONFIG = {
    "SYMBOLS": ["eth-usd","xrp-usd","ltc-usd","ada-usd","doge-usd",
    "sol-usd","wif-usd","ondo-usd","sei-usd","magic-usd","ape-usd",
    "jasmy-usd","wld-usd","syrup-usd","fartcoin-usd","aero-usd",
    "link-usd","hbar-usd","aave-usd","fet-usd","crv-usd","tao-usd",
    "avax-usd","xcn-usd","uni-usd","mkr-usd","toshi-usd","near-usd",
    "algo-usd","trump-usd","bch-usd","inj-usd","pepe-usd","xlm-usd",
    "moodeng-usd","bonk-usd","dot-usd","popcat-usd","arb-usd","icp-usd",
    "qnt-usd","tia-usd","ip-usd","pnut-usd","apt-usd","ena-usd","turbo-usd",
    "bera-usd","pol-usd","mask-usd","pyth-usd","sand-usd","morpho-usd",
    "mana-usd","c98-usd","axs-usd"],
    "GRANULARITY": 3600,  # 1H candles. Use 900 for 15m.
    "LOOP_INTERVAL": 60,  # scan every minute
    "ALERT_COOLDOWN_MIN": 60,
    "MIN_BARS": 120,
    "BREAKOUT_LOOKBACK": 20,
    "BB_PERIOD": 20,
    "BB_STD": 2.0,
    "ATR_PERIOD": 14,
    "ADX_PERIOD": 14,
    "ADX_TREND": 25.0,
    "ADX_RANGE": 20.0,
    "BBW_COMPRESS_PCTL": 25,
    "RSI_TREND_MIN_LONG": 55.0,
    "RSI_TREND_MAX_SHORT": 45.0,
    "RSI_OVERSOLD": 30.0,
    "RSI_OVERBOUGHT": 70.0,
    "SL_ATR_MULT": 1.5,
    "TP_ATR_MULT": 2.5,
    "VOL_MULT_BREAKOUT": 1.2,
    "DISPLAY_TZ": "America/Los_Angeles",
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", "7967738614"),
}

TELEGRAM_API = f"https://api.telegram.org/bot{CONFIG['TELEGRAM_BOT_TOKEN']}/sendMessage"
CBX_BASE = "https://api.exchange.coinbase.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("regime_adaptive_alerts")

# -------------------------------
# Helpers
# -------------------------------
def fmt_price(p: float) -> str:
    if p < 0.001: return f"{p:.8f}"
    if p < 0.01:  return f"{p:.6f}"
    if p < 1:     return f"{p:.4f}"
    if p < 100:   return f"{p:.2f}"
    return f"{p:.2f}"

def now_pt() -> str:
    tz = ZoneInfo(CONFIG["DISPLAY_TZ"])
    return datetime.now(tz=tz).strftime("%Y-%m-%d %H:%M:%S %Z")

# -------------------------------
# Data
# -------------------------------
def fetch_candles(symbol: str, granularity: int) -> pd.DataFrame:
    url = f"{CBX_BASE}/products/{symbol}/candles?granularity={granularity}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[["time","open","high","low","close","volume"]].sort_values("time").reset_index(drop=True)
    return df

# -------------------------------
# Indicators
# -------------------------------
def ema(series: pd.Series, span: int): return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period=14):
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period).mean()
    rs = gain / (loss.replace(0,np.nan))
    return 100 - (100/(1+rs))

def true_range(df): 
    prev = df["close"].shift(1)
    return pd.concat([df["high"]-df["low"], (df["high"]-prev).abs(), (df["low"]-prev).abs()],axis=1).max(axis=1)

def atr(df, period=14): return true_range(df).ewm(alpha=1/period).mean()

def adx(df, period=14):
    up, dn = df["high"].diff(), -df["low"].diff()
    plus_dm  = np.where((up>dn)&(up>0), up, 0.0)
    minus_dm = np.where((dn>up)&(dn>0), dn, 0.0)
    atrs = true_range(df).ewm(alpha=1/period).mean()
    plus_di  = 100*(pd.Series(plus_dm).ewm(alpha=1/period).mean()/atrs)
    minus_di = 100*(pd.Series(minus_dm).ewm(alpha=1/period).mean()/atrs)
    dx = ((plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,np.nan))*100
    return dx.ewm(alpha=1/period).mean()

def bollinger(df, period=20, std=2.0):
    mid = df["close"].rolling(period).mean()
    sd = df["close"].rolling(period).std()
    up, lo = mid+std*sd, mid-std*sd
    width = (up-lo)/mid
    return mid,up,lo,width

def macd(series, fast=12, slow=26, signal=9):
    fast, slow = ema(series,fast), ema(series,slow)
    line = fast-slow
    sig  = ema(line, signal)
    return line, sig, line-sig

# -------------------------------
# Patterns
# -------------------------------
def bullish_engulfing(df):
    if len(df)<2: return False
    prev,last=df.iloc[-2],df.iloc[-1]
    return (prev.close<prev.open and last.close>last.open and last.close>=prev.open and last.open<=prev.close)

def bearish_engulfing(df):
    if len(df)<2: return False
    prev,last=df.iloc[-2],df.iloc[-1]
    return (prev.close>prev.open and last.close<last.open and last.close<=prev.open and last.open>=prev.close)

# -------------------------------
# Signals
# -------------------------------
@dataclass
class Signal:
    symbol:str; regime:str; side:str; entry:float; tp:float; sl:float
    atr:float; adx:float; rsi:float; macd_hist:float; macd_slope:float
    bb_width:float; extras:Dict; asof:datetime

class StrategyEngine:
    def __init__(self,cfg): self.cfg=cfg; self.cooldowns={}
    def in_cooldown(self,s): return (time.time()-self.cooldowns.get(s,0))<self.cfg["ALERT_COOLDOWN_MIN"]*60
    def mark_cooldown(self,s): self.cooldowns[s]=time.time()
    def compute(self,df,symbol)->Optional[Signal]:
        if len(df)<max(self.cfg["MIN_BARS"],self.cfg["BB_PERIOD"]+30): return None
        adx_s, atr_s = adx(df,self.cfg["ADX_PERIOD"]), atr(df,self.cfg["ATR_PERIOD"])
        mid,up,lo,bbw=bollinger(df,self.cfg["BB_PERIOD"],self.cfg["BB_STD"])
        rsi_s, macd_l, sig, hist = rsi(df.close), *macd(df.close)
        last, prev=df.iloc[-1], df.iloc[-2]
        adx_l, atr_l, bbw_l, bbw_p, rsi_l = float(adx_s.iloc[-1]), float(atr_s.iloc[-1]), float(bbw.iloc[-1]), float(bbw.iloc[-2]), float(rsi_s.iloc[-1])
        hist_l, hist_p, slope = float(hist.iloc[-1]), float(hist.iloc[-2]), float(hist.iloc[-1]-hist.iloc[-2])
        bbw_tail = bbw.dropna().iloc[-100:]; bbw_rank=(bbw_tail.rank(pct=True).iloc[-1])*100 if len(bbw_tail)>=20 else 100
        trending = (adx_l>=self.cfg["ADX_TREND"]) and (bbw_l>bbw_p)
        ranging  = (adx_l<=self.cfg["ADX_RANGE"]) and (bbw_rank<=self.cfg["BBW_COMPRESS_PCTL"])
        if trending:
            n=self.cfg["BREAKOUT_LOOKBACK"]
            hi, lo_n = df.high.rolling(n).max().shift(1).iloc[-1], df.low.rolling(n).min().shift(1).iloc[-1]
            vol_sma=df.volume.rolling(20).mean().iloc[-1]; vol_ok=last.volume>=self.cfg["VOL_MULT_BREAKOUT"]*vol_sma
            if last.close>hi and hist_l>0 and slope>0 and rsi_l>=self.cfg["RSI_TREND_MIN_LONG"] and vol_ok:
                return Signal(symbol,"TRENDING","LONG",float(last.close),last.close+self.cfg["TP_ATR_MULT"]*atr_l,last.close-self.cfg["SL_ATR_MULT"]*atr_l,atr_l,adx_l,rsi_l,hist_l,slope,bbw_l,{"hi":hi},last.time)
            if last.close<lo_n and hist_l<0 and slope<0 and rsi_l<=self.cfg["RSI_TREND_MAX_SHORT"] and vol_ok:
                return Signal(symbol,"TRENDING","SHORT",float(last.close),last.close-self.cfg["TP_ATR_MULT"]*atr_l,last.close+self.cfg["SL_ATR_MULT"]*atr_l,atr_l,adx_l,rsi_l,hist_l,slope,bbw_l,{"lo":lo_n},last.time)
        if ranging:
            if last.close<=lo.iloc[-1]+0.15*(up.iloc[-1]-lo.iloc[-1]) and rsi_l<=self.cfg["RSI_OVERSOLD"] and bullish_engulfing(df.tail(2)):
                return Signal(symbol,"RANGING","LONG",float(last.close),last.close+self.cfg["TP_ATR_MULT"]*atr_l,last.close-self.cfg["SL_ATR_MULT"]*atr_l,atr_l,adx_l,rsi_l,hist_l,slope,bbw_l,{"band":"lower"},last.time)
            if last.close>=up.iloc[-1]-0.15*(up.iloc[-1]-lo.iloc[-1]) and rsi_l>=self.cfg["RSI_OVERBOUGHT"] and bearish_engulfing(df.tail(2)):
                return Signal(symbol,"RANGING","SHORT",float(last.close),last.close-self.cfg["TP_ATR_MULT"]*atr_l,last.close+self.cfg["SL_ATR_MULT"]*atr_l,atr_l,adx_l,rsi_l,hist_l,slope,bbw_l,{"band":"upper"},last.time)
        return None

# -------------------------------
# Telegram + Run
# -------------------------------
class Notifier:
    def __init__(self,token,chat): self.api=f"https://api.telegram.org/bot{token}/sendMessage"; self.chat=chat
    def send(self,text):
        try: requests.post(self.api,json={"chat_id":self.chat,"text":text,"parse_mode":"Markdown"},timeout=10)
        except: logger.warning("Telegram failed")

class Runner:
    def __init__(self,cfg): self.cfg=cfg; self.engine=StrategyEngine(cfg); self.notifier=Notifier(cfg["TELEGRAM_BOT_TOKEN"],cfg["TELEGRAM_CHAT_ID"]); self._stop=False
    def stop(self,*_): self._stop=True
    def build_msg(self,s:Signal)->str:
        tz=ZoneInfo(self.cfg["DISPLAY_TZ"]); t=s.asof.tz_convert(tz) if s.asof.tzinfo else s.asof.replace(tzinfo=timezone.utc).astimezone(tz)
        msg=(f"*{s.symbol}* — *{s.side}* ({s.regime})\\nTime: {t:%Y-%m-%d %H:%M:%S %Z}\\nEntry: `{fmt_price(s.entry)}` TP: `{fmt_price(s.tp)}` SL: `{fmt_price(s.sl)}`\\nATR: {s.atr:.4f} | ADX: {s.adx:.1f} | RSI: {s.rsi:.1f}\\nMACD: {s.macd_hist:.5f} slope {s.macd_slope:.5f}\\nBB width {s.bb_width:.5f}")
        return msg
    def loop(self):
        while not self._stop:
            for sym in self.cfg["SYMBOLS"]:
                if self.engine.in_cooldown(sym): continue
                try:
                    sig=self.engine.compute(fetch_candles(sym,self.cfg["GRANULARITY"]),sym)
                    if sig: self.engine.mark_cooldown(sym); self.notifier.send(self.build_msg(sig))
                except Exception as e: logger.warning("scan error %s %s",sym,e)
                time.sleep(0.5)
            time.sleep(self.cfg["LOOP_INTERVAL"])

if __name__=="__main__":
    r=Runner(CONFIG)
    signal.signal(signal.SIGINT,r.stop); signal.signal(signal.SIGTERM,r.stop)
    logger.info("Booting Regime-Adaptive Alerts… %s", now_pt())
    r.loop()
