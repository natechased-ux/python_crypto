
import pandas as pd
import numpy as np

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = (delta.clip(lower=0)).ewm(alpha=1/length, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def true_range(df: pd.DataFrame):
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def adx(df: pd.DataFrame, length: int = 14):
    high, low, close = df['high'], df['low'], df['close']
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    atr_val = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/length, adjust=False).mean() / (atr_val + 1e-12)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/length, adjust=False).mean() / (atr_val + 1e-12)
    dx = ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)) * 100
    adx_val = dx.ewm(alpha=1/length, adjust=False).mean()
    plus_di.index = df.index
    minus_di.index = df.index
    adx_val.index = df.index
    return adx_val, plus_di, minus_di

def bb_width(series: pd.Series, length: int = 20, std_mult: float = 2.0) -> pd.Series:
    ma = series.rolling(length).mean()
    sd = series.rolling(length).std()
    upper = ma + std_mult * sd
    lower = ma - std_mult * sd
    return (upper - lower) / (ma + 1e-12)

def stoch_rsi(series: pd.Series, rsi_len=14, stoch_len=14, smooth_k=3, smooth_d=3):
    r = rsi(series, rsi_len)
    r_min = r.rolling(stoch_len).min()
    r_max = r.rolling(stoch_len).max()
    k = 100 * (r - r_min) / ((r_max - r_min) + 1e-12)
    k = k.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return pd.DataFrame({"k": k, "d": d})
