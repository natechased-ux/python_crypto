# strategies/HybridTrendBreakoutAlert.py
from jesse.strategies import Strategy
import jesse.indicators as ta
from jesse import utils
import os

class HybridTrendBreakoutAlert(Strategy):
    """
    Hybrid Trend + Breakout with alerts and optimizer-ready hyperparameters.
    - Trend: EMA(fast) vs EMA(slow)
    - Breakout: price relative to EMA(fast) ± k * ATR
    - Momentum: ADX + RSI
    - Risk: ATR stops, partial at 2R, trail with ATR, BE after first reduction
    """

    # ===== Defaults (used if not optimizing) =====
    _defaults = dict(
        risk_per_trade=1.0,
        atr_stop_mult=2.0,
        breakout_k=0.5,
        trail_atr_mult=1.5,
        ema_fast_len=50,
        ema_slow_len=200,
        adx_thresh=20,
        rsi_upper=80,
        rsi_lower=20,
    )

    # ----- Hyperparameters for Jesse Optimize -----
    @staticmethod
    def hyperparameters():
        # Keep the search compact; widen later if needed
        return [
            {'name': 'atr_stop_mult', 'type': float, 'min': 1.2, 'max': 3.0, 'default': 2.0},
            {'name': 'breakout_k',    'type': float, 'min': 0.2, 'max': 0.8, 'default': 0.5},
            {'name': 'trail_atr_mult','type': float, 'min': 1.0, 'max': 2.2, 'default': 1.5},
            {'name': 'ema_fast_len',  'type': int,   'min': 20,  'max': 80,  'default': 50},
            {'name': 'ema_slow_len',  'type': int,   'min': 120, 'max': 300, 'default': 200},
            {'name': 'adx_thresh',    'type': int,   'min': 18,  'max': 35,  'default': 20},
            {'name': 'rsi_upper',     'type': int,   'min': 70,  'max': 88,  'default': 80},
            {'name': 'rsi_lower',     'type': int,   'min': 12,  'max': 30,  'default': 20},
            # ⚠️ I recommend NOT optimizing risk_per_trade to avoid sizing-overfit.
            # If you insist, uncomment below and cap it tightly.
            # {'name': 'risk_per_trade','type': float,'min': 0.25,'max': 1.25,'default': 1.0},
        ]

    # ----- Helper to read hp with defaults -----
    def hpv(self, name):
        try:
            return self.hp.get(name, self._defaults[name])
        except Exception:
            # when not in optimize mode, self.hp may not exist
            return self._defaults[name]

    # ----- Indicators / State -----
    @property
    def ema_fast(self):
        return ta.ema(self.candles, int(self.hpv('ema_fast_len')))

    @property
    def ema_slow(self):
        return ta.ema(self.candles, int(self.hpv('ema_slow_len')))

    @property
    def atr(self):
        return ta.atr(self.candles, 14)

    @property
    def adx_val(self):
        return ta.adx(self.candles)

    @property
    def rsi_val(self):
        return ta.rsi(self.candles)

    @property
    def trend(self) -> int:
        if self.ema_fast > self.ema_slow:
            return 1
        elif self.ema_fast < self.ema_slow:
            return -1
        return 0

    # ----- Conditions -----
    def _breakout_up(self) -> bool:
        return self.price > self.ema_fast + self.atr * float(self.hpv('breakout_k'))

    def _breakout_down(self) -> bool:
        return self.price < self.ema_fast - self.atr * float(self.hpv('breakout_k'))

    def _momentum_up(self) -> bool:
        return self.adx_val > float(self.hpv('adx_thresh')) and self.rsi_val > 55

    def _momentum_down(self) -> bool:
        return self.adx_val > float(self.hpv('adx_thresh')) and self.rsi_val < 45

    def _avoid_extreme_rsi(self) -> bool:
        if self.trend == 1 and self.rsi_val >= int(self.hpv('rsi_upper')):
            return False
        if self.trend == -1 and self.rsi_val <= int(self.hpv('rsi_lower')):
            return False
        return True

    # ----- Entries -----
    def should_long(self) -> bool:
        return (
            not self.is_open
            and self.trend == 1
            and self._breakout_up()
            and self._momentum_up()
            and self._avoid_extreme_rsi()
        )

    def go_long(self):
        entry = self.price
        stop = entry - self.atr * float(self.hpv('atr_stop_mult'))
        qty = utils.risk_to_qty(
            self.available_margin,
            float(self.hpv('risk_per_trade')),
            entry, stop,
            fee_rate=self.fee_rate
        )
        self._alert(f"LONG signal {self.symbol} @~{entry:.2f} | SL ~{stop:.2f} | ATR {self.atr:.2f}")
        self.buy = qty, entry  # MARKET

    def should_short(self) -> bool:
        return (
            not self.is_open
            and self.trend == -1
            and self._breakout_down()
            and self._momentum_down()
            and self._avoid_extreme_rsi()
        )

    def go_short(self):
        entry = self.price
        stop = entry + self.atr * float(self.hpv('atr_stop_mult'))
        qty = utils.risk_to_qty(
            self.available_margin,
            float(self.hpv('risk_per_trade')),
            entry, stop,
            fee_rate=self.fee_rate
        )
        self._alert(f"SHORT signal {self.symbol} @~{entry:.2f} | SL ~{stop:.2f} | ATR {self.atr:.2f}")
        self.sell = qty, entry  # MARKET

    def should_cancel_entry(self) -> bool:
        return True

    # ----- Position Management -----
    def on_open_position(self, order) -> None:
        """Set SL/TP immediately after position opens."""
        k = float(self.hpv('atr_stop_mult'))
        if self.is_long:
            sl = self.price - self.atr * k
            r = self.price - sl
            tp1 = self.price + 2 * r  # 2R partial
            self.stop_loss = self.position.qty, sl
            self.take_profit = self.position.qty / 2, tp1
            self._alert(
                f"OPENED LONG {self.symbol}: qty {self.position.qty:.6f} | entry {self.position.entry_price:.2f} | SL {sl:.2f} | TP1 {tp1:.2f}"
            )
        elif self.is_short:
            sl = self.price + self.atr * k
            r = sl - self.price
            tp1 = self.price - 2 * r  # 2R partial
            self.stop_loss = self.position.qty, sl
            self.take_profit = self.position.qty / 2, tp1
            self._alert(
                f"OPENED SHORT {self.symbol}: qty {self.position.qty:.6f} | entry {self.position.entry_price:.2f} | SL {sl:.2f} | TP1 {tp1:.2f}"
            )

    def on_reduced_position(self, order) -> None:
        self.stop_loss = self.position.qty, self.position.entry_price
        side = "LONG" if self.is_long else "SHORT"
        self._alert(f"{side} reduced → SL moved to BE @ {self.position.entry_price:.2f}")

    def update_position(self) -> None:
        trail_k = float(self.hpv('trail_atr_mult'))
        if self.is_long:
            new_sl = max(self.position.entry_price, self.price - self.atr * trail_k)
            self.stop_loss = self.position.qty, new_sl
            if self.trend == -1:
                self._alert("Trend flipped down → liquidating LONG")
                self.liquidate()
        elif self.is_short:
            new_sl = min(self.position.entry_price, self.price + self.atr * trail_k)
            self.stop_loss = self.position.qty, new_sl
            if self.trend == 1:
                self._alert("Trend flipped up → liquidating SHORT")
                self.liquidate()

    def on_close_position(self, order) -> None:
        reason = "TP" if getattr(order, "is_take_profit", False) else "SL" if getattr(order, "is_stop_loss", False) else "Exit"
        self._alert(f"Position closed: {reason}")

    # ----- Filters -----
    def filters(self) -> list:
        return [self._market_ok]

    def _market_ok(self) -> bool:
        return self.atr > 0

    # ----- Alerts -----
    def _alert(self, msg: str):
        self.log(msg)
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return
        try:
            import requests
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {"chat_id": chat_id, "text": f"[{self.exchange} {self.symbol} {self.timeframe}] {msg}"}
            requests.post(url, data=payload, timeout=2)
        except Exception as e:
            self.log(f"Telegram send failed: {e}")
