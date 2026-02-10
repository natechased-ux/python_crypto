
from dataclasses import dataclass
from typing import Optional

@dataclass
class TradeState:
    id: str
    symbol: str
    side: str
    entry: float
    sl: float
    tp: float
    score: int
    risk_fraction: float
    status: str = "OPEN"
    outcome: Optional[str] = None
    r_multiple: float = 0.0

    def is_closed(self) -> bool:
        return self.status.startswith("CLOSED")

    def check_exits(self, last_price: float) -> bool:
        # TODO: implement TP/SL checks in live mode if desired
        return False
