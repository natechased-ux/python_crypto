
from typing import Dict, Any

class ConfluenceScorer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.weights = cfg["confluence"]["weights"]

    def score(self, ctx: Dict[str, Any]) -> int:
        total = 0
        for key, w in self.weights.items():
            if ctx.get(key):
                total += int(w)
        return int(total)

class PositionSizer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.map = {int(k): float(v) for k, v in cfg["sizing"]["score_to_scale"].items()}

    def risk_fraction(self, score: int) -> float:
        base = float(self.cfg["sizing"]["base_risk_fraction"])
        best_mult = 1.0
        for k, mult in sorted(self.map.items()):
            if score >= k:
                best_mult = mult
        return base * best_mult
