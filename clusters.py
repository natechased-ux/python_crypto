
from typing import Dict, Any, Tuple
import math

class ClusterService:
    def __init__(self, cfg):
        self.cfg = cfg

    def _bin_price(self, price: float, bin_width: float) -> float:
        return round(price / bin_width) * bin_width

    def detect(self, symbol: str, mid_price: float, orderbook: Dict[str, Any]) -> Dict[str, Any]:
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return {"support_nearby": False, "resistance_nearby": False, "conflict": False}
        bw_pct = float(self.cfg["clusters"]["bin_width_pct_of_price"])/100.0
        bin_width = max(mid_price * bw_pct, 1e-8)

        def agg(side_rows):
            bins = {}
            for p, size, *_ in side_rows:
                try:
                    price = float(p); sz = float(size)
                except Exception:
                    continue
                bp = self._bin_price(price, bin_width)
                bins[bp] = bins.get(bp, 0.0) + sz
            return bins

        bids = agg(orderbook.get('bids', []))
        asks = agg(orderbook.get('asks', []))

        # Compute percentile threshold
        all_sizes = list(bids.values()) + list(asks.values())
        if not all_sizes:
            return {"support_nearby": False, "resistance_nearby": False, "conflict": False}
        perc = float(self.cfg["clusters"]["percentile_filter"])
        cutoff = sorted(all_sizes)[max(0, int(len(all_sizes)*perc/100)-1)]

        # Keep only whale clusters
        bid_clusters = [(bp, sz) for bp, sz in bids.items() if sz >= cutoff]
        ask_clusters = [(ap, sz) for ap, sz in asks.items() if sz >= cutoff]

        # Find nearest distances
        def nearest(clusters, above: bool):
            best = None
            best_dist = None
            for price, sz in clusters:
                if above and price < mid_price: 
                    continue
                if (not above) and price > mid_price:
                    continue
                dist_pct = abs(price - mid_price) / mid_price * 100.0
                if best_dist is None or dist_pct < best_dist:
                    best = (price, sz, dist_pct)
                    best_dist = dist_pct
            return best

        sup = nearest(bid_clusters, above=False)   # below price
        res = nearest(ask_clusters, above=True)    # above price

        dmin = float(self.cfg["clusters"]["distance_pct_min"])
        dmax = float(self.cfg["clusters"]["distance_pct_max"])
        support_near = bool(sup and dmin <= sup[2] <= dmax)
        resistance_near = bool(res and dmin <= res[2] <= dmax)

        return {
            "support_nearby": support_near,
            "resistance_nearby": resistance_near,
            "support_dist_pct": sup[2] if sup else None,
            "resistance_dist_pct": res[2] if res else None,
            "cutoff_size": cutoff,
        }
