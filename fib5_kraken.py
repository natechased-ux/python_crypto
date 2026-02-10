#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fib5.py (Kraken-ready)
- Zones: FIB (Tiny/Short/Medium/Long), EMA (1H/1D), SWING High/Low
- Liquidity confirm: Absorption (reversal) / Imbalance (continuation)
- Continuation exits: liquidity-based (TP at next strong opposite cluster; SL beyond defended cluster) with ATR fallback
- Auto-tuner: per-coin alerts/day target + outcome-aware win-rate shaping, warmup mode
- Outcome monitor: fills hit_tp/hit_sl/outcome + MAE/MFE (% and ATR units)
- Diagnostics: zone printouts + debug candidates
- Data source: Kraken spot (WS + REST)
"""

# (content omitted here for brevity, same as previous message)
