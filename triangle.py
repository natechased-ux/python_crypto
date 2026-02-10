import pandas as pd
import numpy as np
from scipy.stats import linregress
import requests
from datetime import datetime, timedelta
import time
import json
import os

# ============= TELEGRAM CONFIG =============

TELEGRAM_BOT_TOKEN = "8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw"
TELEGRAM_CHAT_ID = "7967738614"

def send_telegram_alert(message):
    """Send alert via Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    try:
        response = requests.post(url, json=payload)
        return response.status_code == 200
    except Exception as e:
        print(f"Failed to send Telegram: {e}")
        return False

# ============= STATE MANAGEMENT =============

STATE_FILE = 'scanner_state.json'

def load_state():
    """Load previous state to avoid duplicate alerts"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        'last_alerts': {},  # {symbol: timestamp}
        'active_squeezes': {},  # {symbol: {started: timestamp, bars: count}}
        'last_scan': None
    }

def save_state(state):
    """Save state to disk"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

def should_alert(state, symbol, pattern_type):
    """Avoid duplicate alerts within 2 hours"""
    key = f"{symbol}_{pattern_type}"
    if key in state['last_alerts']:
        last_time = datetime.fromisoformat(state['last_alerts'][key])
        if datetime.now() - last_time < timedelta(hours=2):
            return False
    return True

def mark_alerted(state, symbol, pattern_type):
    """Mark that we've alerted for this"""
    key = f"{symbol}_{pattern_type}"
    state['last_alerts'][key] = datetime.now().isoformat()

# ============= DATA FETCHING =============

def get_coinbase_data(symbol='BTC-USD', granularity=900, bars=200):
    """
    Fetch OHLCV data from Coinbase
    granularity=900 for 15min candles
    """
    url = f'https://api.exchange.coinbase.com/products/{symbol}/candles'
    
    try:
        # Fetch recent data without date params (simpler, more reliable)
        response = requests.get(url, params={'granularity': granularity}, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data or len(data) == 0:
            return None
        
        df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values('time').reset_index(drop=True)
        
        for col in ['low', 'high', 'open', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        return df.tail(bars).reset_index(drop=True)
    
    except Exception as e:
        return None

# ============= PATTERN DETECTION =============

def find_fresh_liquidity_sweeps(df, lookback=10, wick_ratio=2.0, check_last_n=2):
    """
    Find liquidity sweeps in last N candles
    STRICTER CRITERIA - only strong wicks
    """
    if len(df) < lookback + 5:
        return []
    
    signals = []
    
    for i in range(len(df) - check_last_n, len(df)):
        if i < lookback + 2:
            continue
            
        recent_low = df['low'].iloc[i-lookback:i-1].min()
        recent_high = df['high'].iloc[i-lookback:i-1].max()
        
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        body_size = abs(current['close'] - current['open'])
        if body_size == 0:
            body_size = 0.0001  # Avoid division by zero
            
        lower_wick = min(current['open'], current['close']) - current['low']
        upper_wick = current['high'] - max(current['open'], current['close'])
        
        candle_range = current['high'] - current['low']
        
        candles_ago = len(df) - 1 - i
        
        # STRICTER CONDITIONS for LONG sweeps
        if current['low'] < recent_low:
            # Calculate sweep depth
            sweep_depth = ((recent_low - current['low']) / recent_low) * 100
            
            # Require:
            # 1. Strong wick (at least 2x body size)
            # 2. Wick is significant portion of candle (at least 40%)
            # 3. Price closed back ABOVE the swept level
            # 4. Noticeable sweep depth (at least 0.1%)
            
            wick_to_body = lower_wick / body_size
            wick_to_range = lower_wick / candle_range if candle_range > 0 else 0
            
            if (wick_to_body >= wick_ratio and 
                wick_to_range >= 0.4 and
                current['close'] > recent_low and
                current['close'] > prev['close'] and
                sweep_depth >= 0.1):
                
                signals.append({
                    'direction': 'LONG',
                    'swept_level': recent_low,
                    'entry_price': current['close'],
                    'candles_ago': candles_ago,
                    'sweep_depth_pct': sweep_depth,
                    'wick_strength': wick_to_body,
                    'wick_pct_of_candle': wick_to_range * 100,
                    'time': current['time']
                })
        
        # STRICTER CONDITIONS for SHORT sweeps
        if current['high'] > recent_high:
            sweep_depth = ((current['high'] - recent_high) / recent_high) * 100
            
            wick_to_body = upper_wick / body_size
            wick_to_range = upper_wick / candle_range if candle_range > 0 else 0
            
            if (wick_to_body >= wick_ratio and 
                wick_to_range >= 0.4 and
                current['close'] < recent_high and
                current['close'] < prev['close'] and
                sweep_depth >= 0.1):
                
                signals.append({
                    'direction': 'SHORT',
                    'swept_level': recent_high,
                    'entry_price': current['close'],
                    'candles_ago': candles_ago,
                    'sweep_depth_pct': sweep_depth,
                    'wick_strength': wick_to_body,
                    'wick_pct_of_candle': wick_to_range * 100,
                    'time': current['time']
                })
    
    return signals

def check_squeeze_status(df, bb_period=20, kc_period=20):
    """
    Check if currently in a squeeze
    """
    if len(df) < max(bb_period, kc_period) + 5:
        return None
    
    df = df.copy()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(bb_period).mean()
    df['bb_std'] = df['close'].rolling(bb_period).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    
    # Keltner Channels
    df['kc_middle'] = df['close'].rolling(kc_period).mean()
    df['tr'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], 
                     abs(x['high'] - x['close']), 
                     abs(x['low'] - x['close'])), axis=1)
    df['atr'] = df['tr'].rolling(kc_period).mean()
    df['kc_upper'] = df['kc_middle'] + 1.5 * df['atr']
    df['kc_lower'] = df['kc_middle'] - 1.5 * df['atr']
    
    # Squeeze condition
    df['squeeze'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
    
    current = df.iloc[-1]
    recent = df.tail(20)
    squeeze_bars = recent['squeeze'].sum()
    
    if current['squeeze'] and squeeze_bars >= 3:
        return {
            'active': True,
            'bars': int(squeeze_bars),
            'bb_width': float(current['bb_upper'] - current['bb_lower'])
        }
    
    return None

def check_trend(df, period=50):
    """
    Simple trend using MA
    """
    if len(df) < period:
        return 'unknown'
    
    ma = df['close'].rolling(period).mean().iloc[-1]
    current = df['close'].iloc[-1]
    
    if current > ma * 1.02:
        return 'uptrend'
    elif current < ma * 0.98:
        return 'downtrend'
    else:
        return 'neutral'

def calculate_confidence(sweep, squeeze, trend):
    """
    Score the setup from 0-100
    UPDATED - stronger emphasis on wick quality
    """
    score = 40  # Lower base score (was 50)
    
    # Wick strength (max +30, was +20)
    if sweep['wick_strength'] >= 5:
        score += 30
    elif sweep['wick_strength'] >= 4:
        score += 25
    elif sweep['wick_strength'] >= 3:
        score += 20
    elif sweep['wick_strength'] >= 2:
        score += 10
    else:
        score += 5
    
    # Wick percentage of candle (max +10, NEW)
    if sweep['wick_pct_of_candle'] >= 60:
        score += 10
    elif sweep['wick_pct_of_candle'] >= 50:
        score += 7
    elif sweep['wick_pct_of_candle'] >= 40:
        score += 5
    
    # Sweep depth (max +10, NEW)
    if sweep['sweep_depth_pct'] >= 0.5:
        score += 10
    elif sweep['sweep_depth_pct'] >= 0.3:
        score += 7
    elif sweep['sweep_depth_pct'] >= 0.15:
        score += 5
    
    # Squeeze active (+20, was +25)
    if squeeze and squeeze['active']:
        score += 20
        # Longer squeeze bonus
        if squeeze['bars'] > 10:
            score += 5
    
    # Trend alignment (+15)
    if (sweep['direction'] == 'LONG' and trend == 'uptrend') or \
       (sweep['direction'] == 'SHORT' and trend == 'downtrend'):
        score += 15
    
    # Recency bonus
    if sweep['candles_ago'] == 0:
        score += 10
    
    return min(score, 100)

# ============= SIGNAL SCORING =============

def find_fresh_liquidity_sweeps(df, lookback=10, wick_ratio=2.0, check_last_n=2):
    """
    Find liquidity sweeps in last N candles
    STRICTER CRITERIA - only strong wicks
    """
    if len(df) < lookback + 5:
        return []
    
    signals = []
    
    for i in range(len(df) - check_last_n, len(df)):
        if i < lookback + 2:
            continue
            
        recent_low = df['low'].iloc[i-lookback:i-1].min()
        recent_high = df['high'].iloc[i-lookback:i-1].max()
        
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        body_size = abs(current['close'] - current['open'])
        if body_size == 0:
            body_size = 0.0001  # Avoid division by zero
            
        lower_wick = min(current['open'], current['close']) - current['low']
        upper_wick = current['high'] - max(current['open'], current['close'])
        
        candle_range = current['high'] - current['low']
        
        candles_ago = len(df) - 1 - i
        
        # STRICTER CONDITIONS for LONG sweeps
        if current['low'] < recent_low:
            # Calculate sweep depth
            sweep_depth = ((recent_low - current['low']) / recent_low) * 100
            
            # Require:
            # 1. Strong wick (at least 2x body size)
            # 2. Wick is significant portion of candle (at least 40%)
            # 3. Price closed back ABOVE the swept level
            # 4. Noticeable sweep depth (at least 0.1%)
            
            wick_to_body = lower_wick / body_size
            wick_to_range = lower_wick / candle_range if candle_range > 0 else 0
            
            if (wick_to_body >= wick_ratio and 
                wick_to_range >= 0.4 and
                current['close'] > recent_low and
                current['close'] > prev['close'] and
                sweep_depth >= 0.1):
                
                signals.append({
                    'direction': 'LONG',
                    'swept_level': recent_low,
                    'entry_price': current['close'],
                    'candles_ago': candles_ago,
                    'sweep_depth_pct': sweep_depth,
                    'wick_strength': wick_to_body,
                    'wick_pct_of_candle': wick_to_range * 100,
                    'time': current['time']
                })
        
        # STRICTER CONDITIONS for SHORT sweeps
        if current['high'] > recent_high:
            sweep_depth = ((current['high'] - recent_high) / recent_high) * 100
            
            wick_to_body = upper_wick / body_size
            wick_to_range = upper_wick / candle_range if candle_range > 0 else 0
            
            if (wick_to_body >= wick_ratio and 
                wick_to_range >= 0.4 and
                current['close'] < recent_high and
                current['close'] < prev['close'] and
                sweep_depth >= 0.1):
                
                signals.append({
                    'direction': 'SHORT',
                    'swept_level': recent_high,
                    'entry_price': current['close'],
                    'candles_ago': candles_ago,
                    'sweep_depth_pct': sweep_depth,
                    'wick_strength': wick_to_body,
                    'wick_pct_of_candle': wick_to_range * 100,
                    'time': current['time']
                })
    
    return signals

# ============= MAIN SCANNER =============

def scan_single_coin(symbol, state, granularity=900):
    """
    Scan one coin for signals
    """
    df = get_coinbase_data(symbol, granularity, bars=200)
    
    if df is None or len(df) < 50:
        return None
    
    # Find fresh sweeps
    sweeps = find_fresh_liquidity_sweeps(df, lookback=10, check_last_n=2)
    
    if len(sweeps) == 0:
        return None
    
    # Check confluence factors
    squeeze = check_squeeze_status(df)
    trend = check_trend(df)
    
    # Update squeeze state
    if squeeze and squeeze['active']:
        if symbol not in state['active_squeezes']:
            state['active_squeezes'][symbol] = {
                'started': datetime.now().isoformat(),
                'bars': squeeze['bars']
            }
        else:
            state['active_squeezes'][symbol]['bars'] = squeeze['bars']
    else:
        if symbol in state['active_squeezes']:
            del state['active_squeezes'][symbol]
    
    # Analyze each sweep
    signals = []
    for sweep in sweeps:
        confidence = calculate_confidence(sweep, squeeze, trend)
        
        # Only alert on high-confidence setups
        if confidence >= 70:
            # Calculate stop loss and target
            if sweep['direction'] == 'LONG':
                stop_loss = sweep['swept_level'] * 0.995  # 0.5% below sweep
                target = sweep['entry_price'] * 1.02  # 2% profit target
            else:
                stop_loss = sweep['swept_level'] * 1.005
                target = sweep['entry_price'] * 0.98
            
            signal = {
                'symbol': symbol,
                'direction': sweep['direction'],
                'confidence': confidence,
                'entry_price': sweep['entry_price'],
                'stop_loss': stop_loss,
                'target': target,
                'swept_level': sweep['swept_level'],
                'candles_ago': sweep['candles_ago'],
                'squeeze_active': squeeze is not None and squeeze['active'],
                'squeeze_bars': squeeze['bars'] if squeeze else 0,
                'trend': trend,
                'wick_strength': sweep['wick_strength'],
                'time': sweep['time']
            }
            signals.append(signal)
    
    return signals if len(signals) > 0 else None

def format_alert_message(signal):
    """
    Format Telegram alert message
    """
    emoji = "🟢" if signal['direction'] == 'LONG' else "🔴"
    
    msg = f"{emoji} <b>{signal['symbol']}</b> - {signal['direction']} SIGNAL\n\n"
    msg += f"⭐ <b>Confidence: {signal['confidence']}%</b>\n\n"
    msg += f"💰 Entry: ${signal['entry_price']:.6f}\n"
    msg += f"🛑 Stop: ${signal['stop_loss']:.6f} ({abs((signal['stop_loss'] - signal['entry_price']) / signal['entry_price'] * 100):.1f}%)\n"
    msg += f"🎯 Target: ${signal['target']:.6f} ({abs((signal['target'] - signal['entry_price']) / signal['entry_price'] * 100):.1f}%)\n\n"
    
    msg += f"📊 <b>Sweep Quality:</b>\n"
    msg += f"  • Swept: ${signal['swept_level']:.6f}\n"
    msg += f"  • Wick: {signal['wick_strength']:.6f}x body ({signal['wick_pct_of_candle']:.0f}% of candle)\n"
    msg += f"  • Sweep depth: {signal.get('sweep_depth_pct', 0):.6f}%\n"
    
    if signal['squeeze_active']:
        msg += f"  • 🔥 Squeeze: {signal['squeeze_bars']} bars\n"
    
    msg += f"  • Trend: {signal['trend']}\n"
    
    if signal['candles_ago'] == 0:
        msg += f"\n⏰ <b>FRESH - Just happened!</b>"
    else:
        msg += f"\n⏰ {signal['candles_ago']} candle(s) ago"
    
    msg += f"\n🕐 {signal['time'].strftime('%Y-%m-%d %H:%M')}"
    
    return msg

def scan_watchlist(symbols, state, granularity=900):
    """
    Scan entire watchlist
    """
    signals_found = []
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning {len(symbols)} coins...")
    
    for i, symbol in enumerate(symbols, 1):
        try:
            signals = scan_single_coin(symbol, state, granularity)
            
            if signals:
                for signal in signals:
                    # Check if we should alert (avoid duplicates)
                    if should_alert(state, symbol, signal['direction']):
                        print(f"  🚨 ALERT: {symbol} {signal['direction']} ({signal['confidence']}%)")
                        
                        # Send Telegram alert
                        msg = format_alert_message(signal)
                        if send_telegram_alert(msg):
                            mark_alerted(state, symbol, signal['direction'])
                            signals_found.append(signal)
                        else:
                            print(f"  ⚠️  Failed to send Telegram alert")
            
            # Rate limiting
            time.sleep(0.3)
            
        except Exception as e:
            print(f"  ❌ Error scanning {symbol}: {e}")
    
    return signals_found

# ============= CONTINUOUS MONITORING =============

def run_continuous_scanner(watchlist, scan_interval_minutes=15, granularity=900):
    """
    Run scanner continuously
    """
    state = load_state()
    
    print(f"\n{'='*70}")
    print(f"🤖 CRYPTO LIQUIDITY SWEEP SCANNER - STARTED")
    print(f"{'='*70}")
    print(f"📱 Telegram alerts enabled")
    print(f"⏱️  Scanning every {scan_interval_minutes} minutes")
    print(f"📊 Timeframe: 15-minute candles")
    print(f"🎯 Watching: {len(watchlist)} coins")
    print(f"{'='*70}\n")
    
    # Send startup notification
    send_telegram_alert(
        f"🤖 <b>Scanner Started</b>\n\n"
        f"Monitoring {len(watchlist)} coins\n"
        f"Scan interval: {scan_interval_minutes} min\n"
        f"Looking for liquidity sweeps + confluence"
    )
    
    scan_count = 0
    
    while True:
        try:
            scan_count += 1
            print(f"\n{'─'*70}")
            print(f"Scan #{scan_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'─'*70}")
            
            signals = scan_watchlist(watchlist, state, granularity)
            
            if len(signals) > 0:
                print(f"\n✅ Found {len(signals)} signal(s) - Alerts sent!")
            else:
                print(f"\n⚪ No new signals")
            
            # Show active squeezes
            if len(state['active_squeezes']) > 0:
                print(f"\n🔒 Active squeezes: {', '.join(state['active_squeezes'].keys())}")
            
            # Save state
            state['last_scan'] = datetime.now().isoformat()
            save_state(state)
            
            # Wait for next scan
            print(f"\n💤 Sleeping for {scan_interval_minutes} minutes...")
            time.sleep(scan_interval_minutes * 60)
            
        except KeyboardInterrupt:
            print(f"\n\n{'='*70}")
            print(f"🛑 Scanner stopped by user")
            print(f"{'='*70}\n")
            send_telegram_alert("🛑 <b>Scanner Stopped</b>")
            break
        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

# ============= USAGE =============

if __name__ == '__main__':
    
    # Your watchlist
    WATCHLIST = [
    "00-USD", "1INCH-USD", "ACH-USD", "ADA-USD", "AERGO-USD", "AERO-USD",
    "AKT-USD", "ALEO-USD", "ALGO-USD", "AMP-USD", "ANKR-USD", "APE-USD", "API3-USD",
    "APR-USD", "APT-USD", "ARB-USD", "ASM-USD", "ASTER-USD", "ATH-USD", "ATOM-USD",
    "AUCTION-USD", "AURORA-USD", "AVAX-USD", "AVNT-USD", "AXL-USD", "AXS-USD",
    "BAL-USD", "BAT-USD", "BERA-USD", "BCH-USD", "BIRB-USD", "BNB-USD", "BNKR-USD",
    "BONK-USD", "BREV-USD", "BTC-USD", "CBETH-USD", "CHZ-USD", "CLANKER-USD",
    "C98-USD", "COMP-USD", "CRV-USD", "CRO-USD", "CTSI-USD", "CVX-USD", "DASH-USD",
    "DEXT-USD", "DOGE-USD", "DOT-USD", "EDGE-USD", "EIGEN-USD", "ELSA-USD", "ENA-USD",
    "ENS-USD", "ETC-USD", "ETH-USD", "ETHFI-USD", "FARTCOIN-USD", "FET-USD",
    "FIGHT-USD", "FIL-USD", "FLR-USD", "FLUID-USD", "GFI-USD", "GHST-USD", "GIGA-USD",
    "GLM-USD", "GRT-USD", "HBAR-USD", "HFT-USD", "HNT-USD", "IMU-USD", "IMX-USD",
    "INJ-USD", "INX-USD", "IP-USD", "IOTX-USD", "IRYS-USD", "JASMY-USD", "JITOSOL-USD",
    "JTO-USD", "JUPITER-USD", "KAITO-USD", "KERNEL-USD", "KITE-USD", "KMNO-USD",
    "KSM-USD", "KTA-USD", "LCX-USD", "LDO-USD", "LIGHTER-USD", "LINK-USD", "LPT-USD",
    "LRC-USD", "LTC-USD", "MAGIC-USD", "MANA-USD", "MANTLE-USD", "MATH-USD",
    "MET-USD", "MINA-USD", "MOG-USD", "MON-USD", "MOODENG-USD", "MORPHO-USD",
    "NEAR-USD", "NKN-USD", "NOICE-USD", "NMR-USD", "OGN-USD", "ONDO-USD", "OP-USD",
    "ORCA-USD", "PAXG-USD", "PEPE-USD", "PENDLE-USD", "PENGU-USD", "PERP-USD",
    "PIRATE-USD", "PLUME-USD", "POPCAT-USD", "POL-USD", "PRIME-USD", "PROMPT-USD",
    "PROVE-USD", "PUMP-USD", "PYTH-USD", "QNT-USD", "RED-USD", "RENDER-USD",
    "REZ-USD", "RLS-USD", "ROSE-USD", "RSR-USD", "SAFE-USD", "SAND-USD",
    "SAPIEN-USD", "SEI-USD", "SENT-USD", "SHIB-USD", "SKL-USD", "SKR-USD", "SKY-USD",
    "SNX-USD", "SOL-USD", "SPK-USD", "SPX-USD", "STX-USD", "STRK-USD", "SUPER-USD",
    "SWFTC-USD", "SYRUP-USD", "TAO-USD", "THQ-USD", "TIA-USD", "TON-USD",
    "TOSHI-USD", "TRAC-USD", "TREE-USD", "TRB-USD", "TRUMP-USD", "TRUST-USD",
    "TROLL-USD", "TURBO-USD", "UNI-USD", "USELESS-USD", "USD1-USD", "VARA-USD",
    "VET-USD", "VOXEL-USD", "VVV-USD", "W-USD", "WET-USD", "WIF-USD", "WLD-USD",
    "WLFI-USD", "XCN-USD", "XLM-USD", "XPL-USD", "XRP-USD", "XTZ-USD", "XYO-USD",
    "YFI-USD", "ZAMA-USD", "ZEC-USD", "ZEN-USD", "ZK-USD", "ZKC-USD", "ZKP-USD",
    "ZORA-USD", "ZRO-USD", "ZRX-USD"

    ]
    
    # Start continuous scanner
    run_continuous_scanner(
        watchlist=WATCHLIST,
        scan_interval_minutes=15,
        granularity=900  # 15-minute candles
    )
