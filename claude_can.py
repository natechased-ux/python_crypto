"""
Coinbase Trading Scanner
Identifies top 3 long and short candidates based on technical indicators
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

class CoinbaseScanner:
    def __init__(self):
        self.base_url = "https://api.exchange.coinbase.com"
        
    def get_trading_pairs(self):
        """Get all USD trading pairs"""
        try:
            response = requests.get(f"{self.base_url}/products")
            products = response.json()
            
            # Filter for USD pairs that are actively trading
            usd_pairs = [
                p['id'] for p in products 
                if p['quote_currency'] == 'USD' 
                and p['status'] == 'online'
                and not p['trading_disabled']
            ]
            return usd_pairs
        except Exception as e:
            print(f"Error fetching trading pairs: {e}")
            return []
    
    def get_candles(self, pair, granularity=3600):
        """
        Get historical candles for a trading pair
        granularity: in seconds (3600 = 1 hour)
        """
        try:
            # Get last 300 candles (max allowed by API)
            url = f"{self.base_url}/products/{pair}/candles"
            params = {
                'granularity': granularity
            }
            
            response = requests.get(url, params=params)
            time.sleep(0.35)  # Rate limiting (3 requests per second)
            
            if response.status_code == 200:
                candles = response.json()
                # Convert to DataFrame [time, low, high, open, close, volume]
                df = pd.DataFrame(candles, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df = df.sort_values('time')
                return df
            else:
                return None
        except Exception as e:
            print(f"Error fetching candles for {pair}: {e}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        deltas = prices.diff()
        gain = (deltas.where(deltas > 0, 0)).rolling(window=period).mean()
        loss = (-deltas.where(deltas < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators"""
        if df is None or len(df) < 30:
            return None
        
        df = df.copy()
        
        # Price change
        df['price_change_pct'] = ((df['close'] - df['close'].shift(24)) / df['close'].shift(24)) * 100
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'] = self.calculate_macd(df['close'])
        
        # Volume trend
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_trend'] = (df['volume'] / df['volume_sma']) - 1
        
        return df
    
    def score_pair(self, df):
        """
        Score a trading pair for long/short potential
        Positive score = bullish (long candidate)
        Negative score = bearish (short candidate)
        """
        if df is None or len(df) < 50:
            return None
        
        latest = df.iloc[-1]
        score = 0
        
        # RSI scoring (oversold = bullish, overbought = bearish)
        if latest['rsi'] < 30:
            score += 2  # Oversold - bullish
        elif latest['rsi'] > 70:
            score -= 2  # Overbought - bearish
        elif latest['rsi'] < 40:
            score += 1
        elif latest['rsi'] > 60:
            score -= 1
        
        # MACD scoring
        if latest['macd'] > latest['macd_signal']:
            score += 1.5  # Bullish crossover
        else:
            score -= 1.5  # Bearish crossover
        
        # Moving average trend
        if latest['close'] > latest['sma_20'] > latest['sma_50']:
            score += 2  # Strong uptrend
        elif latest['close'] < latest['sma_20'] < latest['sma_50']:
            score -= 2  # Strong downtrend
        elif latest['close'] > latest['sma_20']:
            score += 1
        else:
            score -= 1
        
        # Price momentum
        if latest['price_change_pct'] > 5:
            score += 1
        elif latest['price_change_pct'] < -5:
            score -= 1
        
        # Volume confirmation
        if latest['volume_trend'] > 0.3:
            score += 1  # High volume confirms trend
        
        return {
            'score': score,
            'price': latest['close'],
            'rsi': latest['rsi'],
            'price_change_24h': latest['price_change_pct'],
            'volume_trend': latest['volume_trend']
        }
    
    def scan_market(self, limit=None):
        """Scan the market and return scored pairs"""
        print("Fetching trading pairs...")
        pairs = self.get_trading_pairs()
        
        if limit:
            pairs = pairs[:limit]
        
        print(f"Analyzing {len(pairs)} pairs...\n")
        
        results = []
        for i, pair in enumerate(pairs):
            print(f"Analyzing {pair} ({i+1}/{len(pairs)})")
            
            df = self.get_candles(pair)
            df = self.calculate_indicators(df)
            metrics = self.score_pair(df)
            
            if metrics:
                results.append({
                    'pair': pair,
                    **metrics
                })
        
        return pd.DataFrame(results)
    
    def get_top_candidates(self, df, n=3):
        """Get top n long and short candidates"""
        if df.empty:
            return None, None
        
        # Sort by score
        df_sorted = df.sort_values('score', ascending=False)
        
        # Top long candidates (highest positive scores)
        long_candidates = df_sorted.head(n)
        
        # Top short candidates (lowest negative scores)
        short_candidates = df_sorted.tail(n).sort_values('score')
        
        return long_candidates, short_candidates


def main():
    print("="*60)
    print("COINBASE TRADING SCANNER")
    print("="*60)
    print()
    
    scanner = CoinbaseScanner()
    
    # Scan market (limit to 20 pairs for faster execution, remove limit for full scan)
    results = scanner.scan_market(limit=20)
    
    if results.empty:
        print("No results found.")
        return
    
    # Get top candidates
    long_candidates, short_candidates = scanner.get_top_candidates(results, n=3)
    
    print("\n" + "="*60)
    print("TOP 3 LONG CANDIDATES (BULLISH)")
    print("="*60)
    print(long_candidates[['pair', 'score', 'price', 'rsi', 'price_change_24h']].to_string(index=False))
    
    print("\n" + "="*60)
    print("TOP 3 SHORT CANDIDATES (BEARISH)")
    print("="*60)
    print(short_candidates[['pair', 'score', 'price', 'rsi', 'price_change_24h']].to_string(index=False))
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nNote: This is for educational purposes only.")
    print("Always do your own research before trading.")


if __name__ == "__main__":
    main()
