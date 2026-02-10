import requests
import pandas as pd
import time

def get_top_coinbase_pairs(limit=200):
    """
    Fetch top trading pairs from Coinbase by 24h USD volume
    Excludes stablecoins
    """
    print("Fetching Coinbase trading pairs...")
    
    try:
        # Get all products from Coinbase
        url = "https://api.exchange.coinbase.com/products"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        products = response.json()
        
        # Filter for USD pairs only and exclude stablecoins
        stablecoins = ['USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'USDP', 'GUSD', 'PAX', 'PYUSD']
        
        usd_pairs = []
        print(f"Processing {len(products)} products...")
        
        for i, product in enumerate(products):
            if product['id'].endswith('-USD') and product['status'] == 'online':
                base_currency = product['id'].replace('-USD', '')
                
                # Skip stablecoins
                if base_currency in stablecoins:
                    continue
                
                # Get 24h stats
                try:
                    stats_url = f"https://api.exchange.coinbase.com/products/{product['id']}/stats"
                    stats_response = requests.get(stats_url, timeout=5)
                    
                    if stats_response.status_code == 200:
                        stats = stats_response.json()
                        
                        # Get volume in base currency and current price
                        volume_base = float(stats.get('volume', 0))
                        last_price = float(stats.get('last', 0))
                        
                        # Calculate USD volume = base volume * price
                        volume_usd = volume_base * last_price
                        
                        if volume_usd > 0:  # Only include pairs with actual volume
                            usd_pairs.append({
                                'symbol': product['id'],
                                'base': base_currency,
                                'volume_24h_usd': volume_usd,
                                'volume_24h_base': volume_base,
                                'price': last_price,
                                'display_name': product.get('display_name', product['id'])
                            })
                    
                    # Rate limiting - be nice to Coinbase API
                    if i % 10 == 0:
                        print(f"  Processed {i}/{len(products)}...")
                    time.sleep(0.15)
                    
                except Exception as e:
                    # Skip pairs that fail
                    continue
        
        # Sort by USD volume
        df = pd.DataFrame(usd_pairs)
        df = df.sort_values('volume_24h_usd', ascending=False)
        
        # Get top N
        top_pairs = df.head(limit)['symbol'].tolist()
        
        print(f"\n✅ Found {len(top_pairs)} pairs with volume data")
        print(f"\nTop 30 by USD volume:")
        print(f"{'Rank':<6}{'Symbol':<15}{'24h USD Volume':<20}{'Price':<15}")
        print("─" * 60)
        for i, (idx, row) in enumerate(df.head(30).iterrows(), 1):
            print(f"{i:<6}{row['symbol']:<15}${row['volume_24h_usd']:>15,.0f}   ${row['price']:>10,.2f}")
        
        return top_pairs
        
    except Exception as e:
        print(f"❌ Error fetching pairs: {e}")
        print("Falling back to default watchlist...")
        
        # Fallback to a curated list if API fails
        return [
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD',
            'ADA-USD', 'AVAX-USD', 'DOT-USD', 'MATIC-USD', 'LINK-USD',
            'UNI-USD', 'ATOM-USD', 'LTC-USD', 'BCH-USD', 'NEAR-USD',
            'ALGO-USD', 'AAVE-USD', 'FTM-USD', 'SAND-USD', 'MANA-USD',
            'AXS-USD', 'GRT-USD', 'FIL-USD', 'ETC-USD', 'VET-USD',
            'ICP-USD', 'APT-USD', 'ARB-USD', 'OP-USD', 'STX-USD',
            'INJ-USD', 'SUI-USD', 'SEI-USD', 'TIA-USD', 'RUNE-USD',
            'IMX-USD', 'RENDER-USD', 'LDO-USD', 'MKR-USD', 'SNX-USD'
        ]

def save_watchlist(pairs, filename='watchlist.txt'):
    """Save watchlist to file"""
    with open(filename, 'w') as f:
        for pair in pairs:
            f.write(f"{pair}\n")
    print(f"\n✅ Saved {len(pairs)} pairs to {filename}")

def load_watchlist(filename='watchlist.txt'):
    """Load watchlist from file"""
    try:
        with open(filename, 'r') as f:
            pairs = [line.strip() for line in f.readlines() if line.strip()]
        print(f"✅ Loaded {len(pairs)} pairs from {filename}")
        return pairs
    except FileNotFoundError:
        print(f"⚠️  {filename} not found, generating new watchlist...")
        return None

if __name__ == '__main__':
    # Generate watchlist
    print("\n" + "="*70)
    print("GENERATING COINBASE WATCHLIST BY USD VOLUME")
    print("="*70 + "\n")
    
    top_pairs = get_top_coinbase_pairs(200)
    save_watchlist(top_pairs)
    
    print(f"\n{'='*70}")
    print(f"DONE - Use this watchlist in your scanner")
    print(f"{'='*70}\n")

