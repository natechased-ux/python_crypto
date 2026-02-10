import requests
import pandas as pd
import time

# === Configuration ===
SYMBOL = 'xrpusd'
ORDER_BOOK_URL = f'https://www.bitstamp.net/api/v2/order_book/{SYMBOL}/'
STATIC_THRESHOLD = 400000  # Minimum volume for a wall (adjust as needed)
PRICE_BUCKET = 0.01  # Group prices into buckets
PROXIMITY_LIMIT = 20.0  # Max distance from the current price in percentage
POLL_INTERVAL = 30  # Seconds

def fetch_order_book():
    try:
        response = requests.get(ORDER_BOOK_URL)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching order book: {e}")
        return None

def aggregate_order_book(order_book, bucket_size):
    # Create dataframes for bids and asks
    bids = pd.DataFrame(order_book['bids'], columns=['price', 'volume'], dtype=float)
    asks = pd.DataFrame(order_book['asks'], columns=['price', 'volume'], dtype=float)
    
    # Bucket prices
    bids['bucket'] = (bids['price'] // bucket_size) * bucket_size
    asks['bucket'] = (asks['price'] // bucket_size) * bucket_size
    
    # Sum volumes in each bucket
    bid_walls = bids.groupby('bucket')['volume'].sum().reset_index()
    ask_walls = asks.groupby('bucket')['volume'].sum().reset_index()
    
    return bid_walls, ask_walls

def detect_strong_price_walls(order_book, static_threshold, proximity_limit, bucket_size):
    bid_walls, ask_walls = aggregate_order_book(order_book, bucket_size)
    
    # Calculate the current price
    current_price = (float(order_book['bids'][0][0]) + float(order_book['asks'][0][0])) / 2
    
    # Filter for strong walls
    bid_walls = bid_walls[bid_walls['volume'] > static_threshold]
    ask_walls = ask_walls[ask_walls['volume'] > static_threshold]
    
    # Filter walls by proximity to current price
    bid_walls = bid_walls[abs(bid_walls['bucket'] - current_price) / current_price * 100 <= proximity_limit]
    ask_walls = ask_walls[abs(ask_walls['bucket'] - current_price) / current_price * 100 <= proximity_limit]
    
    return bid_walls, ask_walls, current_price

def main():
    while True:
        order_book = fetch_order_book()
        if order_book:
            bid_walls, ask_walls, current_price = detect_strong_price_walls(
                order_book, STATIC_THRESHOLD, PROXIMITY_LIMIT, PRICE_BUCKET
            )
            
            if not bid_walls.empty:
                print(f"💚 Strong Buy Walls near {current_price:.2f}:")
                print(bid_walls)
            else:
                print("No strong buy walls detected.")
            
            if not ask_walls.empty:
                print(f"❤️ Strong Sell Walls near {current_price:.2f}:")
                print(ask_walls)
            else:
                print("No strong sell walls detected.")
        
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
