import pandas as pd
import numpy as np

# === Calculate VWAP ===
def calculate_vwap(df):
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    return df

# === Calculate Order Book Imbalance ===
def calculate_imbalance(df):
    df['imbalance'] = 100 * df['bids_volume'] / (df['bids_volume'] + df['asks_volume'])
    return df

# === Strategy Logic ===
def strategy_logic(df):
    # Ensure columns are numeric
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['vwap'] = pd.to_numeric(df['vwap'], errors='coerce')
    df['imbalance'] = pd.to_numeric(df['imbalance'], errors='coerce')
    
    # Initialize signals list
    signals = []
    for i in range(len(df)):
        if df['close'].iloc[i] > df['vwap'].iloc[i] and df['imbalance'].iloc[i] > 60:
            signals.append("BUY")
        elif df['close'].iloc[i] < df['vwap'].iloc[i] and df['imbalance'].iloc[i] < 40:
            signals.append("SELL")
        else:
            signals.append("HOLD")
    
    # Add signals to DataFrame
    df['signal'] = signals
    return df

# === Main Execution ===
def main():
    # Load historical data (replace with your actual file path)
    file_path = "C:\Users\natec\xrp_bitstamp_data.csv"
    df = pd.read_csv(file_path)
    # Ensure required columns exist
    required_columns = {'close', 'volume', 'bids_volume', 'asks_volume'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Input data must include columns: {required_columns}")

    # Calculate VWAP and imbalance
    df = calculate_vwap(df)
    df = calculate_imbalance(df)

    # Apply strategy logic
    df = strategy_logic(df)

    # Save results
    output_path = '/mnt/data/xrp_strategy_results.csv'
    df.to_csv(output_path, index=False)
    print(f"Strategy results saved to {output_path}")

# === Run Script ===
if __name__ == "__main__":
    main()
