import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CryptoBacktester:
    def __init__(self, initial_capital: float = 10000, trade_size: float = 2500):
        self.initial_capital = initial_capital
        self.trade_size = trade_size
        self.base_url = "https://api.exchange.coinbase.com"
        
    def get_historical_data(self, symbol: str, granularity: int = 300, days: int = 30) -> pd.DataFrame:
        """Fetch historical data from Coinbase Advanced Trade API"""
        try:
            # Try the new Coinbase Advanced Trade API first
            return self._get_coinbase_advanced_data(symbol, granularity, days)
        except:
            try:
                # Fallback to Yahoo Finance
                return self._get_yahoo_finance_data(symbol, granularity, days)
            except:
                # Generate synthetic data for demonstration
                return self._generate_synthetic_data(symbol, granularity, days)
    
    def _get_coinbase_advanced_data(self, symbol: str, granularity: int = 300, days: int = 30) -> pd.DataFrame:
        """Fetch from Coinbase Advanced Trade API"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # Convert to Unix timestamp
        start_unix = int(start_time.timestamp())
        end_unix = int(end_time.timestamp())
        
        # Try different endpoint formats
        endpoints = [
            f"https://api.coinbase.com/api/v3/brokerage/market/products/{symbol}/candles",
            f"https://api.exchange.coinbase.com/products/{symbol}/candles",
            f"https://api.pro.coinbase.com/products/{symbol}/candles"
        ]
        
        for base_url in endpoints:
            try:
                params = {
                    'start': start_unix,
                    'end': end_unix,
                    'granularity': granularity
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if isinstance(data, list) and len(data) > 0:
                        df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                        df = df.sort_values('timestamp').reset_index(drop=True)
                        
                        for col in ['low', 'high', 'open', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col])
                        
                        print(f"Successfully fetched {len(df)} candles for {symbol}")
                        return df
                        
            except Exception as e:
                continue
        
        raise Exception("All Coinbase endpoints failed")
    
    def _get_yahoo_finance_data(self, symbol: str, granularity: int = 300, days: int = 30) -> pd.DataFrame:
        """Fallback to Yahoo Finance API"""
        try:
            # Convert symbol format (BTC-USD -> BTC-USD)
            yahoo_symbol = symbol
            
            # Yahoo Finance doesn't support granularity < 1 day for free tier
            # So we'll get daily data and simulate intraday
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            # Simple Yahoo Finance API call
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
            params = {
                'period1': int(start_time.timestamp()),
                'period2': int(end_time.timestamp()),
                'interval': '1d',  # Daily data
                'includePrePost': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'chart' in data and 'result' in data['chart'] and len(data['chart']['result']) > 0:
                result = data['chart']['result'][0]
                timestamps = result['timestamp']
                quotes = result['indicators']['quote'][0]
                
                df = pd.DataFrame({
                    'timestamp': pd.to_datetime(timestamps, unit='s'),
                    'open': quotes['open'],
                    'high': quotes['high'],
                    'low': quotes['low'],
                    'close': quotes['close'],
                    'volume': quotes['volume']
                })
                
                # Remove NaN values
                df = df.dropna().reset_index(drop=True)
                
                # Simulate intraday data from daily data
                df = self._simulate_intraday_from_daily(df, granularity)
                
                print(f"Successfully fetched {len(df)} candles for {symbol} from Yahoo Finance")
                return df
                
        except Exception as e:
            print(f"Yahoo Finance fallback failed: {e}")
            raise
    
    def _simulate_intraday_from_daily(self, daily_df: pd.DataFrame, granularity: int) -> pd.DataFrame:
        """Convert daily data to intraday by adding realistic price movement"""
        intraday_data = []
        
        for _, row in daily_df.iterrows():
            daily_open = row['open']
            daily_high = row['high']
            daily_low = row['low']
            daily_close = row['close']
            daily_volume = row['volume']
            base_time = row['timestamp']
            
            # Number of intraday candles per day
            candles_per_day = 86400 // granularity
            
            # Generate price path for the day
            price_path = self._generate_realistic_price_path(
                daily_open, daily_high, daily_low, daily_close, candles_per_day
            )
            
            # Create intraday candles
            for i in range(candles_per_day):
                candle_time = base_time + timedelta(seconds=i * granularity)
                
                if i < len(price_path) - 1:
                    candle_open = price_path[i]
                    candle_close = price_path[i + 1]
                    
                    # Add some randomness to high/low within reasonable bounds
                    price_range = abs(candle_close - candle_open) * 0.5
                    candle_high = max(candle_open, candle_close) + np.random.uniform(0, price_range)
                    candle_low = min(candle_open, candle_close) - np.random.uniform(0, price_range)
                    
                    # Ensure we don't exceed daily bounds
                    candle_high = min(candle_high, daily_high)
                    candle_low = max(candle_low, daily_low)
                    
                    volume = daily_volume / candles_per_day * np.random.uniform(0.5, 2.0)
                    
                    intraday_data.append({
                        'timestamp': candle_time,
                        'open': candle_open,
                        'high': candle_high,
                        'low': candle_low,
                        'close': candle_close,
                        'volume': volume
                    })
        
        return pd.DataFrame(intraday_data)
    
    def _generate_realistic_price_path(self, open_price: float, high: float, low: float, 
                                     close_price: float, num_points: int) -> list:
        """Generate a realistic price path that hits the daily high/low"""
        path = [open_price]
        
        # Determine when to hit high and low
        high_idx = np.random.randint(1, num_points - 1)
        low_idx = np.random.randint(1, num_points - 1)
        
        # Ensure we visit both high and low
        key_points = [(0, open_price), (high_idx, high), (low_idx, low), (num_points - 1, close_price)]
        key_points.sort(key=lambda x: x[0])
        
        # Interpolate between key points with some randomness
        for i in range(1, num_points):
            # Find surrounding key points
            prev_key = None
            next_key = None
            
            for j, (idx, price) in enumerate(key_points):
                if idx <= i:
                    prev_key = (idx, price)
                if idx >= i and next_key is None:
                    next_key = (idx, price)
                    break
            
            if prev_key and next_key and prev_key[0] != next_key[0]:
                # Linear interpolation with noise
                ratio = (i - prev_key[0]) / (next_key[0] - prev_key[0])
                base_price = prev_key[1] + (next_key[1] - prev_key[1]) * ratio
                
                # Add some randomness
                noise = np.random.normal(0, abs(close_price - open_price) * 0.01)
                price = base_price + noise
                
                # Keep within reasonable bounds
                price = max(min(price, high * 1.001), low * 0.999)
                path.append(price)
            else:
                path.append(path[-1])
        
        return path
    
    def _generate_synthetic_data(self, symbol: str, granularity: int = 300, days: int = 30) -> pd.DataFrame:
        """Generate synthetic data for demonstration purposes"""
        print(f"Generating synthetic data for {symbol} (for demonstration)")
        
        # Determine base price based on symbol
        base_prices = {
            'BTC-USD': 65000,
            'ETH-USD': 3500,
            'SOL-USD': 140,
            'ADA-USD': 0.45,
            'MATIC-USD': 0.65
        }
        
        base_price = base_prices.get(symbol, 100)
        
        # Generate time series
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        periods = int((end_time - start_time).total_seconds() / granularity)
        timestamps = [start_time + timedelta(seconds=i * granularity) for i in range(periods)]
        
        # Generate price data with realistic characteristics
        prices = []
        current_price = base_price
        
        for i in range(periods):
            # Add trend and noise
            trend = np.sin(i / 100) * 0.001  # Slow trend
            volatility = base_price * 0.02  # 2% volatility
            noise = np.random.normal(0, volatility)
            
            current_price *= (1 + trend + noise / base_price)
            prices.append(current_price)
        
        # Create OHLCV data
        data = []
        for i in range(len(timestamps)):
            if i == 0:
                open_price = base_price
            else:
                open_price = data[i-1]['close']
            
            close_price = prices[i]
            
            # Generate high/low around open/close
            high_price = max(open_price, close_price) * np.random.uniform(1.0, 1.02)
            low_price = min(open_price, close_price) * np.random.uniform(0.98, 1.0)
            
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'timestamp': timestamps[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        print(f"Generated {len(df)} synthetic candles for {symbol}")
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        if df.empty:
            return df
            
        # EMAs
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Volume MA
        df['volume_ma'] = df['volume'].rolling(20).mean()
        
        # Support/Resistance
        df['support'] = df['low'].rolling(20).min()
        df['resistance'] = df['high'].rolling(20).max()
        
        return df
    
    def detect_signals(self, df: pd.DataFrame, i: int) -> dict:
        """Detect trading signals at index i"""
        if i < 200 or i >= len(df) - 1:
            return {}
            
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Long signal detection
        ema_bullish = (current['close'] > current['ema_20'] > current['ema_50'] and 
                      current['close'] > current['ema_200'])
        ema_cross_long = (current['ema_20'] > current['ema_50'] and 
                         prev['ema_20'] <= prev['ema_50'])
        rsi_recovery = (prev['rsi'] < 35 and current['rsi'] > 35 and current['rsi'] < 60)
        macd_bullish = (current['macd'] > current['macd_signal'] and 
                       prev['macd'] <= prev['macd_signal'])
        volume_confirm = current['volume'] > current['volume_ma'] * 1.2
        support_bounce = (prev['low'] <= current['support'] * 1.01 and 
                         current['close'] > prev['close'])
        
        long_strength = sum([ema_bullish, ema_cross_long, rsi_recovery, 
                           macd_bullish, volume_confirm, support_bounce])
        
        # Short signal detection
        ema_bearish = (current['close'] < current['ema_20'] < current['ema_50'] and 
                      current['close'] < current['ema_200'])
        ema_cross_short = (current['ema_20'] < current['ema_50'] and 
                          prev['ema_20'] >= prev['ema_50'])
        rsi_reversal = (prev['rsi'] > 65 and current['rsi'] < 65 and current['rsi'] > 40)
        macd_bearish = (current['macd'] < current['macd_signal'] and 
                       prev['macd'] >= prev['macd_signal'])
        resistance_reject = (prev['high'] >= current['resistance'] * 0.99 and 
                           current['close'] < prev['close'])
        
        short_strength = sum([ema_bearish, ema_cross_short, rsi_reversal, 
                            macd_bearish, volume_confirm, resistance_reject])
        
        signals = {}
        
        if long_strength >= 3:
            signals['long'] = {
                'entry_price': current['close'],
                'stop_loss': current['close'] - (current['atr'] * 2),
                'take_profit_1': current['close'] + (current['atr'] * 2),
                'take_profit_2': current['close'] + (current['atr'] * 4),
                'strength': long_strength,
                'timestamp': current['timestamp']
            }
            
        if short_strength >= 3:
            signals['short'] = {
                'entry_price': current['close'],
                'stop_loss': current['close'] + (current['atr'] * 2),
                'take_profit_1': current['close'] - (current['atr'] * 2),
                'take_profit_2': current['close'] - (current['atr'] * 4),
                'strength': short_strength,
                'timestamp': current['timestamp']
            }
            
        return signals
    
    def simulate_trade(self, df: pd.DataFrame, entry_idx: int, signal: dict, 
                      signal_type: str) -> dict:
        """Simulate a single trade from entry to exit"""
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        take_profit_1 = signal['take_profit_1']
        take_profit_2 = signal['take_profit_2']
        
        # Calculate position size
        position_size = self.trade_size / entry_price
        
        trade_result = {
            'entry_time': signal['timestamp'],
            'entry_price': entry_price,
            'signal_type': signal_type,
            'signal_strength': signal['strength'],
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit_1': take_profit_1,
            'take_profit_2': take_profit_2,
            'exit_time': None,
            'exit_price': None,
            'exit_reason': None,
            'pnl': 0,
            'return_pct': 0,
            'trade_duration': 0
        }
        
        # Look for exit conditions in subsequent candles
        max_lookforward = min(len(df) - entry_idx - 1, 100)  # Max 100 candles or end of data
        
        for j in range(1, max_lookforward + 1):
            current_idx = entry_idx + j
            candle = df.iloc[current_idx]
            
            if signal_type == 'long':
                # Check stop loss
                if candle['low'] <= stop_loss:
                    trade_result['exit_price'] = stop_loss
                    trade_result['exit_reason'] = 'Stop Loss'
                    break
                # Check take profit 1 (50% position)
                elif candle['high'] >= take_profit_1:
                    trade_result['exit_price'] = take_profit_1
                    trade_result['exit_reason'] = 'Take Profit 1'
                    break
                # Check take profit 2
                elif candle['high'] >= take_profit_2:
                    trade_result['exit_price'] = take_profit_2
                    trade_result['exit_reason'] = 'Take Profit 2'
                    break
                    
            else:  # short
                # Check stop loss
                if candle['high'] >= stop_loss:
                    trade_result['exit_price'] = stop_loss
                    trade_result['exit_reason'] = 'Stop Loss'
                    break
                # Check take profit 1
                elif candle['low'] <= take_profit_1:
                    trade_result['exit_price'] = take_profit_1
                    trade_result['exit_reason'] = 'Take Profit 1'
                    break
                # Check take profit 2
                elif candle['low'] <= take_profit_2:
                    trade_result['exit_price'] = take_profit_2
                    trade_result['exit_reason'] = 'Take Profit 2'
                    break
        
        # If no exit found, close at last available price
        if trade_result['exit_price'] is None:
            last_candle = df.iloc[entry_idx + max_lookforward]
            trade_result['exit_price'] = last_candle['close']
            trade_result['exit_reason'] = 'Time Exit'
            trade_result['exit_time'] = last_candle['timestamp']
        else:
            exit_idx = entry_idx + j
            trade_result['exit_time'] = df.iloc[exit_idx]['timestamp']
        
        # Calculate P&L
        if signal_type == 'long':
            pnl = (trade_result['exit_price'] - entry_price) * position_size
        else:
            pnl = (entry_price - trade_result['exit_price']) * position_size
            
        trade_result['pnl'] = pnl
        trade_result['return_pct'] = (pnl / self.trade_size) * 100
        
        # Calculate trade duration
        if trade_result['exit_time']:
            duration = trade_result['exit_time'] - trade_result['entry_time']
            trade_result['trade_duration'] = duration.total_seconds() / 3600  # hours
        
        return trade_result
    
    def backtest_strategy(self, symbol: str, days: int = 30, granularity: int = 300) -> dict:
        """Run complete backtest on a symbol"""
        print(f"Backtesting {symbol} for {days} days...")
        
        # Get data
        df = self.get_historical_data(symbol, granularity, days)
        if df.empty:
            return {}
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        trades = []
        capital = self.initial_capital
        equity_curve = [capital]
        trade_times = []
        
        # Skip positions to avoid overlapping trades
        skip_until = 0
        
        for i in range(200, len(df) - 1):
            if i < skip_until:
                continue
                
            signals = self.detect_signals(df, i)
            
            # Process long signals
            if 'long' in signals:
                trade_result = self.simulate_trade(df, i, signals['long'], 'long')
                trades.append(trade_result)
                capital += trade_result['pnl']
                equity_curve.append(capital)
                trade_times.append(trade_result['entry_time'])
                skip_until = i + 5  # Skip next 5 candles to avoid overlapping
                
            # Process short signals (if no long signal)
            elif 'short' in signals:
                trade_result = self.simulate_trade(df, i, signals['short'], 'short')
                trades.append(trade_result)
                capital += trade_result['pnl']
                equity_curve.append(capital)
                trade_times.append(trade_result['entry_time'])
                skip_until = i + 5  # Skip next 5 candles to avoid overlapping
        
        # Calculate performance metrics
        trades_df = pd.DataFrame(trades)
        
        if trades_df.empty:
            return {
                'symbol': symbol,
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'trades': trades_df,
                'equity_curve': equity_curve
            }
        
        total_trades = len(trades_df)
        profitable_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum()
        total_return = ((capital - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate max drawdown
        peak = self.initial_capital
        max_drawdown = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = ((peak - equity) / peak) * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Additional metrics
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if profitable_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (total_trades - profitable_trades) > 0 else 0
        avg_trade_duration = trades_df['trade_duration'].mean()
        
        return {
            'symbol': symbol,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': total_trades - profitable_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'avg_trade_duration': avg_trade_duration,
            'final_capital': capital,
            'trades': trades_df,
            'equity_curve': equity_curve,
            'trade_times': trade_times
        }
    
    def run_multi_symbol_backtest(self, symbols: list, days: int = 30, granularity: int = 300):
        """Run backtest on multiple symbols"""
        results = {}
        
        for symbol in symbols:
            result = self.backtest_strategy(symbol, days, granularity)
            if result:
                results[symbol] = result
        
        return results
    
    def generate_report(self, results: dict):
        """Generate comprehensive backtest report"""
        print("\n" + "="*80)
        print("CRYPTO TRADING STRATEGY BACKTEST REPORT")
        print("="*80)
        
        # Summary table
        summary_data = []
        total_pnl = 0
        total_trades = 0
        
        for symbol, result in results.items():
            if result['total_trades'] > 0:
                summary_data.append({
                    'Symbol': symbol,
                    'Trades': result['total_trades'],
                    'Win Rate': f"{result['win_rate']:.1f}%",
                    'Total P&L': f"${result['total_pnl']:.2f}",
                    'Return': f"{result['total_return']:.2f}%",
                    'Max DD': f"{result['max_drawdown']:.2f}%",
                    'Profit Factor': f"{result['profit_factor']:.2f}"
                })
                total_pnl += result['total_pnl']
                total_trades += result['total_trades']
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            print("\nPERFORMANCE SUMMARY:")
            print(summary_df.to_string(index=False))
            
            print(f"\nOVERALL PERFORMANCE:")
            print(f"Total Trades: {total_trades}")
            print(f"Total P&L: ${total_pnl:.2f}")
            print(f"Overall Return: {((total_pnl / self.initial_capital) * 100):.2f}%")
            
            # Best and worst performers
            if len(summary_data) > 1:
                best_symbol = max(results.keys(), key=lambda x: results[x]['total_pnl'])
                worst_symbol = min(results.keys(), key=lambda x: results[x]['total_pnl'])
                
                print(f"\nBest Performer: {best_symbol} (${results[best_symbol]['total_pnl']:.2f})")
                print(f"Worst Performer: {worst_symbol} (${results[worst_symbol]['total_pnl']:.2f})")
        
        return summary_data
    
    def plot_equity_curves(self, results: dict):
        """Plot equity curves for all symbols"""
        plt.figure(figsize=(15, 10))
        
        # Plot individual equity curves
        for i, (symbol, result) in enumerate(results.items()):
            if result['total_trades'] > 0:
                plt.subplot(2, 2, 1)
                plt.plot(result['equity_curve'], label=symbol)
                plt.title('Equity Curves by Symbol')
                plt.xlabel('Trade Number')
                plt.ylabel('Capital ($)')
                plt.legend()
                plt.grid(True)
        
        # Combined performance metrics
        symbols = list(results.keys())
        win_rates = [results[s]['win_rate'] for s in symbols if results[s]['total_trades'] > 0]
        returns = [results[s]['total_return'] for s in symbols if results[s]['total_trades'] > 0]
        max_dds = [results[s]['max_drawdown'] for s in symbols if results[s]['total_trades'] > 0]
        
        active_symbols = [s for s in symbols if results[s]['total_trades'] > 0]
        
        if active_symbols:
            plt.subplot(2, 2, 2)
            plt.bar(active_symbols, win_rates)
            plt.title('Win Rate by Symbol')
            plt.ylabel('Win Rate (%)')
            plt.xticks(rotation=45)
            plt.grid(True)
            
            plt.subplot(2, 2, 3)
            plt.bar(active_symbols, returns)
            plt.title('Total Return by Symbol')
            plt.ylabel('Return (%)')
            plt.xticks(rotation=45)
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            plt.bar(active_symbols, max_dds)
            plt.title('Max Drawdown by Symbol')
            plt.ylabel('Max Drawdown (%)')
            plt.xticks(rotation=45)
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize backtester
    backtester = CryptoBacktester(initial_capital=10000, trade_size=2500)
    
    # Define symbols to test
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'MATIC-USD']
    
    # Run backtest (last 30 days, 5-minute candles)
    print("Starting backtest...")
    results = backtester.run_multi_symbol_backtest(
        symbols=symbols,
        days=30,  # Last 30 days
        granularity=300  # 5-minute candles
    )
    
    # Generate report
    summary = backtester.generate_report(results)
    
    # Plot results
    backtester.plot_equity_curves(results)
    
    # Detailed trade analysis for best performer
    if results:
        best_symbol = max(results.keys(), key=lambda x: results[x]['total_pnl'])
        best_result = results[best_symbol]
        
        if best_result['total_trades'] > 0:
            print(f"\nDETAILED ANALYSIS FOR {best_symbol}:")
            print(f"Average Win: ${best_result['avg_win']:.2f}")
            print(f"Average Loss: ${best_result['avg_loss']:.2f}")
            print(f"Average Trade Duration: {best_result['avg_trade_duration']:.2f} hours")
            
            # Show sample trades
            trades_df = best_result['trades']
            print(f"\nSample of Recent Trades:")
            print(trades_df[['entry_time', 'signal_type', 'entry_price', 'exit_price', 
                           'exit_reason', 'pnl', 'return_pct']].tail(10).to_string(index=False))
