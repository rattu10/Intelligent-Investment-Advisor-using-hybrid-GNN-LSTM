import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    """Handles stock data fetching and preprocessing"""
    
    def __init__(self):
        self.cache = {}
    
    def fetch_stock_data(self, symbols, period='1y', interval='1d'):
        """Fetch stock data for given symbols"""
        data = {}
        
        for symbol in symbols:
            try:
                # Check cache first
                cache_key = f"{symbol}_{period}_{interval}"
                if cache_key in self.cache:
                    data[symbol] = self.cache[cache_key]
                    continue
                
                # Fetch data from yfinance
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                
                if df.empty:
                    # Create mock data if fetching fails
                    df = self._create_mock_data(symbol, period)
                
                data[symbol] = df
                self.cache[cache_key] = df
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                # Create mock data as fallback
                data[symbol] = self._create_mock_data(symbol, period)
        
        return data
    
    def _create_mock_data(self, symbol, period='1y'):
        """Create mock stock data for demonstration"""
        # Determine number of days
        if period == '1y':
            days = 365
        elif period == '6mo':
            days = 180
        elif period == '3mo':
            days = 90
        else:
            days = 30
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate mock price data using random walk
        np.random.seed(hash(symbol) % (2**32))  # Consistent seed based on symbol
        
        # Starting price based on symbol
        base_prices = {
            'AAPL': 150.0,
            'GOOGL': 2800.0,
            'MSFT': 380.0,
            'AMZN': 3400.0,
            'TSLA': 800.0,
            'META': 320.0,
            'NVDA': 450.0,
            'JPM': 160.0,
            'V': 240.0,
            'JNJ': 170.0
        }
        
        start_price = base_prices.get(symbol, 100.0)
        
        # Generate price series
        prices = [start_price]
        for _ in range(len(dates) - 1):
            change = np.random.normal(0.001, 0.02)  # 0.1% mean return, 2% volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))  # Ensure positive prices
        
        # Create OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate OHLC based on close price
            volatility = close * 0.02
            high = close + np.random.uniform(0, volatility)
            low = close - np.random.uniform(0, volatility)
            open_price = prices[i-1] if i > 0 else close
            
            # Ensure OHLC logic
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = np.random.randint(1000000, 10000000)
            
            data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def preprocess_data(self, data, normalize=True):
        """Preprocess stock data for model training"""
        processed = {}
        
        for symbol, df in data.items():
            # Calculate returns
            df['Returns'] = df['Close'].pct_change()
            
            # Calculate volatility (rolling std of returns)
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            
            # Calculate price momentum
            df['Momentum'] = df['Close'].pct_change(periods=5)
            
            # Fill NaN values
            try:
                # Try newer pandas method first
                df = df.ffill().fillna(0)
            except AttributeError:
                # Fallback for older pandas
                df = df.fillna(method='forward').fillna(0)
            
            if normalize:
                # Normalize price data
                for col in ['Open', 'High', 'Low', 'Close']:
                    df[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            
            processed[symbol] = df
        
        return processed
