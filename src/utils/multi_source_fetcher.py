#!/usr/bin/env python3
"""
Multi-Source Stock Data Fetcher
Tries multiple reliable APIs to get real-time stock data
Automatically falls back to alternatives if one fails
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import yfinance as yf
from typing import Dict, List, Optional, Tuple

class MultiSourceDataFetcher:
    """
    Multi-source stock data fetcher with automatic fallback
    Tries multiple APIs in order of preference for maximum reliability
    """
    
    def __init__(self):
        """Initialize with API configurations"""
        # API configurations (add your API keys here for better reliability)
        self.api_configs = {
            'alpha_vantage': {
                'base_url': 'https://www.alphavantage.co/query',
                'api_key': 'UK7MXH7V063EO97V',  # Replace with your free API key from alphavantage.co
                'rate_limit': 5,    # calls per minute
                'priority': 1
            },
            'finnhub': {
                'base_url': 'https://finnhub.io/api/v1',
                'api_key': 'demo',  # Replace with your free API key from finnhub.io
                'rate_limit': 60,   # calls per minute
                'priority': 2
            },
            'twelve_data': {
                'base_url': 'https://api.twelvedata.com',
                'api_key': 'demo',  # Replace with your free API key from twelvedata.com
                'rate_limit': 8,    # calls per minute
                'priority': 3
            },
            'polygon': {
                'base_url': 'https://api.polygon.io/v2',
                'api_key': 'demo',  # Replace with your free API key from polygon.io
                'rate_limit': 5,    # calls per minute
                'priority': 4
            }
        }
        
        # Track API usage for rate limiting
        self.api_usage = {}
        
    def get_stock_data(self, symbol: str, period: str = '6mo') -> Optional[pd.DataFrame]:
        """
        Get stock data trying multiple sources in order of preference
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y')
            
        Returns:
            DataFrame with OHLCV data or None if all sources fail
        """
        print(f"🔄 Fetching {symbol} data from multiple sources...")
        
        # Try each API source in order of priority
        sources = [
            ('Alpha Vantage', self._fetch_alpha_vantage),
            ('Finnhub', self._fetch_finnhub),
            ('Twelve Data', self._fetch_twelve_data),
            ('Polygon.io', self._fetch_polygon),
            ('yfinance (fallback)', self._fetch_yfinance)
        ]
        
        for source_name, fetch_func in sources:
            try:
                print(f"   📊 Trying {source_name}...")
                data = fetch_func(symbol, period)
                
                if data is not None and not data.empty and len(data) > 5:
                    print(f"   ✅ {source_name} successful: {len(data)} days of data")
                    return self._standardize_data(data)
                else:
                    print(f"   ⚠️ {source_name} returned insufficient data")
                    
            except Exception as e:
                print(f"   ❌ {source_name} failed: {str(e)}")
                continue
                
        print(f"   🆘 All sources failed for {symbol}")
        return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current/latest price from multiple sources"""
        print(f"💰 Getting current price for {symbol}...")
        
        # Try real-time price sources
        price_sources = [
            ('Alpha Vantage Quote', self._get_price_alpha_vantage),
            ('Finnhub Quote', self._get_price_finnhub),
            ('Twelve Data Quote', self._get_price_twelve_data),
            ('yfinance Info', self._get_price_yfinance)
        ]
        
        for source_name, price_func in price_sources:
            try:
                print(f"   💲 Trying {source_name}...")
                price = price_func(symbol)
                
                if price and price > 0:
                    print(f"   ✅ {source_name}: ${price:.2f}")
                    return float(price)
                else:
                    print(f"   ⚠️ {source_name} returned invalid price")
                    
            except Exception as e:
                print(f"   ❌ {source_name} failed: {str(e)}")
                continue
                
        print(f"   🆘 Could not get current price for {symbol}")
        return None
    
    def get_company_info(self, symbol: str) -> Dict:
        """Get company information from multiple sources"""
        info_sources = [
            self._get_info_alpha_vantage,
            self._get_info_finnhub,
            self._get_info_yfinance
        ]
        
        for get_info_func in info_sources:
            try:
                info = get_info_func(symbol)
                if info and info.get('longName'):
                    return info
            except:
                continue
                
        return {'longName': symbol, 'shortName': symbol}
    
    def _fetch_alpha_vantage(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage API"""
        if not self._check_rate_limit('alpha_vantage'):
            raise Exception("Rate limit exceeded")
            
        # Convert period to Alpha Vantage format
        if period in ['1d', '5d']:
            function = 'TIME_SERIES_INTRADAY'
            interval = '60min'
        else:
            function = 'TIME_SERIES_DAILY'
            interval = None
            
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': self.api_configs['alpha_vantage']['api_key'],
            'outputsize': 'full'
        }
        
        if interval:
            params['interval'] = interval
            
        response = requests.get(self.api_configs['alpha_vantage']['base_url'], 
                              params=params, timeout=10)
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
            
        data = response.json()
        
        # Parse Alpha Vantage response
        if 'Time Series (Daily)' in data:
            ts_data = data['Time Series (Daily)']
        elif 'Time Series (60min)' in data:
            ts_data = data['Time Series (60min)']
        else:
            raise Exception("No time series data found")
            
        # Convert to DataFrame
        df_data = []
        for date_str, values in ts_data.items():
            df_data.append({
                'Date': pd.to_datetime(date_str),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': int(values['5. volume'])
            })
            
        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        return self._filter_by_period(df, period)
    
    def _fetch_finnhub(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch data from Finnhub API"""
        if not self._check_rate_limit('finnhub'):
            raise Exception("Rate limit exceeded")
            
        # Calculate date range
        end_date = datetime.now()
        days_map = {'1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365}
        days = days_map.get(period, 180)
        start_date = end_date - timedelta(days=days)
        
        params = {
            'symbol': symbol,
            'resolution': 'D',
            'from': int(start_date.timestamp()),
            'to': int(end_date.timestamp()),
            'token': self.api_configs['finnhub']['api_key']
        }
        
        response = requests.get(f"{self.api_configs['finnhub']['base_url']}/stock/candle", 
                              params=params, timeout=10)
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
            
        data = response.json()
        
        if data['s'] != 'ok' or not data.get('c'):
            raise Exception("No data returned")
            
        # Convert to DataFrame
        df = pd.DataFrame({
            'Open': data['o'],
            'High': data['h'],
            'Low': data['l'],
            'Close': data['c'],
            'Volume': data['v']
        })
        
        # Add dates
        dates = [datetime.fromtimestamp(ts) for ts in data['t']]
        df.index = pd.DatetimeIndex(dates)
        
        return df
    
    def _fetch_twelve_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch data from Twelve Data API"""
        if not self._check_rate_limit('twelve_data'):
            raise Exception("Rate limit exceeded")
            
        params = {
            'symbol': symbol,
            'interval': '1day',
            'apikey': self.api_configs['twelve_data']['api_key'],
            'outputsize': '5000'
        }
        
        response = requests.get(f"{self.api_configs['twelve_data']['base_url']}/time_series", 
                              params=params, timeout=10)
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
            
        data = response.json()
        
        if 'values' not in data:
            raise Exception("No values in response")
            
        # Convert to DataFrame
        df_data = []
        for item in data['values']:
            df_data.append({
                'Date': pd.to_datetime(item['datetime']),
                'Open': float(item['open']),
                'High': float(item['high']),
                'Low': float(item['low']),
                'Close': float(item['close']),
                'Volume': int(item['volume']) if item['volume'] else 0
            })
            
        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        return self._filter_by_period(df, period)
    
    def _fetch_polygon(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch data from Polygon.io API"""
        if not self._check_rate_limit('polygon'):
            raise Exception("Rate limit exceeded")
            
        # Calculate date range
        end_date = datetime.now()
        days_map = {'1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365}
        days = days_map.get(period, 180)
        start_date = end_date - timedelta(days=days)
        
        url = f"{self.api_configs['polygon']['base_url']}/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        params = {
            'apikey': self.api_configs['polygon']['api_key']
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
            
        data = response.json()
        
        if not data.get('results'):
            raise Exception("No results in response")
            
        # Convert to DataFrame
        df_data = []
        for item in data['results']:
            df_data.append({
                'Date': pd.to_datetime(item['t'], unit='ms'),
                'Open': float(item['o']),
                'High': float(item['h']),
                'Low': float(item['l']),
                'Close': float(item['c']),
                'Volume': int(item['v'])
            })
            
        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def _fetch_yfinance(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Fallback to yfinance"""
        ticker = yf.Ticker(symbol)
        return ticker.history(period=period)
    
    def _get_price_alpha_vantage(self, symbol: str) -> Optional[float]:
        """Get current price from Alpha Vantage"""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_configs['alpha_vantage']['api_key']
        }
        
        response = requests.get(self.api_configs['alpha_vantage']['base_url'], 
                              params=params, timeout=5)
        data = response.json()
        
        quote = data.get('Global Quote', {})
        price = quote.get('05. price')
        return float(price) if price else None
    
    def _get_price_finnhub(self, symbol: str) -> Optional[float]:
        """Get current price from Finnhub"""
        params = {
            'symbol': symbol,
            'token': self.api_configs['finnhub']['api_key']
        }
        
        response = requests.get(f"{self.api_configs['finnhub']['base_url']}/quote", 
                              params=params, timeout=5)
        data = response.json()
        
        return data.get('c')  # current price
    
    def _get_price_twelve_data(self, symbol: str) -> Optional[float]:
        """Get current price from Twelve Data"""
        params = {
            'symbol': symbol,
            'apikey': self.api_configs['twelve_data']['api_key']
        }
        
        response = requests.get(f"{self.api_configs['twelve_data']['base_url']}/price", 
                              params=params, timeout=5)
        data = response.json()
        
        price = data.get('price')
        return float(price) if price else None
    
    def _get_price_yfinance(self, symbol: str) -> Optional[float]:
        """Get current price from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('regularMarketPrice', info.get('previousClose'))
        except:
            return None
    
    def _get_info_alpha_vantage(self, symbol: str) -> Dict:
        """Get company info from Alpha Vantage"""
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': self.api_configs['alpha_vantage']['api_key']
        }
        
        response = requests.get(self.api_configs['alpha_vantage']['base_url'], 
                              params=params, timeout=5)
        data = response.json()
        
        return {
            'longName': data.get('Name', symbol),
            'shortName': data.get('Name', symbol),
            'sector': data.get('Sector'),
            'industry': data.get('Industry')
        }
    
    def _get_info_finnhub(self, symbol: str) -> Dict:
        """Get company info from Finnhub"""
        params = {
            'symbol': symbol,
            'token': self.api_configs['finnhub']['api_key']
        }
        
        response = requests.get(f"{self.api_configs['finnhub']['base_url']}/stock/profile2", 
                              params=params, timeout=5)
        data = response.json()
        
        return {
            'longName': data.get('name', symbol),
            'shortName': data.get('name', symbol),
            'sector': data.get('finnhubIndustry'),
            'industry': data.get('finnhubIndustry')
        }
    
    def _get_info_yfinance(self, symbol: str) -> Dict:
        """Get company info from yfinance"""
        ticker = yf.Ticker(symbol)
        return ticker.info
    
    def _check_rate_limit(self, api_name: str) -> bool:
        """Check if API rate limit allows another call"""
        now = time.time()
        minute_ago = now - 60
        
        if api_name not in self.api_usage:
            self.api_usage[api_name] = []
            
        # Clean old calls
        self.api_usage[api_name] = [
            call_time for call_time in self.api_usage[api_name] 
            if call_time > minute_ago
        ]
        
        # Check limit
        rate_limit = self.api_configs[api_name]['rate_limit']
        if len(self.api_usage[api_name]) >= rate_limit:
            return False
            
        # Record this call
        self.api_usage[api_name].append(now)
        return True
    
    def _standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame format"""
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                if col == 'Volume':
                    df[col] = 1000000  # Default volume
                else:
                    df[col] = df['Close']  # Use close price as fallback
                    
        return df[required_cols]
    
    def _filter_by_period(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Filter DataFrame by period"""
        if period == '1d':
            return df.tail(1)
        elif period == '5d':
            return df.tail(5)
        elif period == '1mo':
            return df.tail(30)
        elif period == '3mo':
            return df.tail(90)
        elif period == '6mo':
            return df.tail(180)
        elif period == '1y':
            return df.tail(365)
        else:
            return df

# Global instance for easy import
multi_fetcher = MultiSourceDataFetcher()

def get_reliable_stock_data(symbol: str, period: str = '6mo') -> Optional[pd.DataFrame]:
    """Easy-to-use function for getting stock data from multiple sources"""
    return multi_fetcher.get_stock_data(symbol, period)

def get_reliable_current_price(symbol: str) -> Optional[float]:
    """Easy-to-use function for getting current price from multiple sources"""
    return multi_fetcher.get_current_price(symbol)

def get_reliable_company_info(symbol: str) -> Dict:
    """Easy-to-use function for getting company info from multiple sources"""
    return multi_fetcher.get_company_info(symbol)

if __name__ == "__main__":
    # Test the multi-source fetcher
    print("🧪 Testing Multi-Source Data Fetcher")
    print("=" * 50)
    
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}:")
        
        # Test historical data
        data = get_reliable_stock_data(symbol, '1mo')
        if data is not None:
            print(f"   ✅ Historical data: {len(data)} days")
            
        # Test current price
        price = get_reliable_current_price(symbol)
        if price:
            print(f"   ✅ Current price: ${price:.2f}")
            
        # Test company info
        info = get_reliable_company_info(symbol)
        print(f"   ✅ Company: {info.get('longName', symbol)}")
        
        print("   " + "-" * 30)
        
    print("\n✅ Multi-source testing completed!")