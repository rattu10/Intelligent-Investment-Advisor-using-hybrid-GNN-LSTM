import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicatorClusteringEngine:
    """
    Advanced Technical Indicator Clustering Module
    Implements clustering-based feature engineering for technical indicators
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.indicator_names = []
        self.clusters = {}
        self.cluster_labels = []
        
    def calculate_technical_indicators(self, data):
        """Calculate comprehensive set of technical indicators"""
        df = data.copy()
        
        # Price-based indicators
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # Momentum indicators
        df['RSI'] = self._calculate_rsi(df['Close'])
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Volatility indicators
        df['Bollinger_Upper'], df['Bollinger_Lower'], df['Bollinger_Width'] = self._calculate_bollinger_bands(df['Close'])
        df['ATR'] = self._calculate_atr(df)
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['OBV'] = self._calculate_obv(df)
        
        # Trend indicators
        df['ADX'] = self._calculate_adx(df)
        df['Aroon_Up'], df['Aroon_Down'] = self._calculate_aroon(df)
        
        # Price position indicators
        df['Williams_R'] = self._calculate_williams_r(df)
        df['Stochastic_K'] = self._calculate_stochastic(df)
        
        # Additional momentum
        df['ROC'] = df['Close'].pct_change(periods=10) * 100
        df['MFI'] = self._calculate_mfi(df)
        
        return df
    
    def perform_clustering(self, data):
        """Cluster technical indicators into correlated and non-correlated groups"""
        indicators_df = self.calculate_technical_indicators(data)
        
        # Select indicator columns (exclude OHLCV)
        indicator_cols = [col for col in indicators_df.columns 
                         if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Handle missing values
        indicators_matrix = indicators_df[indicator_cols].fillna(method='bfill').fillna(method='ffill')
        
        if len(indicators_matrix) < 30:  # Need sufficient data
            return self._create_default_clusters(indicator_cols)
        
        # Take recent data for clustering
        recent_indicators = indicators_matrix.tail(30)
        
        # Standardize indicators
        scaled_indicators = self.scaler.fit_transform(recent_indicators)
        
        # Perform clustering
        cluster_labels = self.kmeans.fit_predict(scaled_indicators.T)  # Transpose to cluster indicators
        
        # Organize indicators by clusters
        clusters = {}
        for i, indicator in enumerate(indicator_cols):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(indicator)
        
        self.clusters = clusters
        self.cluster_labels = cluster_labels
        self.indicator_names = indicator_cols
        
        return {
            'clusters': clusters,
            'cluster_summary': self._generate_cluster_summary(clusters),
            'indicators_data': indicators_matrix.tail(1).to_dict('records')[0],
            'cluster_analysis': self._analyze_clusters(scaled_indicators, cluster_labels)
        }
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        width = (upper - lower) / sma * 100
        return upper, lower, width
    
    def _calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def _calculate_obv(self, data):
        """Calculate On-Balance Volume"""
        obv = []
        obv.append(0)
        
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        return pd.Series(obv, index=data.index)
    
    def _calculate_adx(self, data, period=14):
        """Calculate Average Directional Index (simplified)"""
        high_diff = data['High'].diff()
        low_diff = data['Low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff < 0), 0)
        
        atr = self._calculate_atr(data, period)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = (np.abs(plus_di - minus_di) / np.abs(plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()
        
        return adx.fillna(25)  # Default neutral value
    
    def _calculate_aroon(self, data, period=14):
        """Calculate Aroon indicators"""
        aroon_up = []
        aroon_down = []
        
        for i in range(period, len(data)):
            period_high = data['High'].iloc[i-period:i+1]
            period_low = data['Low'].iloc[i-period:i+1]
            
            high_idx = period_high.idxmax()
            low_idx = period_low.idxmin()
            
            high_periods_since = i - data.index.get_loc(high_idx)
            low_periods_since = i - data.index.get_loc(low_idx)
            
            aroon_up.append(((period - high_periods_since) / period) * 100)
            aroon_down.append(((period - low_periods_since) / period) * 100)
        
        # Pad with initial values
        aroon_up = [50] * period + aroon_up
        aroon_down = [50] * period + aroon_down
        
        return pd.Series(aroon_up, index=data.index), pd.Series(aroon_down, index=data.index)
    
    def _calculate_williams_r(self, data, period=14):
        """Calculate Williams %R"""
        highest_high = data['High'].rolling(window=period).max()
        lowest_low = data['Low'].rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - data['Close']) / (highest_high - lowest_low))
        return williams_r.fillna(-50)
    
    def _calculate_stochastic(self, data, period=14):
        """Calculate Stochastic Oscillator"""
        lowest_low = data['Low'].rolling(window=period).min()
        highest_high = data['High'].rolling(window=period).max()
        
        k_percent = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
        return k_percent.fillna(50)
    
    def _calculate_mfi(self, data, period=14):
        """Calculate Money Flow Index"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        positive_flow = []
        negative_flow = []
        
        for i in range(1, len(data)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.append(money_flow.iloc[i])
                negative_flow.append(0)
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                positive_flow.append(0)
                negative_flow.append(money_flow.iloc[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)
        
        positive_flow = [0] + positive_flow
        negative_flow = [0] + negative_flow
        
        positive_flow_series = pd.Series(positive_flow, index=data.index)
        negative_flow_series = pd.Series(negative_flow, index=data.index)
        
        positive_flow_sum = positive_flow_series.rolling(window=period).sum()
        negative_flow_sum = negative_flow_series.rolling(window=period).sum()
        
        money_flow_ratio = positive_flow_sum / negative_flow_sum
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi.fillna(50)
    
    def _create_default_clusters(self, indicator_cols):
        """Create default clustering when insufficient data"""
        # Group indicators logically
        trend_indicators = [col for col in indicator_cols if any(x in col.lower() for x in ['sma', 'ema', 'adx', 'aroon'])]
        momentum_indicators = [col for col in indicator_cols if any(x in col.lower() for x in ['rsi', 'macd', 'roc', 'williams', 'stochastic'])]
        volatility_indicators = [col for col in indicator_cols if any(x in col.lower() for x in ['bollinger', 'atr', 'volatility'])]
        volume_indicators = [col for col in indicator_cols if any(x in col.lower() for x in ['volume', 'obv', 'mfi'])]
        
        clusters = {
            0: trend_indicators,
            1: momentum_indicators + volatility_indicators,
            2: volume_indicators
        }
        
        return {
            'clusters': clusters,
            'cluster_summary': self._generate_cluster_summary(clusters),
            'indicators_data': {},
            'cluster_analysis': 'Insufficient data for statistical clustering - using logical grouping'
        }
    
    def _generate_cluster_summary(self, clusters):
        """Generate human-readable cluster summary"""
        cluster_descriptions = {
            0: "Trend Following Indicators",
            1: "Momentum & Volatility Indicators", 
            2: "Volume & Money Flow Indicators"
        }
        
        summary = {}
        for cluster_id, indicators in clusters.items():
            desc = cluster_descriptions.get(cluster_id, f"Cluster {cluster_id}")
            summary[desc] = {
                'count': len(indicators),
                'indicators': indicators[:5],  # Show first 5
                'description': self._get_cluster_description(indicators)
            }
        
        return summary
    
    def _get_cluster_description(self, indicators):
        """Get description of what the cluster represents"""
        if any('sma' in ind.lower() or 'ema' in ind.lower() for ind in indicators):
            return "These indicators help identify price trends and direction"
        elif any('rsi' in ind.lower() or 'macd' in ind.lower() for ind in indicators):
            return "These indicators measure momentum and potential reversal points"
        elif any('volume' in ind.lower() or 'obv' in ind.lower() for ind in indicators):
            return "These indicators analyze trading volume and money flow"
        else:
            return "Mixed technical analysis indicators"
    
    def _analyze_clusters(self, scaled_data, cluster_labels):
        """Analyze cluster characteristics"""
        if len(scaled_data) == 0:
            return "No data available for cluster analysis"
            
        cluster_stats = {}
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_data = scaled_data[:, cluster_indices]
            
            cluster_stats[f"Cluster_{label}"] = {
                'size': len(cluster_indices),
                'avg_correlation': np.corrcoef(cluster_data.T).mean() if cluster_data.shape[1] > 1 else 1.0,
                'variance': np.var(cluster_data.mean(axis=1))
            }
        
        return cluster_stats
    
    def get_clustered_features(self, data):
        """Extract features based on clusters for prediction"""
        clustering_result = self.perform_clustering(data)
        
        features = {}
        for cluster_name, cluster_info in clustering_result['cluster_summary'].items():
            indicators = cluster_info['indicators']
            if indicators and cluster_name in clustering_result['indicators_data']:
                # Calculate aggregate features for each cluster
                cluster_values = [clustering_result['indicators_data'].get(ind, 0) for ind in indicators]
                features[f"{cluster_name}_mean"] = np.mean(cluster_values)
                features[f"{cluster_name}_std"] = np.std(cluster_values)
                features[f"{cluster_name}_trend"] = self._calculate_cluster_trend(cluster_values)
        
        return features
    
    def _calculate_cluster_trend(self, values):
        """Calculate trend direction for cluster"""
        if len(values) < 2:
            return 0
        
        # Simple trend calculation
        trend = np.polyfit(range(len(values)), values, 1)[0]
        return 1 if trend > 0 else -1 if trend < 0 else 0