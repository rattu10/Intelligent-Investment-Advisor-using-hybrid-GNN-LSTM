import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import ta

class TechnicalIndicatorClusterer:
    """Clusters technical indicators into correlated and non-correlated groups"""
    
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
    def calculate_indicators(self, df):
        """Calculate various technical indicators"""
        indicators = {}
        
        try:
            # Price-based indicators
            indicators['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
            indicators['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            indicators['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            
            # Exponential Moving Averages
            indicators['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
            indicators['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
            
            # MACD
            indicators['MACD'] = ta.trend.macd_diff(df['Close'])
            indicators['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
            
            # RSI
            indicators['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['Close'])
            indicators['BB_High'] = bb.bollinger_hband()
            indicators['BB_Low'] = bb.bollinger_lband()
            indicators['BB_Width'] = indicators['BB_High'] - indicators['BB_Low']
            
            # Stochastic Oscillator
            indicators['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
            indicators['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
            
            # Volume indicators
            if 'Volume' in df.columns:
                try:
                    indicators['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
                    indicators['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
                except:
                    indicators['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
                    indicators['OBV'] = df['Volume'].cumsum()  # Simplified OBV
            
            # Momentum indicators
            indicators['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
            indicators['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
            
            # Volatility indicators
            indicators['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            
            # Price position indicators
            indicators['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
            
            # Fill NaN values
            for key in indicators:
                try:
                    # Try newer pandas method first
                    indicators[key] = indicators[key].ffill().fillna(0)
                except AttributeError:
                    # Fallback for older pandas
                    indicators[key] = indicators[key].fillna(method='ffill').fillna(0)
                
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            # Fallback indicators
            indicators = self._create_fallback_indicators(df)
        
        return indicators
    
    def _create_fallback_indicators(self, df):
        """Create simple fallback indicators when ta library fails"""
        indicators = {}
        
        # Simple moving averages
        indicators['SMA_10'] = df['Close'].rolling(window=10).mean().fillna(df['Close'])
        indicators['SMA_20'] = df['Close'].rolling(window=20).mean().fillna(df['Close'])
        
        # Simple RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['RSI'] = 100 - (100 / (1 + rs))
        indicators['RSI'] = indicators['RSI'].fillna(50)
        
        # Price momentum
        indicators['Momentum'] = df['Close'].pct_change(periods=5).fillna(0)
        
        # Volatility
        indicators['Volatility'] = df['Close'].rolling(window=20).std().fillna(0)
        
        # Volume-based (if available)
        if 'Volume' in df.columns:
            indicators['Volume_MA'] = df['Volume'].rolling(window=10).mean().fillna(df['Volume'])
        
        return indicators
    
    def cluster_indicators(self, indicators):
        """Cluster indicators based on their correlations"""
        # Convert indicators to DataFrame
        indicator_df = pd.DataFrame(indicators)
        
        # Handle any remaining NaN values
        try:
            # Try newer pandas method first
            indicator_df = indicator_df.ffill().bfill().fillna(0)
        except AttributeError:
            # Fallback for older pandas
            indicator_df = indicator_df.fillna(method='forward').fillna(method='backward').fillna(0)
        
        # Calculate correlation matrix
        corr_matrix = indicator_df.corr().abs()
        
        # Prepare data for clustering
        feature_matrix = indicator_df.values.T  # Transpose to cluster indicators, not time points
        
        # Standardize features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Perform clustering
        cluster_labels = self.kmeans.fit_predict(feature_matrix_scaled)
        
        # Group indicators by cluster
        clusters = {}
        indicator_names = list(indicators.keys())
        
        for i, label in enumerate(cluster_labels):
            cluster_name = f"Cluster_{label}"
            if cluster_name not in clusters:
                clusters[cluster_name] = []
            clusters[cluster_name].append(indicator_names[i])
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(clusters, corr_matrix)
        
        return clusters, cluster_analysis
    
    def _analyze_clusters(self, clusters, corr_matrix):
        """Analyze the characteristics of each cluster"""
        analysis = {}
        
        for cluster_name, indicators in clusters.items():
            if len(indicators) < 2:
                analysis[cluster_name] = {
                    'avg_correlation': 0.0,
                    'cluster_type': 'Independent',
                    'description': 'Single indicator cluster'
                }
                continue
            
            # Calculate average correlation within cluster
            cluster_corrs = []
            for i, ind1 in enumerate(indicators):
                for j, ind2 in enumerate(indicators[i+1:], i+1):
                    if ind1 in corr_matrix.index and ind2 in corr_matrix.index:
                        cluster_corrs.append(corr_matrix.loc[ind1, ind2])
            
            avg_corr = np.mean(cluster_corrs) if cluster_corrs else 0.0
            
            # Classify cluster type
            if avg_corr > 0.7:
                cluster_type = 'Highly Correlated'
                description = 'Indicators with strong positive relationships'
            elif avg_corr > 0.3:
                cluster_type = 'Moderately Correlated'
                description = 'Indicators with moderate relationships'
            else:
                cluster_type = 'Weakly Correlated'
                description = 'Indicators with weak or diverse relationships'
            
            analysis[cluster_name] = {
                'avg_correlation': float(avg_corr),
                'cluster_type': cluster_type,
                'description': description,
                'indicator_count': len(indicators)
            }
        
        return analysis
    
    def process_indicators(self, df):
        """Main method to process indicators and cluster them"""
        # Calculate all indicators
        indicators = self.calculate_indicators(df)
        
        # Cluster indicators
        clusters, cluster_analysis = self.cluster_indicators(indicators)
        
        return {
            'indicators': indicators,
            'clusters': clusters,
            'cluster_analysis': cluster_analysis,
            'close_prices': df['Close'].ffill().fillna(0).values if 'Close' in df.columns else np.zeros(len(df))
        }
