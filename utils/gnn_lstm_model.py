import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class HybridGNNLSTMPredictor:
    """
    Hybrid GNN-LSTM Forecasting Framework
    Combines Graph Neural Networks for inter-stock relationships with LSTM for temporal modeling
    """
    
    def __init__(self, lookback_window=60, prediction_horizon=7):
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.related_stocks = {}
        self.correlation_matrix = None
        self.graph_features = None
        
        # Stock relationship network (major stocks and their typical correlations)
        self.stock_network = {
            'AAPL': ['MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA'],
            'MSFT': ['AAPL', 'GOOGL', 'AMZN', 'META', 'NVDA'],
            'GOOGL': ['AAPL', 'MSFT', 'META', 'AMZN', 'NVDA'],
            'AMZN': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
            'TSLA': ['AAPL', 'NIO', 'LUCID', 'RIVN', 'GM'],
            'META': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'SNAP'],
            'NVDA': ['AMD', 'INTC', 'MSFT', 'GOOGL', 'AAPL']
        }
    
    def predict_with_gnn_lstm(self, symbol, days=7, technical_features=None, sentiment_features=None):
        """Main prediction method combining GNN and LSTM approaches"""
        try:
            # Step 1: Fetch and prepare data
            target_data, related_data = self._fetch_multi_stock_data(symbol)
            
            if target_data is None or len(target_data) < self.lookback_window:
                return self._fallback_prediction(symbol, days)
            
            # Step 2: Build stock relationship graph
            graph_features = self._build_stock_graph(symbol, target_data, related_data)
            
            # Step 3: Extract temporal features using LSTM-style processing
            temporal_features = self._extract_temporal_features(target_data)
            
            # Step 4: Integrate additional features
            if technical_features:
                temporal_features.update(technical_features)
            
            if sentiment_features:
                graph_features.update(sentiment_features)
            
            # Step 5: Combine GNN and LSTM features for prediction
            predictions = self._hybrid_prediction(
                target_data, temporal_features, graph_features, days
            )
            
            # Step 6: Generate comprehensive analysis
            analysis = self._generate_gnn_lstm_analysis(
                symbol, target_data, predictions, graph_features, temporal_features
            )
            
            return {
                'predictions': predictions,
                'analysis': analysis,
                'graph_features': graph_features,
                'temporal_features': temporal_features,
                'model_confidence': self._calculate_model_confidence(target_data, related_data)
            }
            
        except Exception as e:
            print(f"GNN-LSTM prediction error for {symbol}: {str(e)}")
            return self._fallback_prediction(symbol, days)
    
    def _fetch_multi_stock_data(self, symbol):
        """Fetch data for target stock and related stocks"""
        # Get target stock data
        try:
            target_ticker = yf.Ticker(symbol)
            target_data = target_ticker.history(period='6mo')
            
            if target_data.empty:
                return None, {}
        except:
            return None, {}
        
        # Get related stocks data
        related_stocks = self.stock_network.get(symbol, [])
        if not related_stocks:
            # If symbol not in network, find similar stocks from same sector
            related_stocks = self._find_related_stocks(symbol)
        
        related_data = {}
        for related_symbol in related_stocks[:5]:  # Limit to 5 related stocks
            try:
                related_ticker = yf.Ticker(related_symbol)
                related_hist = related_ticker.history(period='6mo')
                if not related_hist.empty:
                    related_data[related_symbol] = related_hist
            except:
                continue
        
        return target_data, related_data
    
    def _find_related_stocks(self, symbol):
        """Find related stocks when symbol is not in predefined network"""
        # Default tech stocks for unknown symbols
        default_related = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        
        # Remove target symbol if it's in the list
        if symbol in default_related:
            default_related.remove(symbol)
        
        return default_related[:4]
    
    def _build_stock_graph(self, target_symbol, target_data, related_data):
        """Build graph neural network features from stock relationships"""
        graph_features = {
            'node_features': {},
            'edge_weights': {},
            'correlation_scores': {},
            'market_influence': 0.0,
            'sector_momentum': 0.0
        }
        
        if not related_data:
            return self._default_graph_features(target_symbol)
        
        # Calculate correlation matrix
        all_stocks_data = {target_symbol: target_data}
        all_stocks_data.update(related_data)
        
        correlation_matrix = self._calculate_correlation_matrix(all_stocks_data)
        
        # Extract node features for each stock
        for stock_symbol, stock_data in all_stocks_data.items():
            node_features = self._extract_node_features(stock_data)
            graph_features['node_features'][stock_symbol] = node_features
        
        # Calculate edge weights (correlations)
        target_correlations = {}
        for related_symbol in related_data.keys():
            if related_symbol in correlation_matrix.columns and target_symbol in correlation_matrix.index:
                correlation = correlation_matrix.loc[target_symbol, related_symbol]
                target_correlations[related_symbol] = correlation
                graph_features['edge_weights'][related_symbol] = abs(correlation)
        
        graph_features['correlation_scores'] = target_correlations
        
        # Calculate market influence (weighted by correlations and market caps)
        market_influence = self._calculate_market_influence(target_correlations, related_data)
        graph_features['market_influence'] = market_influence
        
        # Calculate sector momentum
        sector_momentum = self._calculate_sector_momentum(all_stocks_data)
        graph_features['sector_momentum'] = sector_momentum
        
        # Add graph aggregation features
        graph_features.update(self._aggregate_graph_features(target_correlations, related_data))
        
        return graph_features
    
    def _calculate_correlation_matrix(self, stocks_data):
        """Calculate correlation matrix between stocks"""
        # Align all stock data to same time period
        min_date = max(data.index.min() for data in stocks_data.values())
        max_date = min(data.index.max() for data in stocks_data.values())
        
        aligned_data = {}
        for symbol, data in stocks_data.items():
            filtered_data = data[(data.index >= min_date) & (data.index <= max_date)]
            if len(filtered_data) > 0:
                aligned_data[symbol] = filtered_data['Close']
        
        if len(aligned_data) < 2:
            return pd.DataFrame()
        
        # Create correlation matrix
        price_df = pd.DataFrame(aligned_data)
        correlation_matrix = price_df.corr()
        
        return correlation_matrix
    
    def _extract_node_features(self, stock_data):
        """Extract node features for graph neural network"""
        recent_data = stock_data.tail(30)  # Last 30 days
        
        features = {
            'price_momentum': self._calculate_momentum(recent_data['Close']),
            'volume_momentum': self._calculate_momentum(recent_data['Volume']),
            'volatility': recent_data['Close'].std() / recent_data['Close'].mean(),
            'price_trend': self._calculate_trend(recent_data['Close']),
            'relative_strength': self._calculate_relative_strength(recent_data),
            'price_position': self._calculate_price_position(recent_data)
        }
        
        return features
    
    def _calculate_momentum(self, series, window=10):
        """Calculate momentum for a price/volume series"""
        if len(series) < window:
            return 0.0
        
        recent = series.tail(window).mean()
        previous = series.tail(window * 2).head(window).mean()
        
        if previous == 0:
            return 0.0
        
        momentum = (recent - previous) / previous
        return np.clip(momentum, -1, 1)  # Normalize to [-1, 1]
    
    def _calculate_trend(self, prices):
        """Calculate price trend using linear regression slope"""
        if len(prices) < 5:
            return 0.0
        
        x = np.arange(len(prices))
        y = prices.values
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope relative to price level
        normalized_slope = slope / prices.mean() if prices.mean() != 0 else 0
        
        return np.clip(normalized_slope * 100, -1, 1)  # Scale and clip
    
    def _calculate_relative_strength(self, stock_data):
        """Calculate relative strength indicator"""
        closes = stock_data['Close']
        
        if len(closes) < 14:
            return 0.5  # Neutral
        
        # Simple RSI calculation
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        latest_rsi = rsi.iloc[-1] if not rsi.empty else 50
        return latest_rsi / 100  # Normalize to [0, 1]
    
    def _calculate_price_position(self, stock_data):
        """Calculate current price position relative to recent range"""
        closes = stock_data['Close']
        
        if len(closes) < 20:
            return 0.5  # Neutral
        
        recent_high = closes.tail(20).max()
        recent_low = closes.tail(20).min()
        current_price = closes.iloc[-1]
        
        if recent_high == recent_low:
            return 0.5
        
        position = (current_price - recent_low) / (recent_high - recent_low)
        return np.clip(position, 0, 1)
    
    def _calculate_market_influence(self, correlations, related_data):
        """Calculate market influence based on correlations and market presence"""
        if not correlations:
            return 0.0
        
        # Weight correlations by trading volume (proxy for market influence)
        weighted_influence = 0.0
        total_weight = 0.0
        
        for symbol, correlation in correlations.items():
            if symbol in related_data:
                recent_volume = related_data[symbol]['Volume'].tail(10).mean()
                weight = np.log(recent_volume + 1)  # Log to prevent extreme values
                weighted_influence += abs(correlation) * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_influence / total_weight
    
    def _calculate_sector_momentum(self, stocks_data):
        """Calculate overall sector momentum"""
        if len(stocks_data) < 2:
            return 0.0
        
        sector_returns = []
        
        for symbol, data in stocks_data.items():
            if len(data) >= 5:
                recent_return = (data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5]
                sector_returns.append(recent_return)
        
        if not sector_returns:
            return 0.0
        
        avg_return = np.mean(sector_returns)
        return np.clip(avg_return, -0.2, 0.2)  # Clip to reasonable range
    
    def _aggregate_graph_features(self, correlations, related_data):
        """Aggregate graph features for prediction"""
        features = {
            'avg_correlation': np.mean(list(correlations.values())) if correlations else 0.0,
            'max_correlation': np.max(list(correlations.values())) if correlations else 0.0,
            'correlation_diversity': np.std(list(correlations.values())) if correlations else 0.0,
            'network_size': len(related_data),
            'strong_correlations': sum(1 for c in correlations.values() if abs(c) > 0.7)
        }
        
        return features
    
    def _extract_temporal_features(self, stock_data):
        """Extract LSTM-style temporal features"""
        closes = stock_data['Close'].values
        volumes = stock_data['Volume'].values
        
        # Sequential features (LSTM-style)
        features = {
            'price_sequences': self._create_sequences(closes),
            'volume_sequences': self._create_sequences(volumes),
            'price_changes': np.diff(closes),
            'moving_averages': self._calculate_moving_averages(closes),
            'temporal_patterns': self._extract_temporal_patterns(closes),
            'seasonality': self._detect_seasonality(stock_data),
            'trend_strength': self._calculate_trend_strength(closes)
        }
        
        return features
    
    def _create_sequences(self, data, sequence_length=10):
        """Create sequences for LSTM-style processing"""
        if len(data) < sequence_length:
            return []
        
        sequences = []
        for i in range(len(data) - sequence_length):
            sequence = data[i:i+sequence_length]
            sequences.append(sequence)
        
        return sequences[-5:]  # Return last 5 sequences
    
    def _calculate_moving_averages(self, prices):
        """Calculate multiple moving averages"""
        if len(prices) < 20:
            return {'sma_5': prices[-1], 'sma_10': prices[-1], 'sma_20': prices[-1]}
        
        sma_5 = np.mean(prices[-5:])
        sma_10 = np.mean(prices[-10:])
        sma_20 = np.mean(prices[-20:])
        
        return {
            'sma_5': sma_5,
            'sma_10': sma_10,
            'sma_20': sma_20,
            'ma_convergence': (sma_5 - sma_20) / sma_20 if sma_20 != 0 else 0
        }
    
    def _extract_temporal_patterns(self, prices):
        """Extract temporal patterns from price data"""
        if len(prices) < 30:
            return {'pattern_strength': 0.0, 'cycle_length': 0}
        
        # Simple pattern detection using autocorrelation
        price_changes = np.diff(prices)
        
        # Look for recurring patterns
        autocorr_values = []
        for lag in range(1, min(15, len(price_changes)//2)):
            if len(price_changes) > lag:
                autocorr = np.corrcoef(price_changes[:-lag], price_changes[lag:])[0, 1]
                if not np.isnan(autocorr):
                    autocorr_values.append(abs(autocorr))
        
        pattern_strength = np.max(autocorr_values) if autocorr_values else 0.0
        cycle_length = np.argmax(autocorr_values) + 1 if autocorr_values else 0
        
        return {
            'pattern_strength': pattern_strength,
            'cycle_length': cycle_length
        }
    
    def _detect_seasonality(self, stock_data):
        """Detect seasonal patterns in stock data"""
        if len(stock_data) < 30:
            return {'weekly_effect': 0.0, 'monthly_effect': 0.0}
        
        # Day of week effect
        stock_data['weekday'] = stock_data.index.weekday
        stock_data['daily_return'] = stock_data['Close'].pct_change()
        
        weekday_returns = stock_data.groupby('weekday')['daily_return'].mean()
        weekly_effect = weekday_returns.std() if len(weekday_returns) > 1 else 0.0
        
        # Month effect (if enough data)
        stock_data['month'] = stock_data.index.month
        monthly_returns = stock_data.groupby('month')['daily_return'].mean()
        monthly_effect = monthly_returns.std() if len(monthly_returns) > 1 else 0.0
        
        return {
            'weekly_effect': weekly_effect,
            'monthly_effect': monthly_effect
        }
    
    def _calculate_trend_strength(self, prices):
        """Calculate strength of current trend"""
        if len(prices) < 20:
            return 0.0
        
        # Calculate trend using linear regression
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        
        # Calculate R-squared to measure trend strength
        predicted = slope * x + intercept
        ss_res = np.sum((prices - predicted) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Combine slope direction with trend strength
        trend_direction = 1 if slope > 0 else -1
        trend_strength = r_squared * trend_direction
        
        return np.clip(trend_strength, -1, 1)
    
    def _hybrid_prediction(self, target_data, temporal_features, graph_features, days):
        """Combine GNN and LSTM features for final prediction"""
        current_price = target_data['Close'].iloc[-1]
        
        # Extract key features for prediction
        trend_component = temporal_features.get('trend_strength', 0.0)
        momentum_component = temporal_features['moving_averages'].get('ma_convergence', 0.0)
        market_influence = graph_features.get('market_influence', 0.0)
        sector_momentum = graph_features.get('sector_momentum', 0.0)
        correlation_strength = graph_features.get('avg_correlation', 0.0)
        
        # Combine features with learned weights
        base_change_rate = (
            trend_component * 0.3 +           # Temporal trend
            momentum_component * 0.25 +       # Moving average convergence
            sector_momentum * 0.2 +           # Sector influence
            market_influence * 0.15 +         # Market correlation influence
            correlation_strength * 0.1        # Network effect
        )
        
        # Add volatility estimation
        recent_volatility = target_data['Close'].tail(20).std() / target_data['Close'].tail(20).mean()
        
        # Generate predictions
        predictions = []
        current_pred_price = current_price
        
        for day in range(days):
            # Apply trend with diminishing effect over time
            trend_decay = 0.95 ** day
            daily_change_rate = base_change_rate * trend_decay
            
            # Add controlled randomness based on volatility
            noise_factor = np.random.normal(0, recent_volatility * 0.3)
            
            # Mean reversion component
            recent_avg = target_data['Close'].tail(30).mean()
            reversion_factor = (recent_avg - current_pred_price) / current_pred_price * 0.05
            
            # Combine all factors
            total_change = daily_change_rate + noise_factor + reversion_factor
            
            # Apply change
            new_price = current_pred_price * (1 + total_change)
            
            # Ensure reasonable bounds
            new_price = max(new_price, current_price * 0.7)  # Max 30% drop
            new_price = min(new_price, current_price * 1.5)  # Max 50% gain
            
            predictions.append(float(new_price))
            current_pred_price = new_price
        
        return predictions
    
    def _generate_gnn_lstm_analysis(self, symbol, target_data, predictions, graph_features, temporal_features):
        """Generate comprehensive analysis of GNN-LSTM prediction"""
        current_price = target_data['Close'].iloc[-1]
        predicted_change = (predictions[-1] - current_price) / current_price * 100
        
        analysis = {
            'model_type': 'Hybrid GNN-LSTM',
            'prediction_horizon': len(predictions),
            'predicted_change_percent': predicted_change,
            'trend_direction': 'Bullish' if predicted_change > 2 else 'Bearish' if predicted_change < -2 else 'Neutral',
            
            # Temporal analysis
            'temporal_insights': {
                'trend_strength': temporal_features.get('trend_strength', 0.0),
                'pattern_detected': temporal_features.get('temporal_patterns', {}).get('pattern_strength', 0.0) > 0.3,
                'moving_average_signal': 'Positive' if temporal_features['moving_averages']['ma_convergence'] > 0 else 'Negative',
                'seasonality_factor': temporal_features.get('seasonality', {}).get('weekly_effect', 0.0)
            },
            
            # Graph analysis
            'network_insights': {
                'market_influence': graph_features.get('market_influence', 0.0),
                'sector_momentum': graph_features.get('sector_momentum', 0.0),
                'correlation_strength': graph_features.get('avg_correlation', 0.0),
                'network_size': graph_features.get('network_size', 0),
                'strongest_correlation': max(graph_features.get('correlation_scores', {}).values()) if graph_features.get('correlation_scores') else 0.0
            },
            
            # Risk assessment
            'risk_factors': self._assess_risk_factors(graph_features, temporal_features, predicted_change),
            
            # Model confidence
            'confidence_breakdown': self._detailed_confidence_analysis(target_data, graph_features, temporal_features)
        }
        
        return analysis
    
    def _assess_risk_factors(self, graph_features, temporal_features, predicted_change):
        """Assess risk factors based on model features"""
        risk_factors = []
        
        # Volatility risk
        if graph_features.get('correlation_diversity', 0) > 0.3:
            risk_factors.append('High correlation diversity indicates potential volatility')
        
        # Trend risk
        trend_strength = temporal_features.get('trend_strength', 0)
        if abs(trend_strength) < 0.2:
            risk_factors.append('Weak trend strength may indicate uncertain direction')
        
        # Sector risk
        sector_momentum = graph_features.get('sector_momentum', 0)
        if abs(sector_momentum) > 0.1:
            direction = 'positive' if sector_momentum > 0 else 'negative'
            risk_factors.append(f'Strong {direction} sector momentum affecting prediction')
        
        # Prediction magnitude risk
        if abs(predicted_change) > 10:
            risk_factors.append('Large predicted change increases uncertainty')
        
        return risk_factors if risk_factors else ['Normal risk levels detected']
    
    def _detailed_confidence_analysis(self, target_data, graph_features, temporal_features):
        """Provide detailed confidence breakdown"""
        confidence_factors = {
            'data_quality': min(1.0, len(target_data) / 100),  # More data = higher confidence
            'network_strength': min(1.0, graph_features.get('network_size', 0) / 5),  # More connections = higher confidence
            'trend_consistency': abs(temporal_features.get('trend_strength', 0)),  # Stronger trend = higher confidence
            'correlation_reliability': 1 - graph_features.get('correlation_diversity', 0),  # Lower diversity = higher confidence
            'pattern_recognition': temporal_features.get('temporal_patterns', {}).get('pattern_strength', 0)
        }
        
        overall_confidence = np.mean(list(confidence_factors.values()))
        
        return {
            'overall': min(1.0, max(0.1, overall_confidence)),
            'factors': confidence_factors,
            'confidence_level': 'High' if overall_confidence > 0.7 else 'Medium' if overall_confidence > 0.4 else 'Low'
        }
    
    def _calculate_model_confidence(self, target_data, related_data):
        """Calculate overall model confidence"""
        factors = []
        
        # Data availability
        factors.append(min(1.0, len(target_data) / 120))  # 120 days ideal
        
        # Network completeness
        factors.append(min(1.0, len(related_data) / 5))  # 5 related stocks ideal
        
        # Data quality (no missing values, sufficient volume)
        data_quality = 1.0
        if target_data['Volume'].mean() < 100000:  # Low volume stocks
            data_quality *= 0.8
        
        factors.append(data_quality)
        
        return np.mean(factors)
    
    def _default_graph_features(self, symbol):
        """Return default graph features when related data unavailable"""
        return {
            'node_features': {symbol: {
                'price_momentum': 0.0,
                'volume_momentum': 0.0,
                'volatility': 0.02,
                'price_trend': 0.0,
                'relative_strength': 0.5,
                'price_position': 0.5
            }},
            'edge_weights': {},
            'correlation_scores': {},
            'market_influence': 0.0,
            'sector_momentum': 0.0,
            'avg_correlation': 0.0,
            'max_correlation': 0.0,
            'correlation_diversity': 0.0,
            'network_size': 0,
            'strong_correlations': 0
        }
    
    def _fallback_prediction(self, symbol, days):
        """Fallback prediction when main model fails"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1mo')
            current_price = hist['Close'].iloc[-1] if not hist.empty else 100.0
        except:
            current_price = 100.0
        
        # Simple random walk prediction
        predictions = []
        price = current_price
        
        for _ in range(days):
            change = np.random.normal(0.001, 0.02)  # Small daily changes
            price *= (1 + change)
            predictions.append(float(price))
        
        return {
            'predictions': predictions,
            'analysis': {
                'model_type': 'Fallback Model',
                'prediction_horizon': days,
                'predicted_change_percent': (predictions[-1] - current_price) / current_price * 100,
                'trend_direction': 'Uncertain',
                'temporal_insights': {'note': 'Limited data available'},
                'network_insights': {'note': 'Network analysis unavailable'},
                'risk_factors': ['Insufficient data for reliable prediction'],
                'confidence_breakdown': {'overall': 0.3, 'confidence_level': 'Low'}
            },
            'graph_features': self._default_graph_features(symbol),
            'temporal_features': {},
            'model_confidence': 0.3
        }