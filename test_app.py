#!/usr/bin/env python3
"""
Test script for the Stock Market Forecasting Application
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_processor import DataProcessor
from utils.technical_indicators import TechnicalIndicatorClusterer
from utils.sentiment_analyzer import SentimentAnalyzer
from models.gnn_lstm_model import GNNLSTMPredictor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()
    
    def test_fetch_stock_data(self):
        """Test stock data fetching"""
        data = self.processor.fetch_stock_data(['AAPL'], period='1mo')
        self.assertIn('AAPL', data)
        self.assertIsNotNone(data['AAPL'])
        self.assertTrue(len(data['AAPL']) > 0)
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        data = self.processor.fetch_stock_data(['AAPL'], period='1mo')
        processed = self.processor.preprocess_data(data)
        self.assertIn('AAPL', processed)
        self.assertIn('Returns', processed['AAPL'].columns)

class TestTechnicalIndicators(unittest.TestCase):
    def setUp(self):
        self.clusterer = TechnicalIndicatorClusterer()
        self.processor = DataProcessor()
    
    def test_calculate_indicators(self):
        """Test technical indicator calculation"""
        data = self.processor.fetch_stock_data(['AAPL'], period='3mo')
        indicators = self.clusterer.calculate_indicators(data['AAPL'])
        self.assertIsInstance(indicators, dict)
        self.assertTrue(len(indicators) > 0)
    
    def test_cluster_indicators(self):
        """Test indicator clustering"""
        data = self.processor.fetch_stock_data(['AAPL'], period='3mo')
        result = self.clusterer.process_indicators(data['AAPL'])
        self.assertIn('clusters', result)
        self.assertIn('cluster_analysis', result)

class TestSentimentAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = SentimentAnalyzer()
    
    def test_sentiment_analysis(self):
        """Test basic sentiment analysis"""
        text = "The company reported excellent quarterly results with strong growth."
        sentiment = self.analyzer.analyze_sentiment(text)
        self.assertIn('label', sentiment)
        self.assertIn('score', sentiment)
    
    def test_ontology_extraction(self):
        """Test ontology relationship extraction"""
        text = "Apple announced strong earnings with revenue growth exceeding expectations."
        relations = self.analyzer.extract_ontology_relations(text)
        self.assertIn('events', relations)
        self.assertIn('impacts', relations)
    
    def test_stock_sentiment_analysis(self):
        """Test comprehensive stock sentiment analysis"""
        result = self.analyzer.analyze_stock_sentiment('AAPL')
        self.assertIn('overall_sentiment', result)
        self.assertIn('articles', result)
        self.assertIn('events', result)

class TestGNNLSTMModel(unittest.TestCase):
    def setUp(self):
        self.predictor = GNNLSTMPredictor()
        self.processor = DataProcessor()
        self.clusterer = TechnicalIndicatorClusterer()
        self.analyzer = SentimentAnalyzer()
    
    def test_prediction_pipeline(self):
        """Test the complete prediction pipeline"""
        # Get data
        stock_data = self.processor.fetch_stock_data(['AAPL'], period='1y')
        technical_data = self.clusterer.process_indicators(stock_data['AAPL'])
        sentiment_data = self.analyzer.analyze_stock_sentiment('AAPL')
        
        # Make prediction
        predictions = self.predictor.predict(technical_data, sentiment_data, days_ahead=5)
        
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(p > 0 for p in predictions))  # Prices should be positive

class TestApplicationIntegration(unittest.TestCase):
    def test_full_pipeline(self):
        """Test the complete application pipeline"""
        print("\n=== Testing Full Application Pipeline ===")
        
        # Initialize components
        processor = DataProcessor()
        clusterer = TechnicalIndicatorClusterer()
        analyzer = SentimentAnalyzer()
        predictor = GNNLSTMPredictor()
        
        symbol = 'AAPL'
        print(f"Testing with symbol: {symbol}")
        
        # Step 1: Fetch data
        print("1. Fetching stock data...")
        stock_data = processor.fetch_stock_data([symbol], period='6mo')
        self.assertIn(symbol, stock_data)
        print(f"   ✓ Fetched {len(stock_data[symbol])} data points")
        
        # Step 2: Process technical indicators
        print("2. Processing technical indicators...")
        technical_data = clusterer.process_indicators(stock_data[symbol])
        self.assertIn('clusters', technical_data)
        print(f"   ✓ Created {len(technical_data['clusters'])} indicator clusters")
        
        # Step 3: Analyze sentiment
        print("3. Analyzing sentiment...")
        sentiment_data = analyzer.analyze_stock_sentiment(symbol)
        self.assertIn('overall_sentiment', sentiment_data)
        print(f"   ✓ Overall sentiment: {sentiment_data['overall_sentiment']:.3f}")
        
        # Step 4: Make predictions
        print("4. Generating predictions...")
        predictions = predictor.predict(technical_data, sentiment_data, days_ahead=7)
        self.assertEqual(len(predictions), 7)
        print(f"   ✓ Generated {len(predictions)} price predictions")
        
        # Step 5: Validate results
        print("5. Validating results...")
        current_price = stock_data[symbol]['Close'].iloc[-1]
        print(f"   Current price: ${current_price:.2f}")
        print(f"   Predicted prices: ${predictions[0]:.2f} - ${predictions[-1]:.2f}")
        
        # Basic sanity checks
        self.assertTrue(all(p > 0 for p in predictions))  # Positive prices
        self.assertTrue(all(abs(p - current_price) / current_price < 0.5 for p in predictions))  # Reasonable range
        
        print("   ✓ All validations passed")
        print("\n=== Pipeline Test Completed Successfully ===")

def run_quick_test():
    """Run a quick functionality test"""
    print("\n" + "="*50)
    print("STOCK MARKET FORECASTING - QUICK TEST")
    print("="*50)
    
    try:
        # Test data processing
        print("\n[1/4] Testing Data Processing...")
        processor = DataProcessor()
        data = processor.fetch_stock_data(['AAPL'], period='1mo')
        print("     ✓ Data fetching works")
        
        # Test technical indicators
        print("\n[2/4] Testing Technical Indicators...")
        clusterer = TechnicalIndicatorClusterer()
        tech_result = clusterer.process_indicators(data['AAPL'])
        print(f"     ✓ Generated {len(tech_result['clusters'])} indicator clusters")
        
        # Test sentiment analysis
        print("\n[3/4] Testing Sentiment Analysis...")
        analyzer = SentimentAnalyzer()
        sentiment_result = analyzer.analyze_stock_sentiment('AAPL')
        print(f"     ✓ Sentiment score: {sentiment_result['overall_sentiment']:.3f}")
        
        # Test prediction
        print("\n[4/4] Testing Prediction Model...")
        predictor = GNNLSTMPredictor()
        predictions = predictor.predict(tech_result, sentiment_result, days_ahead=3)
        print(f"     ✓ Generated {len(predictions)} predictions")
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED! ✅")
        print("The application is ready to use.")
        print("Run 'python app.py' to start the web interface.")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check internet connection for data fetching")
        print("3. Try running individual test modules")
        return False

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Stock Market Forecasting Application')
    parser.add_argument('--quick', action='store_true', help='Run quick functionality test')
    parser.add_argument('--full', action='store_true', help='Run full test suite')
    
    args = parser.parse_args()
    
    if args.quick or (not args.full and not args.quick):
        # Default to quick test
        success = run_quick_test()
        sys.exit(0 if success else 1)
    
    elif args.full:
        # Run full test suite
        unittest.main(verbosity=2)
