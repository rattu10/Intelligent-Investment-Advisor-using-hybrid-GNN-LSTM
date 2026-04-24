#!/usr/bin/env python3
"""
Demo script to showcase the Stock Market Forecasting Application functionality
"""

import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_application():
    """Run a complete demo of the application"""
    
    print("\n" + "="*60)
    print("🚀 STOCK MARKET FORECASTING - DEMO")
    print("   Hybrid GNN-LSTM Framework with Ontology-Driven Sentiment")
    print("="*60)
    
    try:
        # Import modules
        from utils.data_processor import DataProcessor
        from utils.technical_indicators import TechnicalIndicatorClusterer
        from utils.sentiment_analyzer import SentimentAnalyzer
        from models.gnn_lstm_model import GNNLSTMPredictor
        
        print("\n✅ All modules imported successfully")
        
        # Initialize components
        print("\n📊 Initializing components...")
        processor = DataProcessor()
        clusterer = TechnicalIndicatorClusterer()
        analyzer = SentimentAnalyzer()
        predictor = GNNLSTMPredictor()
        
        # Demo with Apple stock
        symbol = 'AAPL'
        print(f"\n🍎 Analyzing {symbol} (Apple Inc.)")
        
        # Step 1: Data Processing
        print("\n" + "-"*40)
        print("STEP 1: DATA PROCESSING")
        print("-"*40)
        
        stock_data = processor.fetch_stock_data([symbol], period='6mo')
        df = stock_data[symbol]
        
        print(f"📈 Fetched {len(df)} trading days of data")
        print(f"💰 Current price: ${df['Close'].iloc[-1]:.2f}")
        print(f"📊 Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        print(f"📉 Recent volatility: {df['Close'].pct_change().std()*100:.2f}%")
        
        # Step 2: Technical Analysis
        print("\n" + "-"*40)
        print("STEP 2: TECHNICAL INDICATOR CLUSTERING")
        print("-"*40)
        
        technical_data = clusterer.process_indicators(df)
        
        print(f"🔧 Calculated {len(technical_data['indicators'])} technical indicators")
        print(f"🎯 Created {len(technical_data['clusters'])} indicator clusters:")
        
        for cluster_name, indicators in technical_data['clusters'].items():
            print(f"   • {cluster_name}: {len(indicators)} indicators")
            print(f"     {', '.join(indicators[:3])}{'...' if len(indicators) > 3 else ''}")
        
        # Show cluster analysis
        print("\n📊 Cluster Analysis:")
        for cluster_name, analysis in technical_data['cluster_analysis'].items():
            print(f"   • {cluster_name}: {analysis['cluster_type']}")
            print(f"     Avg Correlation: {analysis['avg_correlation']:.3f}")
        
        # Step 3: Sentiment Analysis
        print("\n" + "-"*40)
        print("STEP 3: ONTOLOGY-DRIVEN SENTIMENT ANALYSIS")
        print("-"*40)
        
        sentiment_data = analyzer.analyze_stock_sentiment(symbol)
        
        sentiment_score = sentiment_data['overall_sentiment']
        if sentiment_score > 0.1:
            sentiment_label = "🟢 POSITIVE"
        elif sentiment_score < -0.1:
            sentiment_label = "🔴 NEGATIVE"
        else:
            sentiment_label = "🟡 NEUTRAL"
        
        print(f"📰 Analyzed {len(sentiment_data['articles'])} news articles")
        print(f"😊 Overall sentiment: {sentiment_label} ({sentiment_score:.3f})")
        print(f"📊 Sentiment volatility: {sentiment_data['sentiment_volatility']:.3f}")
        
        # Show recent events
        if sentiment_data['events']['total_events'] > 0:
            print(f"\n🎯 Detected {sentiment_data['events']['total_events']} market events:")
            for event_type, count in sentiment_data['events']['top_events']:
                print(f"   • {event_type}: {count} occurrences")
        
        # Show sample articles
        print("\n📰 Recent news analysis:")
        for i, article in enumerate(sentiment_data['articles'][:3], 1):
            print(f"   {i}. {article['title'][:60]}...")
            print(f"      Sentiment: {article['sentiment']['label']} ({article['sentiment']['score']:.2f})")
            print(f"      Relevance: {article['relevance_score']:.2f}")
        
        # Step 4: GNN-LSTM Prediction
        print("\n" + "-"*40)
        print("STEP 4: GNN-LSTM HYBRID PREDICTION")
        print("-"*40)
        
        prediction_days = 7
        predictions = predictor.predict(technical_data, sentiment_data, days_ahead=prediction_days)
        
        current_price = df['Close'].iloc[-1]
        final_prediction = predictions[-1]
        price_change = ((final_prediction - current_price) / current_price) * 100
        
        print(f"🤖 Generated {len(predictions)} day price forecast")
        print(f"📈 Current price: ${current_price:.2f}")
        print(f"🎯 {prediction_days}-day prediction: ${final_prediction:.2f}")
        print(f"📊 Expected change: {price_change:+.2f}%")
        
        # Show daily predictions
        print(f"\n📅 Daily predictions for {symbol}:")
        for day, price in enumerate(predictions, 1):
            change = ((price - current_price) / current_price) * 100
            print(f"   Day {day}: ${price:.2f} ({change:+.2f}%)")
        
        # Step 5: Model Insights
        print("\n" + "-"*40)
        print("STEP 5: MODEL INSIGHTS & ARCHITECTURE")
        print("-"*40)
        
        print("🧠 GNN-LSTM Hybrid Architecture:")
        print("   • Graph Neural Network: Captures feature relationships")
        print("   • LSTM Network: Models temporal dependencies")
        print("   • Attention Mechanism: Weights feature importance")
        print("   • Multi-modal Fusion: Combines technical + sentiment data")
        
        print("\n🔬 Technical Innovation:")
        print("   ✓ Clustered technical indicators")
        print("   ✓ Ontology-driven sentiment extraction")
        print("   ✓ Graph-temporal modeling fusion")
        print("   ✓ Interpretable prediction framework")
        
        # Summary
        print("\n" + "="*60)
        print("📋 DEMO SUMMARY")
        print("="*60)
        
        summary = {
            "symbol": symbol,
            "current_price": f"${current_price:.2f}",
            "predicted_price": f"${final_prediction:.2f}",
            "expected_change": f"{price_change:+.2f}%",
            "sentiment_score": f"{sentiment_score:.3f}",
            "technical_clusters": len(technical_data['clusters']),
            "news_articles": len(sentiment_data['articles']),
            "prediction_days": prediction_days
        }
        
        for key, value in summary.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        
        print("\n🎉 Demo completed successfully!")
        print("🌐 Run 'python app.py' to start the web interface")
        print("📱 Then open http://localhost:5000 in your browser")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n💡 Solution: Install dependencies with:")
        print("   pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"\n❌ Demo Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check internet connection")
        print("   2. Verify all dependencies are installed")
        print("   3. Run: python test_app.py --quick")
        return False

def show_web_interface_info():
    """Show information about the web interface"""
    
    print("\n" + "="*60)
    print("🌐 WEB INTERFACE FEATURES")
    print("="*60)
    
    features = [
        "📊 Interactive stock prediction dashboard",
        "📈 Real-time price charts with Plotly.js",
        "🔧 Technical indicator clustering visualization",
        "📰 Sentiment analysis with ontology extraction",
        "🎯 Multi-day prediction forecasts",
        "💫 Modern, responsive UI with Bootstrap 5",
        "🔍 Stock symbol search and autocomplete",
        "📱 Mobile-friendly responsive design"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print("\n🚀 How to use:")
    print("   1. Start the app: python app.py")
    print("   2. Open browser: http://localhost:5000")
    print("   3. Enter stock symbol (e.g., AAPL)")
    print("   4. Select prediction period (5-30 days)")
    print("   5. Click 'Generate Prediction'")
    print("   6. View results, charts, and analysis")
    
    print("\n📊 Available stock symbols:")
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ"]
    print("   " + ", ".join(symbols))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Market Forecasting Demo')
    parser.add_argument('--demo', action='store_true', help='Run full demo')
    parser.add_argument('--web-info', action='store_true', help='Show web interface info')
    
    args = parser.parse_args()
    
    if args.web_info:
        show_web_interface_info()
    elif args.demo or len(sys.argv) == 1:
        # Default to demo
        success = demo_application()
        if success:
            show_web_interface_info()
    
    print("\n" + "="*60)
    print("Thank you for exploring the Stock Market Forecasting Framework!")
    print("="*60)