#!/usr/bin/env python3
"""
Simple test to verify basic functionality without heavy dependencies
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic functionality without PyTorch dependencies"""
    
    print("\n" + "="*50)
    print("🧪 BASIC FUNCTIONALITY TEST")
    print("="*50)
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Data Processor
    total_tests += 1
    try:
        from utils.data_processor import DataProcessor
        processor = DataProcessor()
        data = processor.fetch_stock_data(['AAPL'], period='1mo')
        if 'AAPL' in data and len(data['AAPL']) > 0:
            print("✅ Data Processor: WORKING")
            success_count += 1
        else:
            print("❌ Data Processor: Failed - No data returned")
    except Exception as e:
        print(f"❌ Data Processor: Error - {str(e)[:60]}...")
    
    # Test 2: Technical Indicators
    total_tests += 1
    try:
        from utils.technical_indicators import TechnicalIndicatorClusterer
        clusterer = TechnicalIndicatorClusterer()
        
        # Use the data from previous test if available
        if 'data' in locals() and 'AAPL' in data:
            result = clusterer.process_indicators(data['AAPL'])
            if 'clusters' in result and len(result['clusters']) > 0:
                print("✅ Technical Indicators: WORKING")
                success_count += 1
            else:
                print("❌ Technical Indicators: Failed - No clusters generated")
        else:
            print("❌ Technical Indicators: Failed - No input data")
    except Exception as e:
        print(f"❌ Technical Indicators: Error - {str(e)[:60]}...")
    
    # Test 3: Sentiment Analyzer (basic)
    total_tests += 1
    try:
        from utils.sentiment_analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        
        # Test basic sentiment without heavy model loading
        test_text = "The company reported excellent quarterly results."
        sentiment = analyzer._simple_sentiment_analysis(test_text)
        
        if 'label' in sentiment and 'score' in sentiment:
            print("✅ Sentiment Analyzer: WORKING")
            success_count += 1
        else:
            print("❌ Sentiment Analyzer: Failed - Invalid output format")
    except Exception as e:
        print(f"❌ Sentiment Analyzer: Error - {str(e)[:60]}...")
    
    # Test 4: Flask App Import
    total_tests += 1
    try:
        import app
        if hasattr(app, 'app') and hasattr(app.app, 'run'):
            print("✅ Flask Application: WORKING")
            success_count += 1
        else:
            print("❌ Flask Application: Failed - Invalid app structure")
    except Exception as e:
        print(f"❌ Flask Application: Error - {str(e)[:60]}...")
    
    # Test 5: Configuration
    total_tests += 1
    try:
        import json
        with open('config.json', 'r') as f:
            config = json.load(f)
        if 'model_config' in config and 'app_config' in config:
            print("✅ Configuration: WORKING")
            success_count += 1
        else:
            print("❌ Configuration: Failed - Missing sections")
    except Exception as e:
        print(f"❌ Configuration: Error - {str(e)[:60]}...")
    
    # Summary
    print("\n" + "-"*50)
    print(f"📊 TEST SUMMARY: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 ALL BASIC TESTS PASSED!")
        print("\n💡 Ready for web interface:")
        print("   1. Run: python app.py")
        print("   2. Open: http://localhost:5000")
        print("   3. Note: ML predictions will use fallback methods")
        return True
    elif success_count >= 3:
        print("⚠️  MOSTLY WORKING (some advanced features may be limited)")
        print("\n💡 You can still use the web interface:")
        print("   1. Run: python app.py")
        print("   2. Open: http://localhost:5000")
        print("   3. Note: Some features may use fallback implementations")
        return True
    else:
        print("❌ MULTIPLE FAILURES DETECTED")
        print("\n🔧 Troubleshooting needed:")
        print("   1. Install missing dependencies")
        print("   2. Check internet connection")
        print("   3. Review error messages above")
        return False

def show_installation_guide():
    """Show step-by-step installation guide"""
    
    print("\n" + "="*50)
    print("📦 INSTALLATION GUIDE")
    print("="*50)
    
    print("\n1️⃣ Basic Dependencies (Essential):")
    print("   pip install flask flask-cors pandas numpy yfinance ta")
    
    print("\n2️⃣ Machine Learning Dependencies (Optional):")
    print("   pip install scikit-learn torch transformers")
    
    print("\n3️⃣ Visualization Dependencies (Optional):")
    print("   pip install plotly matplotlib seaborn")
    
    print("\n4️⃣ Advanced Dependencies (Optional):")
    print("   pip install torch-geometric networkx sentence-transformers")
    
    print("\n💡 Quick Start (Minimal Setup):")
    print("   pip install flask flask-cors pandas numpy yfinance")
    print("   python app.py")
    
    print("\n🔧 If you encounter issues:")
    print("   • Try installing dependencies one by one")
    print("   • Use virtual environment: python -m venv venv")
    print("   • Check Python version: python --version (3.8+ required)")
    print("   • Update pip: pip install --upgrade pip")

if __name__ == "__main__":
    print("🚀 Stock Market Forecasting - Basic Test")
    
    success = test_basic_functionality()
    
    if not success:
        show_installation_guide()
    
    print("\n" + "="*50)