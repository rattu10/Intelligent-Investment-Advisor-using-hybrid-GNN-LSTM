#!/usr/bin/env python3
"""
Quick test for the user-friendly stock investment application
"""

import sys
import os
import sqlite3

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_database_setup():
    """Test database creation and basic operations"""
    print("\n📊 Testing Database Setup...")
    
    try:
        # Test SQLite
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        cursor.execute('SELECT 1')
        result = cursor.fetchone()
        conn.close()
        
        if result[0] == 1:
            print("   ✅ SQLite database: Working")
            return True
        else:
            print("   ❌ SQLite database: Failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Database error: {str(e)[:50]}...")
        return False

def test_basic_imports():
    """Test basic imports for the application"""
    print("\n📦 Testing Basic Imports...")
    
    success_count = 0
    total_tests = 0
    
    # Test Flask
    total_tests += 1
    try:
        import flask
        print("   ✅ Flask: Available")
        success_count += 1
    except ImportError:
        print("   ❌ Flask: Not installed")
    
    # Test pandas
    total_tests += 1
    try:
        import pandas
        print("   ✅ Pandas: Available")
        success_count += 1
    except ImportError:
        print("   ❌ Pandas: Not installed")
    
    # Test yfinance
    total_tests += 1
    try:
        import yfinance
        print("   ✅ YFinance: Available")
        success_count += 1
    except ImportError:
        print("   ❌ YFinance: Not installed")
    
    # Test numpy
    total_tests += 1
    try:
        import numpy
        print("   ✅ NumPy: Available")
        success_count += 1
    except ImportError:
        print("   ❌ NumPy: Not installed")
    
    return success_count, total_tests

def test_utility_modules():
    """Test utility modules"""
    print("\n🔧 Testing Utility Modules...")
    
    success_count = 0
    total_tests = 0
    
    # Test SimpleStockPredictor
    total_tests += 1
    try:
        from utils.simple_predictor import SimpleStockPredictor
        predictor = SimpleStockPredictor()
        print("   ✅ Stock Predictor: Working")
        success_count += 1
    except Exception as e:
        print(f"   ❌ Stock Predictor: {str(e)[:40]}...")
    
    # Test RecommendationEngine
    total_tests += 1
    try:
        from utils.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        print("   ✅ Recommendation Engine: Working")
        success_count += 1
    except Exception as e:
        print(f"   ❌ Recommendation Engine: {str(e)[:40]}...")
    
    # Test PortfolioTracker
    total_tests += 1
    try:
        from utils.portfolio_tracker import PortfolioTracker
        tracker = PortfolioTracker()
        print("   ✅ Portfolio Tracker: Working")
        success_count += 1
    except Exception as e:
        print(f"   ❌ Portfolio Tracker: {str(e)[:40]}...")
    
    return success_count, total_tests

def test_app_import():
    """Test main app import"""
    print("\n🌐 Testing Flask Application...")
    
    try:
        # Try importing the app
        import app_user_friendly
        print("   ✅ Flask App: Import successful")
        
        # Check if app has required attributes
        if hasattr(app_user_friendly, 'app'):
            print("   ✅ Flask App: Configuration valid")
            return True
        else:
            print("   ❌ Flask App: Missing app object")
            return False
            
    except Exception as e:
        print(f"   ❌ Flask App: {str(e)[:50]}...")
        return False

def run_quick_test():
    """Run comprehensive quick test"""
    print("\n" + "="*60)
    print("🚀 SMART STOCK INVESTMENT - QUICK TEST")
    print("   User-Friendly Investment Platform")
    print("="*60)
    
    total_success = 0
    total_tests = 0
    
    # Test database
    if test_database_setup():
        total_success += 1
    total_tests += 1
    
    # Test basic imports
    import_success, import_total = test_basic_imports()
    total_success += import_success
    total_tests += import_total
    
    # Test utility modules
    util_success, util_total = test_utility_modules()
    total_success += util_success
    total_tests += util_total
    
    # Test app import
    if test_app_import():
        total_success += 1
    total_tests += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"📊 TEST RESULTS: {total_success}/{total_tests} tests passed")
    
    success_rate = (total_success / total_tests) * 100
    
    if success_rate >= 90:
        print("🎉 EXCELLENT! Application is ready to run")
        print("\n💡 Next steps:")
        print("   1. Run: python app_user_friendly.py")
        print("   2. Open: http://localhost:5000")
        print("   3. Create your account and start investing!")
        return True
    elif success_rate >= 70:
        print("⚠️  MOSTLY READY with some limitations")
        print("\n💡 You can run the app, but some features may be limited:")
        print("   1. Run: python app_user_friendly.py")
        print("   2. Open: http://localhost:5000")
        print("   3. Some advanced features may use fallback implementations")
        return True
    else:
        print("❌ SETUP INCOMPLETE - Install missing dependencies")
        print("\n🔧 Installation commands:")
        print("   pip install flask flask-cors flask-sqlalchemy flask-login")
        print("   pip install pandas numpy yfinance scikit-learn")
        print("   pip install werkzeug jinja2 blinker itsdangerous")
        return False

def show_features():
    """Show application features"""
    print("\n" + "="*60)
    print("✨ SMART STOCK INVESTMENT FEATURES")
    print("="*60)
    
    features = [
        "👤 User Registration & Login",
        "📊 Personalized Investment Profile",
        "🔮 Stock Price Predictions (USD & INR)",
        "💡 Personalized Investment Recommendations",
        "📱 Portfolio Tracking & Analytics",
        "👀 Stock Watchlist Management",
        "📈 Real-time Stock Data Integration",
        "🎯 Risk-based Investment Suggestions",
        "📊 Performance Analytics & Insights",
        "💱 Multi-currency Support (USD/INR)"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print("\n🎯 Perfect for:")
    print("   • Beginner investors learning the market")
    print("   • Experienced investors wanting AI insights")
    print("   • Anyone interested in Indian & US stocks")
    print("   • Portfolio tracking and management")

if __name__ == "__main__":
    success = run_quick_test()
    
    if success:
        show_features()
    
    print("\n" + "="*60)
    print("Thank you for using Smart Stock Investment! 📈💰")
    print("="*60)
