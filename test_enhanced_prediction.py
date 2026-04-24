#!/usr/bin/env python3
"""
Test script to validate the enhanced prediction system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.simple_predictor import SimpleStockPredictor
import json

def test_prediction(symbol="AAPL", days=7):
    """Test the enhanced prediction system"""
    print(f"Testing enhanced prediction for {symbol} over {days} days...")
    print("=" * 60)
    
    predictor = SimpleStockPredictor()
    
    try:
        result = predictor.predict_stock(symbol, days)
        
        print(f"✅ Stock: {symbol}")
        print(f"✅ Company: {result['company_name']}")
        print(f"✅ Current Price: ${result['current_price']:.2f}")
        print(f"✅ Predicted Price: ${result['predictions'][-1]:.2f}")
        print(f"✅ Confidence: {result['confidence']}")
        print(f"✅ Last Updated: {result['last_updated']}")
        
        print(f"\n📊 Prediction Summary:")
        for key, value in result['summary'].items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n💡 Recommendation:")
        rec = result['recommendation']
        print(f"   Action: {rec['action']}")
        print(f"   Reason: {rec['reason']}")
        print(f"   Risk Level: {rec['risk_level']}")
        print(f"   Confidence: {rec['confidence']}")
        
        print(f"\n📅 Prediction Dates:")
        for i, date in enumerate(result['prediction_dates']):
            print(f"   Day {i+1}: {date} - ${result['predictions'][i]:.2f}")
        
        print(f"\n📈 Historical Data Points: {len(result['historical_data']['dates'])}")
        print(f"   Date Range: {result['historical_data']['dates'][0]} to {result['historical_data']['dates'][-1]}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_multiple_stocks():
    """Test multiple popular stocks"""
    stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "RELIANCE.NS", "TCS.NS"]
    
    print("Testing Enhanced Prediction System")
    print("=" * 60)
    
    success_count = 0
    
    for stock in stocks:
        print(f"\n🔍 Testing {stock}...")
        try:
            predictor = SimpleStockPredictor()
            result = predictor.predict_stock(stock, 5)
            
            print(f"   ✅ {stock}: ${result['current_price']:.2f} -> ${result['predictions'][-1]:.2f}")
            print(f"   📊 Trend: {result['summary']['trend']}")
            print(f"   💡 Action: {result['recommendation']['action']}")
            
            success_count += 1
        except Exception as e:
            print(f"   ❌ {stock}: Error - {str(e)}")
    
    print(f"\n📊 Results: {success_count}/{len(stocks)} stocks tested successfully")
    return success_count == len(stocks)

if __name__ == "__main__":
    print("🚀 Enhanced Stock Prediction System Test")
    print("=" * 60)
    
    # Test single stock with detailed output
    test_prediction("AAPL", 7)
    
    print("\n" + "=" * 60)
    
    # Test multiple stocks
    test_multiple_stocks()
    
    print("\n🎉 Testing completed!")