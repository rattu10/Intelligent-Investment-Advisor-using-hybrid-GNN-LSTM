#!/usr/bin/env python3
"""
Test script for the improved stock predictor
Shows accurate predictions with confidence metrics
"""

import sys
import os

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.improved_predictor import ImprovedStockPredictor
import json

def test_improved_predictions():
    """Test the improved predictor with real stocks"""
    print("🧪 Testing Improved Stock Predictor")
    print("=" * 60)
    
    # Initialize improved predictor
    predictor = ImprovedStockPredictor()
    
    # Test with popular stocks
    test_symbols = ['GOOGL', 'AAPL', 'MSFT', 'TSLA']
    
    for symbol in test_symbols:
        print(f"\n📈 Testing {symbol}...")
        print("-" * 40)
        
        try:
            # Get prediction
            result = predictor.predict_stock(symbol, days=7)
            
            if result:
                # Display key results
                print(f"✅ Company: {result['company_name']}")
                print(f"💰 Current Price: ${result['current_price']:.2f}")
                print(f"🔮 7-day Prediction: ${result['predictions'][-1]:.2f}")
                
                # Calculate and display change
                change = (result['predictions'][-1] - result['current_price']) / result['current_price'] * 100
                change_color = "📈" if change > 0 else "📉"
                print(f"{change_color} Expected Change: {change:+.1f}%")
                
                # Display summary
                summary = result['summary']
                print(f"📊 Trend: {summary['trend']}")
                print(f"📈 Volatility: {summary['volatility']}")
                
                # Display accuracy metrics
                if 'accuracy_metrics' in result:
                    acc = result['accuracy_metrics']
                    print(f"🎯 Model Accuracy: {acc['average_accuracy']}")
                    print(f"🔍 Confidence: {acc['prediction_confidence']}")
                    print(f"📋 Reliability: {acc['model_reliability']}")
                
                # Display recommendation
                rec = result['recommendation']
                print(f"💡 Recommendation: {rec['action']} ({rec['confidence']} confidence)")
                print(f"📝 Reason: {rec['reason']}")
                
                print(f"⏰ Last Updated: {result['last_updated']}")
                
            else:
                print(f"❌ Failed to get prediction for {symbol}")
                
        except Exception as e:
            print(f"❌ Error testing {symbol}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ Improved Predictor Test Complete!")
    print("\nKey Improvements:")
    print("• Realistic trend assessment (fixed 'strongly declining' bias)")
    print("• Accuracy metrics with percentage scores")
    print("• Better prediction algorithms with proper bounds")
    print("• Multi-source data fetching for reliability")
    print("• Confidence levels based on data quality")

if __name__ == "__main__":
    test_improved_predictions()