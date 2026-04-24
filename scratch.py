import sys
sys.path.append('.')
from utils.advanced_hybrid_predictor import AdvancedHybridPredictor

if __name__ == '__main__':
    predictor = AdvancedHybridPredictor()
    print("Testing predictor...")
    result = predictor.predict_stock('AAPL', days=5)
    print("Prediction result keys:", result.keys())
    print("Predicted prices:", result.get('predictions'))
    print("Current price:", result.get('current_price'))
