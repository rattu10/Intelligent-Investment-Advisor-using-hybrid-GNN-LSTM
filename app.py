from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from models.gnn_lstm_model import GNNLSTMPredictor
from utils.data_processor import DataProcessor
from utils.sentiment_analyzer import SentimentAnalyzer
from utils.technical_indicators import TechnicalIndicatorClusterer
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Initialize components
data_processor = DataProcessor()
sentiment_analyzer = SentimentAnalyzer()
technical_clusterer = TechnicalIndicatorClusterer()
model_predictor = GNNLSTMPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_stock():
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL')
        days_ahead = data.get('days_ahead', 5)
        
        # Fetch stock data
        stock_data = data_processor.fetch_stock_data([symbol], period='1y')
        
        # Process technical indicators
        technical_data = technical_clusterer.process_indicators(stock_data[symbol])
        
        # Get sentiment data
        sentiment_data = sentiment_analyzer.analyze_stock_sentiment(symbol)
        
        # Make predictions
        predictions = model_predictor.predict(
            technical_data, 
            sentiment_data, 
            days_ahead=days_ahead
        )
        
        # Get current price for reference
        current_price = stock_data[symbol]['Close'].iloc[-1]
        
        response = {
            'success': True,
            'symbol': symbol,
            'current_price': float(current_price),
            'predictions': predictions.tolist(),
            'technical_clusters': technical_data['clusters'],
            'sentiment_score': sentiment_data['overall_sentiment'],
            'sentiment_events': sentiment_data['events'][:5]  # Top 5 events
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stocks/search', methods=['GET'])
def search_stocks():
    query = request.args.get('q', '')
    # Simple stock symbol suggestions
    common_stocks = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.'},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
        {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
        {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
        {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
        {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
        {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'},
        {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.'},
        {'symbol': 'V', 'name': 'Visa Inc.'},
        {'symbol': 'JNJ', 'name': 'Johnson & Johnson'}
    ]
    
    if query:
        filtered = [s for s in common_stocks if query.upper() in s['symbol'] or query.lower() in s['name'].lower()]
        return jsonify(filtered[:5])
    
    return jsonify(common_stocks[:5])

@app.route('/api/technical-analysis/<symbol>')
def get_technical_analysis(symbol):
    try:
        # Fetch recent stock data
        stock_data = data_processor.fetch_stock_data([symbol], period='3mo')
        
        # Process technical indicators
        technical_data = technical_clusterer.process_indicators(stock_data[symbol])
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'indicators': technical_data['indicators'],
            'clusters': technical_data['clusters'],
            'cluster_analysis': technical_data['cluster_analysis']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/sentiment/<symbol>')
def get_sentiment_analysis(symbol):
    try:
        sentiment_data = sentiment_analyzer.analyze_stock_sentiment(symbol)
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'sentiment_data': sentiment_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
