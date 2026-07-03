from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import json
import requests
from utils.simple_predictor import SimpleStockPredictor
from utils.improved_predictor import ImprovedStockPredictor
from utils.advanced_hybrid_predictor import AdvancedHybridPredictor
from utils.recommendation_engine import RecommendationEngine
from utils.portfolio_tracker import PortfolioTracker
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stock_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
CORS(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize components
predictor = AdvancedHybridPredictor()  # Use advanced GNN-LSTM and Ontology hybrid predictor
recommendation_engine = RecommendationEngine()
portfolio_tracker = PortfolioTracker()

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    full_name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    monthly_income = db.Column(db.Float, default=0.0)
    investment_experience = db.Column(db.String(20), default='beginner')  # beginner, intermediate, expert
    risk_tolerance = db.Column(db.String(20), default='low')  # low, medium, high
    investment_goals = db.Column(db.Text)
    preferred_sectors = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    watchlist = db.relationship('Watchlist', backref='user', lazy=True, cascade='all, delete-orphan')
    portfolio = db.relationship('Portfolio', backref='user', lazy=True, cascade='all, delete-orphan')

# Watchlist Model
class Watchlist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    company_name = db.Column(db.String(100))
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

# Portfolio Model
class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    company_name = db.Column(db.String(100))
    shares = db.Column(db.Float, nullable=False)
    purchase_price = db.Column(db.Float, nullable=False)
    purchase_date = db.Column(db.DateTime, default=datetime.utcnow)
    current_price = db.Column(db.Float, default=0.0)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Utility Functions
def get_currency_rate():
    """Get USD to INR conversion rate"""
    try:
        # Using a free API for currency conversion
        response = requests.get('https://api.exchangerate-api.com/v4/latest/USD')
        data = response.json()
        return data['rates'].get('INR', 82.0)  # Fallback rate
    except:
        return 82.0  # Default rate if API fails

def format_currency(amount, currency='USD'):
    """Format currency with proper symbols"""
    if currency == 'USD':
        return f"${amount:,.2f}"
    else:  # INR
        return f"₹{amount:,.2f}"

def get_stock_info(symbol):
    """Get basic stock information"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0)
        }
    except:
        return {
            'name': symbol,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': 0
        }

# Routes
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        full_name = data.get('full_name')
        
        # Check if user exists
        if User.query.filter_by(username=username).first():
            if request.is_json:
                return jsonify({'success': False, 'message': 'Username already exists'})
            flash('Username already exists')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            if request.is_json:
                return jsonify({'success': False, 'message': 'Email already exists'})
            flash('Email already exists')
            return render_template('register.html')
        
        # Create new user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            full_name=full_name
        )
        
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        
        if request.is_json:
            return jsonify({'success': True, 'message': 'Registration successful'})
        
        flash('Registration successful!')
        return redirect(url_for('profile_setup'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        
        username = data.get('username')
        password = data.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            if request.is_json:
                return jsonify({'success': True, 'message': 'Login successful'})
            return redirect(url_for('dashboard'))
        else:
            if request.is_json:
                return jsonify({'success': False, 'message': 'Invalid username or password'})
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/profile-setup')
@login_required
def profile_setup():
    return render_template('profile_setup.html')

@app.route('/api/profile', methods=['POST'])
@login_required
def update_profile():
    data = request.get_json()
    
    current_user.age = data.get('age')
    current_user.monthly_income = float(data.get('monthly_income', 0))
    current_user.investment_experience = data.get('investment_experience')
    current_user.risk_tolerance = data.get('risk_tolerance')
    current_user.investment_goals = data.get('investment_goals')
    current_user.preferred_sectors = json.dumps(data.get('preferred_sectors', []))
    
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Profile updated successfully'})

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's portfolio summary
    holdings = Portfolio.query.filter_by(user_id=current_user.id).all()
    portfolio_value = portfolio_tracker.get_portfolio_value(holdings)
    watchlist_items = Watchlist.query.filter_by(user_id=current_user.id).all()
    
    return render_template('dashboard.html', 
                         portfolio_value=portfolio_value,
                         watchlist_count=len(watchlist_items))

@app.route('/predict')
@login_required
def predict_page():
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
@login_required
def predict_stock():
    data = request.json
    symbol = data.get('symbol', '').upper()
    days = int(data.get('days', 7))
    
    try:
        # Get enhanced stock prediction
        prediction_data = predictor.predict_stock(symbol, days)
        
        # Get currency rate
        usd_to_inr = get_currency_rate()
        
        # Convert prices to both currencies
        current_price_usd = prediction_data['current_price']
        current_price_inr = current_price_usd * usd_to_inr
        
        predictions_inr = [price * usd_to_inr for price in prediction_data['predictions']]
        
        # Convert historical data to INR as well
        historical_data_inr = {
            'dates': prediction_data['historical_data']['dates'],
            'prices': [price * usd_to_inr for price in prediction_data['historical_data']['prices']],
            'volumes': prediction_data['historical_data']['volumes'],
            'highs': [price * usd_to_inr for price in prediction_data['historical_data']['highs']],
            'lows': [price * usd_to_inr for price in prediction_data['historical_data']['lows']]
        }
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'company_name': prediction_data['company_name'],
            'current_price': {
                'usd': current_price_usd,
                'inr': current_price_inr,
                'usd_formatted': format_currency(current_price_usd, 'USD'),
                'inr_formatted': format_currency(current_price_inr, 'INR')
            },
            'predictions': {
                'usd': prediction_data['predictions'],
                'inr': predictions_inr
            },
            'prediction_dates': prediction_data['prediction_dates'],
            'historical_data': {
                'usd': prediction_data['historical_data'],
                'inr': historical_data_inr
            },
            'prediction_summary': prediction_data['summary'],
            'recommendation': prediction_data['recommendation'],
            'confidence': prediction_data['confidence'],
            'last_updated': prediction_data['last_updated']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/recommendations')
@login_required
def recommendations_page():
    return render_template('recommendations.html')

@app.route('/api/recommendations')
@login_required
def get_recommendations():
    try:
        # Get personalized recommendations
        recommendations = recommendation_engine.get_personalized_recommendations(current_user)
        
        # Convert to both currencies
        usd_to_inr = get_currency_rate()
        
        for rec in recommendations:
            rec['price_usd_formatted'] = format_currency(rec['current_price'], 'USD')
            rec['price_inr_formatted'] = format_currency(rec['current_price'] * usd_to_inr, 'INR')
            rec['target_price_usd_formatted'] = format_currency(rec['target_price'], 'USD')
            rec['target_price_inr_formatted'] = format_currency(rec['target_price'] * usd_to_inr, 'INR')
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/portfolio')
@login_required
def portfolio_page():
    return render_template('portfolio.html')

@app.route('/api/portfolio')
@login_required
def get_portfolio():
    try:
        holdings = Portfolio.query.filter_by(user_id=current_user.id).all()
        portfolio_data = portfolio_tracker.get_user_portfolio(holdings)
        
        # Convert to both currencies
        usd_to_inr = get_currency_rate()
        
        for item in portfolio_data['holdings']:
            item['current_value_inr'] = item['current_value'] * usd_to_inr
            item['total_return_inr'] = item['total_return'] * usd_to_inr
            item['current_value_formatted'] = format_currency(item['current_value'], 'USD')
            item['current_value_inr_formatted'] = format_currency(item['current_value_inr'], 'INR')
        
        portfolio_data['total_value_inr'] = portfolio_data['total_value'] * usd_to_inr
        portfolio_data['total_return_inr'] = portfolio_data['total_return'] * usd_to_inr
        
        return jsonify({
            'success': True,
            'portfolio': portfolio_data,
            'usd_to_inr_rate': usd_to_inr
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/add-to-watchlist', methods=['POST'])
@login_required
def add_to_watchlist():
    data = request.json
    symbol = data.get('symbol', '').upper()
    
    # Check if already in watchlist
    existing = Watchlist.query.filter_by(user_id=current_user.id, symbol=symbol).first()
    if existing:
        return jsonify({'success': False, 'message': 'Stock already in watchlist'})
    
    # Get stock info
    stock_info = get_stock_info(symbol)
    
    # Add to watchlist
    watchlist_item = Watchlist(
        user_id=current_user.id,
        symbol=symbol,
        company_name=stock_info['name']
    )
    
    db.session.add(watchlist_item)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Added to watchlist'})

@app.route('/api/add-to-portfolio', methods=['POST'])
@login_required
def add_to_portfolio():
    data = request.json
    symbol = data.get('symbol', '').upper()
    shares = float(data.get('shares', 0))
    purchase_price = float(data.get('purchase_price', 0))
    
    # Get stock info
    stock_info = get_stock_info(symbol)
    
    # Add to portfolio
    portfolio_item = Portfolio(
        user_id=current_user.id,
        symbol=symbol,
        company_name=stock_info['name'],
        shares=shares,
        purchase_price=purchase_price
    )
    
    db.session.add(portfolio_item)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Added to portfolio'})

@app.route('/api/remove-from-portfolio', methods=['POST'])
@login_required
def remove_from_portfolio():
    data = request.json
    symbol = data.get('symbol', '').upper()
    
    # Remove from portfolio
    Portfolio.query.filter_by(user_id=current_user.id, symbol=symbol).delete()
    db.session.commit()
    
    return jsonify({'success': True, 'message': f'Removed {symbol} from portfolio'})

@app.route('/api/search-stocks')
def search_stocks():
    query = request.args.get('q', '').upper()
    
    # Popular Indian and US stocks
    stocks = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.', 'exchange': 'NASDAQ'},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'exchange': 'NASDAQ'},
        {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'exchange': 'NASDAQ'},
        {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'exchange': 'NASDAQ'},
        {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'exchange': 'NASDAQ'},
        {'symbol': 'RELIANCE.NS', 'name': 'Reliance Industries Ltd', 'exchange': 'NSE'},
        {'symbol': 'TCS.NS', 'name': 'Tata Consultancy Services', 'exchange': 'NSE'},
        {'symbol': 'INFY.NS', 'name': 'Infosys Limited', 'exchange': 'NSE'},
        {'symbol': 'HDFCBANK.NS', 'name': 'HDFC Bank Limited', 'exchange': 'NSE'},
        {'symbol': 'ICICIBANK.NS', 'name': 'ICICI Bank Limited', 'exchange': 'NSE'}
    ]
    
    if query:
        filtered = [s for s in stocks if query in s['symbol'] or query.lower() in s['name'].lower()]
        return jsonify(filtered[:10])
    
    return jsonify(stocks[:10])

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    print("\n🚀 Starting Intelligent Investment Advisor App")
    print("=" * 50)
    print("🌐 Open your browser: http://localhost:5000")
    print("✨ Features: Login, Predictions, Recommendations, Portfolio")
    print("💰 Currency: USD & INR support")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
