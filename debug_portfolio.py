#!/usr/bin/env python3
"""
Debug script to check portfolio functionality and create sample data
"""

import sys
import os

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app_user_friendly import app, db, User, Portfolio, Watchlist
from werkzeug.security import generate_password_hash
from datetime import datetime
import yfinance as yf

def create_database_tables():
    """Create all database tables"""
    print("🗄️ Creating database tables...")
    with app.app_context():
        try:
            db.create_all()
            print("✅ Database tables created successfully!")
            return True
        except Exception as e:
            print(f"❌ Error creating tables: {str(e)}")
            return False

def create_demo_user():
    """Create a demo user if doesn't exist"""
    print("👤 Creating demo user...")
    with app.app_context():
        try:
            # Check if demo user exists
            demo_user = User.query.filter_by(username='demo').first()
            if demo_user:
                print("✅ Demo user already exists!")
                return demo_user.id
            
            # Create demo user
            demo_user = User(
                username='demo',
                email='demo@example.com',
                password_hash=generate_password_hash('demo123'),
                full_name='Demo User',
                age=30,
                monthly_income=50000.0,
                investment_experience='beginner',
                risk_tolerance='medium'
            )
            
            db.session.add(demo_user)
            db.session.commit()
            print("✅ Demo user created successfully!")
            return demo_user.id
            
        except Exception as e:
            print(f"❌ Error creating demo user: {str(e)}")
            return None

def add_sample_portfolio_data(user_id):
    """Add some sample portfolio holdings"""
    print("📊 Adding sample portfolio data...")
    with app.app_context():
        try:
            # Check if portfolio data already exists
            existing = Portfolio.query.filter_by(user_id=user_id).first()
            if existing:
                print("✅ Portfolio data already exists!")
                return
            
            # Sample portfolio holdings
            sample_holdings = [
                {
                    'symbol': 'AAPL',
                    'company_name': 'Apple Inc.',
                    'shares': 10,
                    'purchase_price': 150.00
                },
                {
                    'symbol': 'GOOGL',
                    'company_name': 'Alphabet Inc.',
                    'shares': 5,
                    'purchase_price': 2800.00
                },
                {
                    'symbol': 'MSFT',
                    'company_name': 'Microsoft Corporation',
                    'shares': 8,
                    'purchase_price': 350.00
                },
                {
                    'symbol': 'TSLA',
                    'company_name': 'Tesla, Inc.',
                    'shares': 3,
                    'purchase_price': 800.00
                }
            ]
            
            for holding in sample_holdings:
                portfolio_item = Portfolio(
                    user_id=user_id,
                    symbol=holding['symbol'],
                    company_name=holding['company_name'],
                    shares=holding['shares'],
                    purchase_price=holding['purchase_price'],
                    purchase_date=datetime.now()
                )
                db.session.add(portfolio_item)
            
            db.session.commit()
            print("✅ Sample portfolio data added successfully!")
            
        except Exception as e:
            print(f"❌ Error adding portfolio data: {str(e)}")

def test_portfolio_api(user_id):
    """Test the portfolio API directly"""
    print("🧪 Testing portfolio API...")
    with app.app_context():
        try:
            from utils.portfolio_tracker import PortfolioTracker
            tracker = PortfolioTracker()
            
            portfolio_data = tracker.get_user_portfolio(user_id)
            print("✅ Portfolio API working!")
            print(f"   📈 Total holdings: {len(portfolio_data['holdings'])}")
            print(f"   💰 Total value: ${portfolio_data['total_value']:.2f}")
            print(f"   📊 Return: {portfolio_data['return_percentage']:.2f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Error testing portfolio API: {str(e)}")
            return False

def test_api_endpoint():
    """Test the actual Flask API endpoint"""
    print("🌐 Testing Flask API endpoint...")
    with app.test_client() as client:
        with app.app_context():
            try:
                # Login as demo user
                response = client.post('/login', data={
                    'username': 'demo',
                    'password': 'demo123'
                }, follow_redirects=True)
                
                if response.status_code == 200:
                    print("✅ Login successful!")
                    
                    # Test portfolio API
                    api_response = client.get('/api/portfolio')
                    print(f"   📡 API Response Status: {api_response.status_code}")
                    
                    if api_response.status_code == 200:
                        data = api_response.get_json()
                        print(f"   📊 API Response: {data.get('success', False)}")
                        if data.get('success'):
                            portfolio = data.get('portfolio', {})
                            print(f"   💼 Holdings: {len(portfolio.get('holdings', []))}")
                        return True
                    else:
                        print(f"   ❌ API Error: {api_response.data}")
                else:
                    print(f"❌ Login failed: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error testing API endpoint: {str(e)}")
                
    return False

def main():
    """Main debug function"""
    print("🔧 Portfolio Debug Tool")
    print("=" * 50)
    
    # Step 1: Create database tables
    if not create_database_tables():
        return
    
    # Step 2: Create demo user
    user_id = create_demo_user()
    if not user_id:
        return
    
    # Step 3: Add sample portfolio data
    add_sample_portfolio_data(user_id)
    
    # Step 4: Test portfolio API
    if not test_portfolio_api(user_id):
        return
    
    # Step 5: Test Flask API endpoint
    test_api_endpoint()
    
    print("\n" + "=" * 50)
    print("✅ Portfolio Debug Complete!")
    print("\nDemo User Credentials:")
    print("   Username: demo")
    print("   Password: demo123")
    print("\nTo test:")
    print("1. Run 'python app_user_friendly.py'")
    print("2. Login with demo credentials")
    print("3. Navigate to Portfolio page")

if __name__ == "__main__":
    main()