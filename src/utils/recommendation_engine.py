import yfinance as yf
import numpy as np
import json
from datetime import datetime, timedelta

class RecommendationEngine:
    """Personalized stock recommendation engine"""
    
    def __init__(self):
        # Stock categories and their characteristics
        self.stock_categories = {
            'conservative': {
                'symbols': ['AAPL', 'MSFT', 'JNJ', 'PG', 'KO'],
                'description': 'Safe, reliable companies with steady growth',
                'risk': 'Low',
                'expected_return': '8-12%'
            },
            'growth': {
                'symbols': ['GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'],
                'description': 'High-growth companies with innovation potential',
                'risk': 'Medium-High',
                'expected_return': '15-25%'
            },
            'dividend': {
                'symbols': ['T', 'VZ', 'XOM', 'CVX', 'IBM'],
                'description': 'Companies that pay regular dividends',
                'risk': 'Low-Medium',
                'expected_return': '6-10%'
            },
            'indian_blue_chip': {
                'symbols': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS'],
                'description': 'Top Indian companies with strong fundamentals',
                'risk': 'Medium',
                'expected_return': '12-18%'
            },
            'tech': {
                'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'],
                'description': 'Technology companies driving innovation',
                'risk': 'Medium-High',
                'expected_return': '15-30%'
            }
        }
    
    def get_personalized_recommendations(self, user):
        """Get personalized stock recommendations based on user profile"""
        try:
            # Parse user preferences
            risk_tolerance = user.risk_tolerance or 'medium'
            investment_experience = user.investment_experience or 'beginner'
            monthly_income = user.monthly_income or 50000
            
            # Get preferred sectors
            try:
                preferred_sectors = json.loads(user.preferred_sectors or '[]')
            except:
                preferred_sectors = []
            
            # Select appropriate stock categories
            selected_categories = self._select_categories(risk_tolerance, investment_experience, preferred_sectors)
            
            # Get stock recommendations
            recommendations = []
            
            for category in selected_categories:
                stocks = self._get_category_stocks(category, limit=2)
                recommendations.extend(stocks)
            
            # Add investment amount suggestions
            for rec in recommendations:
                rec['suggested_investment'] = self._calculate_suggested_investment(monthly_income, risk_tolerance)
                rec['time_horizon'] = self._get_time_horizon(investment_experience)
            
            return recommendations[:6]  # Return top 6 recommendations
            
        except Exception as e:
            # Return demo recommendations
            return self._get_demo_recommendations()
    
    def _select_categories(self, risk_tolerance, experience, preferred_sectors):
        """Select stock categories based on user profile"""
        categories = []
        
        # Based on risk tolerance
        if risk_tolerance == 'low':
            categories.extend(['conservative', 'dividend'])
        elif risk_tolerance == 'medium':
            categories.extend(['conservative', 'growth', 'indian_blue_chip'])
        else:  # high risk
            categories.extend(['growth', 'tech'])
        
        # Based on experience
        if experience == 'beginner':
            if 'growth' in categories:
                categories.remove('growth')
            categories.append('conservative')
        
        # Based on preferred sectors
        if 'technology' in preferred_sectors:
            categories.append('tech')
        if 'banking' in preferred_sectors:
            categories.append('indian_blue_chip')
        
        # Remove duplicates and limit
        return list(set(categories))[:3]
    
    def _get_category_stocks(self, category, limit=2):
        """Get stocks from a specific category with analysis"""
        category_info = self.stock_categories.get(category, self.stock_categories['conservative'])
        stocks = category_info['symbols'][:limit]
        
        recommendations = []
        
        for symbol in stocks:
            try:
                stock_data = self._analyze_stock(symbol)
                stock_data.update({
                    'category': category,
                    'category_description': category_info['description'],
                    'risk_level': category_info['risk'],
                    'expected_return': category_info['expected_return']
                })
                recommendations.append(stock_data)
            except:
                # Add demo data if real data fails
                recommendations.append(self._get_demo_stock_data(symbol, category_info))
        
        return recommendations
    
    def _analyze_stock(self, symbol):
        """Analyze individual stock"""
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='3mo')
        info = ticker.info
        
        current_price = hist['Close'].iloc[-1]
        
        # Simple analysis
        month_change = (current_price - hist['Close'].iloc[-30]) / hist['Close'].iloc[-30] * 100
        volatility = np.std(hist['Close'].pct_change()) * 100
        
        # Target price (simple estimation)
        target_price = current_price * (1 + np.random.uniform(0.05, 0.15))  # 5-15% target
        
        # Generate simple reason
        if month_change > 5:
            reason = "Strong recent performance and positive momentum"
        elif month_change > 0:
            reason = "Steady growth with good fundamentals"
        else:
            reason = "Attractive entry point with long-term potential"
        
        return {
            'symbol': symbol,
            'company_name': info.get('longName', symbol),
            'current_price': current_price,
            'target_price': target_price,
            'potential_gain': ((target_price - current_price) / current_price * 100),
            'recent_performance': f"{month_change:+.1f}%",
            'reason': reason,
            'sector': info.get('sector', 'Unknown')
        }
    
    def _calculate_suggested_investment(self, monthly_income, risk_tolerance):
        """Calculate suggested investment amount"""
        # Suggest 10-20% of monthly income based on risk tolerance
        if risk_tolerance == 'low':
            percentage = 0.10
        elif risk_tolerance == 'medium':
            percentage = 0.15
        else:
            percentage = 0.20
        
        suggested_amount = monthly_income * percentage
        
        # Round to nearest 1000
        return round(suggested_amount / 1000) * 1000
    
    def _get_time_horizon(self, experience):
        """Get recommended investment time horizon"""
        if experience == 'beginner':
            return "1-2 years"
        elif experience == 'intermediate':
            return "2-5 years"
        else:
            return "5+ years"
    
    def _get_demo_stock_data(self, symbol, category_info):
        """Generate demo stock data when real data is unavailable"""
        base_prices = {
            'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 380.0,
            'AMZN': 3400.0, 'TSLA': 800.0, 'RELIANCE.NS': 2500.0
        }
        
        current_price = base_prices.get(symbol, 100.0)
        target_price = current_price * 1.12  # 12% target
        
        return {
            'symbol': symbol,
            'company_name': symbol.replace('.NS', '') + ' Corporation',
            'current_price': current_price,
            'target_price': target_price,
            'potential_gain': 12.0,
            'recent_performance': '+3.2%',
            'reason': 'Strong fundamentals and good growth prospects',
            'sector': 'Technology',
            'category_description': category_info['description'],
            'risk_level': category_info['risk'],
            'expected_return': category_info['expected_return']
        }
    
    def _get_demo_recommendations(self):
        """Get demo recommendations when user data is not available"""
        return [
            {
                'symbol': 'AAPL',
                'company_name': 'Apple Inc.',
                'current_price': 150.0,
                'target_price': 170.0,
                'potential_gain': 13.3,
                'recent_performance': '+5.2%',
                'reason': 'Strong product ecosystem and loyal customer base',
                'sector': 'Technology',
                'risk_level': 'Medium',
                'expected_return': '12-18%',
                'suggested_investment': 10000,
                'time_horizon': '2-5 years'
            },
            {
                'symbol': 'RELIANCE.NS',
                'company_name': 'Reliance Industries',
                'current_price': 2500.0,
                'target_price': 2800.0,
                'potential_gain': 12.0,
                'recent_performance': '+3.1%',
                'reason': 'Diversified business model with strong fundamentals',
                'sector': 'Energy',
                'risk_level': 'Medium',
                'expected_return': '10-15%',
                'suggested_investment': 15000,
                'time_horizon': '2-5 years'
            }
        ]
