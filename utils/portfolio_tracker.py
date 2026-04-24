import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy

class PortfolioTracker:
    """Track and analyze user portfolio performance"""
    
    def __init__(self):
        self.cache = {}  # Simple caching for stock prices
    
    def get_user_portfolio(self, holdings):
        """Get complete portfolio analysis for a user"""
        try:
            
            if not holdings:
                return {
                    'holdings': [],
                    'total_value': 0,
                    'total_investment': 0,
                    'total_return': 0,
                    'return_percentage': 0,
                    'summary': 'No investments yet. Start building your portfolio!'
                }
            
            portfolio_data = []
            total_value = 0
            total_investment = 0
            
            for holding in holdings:
                # Get current price robustly
                try:
                    current_price = self._get_current_price(holding.symbol)
                except Exception:
                    current_price = holding.current_price if holding.current_price > 0 else holding.purchase_price
                
                # Calculate values
                investment_value = holding.shares * holding.purchase_price
                current_value = holding.shares * current_price
                total_return = current_value - investment_value
                return_percentage = (total_return / investment_value * 100) if investment_value > 0 else 0
                
                portfolio_item = {
                    'symbol': holding.symbol,
                    'company_name': holding.company_name,
                    'shares': holding.shares,
                    'purchase_price': holding.purchase_price,
                    'current_price': current_price,
                    'investment_value': investment_value,
                    'current_value': current_value,
                    'total_return': total_return,
                    'return_percentage': return_percentage,
                    'purchase_date': holding.purchase_date.strftime('%Y-%m-%d') if holding.purchase_date else 'Unknown'
                }
                
                portfolio_data.append(portfolio_item)
                total_value += current_value
                total_investment += investment_value
            
            total_return = total_value - total_investment
            return_percentage = (total_return / total_investment * 100) if total_investment > 0 else 0
            
            # Generate summary
            summary = self._generate_portfolio_summary(return_percentage, len(holdings))
            
            return {
                'holdings': portfolio_data,
                'total_value': total_value,
                'total_investment': total_investment,
                'total_return': total_return,
                'return_percentage': return_percentage,
                'summary': summary,
                'diversification_score': self._calculate_diversification_score(portfolio_data)
            }
            
        except Exception as e:
            print(f"Portfolio error: {e}")
            if 'holdings' in locals() and holdings:
                return {
                    'holdings': [{'symbol': h.symbol, 'company_name': h.company_name, 'shares': h.shares, 'purchase_price': h.purchase_price, 'current_price': h.purchase_price, 'investment_value': h.shares*h.purchase_price, 'current_value': h.shares*h.purchase_price, 'total_return': 0, 'return_percentage': 0, 'purchase_date': 'Unknown'} for h in holdings],
                    'total_value': sum(h.shares*h.purchase_price for h in holdings),
                    'total_investment': sum(h.shares*h.purchase_price for h in holdings),
                    'total_return': 0,
                    'return_percentage': 0,
                    'summary': 'Error fetching live prices. Showing offline portfolio values.',
                    'diversification_score': 50
                }
            return self._get_demo_portfolio()
    
    def get_portfolio_value(self, holdings):
        """Get simple portfolio value for dashboard"""
        try:
            
            if not holdings:
                return 0
            
            total_value = 0
            for holding in holdings:
                try:
                    current_price = self._get_current_price(holding.symbol)
                except Exception:
                    current_price = holding.current_price if holding.current_price > 0 else holding.purchase_price
                total_value += holding.shares * current_price
            
            return total_value
            
        except:
            return 0
    
    def _get_current_price(self, symbol):
        """Get current stock price with caching"""
        # Check cache first (cache for 5 minutes)
        cache_key = symbol
        current_time = datetime.now()
        
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if (current_time - cached_data['timestamp']).seconds < 300:  # 5 minutes
                return cached_data['price']
        
        try:
            # Fetch current price
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d')
            
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                
                # Cache the result
                self.cache[cache_key] = {
                    'price': price,
                    'timestamp': current_time
                }
                
                return price
            else:
                return 100.0  # Fallback price
                
        except Exception as e:
            # Return demo price based on symbol
            demo_prices = {
                'AAPL': 150.0,
                'GOOGL': 2800.0,
                'MSFT': 380.0,
                'RELIANCE.NS': 2500.0,
                'TCS.NS': 3200.0
            }
            return demo_prices.get(symbol, 100.0)
    
    def _generate_portfolio_summary(self, return_percentage, num_holdings):
        """Generate user-friendly portfolio summary"""
        if return_percentage > 10:
            performance = "Excellent! Your portfolio is performing very well."
        elif return_percentage > 5:
            performance = "Good performance! You're on the right track."
        elif return_percentage > 0:
            performance = "Positive returns! Your investments are growing."
        elif return_percentage > -5:
            performance = "Slight decline, but this is normal in investing."
        else:
            performance = "Portfolio is down, consider reviewing your investments."
        
        diversification = ""
        if num_holdings < 3:
            diversification = " Consider adding more stocks to diversify your portfolio."
        elif num_holdings > 10:
            diversification = " You have good diversification across multiple stocks."
        
        return performance + diversification
    
    def _calculate_diversification_score(self, portfolio_data):
        """Calculate portfolio diversification score"""
        if not portfolio_data:
            return 0
        
        # Simple diversification based on number of holdings and sector spread
        num_holdings = len(portfolio_data)
        
        # Score based on number of holdings
        if num_holdings >= 10:
            holding_score = 100
        elif num_holdings >= 5:
            holding_score = 80
        elif num_holdings >= 3:
            holding_score = 60
        else:
            holding_score = 30
        
        # Check for concentration risk (no single holding > 30%)
        total_value = sum(item['current_value'] for item in portfolio_data)
        max_concentration = max(item['current_value'] / total_value for item in portfolio_data) * 100
        
        if max_concentration > 50:
            concentration_score = 20
        elif max_concentration > 30:
            concentration_score = 60
        else:
            concentration_score = 100
        
        # Combined score
        final_score = (holding_score + concentration_score) / 2
        return round(final_score)
    
    def get_portfolio_insights(self, user_id):
        """Get insights and recommendations for portfolio improvement"""
        portfolio = self.get_user_portfolio(user_id)
        
        insights = []
        
        # Performance insights
        if portfolio['return_percentage'] > 15:
            insights.append({
                'type': 'success',
                'title': 'Excellent Performance!',
                'message': 'Your portfolio is outperforming market averages.'
            })
        elif portfolio['return_percentage'] < -10:
            insights.append({
                'type': 'warning',
                'title': 'Review Required',
                'message': 'Consider reviewing underperforming stocks.'
            })
        
        # Diversification insights
        if len(portfolio['holdings']) < 3:
            insights.append({
                'type': 'info',
                'title': 'Diversify Your Portfolio',
                'message': 'Add more stocks from different sectors to reduce risk.'
            })
        
        # Risk insights
        if portfolio.get('diversification_score', 0) < 50:
            insights.append({
                'type': 'warning',
                'title': 'High Concentration Risk',
                'message': 'Your portfolio is concentrated in few stocks. Consider spreading investments.'
            })
        
        return insights
    
    def _get_demo_portfolio(self):
        """Generate demo portfolio data"""
        return {
            'holdings': [
                {
                    'symbol': 'AAPL',
                    'company_name': 'Apple Inc.',
                    'shares': 10,
                    'purchase_price': 140.0,
                    'current_price': 150.0,
                    'investment_value': 1400.0,
                    'current_value': 1500.0,
                    'total_return': 100.0,
                    'return_percentage': 7.14,
                    'purchase_date': '2024-01-15'
                }
            ],
            'total_value': 1500.0,
            'total_investment': 1400.0,
            'total_return': 100.0,
            'return_percentage': 7.14,
            'summary': 'Good start! Your portfolio is showing positive returns.',
            'diversification_score': 40
        }
