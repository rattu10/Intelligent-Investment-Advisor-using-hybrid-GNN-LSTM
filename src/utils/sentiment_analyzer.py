import requests
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timedelta
import re

class SentimentAnalyzer:
    """Ontology-driven sentiment analysis for financial news"""
    
    def __init__(self):
        # Initialize sentiment analysis model
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert"
            )
        except:
            # Fallback to general sentiment model
            self.sentiment_pipeline = pipeline("sentiment-analysis")
        
        # Define financial ontology relationships
        self.financial_ontology = {
            'market_events': {
                'earnings': ['earnings', 'revenue', 'profit', 'quarterly results'],
                'acquisitions': ['acquisition', 'merger', 'buyout', 'takeover'],
                'partnerships': ['partnership', 'collaboration', 'joint venture'],
                'product_launch': ['launch', 'release', 'unveil', 'introduce'],
                'regulatory': ['regulation', 'compliance', 'SEC', 'FDA approval'],
                'leadership': ['CEO', 'executive', 'leadership', 'appointment']
            },
            'market_impact': {
                'positive': ['growth', 'increase', 'gain', 'surge', 'rise', 'bullish'],
                'negative': ['decline', 'fall', 'drop', 'loss', 'bearish', 'crash'],
                'neutral': ['stable', 'unchanged', 'maintain', 'steady']
            }
        }
    
    def get_financial_news(self, symbol, days_back=7):
        """Fetch financial news for a given stock symbol"""
        # Mock news data for demonstration
        # In production, you would integrate with news APIs like NewsAPI, Alpha Vantage, etc.
        
        mock_news = [
            {
                'title': f'{symbol} reports strong quarterly earnings beating expectations',
                'content': f'{symbol} announced robust financial results with revenue growth of 15% year-over-year. The company exceeded analyst expectations for both earnings and revenue.',
                'date': datetime.now() - timedelta(days=1),
                'source': 'Financial Times'
            },
            {
                'title': f'{symbol} announces strategic partnership with major tech company',
                'content': f'{symbol} has entered into a strategic partnership that is expected to drive innovation and expand market reach.',
                'date': datetime.now() - timedelta(days=2),
                'source': 'Reuters'
            },
            {
                'title': f'Market volatility affects {symbol} stock performance',
                'content': f'Recent market turbulence has impacted {symbol} shares, though analysts remain optimistic about long-term prospects.',
                'date': datetime.now() - timedelta(days=3),
                'source': 'Bloomberg'
            },
            {
                'title': f'{symbol} invests heavily in R&D for future growth',
                'content': f'{symbol} has announced significant investments in research and development, signaling commitment to innovation and market leadership.',
                'date': datetime.now() - timedelta(days=4),
                'source': 'Wall Street Journal'
            },
            {
                'title': f'Analysts upgrade {symbol} with positive outlook',
                'content': f'Leading financial analysts have upgraded their rating for {symbol} citing strong fundamentals and growth potential.',
                'date': datetime.now() - timedelta(days=5),
                'source': 'MarketWatch'
            }
        ]
        
        return mock_news
    
    def extract_ontology_relations(self, text):
        """Extract structured relationships from text using financial ontology"""
        relations = {
            'events': [],
            'entities': [],
            'sentiments': [],
            'impacts': []
        }
        
        text_lower = text.lower()
        
        # Extract market events
        for event_type, keywords in self.financial_ontology['market_events'].items():
            for keyword in keywords:
                if keyword in text_lower:
                    relations['events'].append({
                        'type': event_type,
                        'keyword': keyword,
                        'context': self._extract_context(text, keyword)
                    })
        
        # Extract market impact indicators
        for impact_type, keywords in self.financial_ontology['market_impact'].items():
            for keyword in keywords:
                if keyword in text_lower:
                    relations['impacts'].append({
                        'type': impact_type,
                        'keyword': keyword,
                        'strength': self._assess_impact_strength(text, keyword)
                    })
        
        return relations
    
    def _extract_context(self, text, keyword, window=50):
        """Extract context around a keyword"""
        pattern = rf'\b\w*{keyword}\w*\b'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            return text[start:end].strip()
        
        return ""
    
    def _assess_impact_strength(self, text, keyword):
        """Assess the strength of market impact based on surrounding words"""
        intensifiers = ['significant', 'major', 'substantial', 'dramatic', 'sharp']
        diminishers = ['slight', 'minor', 'modest', 'gradual', 'small']
        
        context = self._extract_context(text, keyword, 30).lower()
        
        if any(intensifier in context for intensifier in intensifiers):
            return 'high'
        elif any(diminisher in context for diminisher in diminishers):
            return 'low'
        else:
            return 'medium'
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using the pre-trained model"""
        try:
            result = self.sentiment_pipeline(text[:512])  # Truncate to model limit
            
            if isinstance(result, list):
                result = result[0]
            
            # Normalize sentiment scores
            label = result['label'].upper()
            score = result['score']
            
            # Convert to unified scale
            if label in ['POSITIVE', 'POS']:
                sentiment_score = score
            elif label in ['NEGATIVE', 'NEG']:
                sentiment_score = -score
            else:  # NEUTRAL
                sentiment_score = 0.0
            
            return {
                'label': label,
                'score': sentiment_score,
                'confidence': score
            }
            
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            # Fallback to simple keyword-based analysis
            return self._simple_sentiment_analysis(text)
    
    def _simple_sentiment_analysis(self, text):
        """Simple fallback sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'positive', 'growth', 'profit', 'gain']
        negative_words = ['bad', 'poor', 'negative', 'loss', 'decline', 'fall', 'drop']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return {'label': 'POSITIVE', 'score': 0.6, 'confidence': 0.6}
        elif neg_count > pos_count:
            return {'label': 'NEGATIVE', 'score': -0.6, 'confidence': 0.6}
        else:
            return {'label': 'NEUTRAL', 'score': 0.0, 'confidence': 0.5}
    
    def analyze_stock_sentiment(self, symbol):
        """Comprehensive sentiment analysis for a stock"""
        # Get news articles
        news_articles = self.get_financial_news(symbol)
        
        analyzed_articles = []
        sentiment_scores = []
        all_events = []
        
        for article in news_articles:
            # Combine title and content for analysis
            full_text = f"{article['title']} {article['content']}"
            
            # Analyze sentiment
            sentiment = self.analyze_sentiment(full_text)
            
            # Extract ontology relations
            relations = self.extract_ontology_relations(full_text)
            
            analyzed_article = {
                'title': article['title'],
                'date': article['date'].strftime('%Y-%m-%d'),
                'source': article['source'],
                'sentiment': sentiment,
                'relations': relations,
                'relevance_score': self._calculate_relevance(full_text, symbol)
            }
            
            analyzed_articles.append(analyzed_article)
            sentiment_scores.append(sentiment['score'])
            all_events.extend(relations['events'])
        
        # Calculate overall sentiment
        if sentiment_scores:
            overall_sentiment = np.mean(sentiment_scores)
            sentiment_std = np.std(sentiment_scores)
        else:
            overall_sentiment = 0.0
            sentiment_std = 0.0
        
        # Aggregate events
        event_summary = self._summarize_events(all_events)
        
        return {
            'overall_sentiment': float(overall_sentiment),
            'sentiment_volatility': float(sentiment_std),
            'articles': analyzed_articles[:5],  # Return top 5 articles
            'events': event_summary,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _calculate_relevance(self, text, symbol):
        """Calculate relevance score of article to the stock"""
        symbol_mentions = text.upper().count(symbol.upper())
        text_length = len(text.split())
        
        # Base relevance on symbol mentions and financial keywords
        financial_keywords = ['stock', 'share', 'market', 'investor', 'trading']
        keyword_score = sum(1 for keyword in financial_keywords if keyword in text.lower())
        
        relevance = (symbol_mentions * 0.4 + keyword_score * 0.1) / max(text_length / 100, 1)
        return min(relevance, 1.0)  # Cap at 1.0
    
    def _summarize_events(self, events):
        """Summarize extracted events"""
        event_counts = {}
        for event in events:
            event_type = event['type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            'total_events': len(events),
            'event_types': event_counts,
            'top_events': sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        }
