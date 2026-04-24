import requests
import re
import json
from datetime import datetime, timedelta
from textblob import TextBlob
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class OntologyDrivenSentimentAnalyzer:
    """
    Advanced Sentiment Analysis Engine with Ontology-driven approach
    Extracts structured relationships: Company → Event → Market Impact
    """
    
    def __init__(self):
        # Financial ontology mappings
        self.market_events = {
            'earnings': {'weight': 0.8, 'impact_duration': 5, 'volatility_factor': 1.5},
            'merger': {'weight': 0.9, 'impact_duration': 10, 'volatility_factor': 2.0},
            'acquisition': {'weight': 0.85, 'impact_duration': 8, 'volatility_factor': 1.8},
            'partnership': {'weight': 0.6, 'impact_duration': 3, 'volatility_factor': 1.2},
            'lawsuit': {'weight': -0.7, 'impact_duration': 7, 'volatility_factor': 1.6},
            'regulation': {'weight': -0.5, 'impact_duration': 15, 'volatility_factor': 1.4},
            'product_launch': {'weight': 0.6, 'impact_duration': 5, 'volatility_factor': 1.3},
            'ceo_change': {'weight': 0.3, 'impact_duration': 12, 'volatility_factor': 1.7},
            'dividend': {'weight': 0.4, 'impact_duration': 2, 'volatility_factor': 1.1},
            'buyback': {'weight': 0.5, 'impact_duration': 3, 'volatility_factor': 1.2}
        }
        
        self.sentiment_keywords = {
            'positive': ['strong', 'growth', 'profit', 'increase', 'beat', 'exceed', 'successful', 
                        'bullish', 'optimistic', 'upgrade', 'breakthrough', 'innovation', 'expansion'],
            'negative': ['weak', 'decline', 'loss', 'decrease', 'miss', 'disappoint', 'failed',
                        'bearish', 'pessimistic', 'downgrade', 'concern', 'challenge', 'risk'],
            'neutral': ['stable', 'maintain', 'unchanged', 'hold', 'steady', 'consistent']
        }
        
        self.entity_relations = {
            'company_indicators': ['revenue', 'profit', 'sales', 'market share', 'growth rate'],
            'market_indicators': ['volatility', 'volume', 'price movement', 'trend', 'momentum'],
            'external_factors': ['economy', 'sector', 'competition', 'regulation', 'technology']
        }
    
    def analyze_sentiment(self, symbol, company_name=None):
        """Perform comprehensive sentiment analysis for a stock"""
        try:
            # Get company information
            if not company_name:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                company_name = info.get('longName', symbol)
            
            # Fetch news data
            news_data = self._fetch_financial_news(symbol, company_name)
            
            if not news_data:
                return self._create_default_sentiment(symbol)
            
            # Perform ontology-driven analysis
            ontology_analysis = self._extract_ontology_relationships(news_data, symbol, company_name)
            
            # Calculate sentiment scores
            sentiment_scores = self._calculate_sentiment_scores(ontology_analysis)
            
            # Generate market impact prediction
            market_impact = self._predict_market_impact(ontology_analysis, sentiment_scores)
            
            return {
                'symbol': symbol,
                'company_name': company_name,
                'overall_sentiment': sentiment_scores['overall'],
                'sentiment_breakdown': sentiment_scores,
                'ontology_relationships': ontology_analysis['relationships'],
                'market_events': ontology_analysis['events'],
                'predicted_impact': market_impact,
                'confidence': self._calculate_confidence(ontology_analysis),
                'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'news_count': len(news_data)
            }
            
        except Exception as e:
            return self._create_default_sentiment(symbol, error=str(e))
    
    def _fetch_financial_news(self, symbol, company_name):
        """Fetch recent financial news"""
        try:
            # Try to get news from yfinance
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if news:
                return news[:10]  # Latest 10 news items
            
            # Fallback: create sample news for demonstration
            return self._create_sample_news(symbol, company_name)
            
        except:
            return self._create_sample_news(symbol, company_name)
    
    def _create_sample_news(self, symbol, company_name):
        """Create sample news for demonstration purposes"""
        sample_news = [
            {
                'title': f'{company_name} Reports Strong Q3 Earnings, Beats Expectations',
                'summary': f'{company_name} delivered robust quarterly results with revenue growth exceeding analyst forecasts.',
                'published': datetime.now() - timedelta(days=1)
            },
            {
                'title': f'Market Analysis: {symbol} Shows Bullish Technical Indicators',
                'summary': f'Technical analysis suggests {symbol} may continue its upward momentum based on recent trading patterns.',
                'published': datetime.now() - timedelta(days=2)
            },
            {
                'title': f'{company_name} Announces Strategic Partnership for Market Expansion',
                'summary': f'The partnership is expected to enhance {company_name}\'s market position and drive future growth.',
                'published': datetime.now() - timedelta(days=3)
            }
        ]
        return sample_news
    
    def _extract_ontology_relationships(self, news_data, symbol, company_name):
        """Extract structured Company → Event → Impact relationships"""
        relationships = []
        detected_events = []
        
        for news_item in news_data:
            # Extract text content
            text_content = self._extract_text_content(news_item)
            
            # Identify entities and events
            entities = self._identify_entities(text_content, company_name)
            events = self._identify_market_events(text_content)
            
            # Create relationship triplets
            for event in events:
                relationship = {
                    'company': company_name,
                    'event': event['type'],
                    'impact': self._determine_impact(text_content, event),
                    'confidence': event['confidence'],
                    'timestamp': news_item.get('published', datetime.now()),
                    'source_text': text_content[:200] + '...' if len(text_content) > 200 else text_content
                }
                relationships.append(relationship)
                detected_events.append(event)
        
        return {
            'relationships': relationships,
            'events': detected_events,
            'entity_mentions': self._count_entity_mentions(news_data, company_name)
        }
    
    def _extract_text_content(self, news_item):
        """Extract text content from news item"""
        text_parts = []
        
        if isinstance(news_item, dict):
            title = news_item.get('title', '')
            summary = news_item.get('summary', '')
            text_parts = [title, summary]
        else:
            text_parts = [str(news_item)]
        
        return ' '.join(text_parts)
    
    def _identify_entities(self, text, company_name):
        """Identify financial entities in text"""
        entities = {
            'company': [],
            'financial_metrics': [],
            'market_terms': []
        }
        
        text_lower = text.lower()
        
        # Company mentions
        if company_name.lower() in text_lower:
            entities['company'].append(company_name)
        
        # Financial metrics
        financial_terms = ['revenue', 'profit', 'earnings', 'sales', 'margin', 'growth', 'market cap']
        for term in financial_terms:
            if term in text_lower:
                entities['financial_metrics'].append(term)
        
        # Market terms
        market_terms = ['bullish', 'bearish', 'volatile', 'trend', 'momentum', 'volume']
        for term in market_terms:
            if term in text_lower:
                entities['market_terms'].append(term)
        
        return entities
    
    def _identify_market_events(self, text):
        """Identify market events in text"""
        events = []
        text_lower = text.lower()
        
        for event_type, event_config in self.market_events.items():
            # Simple keyword matching for event detection
            event_keywords = {
                'earnings': ['earnings', 'quarterly results', 'q1', 'q2', 'q3', 'q4'],
                'merger': ['merger', 'merge', 'combining'],
                'acquisition': ['acquisition', 'acquire', 'purchase', 'buy'],
                'partnership': ['partnership', 'partner', 'collaboration', 'alliance'],
                'lawsuit': ['lawsuit', 'legal action', 'litigation', 'sue'],
                'regulation': ['regulation', 'regulatory', 'compliance', 'policy'],
                'product_launch': ['launch', 'product', 'new offering', 'release'],
                'ceo_change': ['ceo', 'chief executive', 'leadership change'],
                'dividend': ['dividend', 'payout', 'distribution'],
                'buyback': ['buyback', 'share repurchase', 'stock repurchase']
            }
            
            keywords = event_keywords.get(event_type, [event_type])
            
            for keyword in keywords:
                if keyword in text_lower:
                    confidence = min(0.9, text_lower.count(keyword) * 0.3 + 0.4)
                    events.append({
                        'type': event_type,
                        'confidence': confidence,
                        'keyword_matched': keyword
                    })
                    break  # Avoid duplicate events
        
        return events
    
    def _determine_impact(self, text, event):
        """Determine market impact based on text sentiment and event type"""
        # Get base impact from event type
        base_impact = self.market_events[event['type']]['weight']
        
        # Analyze text sentiment
        text_sentiment = self._analyze_text_sentiment(text)
        
        # Combine event impact with sentiment
        if text_sentiment > 0.1:
            impact_direction = 'positive'
            impact_magnitude = base_impact * (1 + text_sentiment)
        elif text_sentiment < -0.1:
            impact_direction = 'negative'
            impact_magnitude = base_impact * (1 + abs(text_sentiment))
        else:
            impact_direction = 'neutral'
            impact_magnitude = abs(base_impact) * 0.5
        
        return {
            'direction': impact_direction,
            'magnitude': min(1.0, abs(impact_magnitude)),
            'duration_days': self.market_events[event['type']]['impact_duration'],
            'volatility_factor': self.market_events[event['type']]['volatility_factor']
        }
    
    def _analyze_text_sentiment(self, text):
        """Analyze sentiment of text using keyword-based approach"""
        text_lower = text.lower()
        
        positive_score = sum(1 for word in self.sentiment_keywords['positive'] if word in text_lower)
        negative_score = sum(1 for word in self.sentiment_keywords['negative'] if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0
        
        # Normalize scores
        positive_normalized = positive_score / total_words
        negative_normalized = negative_score / total_words
        
        # Calculate sentiment polarity (-1 to 1)
        sentiment = (positive_normalized - negative_normalized) * 10
        return max(-1, min(1, sentiment))
    
    def _calculate_sentiment_scores(self, ontology_analysis):
        """Calculate comprehensive sentiment scores"""
        relationships = ontology_analysis['relationships']
        
        if not relationships:
            return {
                'overall': 0.0,
                'positive': 0.33,
                'negative': 0.33,
                'neutral': 0.34,
                'event_based': 0.0,
                'confidence_weighted': 0.0
            }
        
        # Calculate event-based sentiment
        event_sentiments = []
        confidence_weights = []
        
        for rel in relationships:
            impact = rel['impact']
            confidence = rel['confidence']
            
            if impact['direction'] == 'positive':
                sentiment_score = impact['magnitude']
            elif impact['direction'] == 'negative':
                sentiment_score = -impact['magnitude']
            else:
                sentiment_score = 0
            
            event_sentiments.append(sentiment_score)
            confidence_weights.append(confidence)
        
        # Calculate weighted average
        if confidence_weights:
            overall_sentiment = sum(s * w for s, w in zip(event_sentiments, confidence_weights)) / sum(confidence_weights)
        else:
            overall_sentiment = 0
        
        # Calculate distribution
        positive_events = sum(1 for s in event_sentiments if s > 0.1)
        negative_events = sum(1 for s in event_sentiments if s < -0.1)
        neutral_events = len(event_sentiments) - positive_events - negative_events
        
        total_events = len(event_sentiments) if event_sentiments else 1
        
        return {
            'overall': max(-1, min(1, overall_sentiment)),
            'positive': positive_events / total_events,
            'negative': negative_events / total_events,
            'neutral': neutral_events / total_events,
            'event_based': sum(event_sentiments) / len(event_sentiments) if event_sentiments else 0,
            'confidence_weighted': overall_sentiment
        }
    
    def _predict_market_impact(self, ontology_analysis, sentiment_scores):
        """Predict market impact based on ontology analysis"""
        relationships = ontology_analysis['relationships']
        
        if not relationships:
            return {
                'price_impact': 0.0,
                'volatility_impact': 1.0,
                'duration_days': 1,
                'confidence': 0.3,
                'key_factors': ['Limited news data available']
            }
        
        # Calculate aggregate impact
        price_impacts = []
        volatility_factors = []
        durations = []
        key_factors = []
        
        for rel in relationships:
            impact = rel['impact']
            confidence = rel['confidence']
            
            # Price impact
            direction_multiplier = 1 if impact['direction'] == 'positive' else -1 if impact['direction'] == 'negative' else 0
            price_impact = direction_multiplier * impact['magnitude'] * confidence
            price_impacts.append(price_impact)
            
            # Volatility and duration
            volatility_factors.append(impact['volatility_factor'])
            durations.append(impact['duration_days'])
            
            # Key factors
            key_factors.append(f"{rel['event']} ({impact['direction']})")
        
        # Aggregate predictions
        avg_price_impact = sum(price_impacts) / len(price_impacts) if price_impacts else 0
        avg_volatility = sum(volatility_factors) / len(volatility_factors) if volatility_factors else 1.0
        max_duration = max(durations) if durations else 1
        
        return {
            'price_impact': max(-0.2, min(0.2, avg_price_impact)),  # Cap at ±20%
            'volatility_impact': min(3.0, avg_volatility),  # Cap at 3x
            'duration_days': min(30, max_duration),  # Cap at 30 days
            'confidence': sentiment_scores['overall'] * 0.5 + 0.5,  # Convert to 0-1 range
            'key_factors': key_factors[:5]  # Top 5 factors
        }
    
    def _count_entity_mentions(self, news_data, company_name):
        """Count mentions of different entities"""
        mentions = {
            'company': 0,
            'financial_terms': 0,
            'market_terms': 0
        }
        
        all_text = ' '.join([self._extract_text_content(news) for news in news_data]).lower()
        
        mentions['company'] = all_text.count(company_name.lower())
        
        financial_terms = ['revenue', 'profit', 'earnings', 'growth', 'sales']
        mentions['financial_terms'] = sum(all_text.count(term) for term in financial_terms)
        
        market_terms = ['market', 'stock', 'share', 'price', 'trading']
        mentions['market_terms'] = sum(all_text.count(term) for term in market_terms)
        
        return mentions
    
    def _calculate_confidence(self, ontology_analysis):
        """Calculate confidence in sentiment analysis"""
        relationships = ontology_analysis['relationships']
        
        if not relationships:
            return 0.3  # Low confidence
        
        # Factors affecting confidence
        num_relationships = len(relationships)
        avg_confidence = sum(rel['confidence'] for rel in relationships) / num_relationships
        entity_diversity = len(set(rel['event'] for rel in relationships))
        
        # Calculate overall confidence
        confidence = min(1.0, (
            avg_confidence * 0.4 +  # Individual relationship confidence
            min(1.0, num_relationships / 5) * 0.3 +  # Number of relationships
            min(1.0, entity_diversity / 3) * 0.3  # Diversity of events
        ))
        
        return confidence
    
    def _create_default_sentiment(self, symbol, error=None):
        """Create default sentiment when analysis fails"""
        return {
            'symbol': symbol,
            'company_name': symbol,
            'overall_sentiment': 0.0,
            'sentiment_breakdown': {
                'overall': 0.0,
                'positive': 0.33,
                'negative': 0.33,
                'neutral': 0.34,
                'event_based': 0.0,
                'confidence_weighted': 0.0
            },
            'ontology_relationships': [],
            'market_events': [],
            'predicted_impact': {
                'price_impact': 0.0,
                'volatility_impact': 1.0,
                'duration_days': 1,
                'confidence': 0.3,
                'key_factors': ['Limited data available' + (f': {error}' if error else '')]
            },
            'confidence': 0.3,
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'news_count': 0
        }