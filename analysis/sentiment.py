# sentiment.py
import pandas as pd
import numpy as np
import logging
import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Union, Optional
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time
import random

class SentimentAnalyzer:
    """Sentiment analyzer for Indian financial news."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Download required NLTK resources if not already present
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            self.logger.info("Downloading NLTK resources...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # List of Indian financial news sources
        self.news_sources = self.config.get('news_sources', [
            'economictimes.indiatimes.com',
            'moneycontrol.com',
            'livemint.com',
            'businesstoday.in',
            'business-standard.com'
        ])
        
        # Dictionary mapping stock symbols to company names and keywords
        self.symbol_to_company = {}
        self._load_company_mappings()
        
    def _load_company_mappings(self) -> None:
        """Load mapping from stock symbols to company names and keywords."""
        # In a real implementation, this would load from a database or API
        # For demo, use a small hardcoded set
        self.symbol_to_company = {
            'RELIANCE': {
                'name': 'Reliance Industries Limited',
                'keywords': ['reliance', 'ril', 'mukesh ambani', 'jio']
            },
            'TCS': {
                'name': 'Tata Consultancy Services',
                'keywords': ['tata consultancy', 'tcs', 'tata', 'it services']
            },
            'HDFCBANK': {
                'name': 'HDFC Bank',
                'keywords': ['hdfc bank', 'hdfc', 'bank']
            },
            'INFY': {
                'name': 'Infosys',
                'keywords': ['infosys', 'infy', 'it services']
            },
            'HINDUNILVR': {
                'name': 'Hindustan Unilever',
                'keywords': ['hindustan unilever', 'hul', 'consumer goods']
            }
        }
        
        # If an external file path is provided in config, load from there
        mapping_file = self.config.get('company_mapping_file')
        if mapping_file:
            try:
                df = pd.read_csv(mapping_file)
                # Process the dataframe and update the dictionary
                for _, row in df.iterrows():
                    self.symbol_to_company[row['symbol']] = {
                        'name': row['company_name'],
                        'keywords': row.get('keywords', '').lower().split(',')
                    }
            except Exception as e:
                self.logger.error(f"Error loading company mappings: {str(e)}")
    
    def fetch_news(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Fetch recent news for a given stock symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            List of news articles with source, title, date, url, and content
        """
        articles = []
        
        # Get company info
        company_info = self.symbol_to_company.get(symbol)
        if not company_info:
            self.logger.warning(f"No company info found for symbol {symbol}")
            return articles
        
        # In a real implementation, this would use a news API or web scraping
        # For demo purposes, simulate a few articles
        self.logger.info(f"Fetching news for {symbol} ({company_info['name']})")
        
        try:
            # Simulate API call or web scraping
            # In reality, this would use newsapi.org, Google News API, or custom scraping
            simulated_articles = self._simulate_news_articles(symbol, company_info, days)
            articles.extend(simulated_articles)
            
        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {str(e)}")
        
        return articles
    
    def _simulate_news_articles(self, symbol: str, company_info: Dict[str, Any], 
                              days: int) -> List[Dict[str, Any]]:
        """Simulate news articles for testing purposes."""
        articles = []
        company_name = company_info['name']
        
        # Random news topics and sentiments for simulation
        topics = [
            ('quarterly results', 0.7),
            ('expansion plans', 0.8),
            ('new product launch', 0.9),
            ('management changes', 0.3),
            ('regulatory issues', -0.5),
            ('competition concerns', -0.6),
            ('market share loss', -0.8),
            ('analyst upgrades', 0.6),
            ('analyst downgrades', -0.7),
            ('merger talks', 0.4),
            ('layoffs', -0.9),
            ('industry outlook', 0.2)
        ]
        
        # Generate random number of articles
        num_articles = random.randint(2, 6)
        
        for i in range(num_articles):
            # Select random topic
            topic, sentiment_bias = random.choice(topics)
            
            # Generate random date within specified days
            days_ago = random.randint(0, days - 1)
            date = pd.Timestamp.now() - pd.Timedelta(days=days_ago)
            date_str = date.strftime('%Y-%m-%d')
            
            # Randomize source
            source = random.choice(self.news_sources)
            
            # Generate title
            title = f"{company_name} {topic.title()}: What Investors Need to Know"
            
            # Generate content with sentiment bias
            content = self._generate_article_content(company_name, topic, sentiment_bias)
            
            # Create URL
            topic_slug = topic.lower().replace(' ', '-')
            url = f"https://{source}/markets/{date.year}/{date.month:02d}/{date.day:02d}/{symbol.lower()}-{topic_slug}"
            
            article = {
                'symbol': symbol,
                'source': source,
                'title': title,
                'date': date_str,
                'url': url,
                'content': content
            }
            articles.append(article)
        
        return articles
    
    def _generate_article_content(self, company_name: str, topic: str, 
                                sentiment_bias: float) -> str:
        """Generate synthetic article content with a given sentiment bias."""
        # Positive phrases
        positive_phrases = [
            "exceeded expectations",
            "strong performance",
            "beat analyst estimates",
            "positive outlook",
            "growth trajectory",
            "strategic advantage",
            "market leadership",
            "innovative approach",
            "value creation",
            "robust demand"
        ]
        
        # Negative phrases
        negative_phrases = [
            "missed expectations",
            "weak performance",
            "fell short of estimates",
            "challenging outlook",
            "slowing growth",
            "competitive pressures",
            "market share erosion",
            "lagging innovation",
            "value destruction",
            "softening demand"
        ]
        
        # Neutral phrases
        neutral_phrases = [
            "announced today",
            "according to sources",
            "reported recently",
            "disclosed in a statement",
            "as per industry trends",
            "market participants noted",
            "analysts commented",
            "investors are watching",
            "in line with forecasts",
            "as expected by the market"
        ]
        
        # Select phrases based on sentiment bias
        if sentiment_bias > 0.3:
            primary_phrases = positive_phrases
            secondary_phrases = neutral_phrases
            tertiary_phrases = negative_phrases
            tone = "positive"
        elif sentiment_bias < -0.3:
            primary_phrases = negative_phrases
            secondary_phrases = neutral_phrases
            tertiary_phrases = positive_phrases
            tone = "negative"
        else:
            primary_phrases = neutral_phrases
            secondary_phrases = positive_phrases
            tertiary_phrases = negative_phrases
            tone = "neutral"
        
        # Assemble paragraphs
        paragraphs = []
        
        # Intro paragraph
        intro = f"{company_name} {random.choice(primary_phrases)} regarding {topic} today. "
        intro += f"The company {random.choice(primary_phrases)} in the recent {random.choice(['quarter', 'fiscal year', 'period'])}. "
        intro += f"Investors {random.choice(['are closely watching', 'have shown interest in', 'are analyzing'])} these developments."
        paragraphs.append(intro)
        
        # Details paragraph
        details = f"In terms of {topic}, {company_name} {random.choice(primary_phrases)}. "
        details += f"The management emphasized that {random.choice(primary_phrases)}. "
        if tone == "positive":
            details += f"However, some analysts noted that {random.choice(tertiary_phrases)}."
        elif tone == "negative":
            details += f"However, some analysts noted that {random.choice(tertiary_phrases)}."
        else:
            details += f"Analysts have {random.choice(['mixed', 'varied', 'different'])} views on these developments."
        paragraphs.append(details)
        
        # Market reaction paragraph
        reaction = f"The market reaction was {random.choice(['swift', 'measured', 'significant'])}. "
        reaction += f"The stock {random.choice(['moved', 'traded', 'performed'])} {random.choice(['in line with', 'contrary to', 'better than'])} the broader market. "
        reaction += f"Trading volumes were {random.choice(['elevated', 'normal', 'lower than average'])}."
        paragraphs.append(reaction)
        
        # Outlook paragraph
        outlook = f"Looking ahead, {company_name} {random.choice(secondary_phrases)} for the coming quarters. "
        outlook += f"The {random.choice(['management', 'CEO', 'executives'])} {random.choice(['expressed confidence', 'remained cautious', 'provided guidance'])} regarding future performance. "
        outlook += f"The {topic} is expected to {random.choice(['continue to impact', 'play a crucial role in', 'be a key factor for'])} the company's strategy."
        paragraphs.append(outlook)
        
        # Join paragraphs
        return "\n\n".join(paragraphs)
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Raw text content
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        # Join tokens back into a string
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text content.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        # Preprocess text
        preprocessed_text = self.preprocess_text(text)
        
        # Use TextBlob for sentiment analysis
        blob = TextBlob(preprocessed_text)
        
        # Get polarity and subjectivity
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': sentiment
        }
    
    def analyze_news_for_symbol(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Analyze news sentiment for a specific stock symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Fetch news articles
        articles = self.fetch_news(symbol, days)
        
        if not articles:
            self.logger.warning(f"No news found for {symbol}")
            return {
                'symbol': symbol,
                'articles_count': 0,
                'average_sentiment': 0,
                'sentiment_trend': 'neutral',
                'recent_sentiment': 'neutral',
                'articles': []
            }
        
        # Analyze sentiment for each article
        for article in articles:
            article['sentiment'] = self.analyze_sentiment(article['content'])
        
        # Calculate average sentiment
        polarities = [article['sentiment']['polarity'] for article in articles]
        average_sentiment = sum(polarities) / len(polarities)
        
        # Determine sentiment trend
        # Sort articles by date
        sorted_articles = sorted(articles, key=lambda x: x['date'])
        
        # Get polarities by date
        trend_data = [(article['date'], article['sentiment']['polarity']) for article in sorted_articles]
        
        # Calculate trend direction (positive slope = improving sentiment)
        if len(trend_data) >= 2:
            x = list(range(len(trend_data)))
            y = [polarity for _, polarity in trend_data]
            
            # Simple linear regression
            n = len(x)
            if n > 1:
                slope = (n * sum(x*y for x, y in zip(x, y)) - sum(x) * sum(y)) / (n * sum(x*x for x in x) - sum(x)**2)
                
                if slope > 0.05:
                    sentiment_trend = 'improving'
                elif slope < -0.05:
                    sentiment_trend = 'deteriorating'
                else:
                    sentiment_trend = 'stable'
            else:
                sentiment_trend = 'stable'
        else:
            sentiment_trend = 'stable'
        
        # Get recent sentiment (from the most recent article)
        recent_sentiment = sorted_articles[-1]['sentiment']['sentiment'] if sorted_articles else 'neutral'
        
        return {
            'symbol': symbol,
            'articles_count': len(articles),
            'average_sentiment': average_sentiment,
            'sentiment_trend': sentiment_trend,
            'recent_sentiment': recent_sentiment,
            'articles': articles
        }
    
    def analyze_news_for_multiple_symbols(self, symbols: List[str], 
                                       days: int = 7) -> Dict[str, Any]:
        """
        Analyze news sentiment for multiple stock symbols.
        
        Args:
            symbols: List of stock symbols
            days: Number of days to look back
            
        Returns:
            Dictionary with sentiment analysis results for all symbols
        """
        results = {}
        
        for symbol in symbols:
            try:
                symbol_results = self.analyze_news_for_symbol(symbol, days)
                results[symbol] = symbol_results
                
                # Add a small delay to avoid overloading news sources
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error analyzing news for {symbol}: {str(e)}")
                results[symbol] = {
                    'symbol': symbol,
                    'error': str(e),
                    'articles_count': 0,
                    'average_sentiment': 0,
                    'sentiment_trend': 'neutral',
                    'recent_sentiment': 'neutral',
                    'articles': []
                }
        
        return results
    
    def get_market_sentiment(self, index_symbols: List[str] = None) -> Dict[str, Any]:
        """
        Get overall market sentiment based on major indices and financial news.
        
        Args:
            index_symbols: List of index symbols (default uses major Indian indices)
            
        Returns:
            Dictionary with market sentiment analysis
        """
        # Default to major Indian indices if not provided
        if not index_symbols:
            index_symbols = ['NIFTY50', 'SENSEX', 'BANKNIFTY']
        
        # Analyze news for indices
        market_news = self.analyze_news_for_multiple_symbols(index_symbols)
        
        # Calculate overall market sentiment
        polarities = []
        for symbol, data in market_news.items():
            if data['articles_count'] > 0:
                polarities.append(data['average_sentiment'])
        
        if not polarities:
            overall_sentiment = 'neutral'
            average_polarity = 0
        else:
            average_polarity = sum(polarities) / len(polarities)
            
            if average_polarity > 0.2:
                overall_sentiment = 'bullish'
            elif average_polarity > 0.05:
                overall_sentiment = 'slightly_bullish'
            elif average_polarity < -0.2:
                overall_sentiment = 'bearish'
            elif average_polarity < -0.05:
                overall_sentiment = 'slightly_bearish'
            else:
                overall_sentiment = 'neutral'
        
        return {
            'overall_sentiment': overall_sentiment,
            'average_polarity': average_polarity,
            'index_data': market_news
        }