"""
News ingestion and enrichment with embeddings and risk assessment.
"""
import hashlib
import os
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import requests
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from ..db import crud


class NewsIngester:
    """Handles news ingestion and enrichment"""
    
    def __init__(self, config: dict):
        """
        Initialize news ingester.
        
        Args:
            config: News configuration dictionary
        """
        self.config = config
        self.providers = config.get('providers', [])
        self.risk_tags = config.get('risk_tags_thresholds', {})
        self.embedding_model_name = config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_model = None
        self.simulation_mode = False
    
    def _load_embedding_model(self):
        """Lazy load embedding model"""
        if self.embedding_model is None:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                print(f"Error loading embedding model: {e}")
                self.simulation_mode = True
    
    def fetch_news(
        self,
        symbols: Optional[List[str]] = None,
        lookback_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles from configured providers.
        
        Args:
            symbols: Optional list of symbols to filter by
            lookback_hours: Hours to look back for news
            
        Returns:
            List of news articles
        """
        all_articles = []
        
        for provider in self.providers:
            if not provider.get('enabled', True):
                continue
            
            provider_type = provider.get('type')
            
            if provider_type == 'newsapi':
                articles = self._fetch_newsapi(symbols, lookback_hours, provider)
            elif provider_type == 'rss':
                articles = self._fetch_rss(provider)
            else:
                print(f"Unknown provider type: {provider_type}")
                continue
            
            all_articles.extend(articles)
        
        # If simulation mode or no providers, generate mock data
        if not all_articles and self.simulation_mode:
            all_articles = self._generate_mock_news(symbols, lookback_hours)
        
        return all_articles
    
    def _fetch_newsapi(
        self,
        symbols: Optional[List[str]],
        lookback_hours: int,
        provider: dict
    ) -> List[Dict[str, Any]]:
        """Fetch news from NewsAPI"""
        api_key_env = provider.get('api_key_env', 'NEWSAPI_KEY')
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            print(f"NewsAPI key not found in env var: {api_key_env}")
            return []
        
        articles = []
        from_date = datetime.now() - timedelta(hours=lookback_hours)
        
        # Build query
        if symbols:
            # For stock news, use company names or ticker symbols
            query = ' OR '.join(symbols[:5])  # Limit to first 5 symbols
        else:
            query = 'stock market OR trading OR finance'
        
        try:
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': query,
                'from': from_date.isoformat(),
                'sortBy': 'publishedAt',
                'apiKey': api_key,
                'language': 'en'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for article in data.get('articles', []):
                articles.append({
                    'source': article.get('source', {}).get('name', 'NewsAPI'),
                    'title': article.get('title', ''),
                    'body': article.get('description', '') + ' ' + article.get('content', ''),
                    'url': article.get('url', ''),
                    'published_at': datetime.fromisoformat(
                        article.get('publishedAt', '').replace('Z', '+00:00')
                    ),
                    'symbols': self._extract_symbols_from_text(
                        article.get('title', '') + ' ' + article.get('description', ''),
                        symbols or []
                    )
                })
        except Exception as e:
            print(f"Error fetching NewsAPI: {e}")
        
        return articles
    
    def _fetch_rss(self, provider: dict) -> List[Dict[str, Any]]:
        """Fetch news from RSS feeds"""
        import feedparser
        
        articles = []
        urls = provider.get('urls', [])
        
        for url in urls:
            try:
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:20]:  # Limit to 20 articles per feed
                    published_at = datetime.now()
                    if hasattr(entry, 'published_parsed'):
                        published_at = datetime(*entry.published_parsed[:6])
                    
                    articles.append({
                        'source': feed.feed.get('title', 'RSS'),
                        'title': entry.get('title', ''),
                        'body': entry.get('summary', ''),
                        'url': entry.get('link', ''),
                        'published_at': published_at,
                        'symbols': []
                    })
            except Exception as e:
                print(f"Error fetching RSS feed {url}: {e}")
        
        return articles
    
    def _extract_symbols_from_text(self, text: str, symbols: List[str]) -> List[str]:
        """Extract mentioned symbols from text"""
        mentioned = []
        text_upper = text.upper()
        
        for symbol in symbols:
            symbol_base = symbol.split('.')[0]  # Remove suffix like .NS
            if symbol_base in text_upper:
                mentioned.append(symbol)
        
        return mentioned
    
    def deduplicate_news(
        self,
        db: Session,
        articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate news articles using content hash.
        
        Args:
            db: Database session
            articles: List of articles to deduplicate
            
        Returns:
            List of unique articles not already in database
        """
        unique_articles = []
        
        for article in articles:
            # Compute content hash
            content = f"{article['title']}{article['body']}"
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Check if already exists
            existing = crud.get_news_by_hash(db, content_hash)
            if existing is None:
                article['hash'] = content_hash
                unique_articles.append(article)
        
        return unique_articles
    
    def enrich_news(
        self,
        articles: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Enrich news articles with embeddings and risk assessment.
        
        Args:
            articles: List of articles to enrich
            
        Returns:
            List of tuples (article, enrichment_data)
        """
        self._load_embedding_model()
        enriched = []
        
        for article in articles:
            # Generate embedding
            text = f"{article['title']} {article['body']}"
            
            if self.embedding_model is not None:
                try:
                    embedding_vec = self.embedding_model.encode(text)
                    embedding_bytes = pickle.dumps(embedding_vec)
                except Exception as e:
                    print(f"Error generating embedding: {e}")
                    embedding_bytes = None
            else:
                embedding_bytes = None
            
            # Assess risk
            risk_tags, risk_score = self._assess_risk(text)
            
            # Generate summary (simplified - use first 3 sentences)
            summary = self._generate_summary(article['body'])
            
            enrichment = {
                'embedding': embedding_bytes,
                'risk_tags': risk_tags,
                'risk_score': risk_score,
                'summary': summary
            }
            
            enriched.append((article, enrichment))
        
        return enriched
    
    def _assess_risk(self, text: str) -> Tuple[List[str], float]:
        """Assess risk from news text"""
        text_lower = text.lower()
        risk_tags = []
        risk_score = 0.0
        
        # Check for high risk keywords
        for tag in self.risk_tags.get('high_risk', []):
            if tag.lower() in text_lower:
                risk_tags.append(f"high:{tag}")
                risk_score += 0.5
        
        # Check for medium risk keywords
        for tag in self.risk_tags.get('medium_risk', []):
            if tag.lower() in text_lower:
                risk_tags.append(f"medium:{tag}")
                risk_score += 0.3
        
        # Check for low risk keywords
        for tag in self.risk_tags.get('low_risk', []):
            if tag.lower() in text_lower:
                risk_tags.append(f"low:{tag}")
                risk_score += 0.1
        
        # Normalize score
        risk_score = min(1.0, risk_score)
        
        return risk_tags, risk_score
    
    def _generate_summary(self, text: str) -> str:
        """Generate 3-bullet summary (simplified)"""
        if not text:
            return ""
        
        # Split into sentences
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Take first 3 sentences
        summary_sentences = sentences[:3]
        summary = '\n• ' + '\n• '.join(summary_sentences)
        
        return summary
    
    def save_news(
        self,
        db: Session,
        enriched_articles: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    ):
        """
        Save news articles and enrichment to database.
        
        Args:
            db: Database session
            enriched_articles: List of (article, enrichment) tuples
        """
        for article, enrichment in enriched_articles:
            # Save raw news
            news_raw = crud.create_news_raw(
                db=db,
                ts_published=article['published_at'],
                source=article['source'],
                title=article['title'],
                content_hash=article['hash'],
                body=article.get('body'),
                url=article.get('url'),
                symbols=article.get('symbols')
            )
            
            # Save enriched data
            crud.create_news_enriched(
                db=db,
                news_id=news_raw.news_id,
                embedding=enrichment['embedding'],
                risk_tags=enrichment['risk_tags'],
                risk_score=enrichment['risk_score'],
                summary=enrichment['summary']
            )
    
    def _generate_mock_news(
        self,
        symbols: Optional[List[str]],
        lookback_hours: int
    ) -> List[Dict[str, Any]]:
        """Generate mock news for simulation mode"""
        import random
        
        articles = []
        num_articles = random.randint(5, 15)
        
        templates = [
            "Company {symbol} reports strong quarterly earnings",
            "{symbol} announces new product launch",
            "Analysts upgrade {symbol} to buy rating",
            "{symbol} faces regulatory investigation",
            "Market volatility impacts {symbol} trading"
        ]
        
        for i in range(num_articles):
            symbol = random.choice(symbols) if symbols else "MARKET"
            title = random.choice(templates).format(symbol=symbol)
            
            articles.append({
                'source': 'MockNews',
                'title': title,
                'body': f"This is a simulated news article about {symbol}. " * 5,
                'url': f'https://example.com/news/{i}',
                'published_at': datetime.now() - timedelta(hours=random.randint(0, lookback_hours)),
                'symbols': [symbol] if symbol != "MARKET" else []
            })
        
        return articles
