import os
import httpx
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from textblob import TextBlob
import asyncio
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class SentimentService:
    """Service for analyzing market sentiment from news and social media."""
    
    def __init__(self):
        self.news_api_key = os.getenv("NEWS_API_KEY", "")
        self.twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN", "")
        self.reddit_client_id = os.getenv("REDDIT_CLIENT_ID", "")
        self.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY", "")
        
        # Check if we have at least one API key
        if not any([self.news_api_key, self.twitter_bearer_token, self.finnhub_api_key]):
            logger.warning("No sentiment API keys configured. Sentiment analysis will use fallback methods.")
    
    async def get_sentiment_for_tickers(self, tickers: List[str], days: int = 7) -> Dict[str, Any]:
        """Get sentiment analysis for a list of tickers from multiple sources."""
        try:
            # Create tasks for parallel execution
            tasks = []
            
            # Add tasks based on available API keys
            if self.news_api_key:
                tasks.append(self._get_news_sentiment(tickers, days))
            
            if self.twitter_bearer_token:
                tasks.append(self._get_social_media_sentiment(tickers, days, source="twitter"))
            
            if self.reddit_client_id and self.reddit_client_secret:
                tasks.append(self._get_social_media_sentiment(tickers, days, source="reddit"))
            
            if self.finnhub_api_key:
                tasks.append(self._get_finnhub_sentiment(tickers, days))
            
            # Always add the fallback sentiment analysis
            tasks.append(self._get_fallback_sentiment(tickers, days))
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results, filtering out exceptions
            combined_sentiment = {
                "overall_sentiment": {},
                "news_sentiment": {},
                "social_sentiment": {},
                "sources": [],
                "sentiment_trends": {},
                "keywords": {},
                "timestamp": datetime.now().isoformat()
            }
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in sentiment analysis: {str(result)}")
                    continue
                
                # Update combined sentiment with valid results
                if "source" in result:
                    combined_sentiment["sources"].append(result["source"])
                
                if "overall_sentiment" in result:
                    for ticker, sentiment in result["overall_sentiment"].items():
                        if ticker not in combined_sentiment["overall_sentiment"]:
                            combined_sentiment["overall_sentiment"][ticker] = sentiment
                        else:
                            # Average with existing sentiment
                            existing = combined_sentiment["overall_sentiment"][ticker]
                            combined_sentiment["overall_sentiment"][ticker] = (existing + sentiment) / 2
                
                # Merge other sentiment data
                for key in ["news_sentiment", "social_sentiment", "sentiment_trends", "keywords"]:
                    if key in result:
                        combined_sentiment[key].update(result[key])
            
            # Calculate final sentiment scores
            for ticker in tickers:
                if ticker not in combined_sentiment["overall_sentiment"]:
                    combined_sentiment["overall_sentiment"][ticker] = 0.0
            
            return combined_sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                "overall_sentiment": {ticker: 0.0 for ticker in tickers},
                "sources": ["fallback"],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_news_sentiment(self, tickers: List[str], days: int) -> Dict[str, Any]:
        """Get sentiment from news articles using NewsAPI."""
        try:
            news_sentiment = {}
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                for ticker in tickers:
                    # Query NewsAPI for articles related to the ticker
                    response = await client.get(
                        "https://newsapi.org/v2/everything",
                        params={
                            "q": f"{ticker} OR {self._get_company_name(ticker)}",
                            "from": from_date,
                            "sortBy": "publishedAt",
                            "language": "en",
                            "apiKey": self.news_api_key
                        }
                    )
                    
                    if response.status_code != 200:
                        logger.warning(f"NewsAPI request failed: {response.text}")
                        continue
                    
                    data = response.json()
                    articles = data.get("articles", [])
                    
                    if not articles:
                        news_sentiment[ticker] = {"score": 0, "article_count": 0, "keywords": []}
                        continue
                    
                    # Analyze sentiment of each article
                    scores = []
                    all_text = ""
                    
                    for article in articles:
                        title = article.get("title", "")
                        description = article.get("description", "")
                        content = article.get("content", "")
                        
                        text = f"{title}. {description}. {content}"
                        all_text += text + " "
                        
                        blob = TextBlob(text)
                        scores.append(blob.sentiment.polarity)
                    
                    # Extract keywords
                    keywords = self._extract_keywords(all_text)
                    
                    # Calculate average sentiment
                    avg_sentiment = sum(scores) / len(scores) if scores else 0
                    
                    news_sentiment[ticker] = {
                        "score": avg_sentiment,
                        "article_count": len(articles),
                        "keywords": keywords[:10]  # Top 10 keywords
                    }
            
            # Create overall result
            result = {
                "source": "news_api",
                "overall_sentiment": {ticker: data["score"] for ticker, data in news_sentiment.items()},
                "news_sentiment": news_sentiment,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting news sentiment: {str(e)}")
            raise
    
    async def _get_social_media_sentiment(self, tickers: List[str], days: int, source: str) -> Dict[str, Any]:
        """Get sentiment from social media (Twitter/Reddit)."""
        try:
            social_sentiment = {}
            
            if source == "twitter" and self.twitter_bearer_token:
                # Use Twitter API v2
                async with httpx.AsyncClient(timeout=30.0) as client:
                    for ticker in tickers:
                        # Query Twitter for recent tweets about the ticker
                        end_time = datetime.now()
                        start_time = end_time - timedelta(days=days)
                        
                        response = await client.get(
                            "https://api.twitter.com/2/tweets/search/recent",
                            headers={"Authorization": f"Bearer {self.twitter_bearer_token}"},
                            params={
                                "query": f"${ticker} OR {self._get_company_name(ticker)} lang:en -is:retweet",
                                "max_results": 100,
                                "start_time": start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                                "end_time": end_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                                "tweet.fields": "created_at,public_metrics"
                            }
                        )
                        
                        if response.status_code != 200:
                            logger.warning(f"Twitter API request failed: {response.text}")
                            continue
                        
                        data = response.json()
                        tweets = data.get("data", [])
                        
                        if not tweets:
                            social_sentiment[ticker] = {"score": 0, "tweet_count": 0, "keywords": []}
                            continue
                        
                        # Analyze sentiment of each tweet
                        scores = []
                        all_text = ""
                        
                        for tweet in tweets:
                            text = tweet.get("text", "")
                            all_text += text + " "
                            
                            blob = TextBlob(text)
                            scores.append(blob.sentiment.polarity)
                        
                        # Extract keywords
                        keywords = self._extract_keywords(all_text)
                        
                        # Calculate average sentiment
                        avg_sentiment = sum(scores) / len(scores) if scores else 0
                        
                        social_sentiment[ticker] = {
                            "score": avg_sentiment,
                            "tweet_count": len(tweets),
                            "keywords": keywords[:10]  # Top 10 keywords
                        }
            
            elif source == "reddit" and self.reddit_client_id and self.reddit_client_secret:
                # Use Reddit API
                async with httpx.AsyncClient(timeout=30.0) as client:
                    # Get OAuth token
                    auth_response = await client.post(
                        "https://www.reddit.com/api/v1/access_token",
                        auth=(self.reddit_client_id, self.reddit_client_secret),
                        data={"grant_type": "client_credentials"},
                        headers={"User-Agent": "SmartPortfolio/1.0"}
                    )
                    
                    if auth_response.status_code != 200:
                        logger.warning(f"Reddit authentication failed: {auth_response.text}")
                        return {"source": "reddit", "overall_sentiment": {ticker: 0.0 for ticker in tickers}}
                    
                    token_data = auth_response.json()
                    access_token = token_data.get("access_token")
                    
                    for ticker in tickers:
                        # Search for posts about the ticker
                        search_response = await client.get(
                            f"https://oauth.reddit.com/search",
                            headers={
                                "Authorization": f"Bearer {access_token}",
                                "User-Agent": "SmartPortfolio/1.0"
                            },
                            params={
                                "q": f"{ticker} OR {self._get_company_name(ticker)}",
                                "sort": "relevance",
                                "t": "week",
                                "limit": 100
                            }
                        )
                        
                        if search_response.status_code != 200:
                            logger.warning(f"Reddit search failed: {search_response.text}")
                            continue
                        
                        posts = search_response.json().get("data", {}).get("children", [])
                        
                        if not posts:
                            social_sentiment[ticker] = {"score": 0, "post_count": 0, "keywords": []}
                            continue
                        
                        # Analyze sentiment of each post
                        scores = []
                        all_text = ""
                        
                        for post in posts:
                            title = post.get("data", {}).get("title", "")
                            selftext = post.get("data", {}).get("selftext", "")
                            
                            text = f"{title}. {selftext}"
                            all_text += text + " "
                            
                            blob = TextBlob(text)
                            scores.append(blob.sentiment.polarity)
                        
                        # Extract keywords
                        keywords = self._extract_keywords(all_text)
                        
                        # Calculate average sentiment
                        avg_sentiment = sum(scores) / len(scores) if scores else 0
                        
                        social_sentiment[ticker] = {
                            "score": avg_sentiment,
                            "post_count": len(posts),
                            "keywords": keywords[:10]  # Top 10 keywords
                        }
            
            # Create overall result
            result = {
                "source": source,
                "overall_sentiment": {ticker: data["score"] for ticker, data in social_sentiment.items()},
                "social_sentiment": social_sentiment,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting social media sentiment from {source}: {str(e)}")
            raise
    
    async def _get_finnhub_sentiment(self, tickers: List[str], days: int) -> Dict[str, Any]:
        """Get sentiment from Finnhub API."""
        try:
            sentiment_data = {}
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                for ticker in tickers:
                    response = await client.get(
                        "https://finnhub.io/api/v1/news-sentiment",
                        params={
                            "symbol": ticker,
                            "token": self.finnhub_api_key
                        }
                    )
                    
                    if response.status_code != 200:
                        logger.warning(f"Finnhub API request failed: {response.text}")
                        continue
                    
                    data = response.json()
                    
                    # Extract sentiment score
                    sentiment_score = data.get("sentiment", {}).get("sentiment", 0)
                    buzz = data.get("buzz", {})
                    
                    sentiment_data[ticker] = {
                        "score": sentiment_score,
                        "buzz_score": buzz.get("buzz", 0),
                        "article_count": buzz.get("articlesInLastWeek", 0)
                    }
            
            # Create overall result
            result = {
                "source": "finnhub",
                "overall_sentiment": {ticker: data["score"] for ticker, data in sentiment_data.items()},
                "news_sentiment": sentiment_data,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting Finnhub sentiment: {str(e)}")
            raise
    
    async def _get_fallback_sentiment(self, tickers: List[str], days: int) -> Dict[str, Any]:
        """Fallback method for sentiment analysis when APIs are not available."""
        try:
            # Use yfinance to get recent price movements as a proxy for sentiment
            import yfinance as yf
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            sentiment_data = {}
            
            for ticker in tickers:
                try:
                    # Download historical data
                    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    
                    if stock_data.empty:
                        sentiment_data[ticker] = {"score": 0, "trend": "neutral"}
                        continue
                    
                    # Calculate price change
                    first_price = stock_data["Close"].iloc[0] if len(stock_data) > 0 else 0
                    last_price = stock_data["Close"].iloc[-1] if len(stock_data) > 0 else 0
                    
                    if first_price == 0:
                        price_change = 0
                    else:
                        price_change = (last_price - first_price) / first_price
                    
                    # Map price change to sentiment score (-1 to 1)
                    # Using a sigmoid-like function to map any price change to -1 to 1 range
                    sentiment_score = 2 / (1 + np.exp(-10 * price_change)) - 1
                    
                    # Determine trend
                    if price_change > 0.03:
                        trend = "bullish"
                    elif price_change < -0.03:
                        trend = "bearish"
                    else:
                        trend = "neutral"
                    
                    sentiment_data[ticker] = {
                        "score": sentiment_score,
                        "trend": trend,
                        "price_change": price_change
                    }
                    
                except Exception as ticker_error:
                    logger.error(f"Error processing fallback sentiment for {ticker}: {str(ticker_error)}")
                    sentiment_data[ticker] = {"score": 0, "trend": "neutral"}
            
            # Create overall result
            result = {
                "source": "price_movement",
                "overall_sentiment": {ticker: data["score"] for ticker, data in sentiment_data.items()},
                "sentiment_trends": {ticker: data["trend"] for ticker, data in sentiment_data.items()},
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting fallback sentiment: {str(e)}")
            return {
                "source": "fallback",
                "overall_sentiment": {ticker: 0.0 for ticker in tickers},
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            from nltk.corpus import stopwords
            import nltk
            
            # Download stopwords if needed
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
            
            # Get English stopwords
            stop_words = set(stopwords.words('english'))
            
            # Add financial stopwords
            financial_stopwords = [
                'stock', 'market', 'price', 'share', 'shares', 'company', 'companies',
                'investor', 'investors', 'trading', 'trader', 'traders', 'buy', 'sell',
                'investment', 'investments', 'profit', 'loss', 'gains', 'earnings',
                'quarter', 'quarterly', 'annual', 'year', 'month', 'week', 'day'
            ]
            stop_words.update(financial_stopwords)
            
            # Create vectorizer
            vectorizer = CountVectorizer(
                max_features=50,
                stop_words=list(stop_words),
                ngram_range=(1, 2)
            )
            
            # Extract features
            X = vectorizer.fit_transform([text])
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Get counts
            counts = X.toarray()[0]
            
            # Create dictionary of feature -> count
            keyword_counts = {feature_names[i]: counts[i] for i in range(len(feature_names))}
            
            # Sort by count
            sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Return just the keywords
            return [keyword for keyword, _ in sorted_keywords]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def _get_company_name(self, ticker: str) -> str:
        """Get company name from ticker symbol."""
        # This is a simplified version. In production, you would use a proper lookup service
        # or database to get the company name.
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if 'shortName' in info:
                return info['shortName']
            elif 'longName' in info:
                return info['longName']
            else:
                return ticker
                
        except Exception:
            return ticker
