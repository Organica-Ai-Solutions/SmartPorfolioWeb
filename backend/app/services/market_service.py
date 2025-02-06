import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models import MarketData, Portfolio, Notification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class MarketService:
    def __init__(self, db: Session):
        self.db = db

    def fetch_and_store_market_data(self, tickers: List[str], days: int = 30) -> Dict[str, pd.DataFrame]:
        """Fetch and store market data for the given tickers."""
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()
        
        market_data = {}
        for ticker in tickers:
            try:
                # Fetch data from Yahoo Finance
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                
                # Store data in database
                for index, row in hist.iterrows():
                    market_data_entry = MarketData(
                        ticker=ticker,
                        timestamp=index,
                        open_price=float(row['Open']),
                        high_price=float(row['High']),
                        low_price=float(row['Low']),
                        close_price=float(row['Close']),
                        volume=int(row['Volume']),
                        additional_data={
                            'dividends': float(row.get('Dividends', 0)),
                            'stock_splits': float(row.get('Stock Splits', 0))
                        }
                    )
                    self.db.add(market_data_entry)
                
                market_data[ticker] = hist
            
            except Exception as e:
                print(f"Error fetching data for {ticker}: {str(e)}")
        
        self.db.commit()
        return market_data

    def analyze_market_conditions(self, tickers: List[str], lookback_days: int = 30) -> Dict:
        """Analyze current market conditions and trends."""
        market_data = self.fetch_and_store_market_data(tickers, lookback_days)
        
        # Calculate market metrics
        metrics = {
            "market_trends": self._calculate_market_trends(market_data),
            "volatility_analysis": self._analyze_volatility(market_data),
            "correlation_analysis": self._analyze_correlations(market_data),
            "technical_indicators": self._calculate_technical_indicators(market_data),
            "market_regime": self._detect_market_regime(market_data)
        }
        
        return metrics

    def _calculate_market_trends(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate market trends for each ticker."""
        trends = {}
        for ticker, data in market_data.items():
            # Calculate moving averages
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            
            # Calculate momentum indicators
            data['RSI'] = self._calculate_rsi(data['Close'])
            data['MACD'], data['Signal'] = self._calculate_macd(data['Close'])
            
            latest = data.iloc[-1]
            trends[ticker] = {
                "current_price": float(latest['Close']),
                "ma20": float(latest['MA20']),
                "ma50": float(latest['MA50']),
                "rsi": float(latest['RSI']),
                "macd": float(latest['MACD']),
                "macd_signal": float(latest['Signal']),
                "trend": "bullish" if latest['MA20'] > latest['MA50'] else "bearish"
            }
        
        return trends

    def _analyze_volatility(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze volatility patterns."""
        volatility = {}
        for ticker, data in market_data.items():
            returns = data['Close'].pct_change()
            volatility[ticker] = {
                "daily_volatility": float(returns.std()),
                "annualized_volatility": float(returns.std() * np.sqrt(252)),
                "high_low_range": float((data['High'] - data['Low']).mean()),
                "average_true_range": float(self._calculate_atr(data))
            }
        
        return volatility

    def _analyze_correlations(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze correlations between assets."""
        # Create returns DataFrame
        returns_data = pd.DataFrame()
        for ticker, data in market_data.items():
            returns_data[ticker] = data['Close'].pct_change()
        
        # Calculate correlation matrix
        correlation_matrix = returns_data.corr()
        
        # Perform PCA analysis
        scaler = StandardScaler()
        pca = PCA()
        scaled_returns = scaler.fit_transform(returns_data.fillna(0))
        pca_result = pca.fit_transform(scaled_returns)
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "principal_components": pca_result.tolist()
        }

    def _calculate_technical_indicators(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate various technical indicators."""
        indicators = {}
        for ticker, data in market_data.items():
            indicators[ticker] = {
                "bollinger_bands": self._calculate_bollinger_bands(data['Close']),
                "momentum": self._calculate_momentum(data['Close']),
                "volume_analysis": {
                    "average_volume": float(data['Volume'].mean()),
                    "volume_trend": "increasing" if data['Volume'].iloc[-1] > data['Volume'].mean() else "decreasing"
                }
            }
        
        return indicators

    def _detect_market_regime(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Detect current market regime using various metrics."""
        # Combine all close prices
        closes = pd.DataFrame({ticker: data['Close'] for ticker, data in market_data.items()})
        returns = closes.pct_change()
        
        # Calculate regime indicators
        volatility = returns.std() * np.sqrt(252)
        correlation = returns.corr().mean().mean()
        trend = (closes.iloc[-1] / closes.iloc[0] - 1).mean()
        
        # Determine regime
        regime = "normal"
        if volatility.mean() > 0.25:  # High volatility
            regime = "high_volatility"
        elif correlation > 0.7:  # High correlation
            regime = "risk_off"
        elif trend < -0.1:  # Downtrend
            regime = "bear_market"
        elif trend > 0.1:  # Uptrend
            regime = "bull_market"
        
        return {
            "regime": regime,
            "metrics": {
                "average_volatility": float(volatility.mean()),
                "average_correlation": float(correlation),
                "market_trend": float(trend)
            }
        }

    @staticmethod
    def _calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calculate_macd(prices: pd.Series) -> tuple:
        """Calculate MACD and Signal line."""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

    @staticmethod
    def _calculate_bollinger_bands(prices: pd.Series, window: int = 20) -> Dict:
        """Calculate Bollinger Bands."""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = ma + (std * 2)
        lower_band = ma - (std * 2)
        return {
            "middle_band": float(ma.iloc[-1]),
            "upper_band": float(upper_band.iloc[-1]),
            "lower_band": float(lower_band.iloc[-1])
        }

    @staticmethod
    def _calculate_momentum(prices: pd.Series, period: int = 10) -> float:
        """Calculate price momentum."""
        return float(prices.iloc[-1] / prices.iloc[-period] - 1)

    @staticmethod
    def _calculate_atr(data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.iloc[-1] 