import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from enum import Enum
import asyncio
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Import our services
from .macro_service import MacroService, MarketRegime
from .sentiment_service import SentimentService
from .options_service import OptionsService

logger = logging.getLogger(__name__)

class AssetClass(str, Enum):
    STOCKS = "stocks"
    BONDS = "bonds"
    COMMODITIES = "commodities"
    CRYPTO = "crypto"
    CASH = "cash"
    REAL_ESTATE = "real_estate"

class AllocationSignal(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class TrendDirection(str, Enum):
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

class DynamicAllocationService:
    """Service for dynamic asset allocation based on market conditions."""
    
    def __init__(self):
        self.macro_service = MacroService()
        self.sentiment_service = SentimentService()
        self.options_service = OptionsService()
        
        # Asset class mappings for tickers
        self.asset_class_mapping = {
            # Stocks
            "SPY": AssetClass.STOCKS, "QQQ": AssetClass.STOCKS, "IWM": AssetClass.STOCKS,
            "VTI": AssetClass.STOCKS, "VEA": AssetClass.STOCKS, "VWO": AssetClass.STOCKS,
            "AAPL": AssetClass.STOCKS, "MSFT": AssetClass.STOCKS, "GOOGL": AssetClass.STOCKS,
            "AMZN": AssetClass.STOCKS, "TSLA": AssetClass.STOCKS, "META": AssetClass.STOCKS,
            
            # Bonds
            "TLT": AssetClass.BONDS, "IEF": AssetClass.BONDS, "SHY": AssetClass.BONDS,
            "BND": AssetClass.BONDS, "AGG": AssetClass.BONDS, "HYG": AssetClass.BONDS,
            "LQD": AssetClass.BONDS, "TIP": AssetClass.BONDS,
            
            # Commodities
            "GLD": AssetClass.COMMODITIES, "SLV": AssetClass.COMMODITIES, "USO": AssetClass.COMMODITIES,
            "DBA": AssetClass.COMMODITIES, "DBC": AssetClass.COMMODITIES, "PDBC": AssetClass.COMMODITIES,
            
            # Crypto
            "BTC-USD": AssetClass.CRYPTO, "ETH-USD": AssetClass.CRYPTO, "ADA-USD": AssetClass.CRYPTO,
            "SOL-USD": AssetClass.CRYPTO, "DOT-USD": AssetClass.CRYPTO, "MATIC-USD": AssetClass.CRYPTO,
            
            # Real Estate
            "VNQ": AssetClass.REAL_ESTATE, "REIT": AssetClass.REAL_ESTATE, "IYR": AssetClass.REAL_ESTATE,
            
            # Cash/Money Market
            "SHV": AssetClass.CASH, "BIL": AssetClass.CASH, "VMOT": AssetClass.CASH
        }
    
    async def get_dynamic_allocation(
        self, 
        tickers: List[str], 
        base_allocation: Dict[str, float],
        risk_tolerance: str = "medium",
        lookback_days: int = 252
    ) -> Dict[str, Any]:
        """Get dynamic asset allocation based on current market conditions."""
        try:
            # Get market data and indicators
            market_data = await self._get_comprehensive_market_data(tickers, lookback_days)
            
            # Determine tactical allocation based on market regime
            tactical_allocation = await self._get_tactical_allocation(
                tickers, base_allocation, market_data, risk_tolerance
            )
            
            # Apply dynamic risk budgeting
            risk_adjusted_allocation = await self._apply_dynamic_risk_budgeting(
                tactical_allocation, market_data, risk_tolerance
            )
            
            # Apply momentum signals
            final_allocation = await self._apply_momentum_signals(
                risk_adjusted_allocation, market_data, tickers
            )
            
            # Generate allocation insights
            insights = await self._generate_allocation_insights(
                base_allocation, final_allocation, market_data
            )
            
            return {
                "base_allocation": base_allocation,
                "tactical_allocation": tactical_allocation,
                "risk_adjusted_allocation": risk_adjusted_allocation,
                "final_allocation": final_allocation,
                "allocation_insights": insights,
                "market_data_summary": market_data.get("summary", {}),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in dynamic allocation: {str(e)}")
            return {
                "base_allocation": base_allocation,
                "final_allocation": base_allocation,  # Fallback to base allocation
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_comprehensive_market_data(self, tickers: List[str], lookback_days: int) -> Dict[str, Any]:
        """Gather comprehensive market data from all sources."""
        try:
            # Create tasks for parallel data collection
            tasks = [
                self.macro_service.get_macro_indicators(),
                self.sentiment_service.get_sentiment_for_tickers(tickers),
                self.options_service.get_options_data(tickers),
                self._get_price_momentum_data(tickers, lookback_days),
                self._get_volatility_data(tickers, lookback_days)
            ]
            
            # Execute all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            market_data = {
                "macro_indicators": results[0] if not isinstance(results[0], Exception) else {},
                "sentiment_data": results[1] if not isinstance(results[1], Exception) else {},
                "options_data": results[2] if not isinstance(results[2], Exception) else {},
                "momentum_data": results[3] if not isinstance(results[3], Exception) else {},
                "volatility_data": results[4] if not isinstance(results[4], Exception) else {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Create summary
            market_data["summary"] = self._create_market_summary(market_data)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting comprehensive market data: {str(e)}")
            return {"error": str(e)}
    
    async def _get_tactical_allocation(
        self, 
        tickers: List[str], 
        base_allocation: Dict[str, float],
        market_data: Dict[str, Any],
        risk_tolerance: str
    ) -> Dict[str, float]:
        """Create tactical allocation based on market regime."""
        try:
            # Get current market regime
            macro_data = market_data.get("macro_indicators", {})
            market_regime_data = macro_data.get("market_regime", {})
            current_regime = market_regime_data.get("current_regime", "unknown")
            
            # Get asset class allocations
            asset_class_allocation = self._get_asset_class_allocation(tickers, base_allocation)
            
            # Define tactical rules based on market regimes
            tactical_rules = self._get_tactical_rules_by_regime(current_regime, risk_tolerance)
            
            # Apply tactical adjustments
            tactical_allocation = {}
            
            for ticker, base_weight in base_allocation.items():
                asset_class = self.asset_class_mapping.get(ticker, AssetClass.STOCKS)
                
                # Get tactical multiplier for this asset class
                tactical_multiplier = tactical_rules.get(asset_class, 1.0)
                
                # Apply sentiment adjustments
                sentiment_multiplier = self._get_sentiment_multiplier(
                    ticker, market_data.get("sentiment_data", {})
                )
                
                # Calculate adjusted weight
                adjusted_weight = base_weight * tactical_multiplier * sentiment_multiplier
                tactical_allocation[ticker] = adjusted_weight
            
            # Normalize weights to sum to 1
            total_weight = sum(tactical_allocation.values())
            if total_weight > 0:
                tactical_allocation = {
                    ticker: weight / total_weight 
                    for ticker, weight in tactical_allocation.items()
                }
            else:
                tactical_allocation = base_allocation
            
            return tactical_allocation
            
        except Exception as e:
            logger.error(f"Error in tactical allocation: {str(e)}")
            return base_allocation
    
    def _get_tactical_rules_by_regime(self, regime: str, risk_tolerance: str) -> Dict[AssetClass, float]:
        """Get tactical allocation rules based on market regime and risk tolerance."""
        # Base multipliers by regime
        regime_rules = {
            MarketRegime.EXPANSION: {
                AssetClass.STOCKS: 1.1,
                AssetClass.BONDS: 0.9,
                AssetClass.COMMODITIES: 1.0,
                AssetClass.CRYPTO: 1.2,
                AssetClass.CASH: 0.8,
                AssetClass.REAL_ESTATE: 1.05
            },
            MarketRegime.PEAK: {
                AssetClass.STOCKS: 0.9,
                AssetClass.BONDS: 1.1,
                AssetClass.COMMODITIES: 1.1,
                AssetClass.CRYPTO: 0.8,
                AssetClass.CASH: 1.2,
                AssetClass.REAL_ESTATE: 0.95
            },
            MarketRegime.CONTRACTION: {
                AssetClass.STOCKS: 0.7,
                AssetClass.BONDS: 1.3,
                AssetClass.COMMODITIES: 0.8,
                AssetClass.CRYPTO: 0.6,
                AssetClass.CASH: 1.5,
                AssetClass.REAL_ESTATE: 0.8
            },
            MarketRegime.TROUGH: {
                AssetClass.STOCKS: 1.2,
                AssetClass.BONDS: 1.0,
                AssetClass.COMMODITIES: 1.1,
                AssetClass.CRYPTO: 1.3,
                AssetClass.CASH: 0.7,
                AssetClass.REAL_ESTATE: 1.1
            },
            MarketRegime.RECOVERY: {
                AssetClass.STOCKS: 1.15,
                AssetClass.BONDS: 0.95,
                AssetClass.COMMODITIES: 1.05,
                AssetClass.CRYPTO: 1.25,
                AssetClass.CASH: 0.8,
                AssetClass.REAL_ESTATE: 1.1
            },
            MarketRegime.STAGFLATION: {
                AssetClass.STOCKS: 0.8,
                AssetClass.BONDS: 0.7,
                AssetClass.COMMODITIES: 1.4,
                AssetClass.CRYPTO: 1.1,
                AssetClass.CASH: 1.1,
                AssetClass.REAL_ESTATE: 1.2
            },
            MarketRegime.REFLATION: {
                AssetClass.STOCKS: 1.05,
                AssetClass.BONDS: 0.9,
                AssetClass.COMMODITIES: 1.2,
                AssetClass.CRYPTO: 1.1,
                AssetClass.CASH: 0.9,
                AssetClass.REAL_ESTATE: 1.15
            },
            MarketRegime.DISINFLATION: {
                AssetClass.STOCKS: 1.1,
                AssetClass.BONDS: 1.2,
                AssetClass.COMMODITIES: 0.8,
                AssetClass.CRYPTO: 0.9,
                AssetClass.CASH: 0.9,
                AssetClass.REAL_ESTATE: 1.0
            }
        }
        
        # Get base rules for the regime
        base_rules = regime_rules.get(regime, {
            AssetClass.STOCKS: 1.0,
            AssetClass.BONDS: 1.0,
            AssetClass.COMMODITIES: 1.0,
            AssetClass.CRYPTO: 1.0,
            AssetClass.CASH: 1.0,
            AssetClass.REAL_ESTATE: 1.0
        })
        
        # Adjust based on risk tolerance
        risk_adjustments = {
            "low": {
                AssetClass.STOCKS: 0.9,
                AssetClass.BONDS: 1.1,
                AssetClass.COMMODITIES: 0.8,
                AssetClass.CRYPTO: 0.7,
                AssetClass.CASH: 1.2,
                AssetClass.REAL_ESTATE: 0.9
            },
            "medium": {
                AssetClass.STOCKS: 1.0,
                AssetClass.BONDS: 1.0,
                AssetClass.COMMODITIES: 1.0,
                AssetClass.CRYPTO: 1.0,
                AssetClass.CASH: 1.0,
                AssetClass.REAL_ESTATE: 1.0
            },
            "high": {
                AssetClass.STOCKS: 1.1,
                AssetClass.BONDS: 0.9,
                AssetClass.COMMODITIES: 1.2,
                AssetClass.CRYPTO: 1.3,
                AssetClass.CASH: 0.8,
                AssetClass.REAL_ESTATE: 1.1
            }
        }
        
        risk_adj = risk_adjustments.get(risk_tolerance, risk_adjustments["medium"])
        
        # Combine regime rules with risk adjustments
        final_rules = {}
        for asset_class in AssetClass:
            final_rules[asset_class] = base_rules.get(asset_class, 1.0) * risk_adj.get(asset_class, 1.0)
        
        return final_rules
    
    async def _apply_dynamic_risk_budgeting(
        self, 
        allocation: Dict[str, float], 
        market_data: Dict[str, Any],
        risk_tolerance: str
    ) -> Dict[str, float]:
        """Apply dynamic risk budgeting based on volatility forecasts."""
        try:
            volatility_data = market_data.get("volatility_data", {})
            options_data = market_data.get("options_data", {})
            
            # Calculate volatility-adjusted weights
            risk_adjusted_allocation = {}
            
            for ticker, weight in allocation.items():
                # Get historical volatility
                hist_vol = volatility_data.get(ticker, {}).get("historical_volatility", 0.2)
                
                # Get implied volatility if available
                implied_vol = None
                if ticker in options_data.get("implied_volatility", {}):
                    implied_vol = options_data["implied_volatility"][ticker]
                
                # Calculate volatility forecast
                vol_forecast = self._calculate_volatility_forecast(hist_vol, implied_vol)
                
                # Calculate risk budget adjustment
                risk_adjustment = self._calculate_risk_adjustment(vol_forecast, risk_tolerance)
                
                # Apply adjustment
                risk_adjusted_allocation[ticker] = weight * risk_adjustment
            
            # Normalize weights
            total_weight = sum(risk_adjusted_allocation.values())
            if total_weight > 0:
                risk_adjusted_allocation = {
                    ticker: weight / total_weight 
                    for ticker, weight in risk_adjusted_allocation.items()
                }
            else:
                risk_adjusted_allocation = allocation
            
            return risk_adjusted_allocation
            
        except Exception as e:
            logger.error(f"Error in dynamic risk budgeting: {str(e)}")
            return allocation
    
    async def _apply_momentum_signals(
        self, 
        allocation: Dict[str, float], 
        market_data: Dict[str, Any], 
        tickers: List[str]
    ) -> Dict[str, float]:
        """Apply momentum-based trend following signals."""
        try:
            momentum_data = market_data.get("momentum_data", {})
            
            # Calculate momentum adjustments
            momentum_allocation = {}
            
            for ticker, weight in allocation.items():
                # Get momentum signals
                momentum_signals = momentum_data.get(ticker, {})
                
                # Calculate momentum score
                momentum_score = self._calculate_momentum_score(momentum_signals)
                
                # Get momentum multiplier
                momentum_multiplier = self._get_momentum_multiplier(momentum_score)
                
                # Apply momentum adjustment
                momentum_allocation[ticker] = weight * momentum_multiplier
            
            # Normalize weights
            total_weight = sum(momentum_allocation.values())
            if total_weight > 0:
                momentum_allocation = {
                    ticker: weight / total_weight 
                    for ticker, weight in momentum_allocation.items()
                }
            else:
                momentum_allocation = allocation
            
            return momentum_allocation
            
        except Exception as e:
            logger.error(f"Error applying momentum signals: {str(e)}")
            return allocation
    
    async def _get_price_momentum_data(self, tickers: List[str], lookback_days: int) -> Dict[str, Any]:
        """Get price momentum data for tickers."""
        try:
            momentum_data = {}
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            for ticker in tickers:
                try:
                    # Download price data
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    
                    if data.empty:
                        continue
                    
                    # Calculate momentum indicators
                    close_prices = data['Close']
                    
                    # Moving averages
                    ma_20 = close_prices.rolling(window=20).mean()
                    ma_50 = close_prices.rolling(window=50).mean()
                    ma_200 = close_prices.rolling(window=200).mean()
                    
                    # Price momentum
                    current_price = close_prices.iloc[-1]
                    price_1m = close_prices.iloc[-21] if len(close_prices) > 21 else close_prices.iloc[0]
                    price_3m = close_prices.iloc[-63] if len(close_prices) > 63 else close_prices.iloc[0]
                    price_6m = close_prices.iloc[-126] if len(close_prices) > 126 else close_prices.iloc[0]
                    price_12m = close_prices.iloc[-252] if len(close_prices) > 252 else close_prices.iloc[0]
                    
                    # Calculate momentum percentages
                    momentum_1m = (current_price - price_1m) / price_1m if price_1m > 0 else 0
                    momentum_3m = (current_price - price_3m) / price_3m if price_3m > 0 else 0
                    momentum_6m = (current_price - price_6m) / price_6m if price_6m > 0 else 0
                    momentum_12m = (current_price - price_12m) / price_12m if price_12m > 0 else 0
                    
                    # RSI calculation
                    rsi = self._calculate_rsi(close_prices)
                    
                    # MACD calculation
                    macd, macd_signal = self._calculate_macd(close_prices)
                    
                    # Trend direction
                    trend_direction = self._determine_trend_direction(
                        current_price, ma_20.iloc[-1], ma_50.iloc[-1], ma_200.iloc[-1]
                    )
                    
                    momentum_data[ticker] = {
                        "current_price": float(current_price),
                        "ma_20": float(ma_20.iloc[-1]) if not pd.isna(ma_20.iloc[-1]) else None,
                        "ma_50": float(ma_50.iloc[-1]) if not pd.isna(ma_50.iloc[-1]) else None,
                        "ma_200": float(ma_200.iloc[-1]) if not pd.isna(ma_200.iloc[-1]) else None,
                        "momentum_1m": float(momentum_1m),
                        "momentum_3m": float(momentum_3m),
                        "momentum_6m": float(momentum_6m),
                        "momentum_12m": float(momentum_12m),
                        "rsi": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
                        "macd": float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else None,
                        "macd_signal": float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else None,
                        "trend_direction": trend_direction
                    }
                    
                except Exception as ticker_error:
                    logger.error(f"Error getting momentum data for {ticker}: {str(ticker_error)}")
            
            return momentum_data
            
        except Exception as e:
            logger.error(f"Error getting price momentum data: {str(e)}")
            return {}
    
    async def _get_volatility_data(self, tickers: List[str], lookback_days: int) -> Dict[str, Any]:
        """Get volatility data for tickers."""
        try:
            volatility_data = {}
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            for ticker in tickers:
                try:
                    # Download price data
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    
                    if data.empty:
                        continue
                    
                    # Calculate returns
                    returns = data['Close'].pct_change().dropna()
                    
                    # Historical volatility (annualized)
                    hist_volatility = returns.std() * np.sqrt(252)
                    
                    # Rolling volatilities
                    rolling_30d = returns.rolling(window=30).std() * np.sqrt(252)
                    rolling_60d = returns.rolling(window=60).std() * np.sqrt(252)
                    rolling_90d = returns.rolling(window=90).std() * np.sqrt(252)
                    
                    # GARCH-like volatility clustering detection
                    vol_clustering = self._detect_volatility_clustering(returns)
                    
                    volatility_data[ticker] = {
                        "historical_volatility": float(hist_volatility),
                        "rolling_30d": float(rolling_30d.iloc[-1]) if not pd.isna(rolling_30d.iloc[-1]) else None,
                        "rolling_60d": float(rolling_60d.iloc[-1]) if not pd.isna(rolling_60d.iloc[-1]) else None,
                        "rolling_90d": float(rolling_90d.iloc[-1]) if not pd.isna(rolling_90d.iloc[-1]) else None,
                        "volatility_clustering": vol_clustering,
                        "volatility_regime": "high" if hist_volatility > 0.3 else "medium" if hist_volatility > 0.15 else "low"
                    }
                    
                except Exception as ticker_error:
                    logger.error(f"Error getting volatility data for {ticker}: {str(ticker_error)}")
            
            return volatility_data
            
        except Exception as e:
            logger.error(f"Error getting volatility data: {str(e)}")
            return {}
    
    def _get_asset_class_allocation(self, tickers: List[str], allocation: Dict[str, float]) -> Dict[AssetClass, float]:
        """Get allocation by asset class."""
        asset_class_allocation = {}
        
        for ticker, weight in allocation.items():
            asset_class = self.asset_class_mapping.get(ticker, AssetClass.STOCKS)
            
            if asset_class not in asset_class_allocation:
                asset_class_allocation[asset_class] = 0
            
            asset_class_allocation[asset_class] += weight
        
        return asset_class_allocation
    
    def _get_sentiment_multiplier(self, ticker: str, sentiment_data: Dict[str, Any]) -> float:
        """Get sentiment-based multiplier for a ticker."""
        try:
            overall_sentiment = sentiment_data.get("overall_sentiment", {})
            ticker_sentiment = overall_sentiment.get(ticker, 0.0)
            
            # Convert sentiment score (-1 to 1) to multiplier (0.8 to 1.2)
            # Neutral sentiment (0) = 1.0 multiplier
            # Positive sentiment (1) = 1.2 multiplier
            # Negative sentiment (-1) = 0.8 multiplier
            multiplier = 1.0 + (ticker_sentiment * 0.2)
            
            # Clamp to reasonable range
            return max(0.7, min(1.3, multiplier))
            
        except Exception as e:
            logger.error(f"Error calculating sentiment multiplier for {ticker}: {str(e)}")
            return 1.0
    
    def _calculate_volatility_forecast(self, hist_vol: float, implied_vol: Optional[float]) -> float:
        """Calculate volatility forecast combining historical and implied volatility."""
        if implied_vol is not None and implied_vol > 0:
            # Weight implied volatility higher as it's forward-looking
            return 0.3 * hist_vol + 0.7 * implied_vol
        else:
            return hist_vol
    
    def _calculate_risk_adjustment(self, vol_forecast: float, risk_tolerance: str) -> float:
        """Calculate risk adjustment based on volatility forecast and risk tolerance."""
        # Define target volatility by risk tolerance
        target_vol = {
            "low": 0.10,      # 10% target volatility
            "medium": 0.15,   # 15% target volatility
            "high": 0.25      # 25% target volatility
        }.get(risk_tolerance, 0.15)
        
        # Calculate adjustment factor
        # If asset volatility is higher than target, reduce allocation
        # If asset volatility is lower than target, could increase allocation
        if vol_forecast > 0:
            adjustment = target_vol / vol_forecast
            # Clamp adjustment to reasonable range
            return max(0.5, min(1.5, adjustment))
        else:
            return 1.0
    
    def _calculate_momentum_score(self, momentum_signals: Dict[str, Any]) -> float:
        """Calculate overall momentum score from various signals."""
        try:
            scores = []
            
            # Price momentum scores
            momentum_1m = momentum_signals.get("momentum_1m", 0)
            momentum_3m = momentum_signals.get("momentum_3m", 0)
            momentum_6m = momentum_signals.get("momentum_6m", 0)
            momentum_12m = momentum_signals.get("momentum_12m", 0)
            
            # Weight recent momentum higher
            weighted_momentum = (
                0.4 * momentum_1m +
                0.3 * momentum_3m +
                0.2 * momentum_6m +
                0.1 * momentum_12m
            )
            scores.append(weighted_momentum)
            
            # RSI score (convert to -1 to 1 scale)
            rsi = momentum_signals.get("rsi")
            if rsi is not None:
                rsi_score = (rsi - 50) / 50  # Convert 0-100 scale to -1 to 1
                scores.append(rsi_score)
            
            # MACD score
            macd = momentum_signals.get("macd", 0)
            macd_signal = momentum_signals.get("macd_signal", 0)
            if macd != 0 and macd_signal != 0:
                macd_score = 1 if macd > macd_signal else -1
                scores.append(macd_score * 0.5)  # Moderate weight
            
            # Moving average trend score
            current_price = momentum_signals.get("current_price", 0)
            ma_20 = momentum_signals.get("ma_20", 0)
            ma_50 = momentum_signals.get("ma_50", 0)
            
            if current_price > 0 and ma_20 > 0 and ma_50 > 0:
                if current_price > ma_20 > ma_50:
                    scores.append(1.0)  # Strong uptrend
                elif current_price > ma_20:
                    scores.append(0.5)  # Moderate uptrend
                elif current_price < ma_20 < ma_50:
                    scores.append(-1.0)  # Strong downtrend
                else:
                    scores.append(-0.5)  # Moderate downtrend
            
            # Average all scores
            if scores:
                return sum(scores) / len(scores)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating momentum score: {str(e)}")
            return 0.0
    
    def _get_momentum_multiplier(self, momentum_score: float) -> float:
        """Get allocation multiplier based on momentum score."""
        # Convert momentum score (-1 to 1) to multiplier (0.7 to 1.3)
        # Strong positive momentum = increase allocation
        # Strong negative momentum = decrease allocation
        multiplier = 1.0 + (momentum_score * 0.3)
        
        # Clamp to reasonable range
        return max(0.6, min(1.4, multiplier))
    
    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line."""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal
    
    def _determine_trend_direction(self, current_price: float, ma_20: float, ma_50: float, ma_200: float) -> TrendDirection:
        """Determine trend direction based on moving averages."""
        if pd.isna(ma_20) or pd.isna(ma_50) or pd.isna(ma_200):
            return TrendDirection.SIDEWAYS
        
        if current_price > ma_20 > ma_50 > ma_200:
            # All moving averages aligned upward
            pct_above_ma200 = (current_price - ma_200) / ma_200
            return TrendDirection.STRONG_UPTREND if pct_above_ma200 > 0.1 else TrendDirection.UPTREND
        elif current_price < ma_20 < ma_50 < ma_200:
            # All moving averages aligned downward
            pct_below_ma200 = (ma_200 - current_price) / ma_200
            return TrendDirection.STRONG_DOWNTREND if pct_below_ma200 > 0.1 else TrendDirection.DOWNTREND
        else:
            return TrendDirection.SIDEWAYS
    
    def _detect_volatility_clustering(self, returns: pd.Series) -> bool:
        """Detect if there's volatility clustering in returns."""
        try:
            # Calculate squared returns (proxy for volatility)
            squared_returns = returns ** 2
            
            # Test for autocorrelation in squared returns
            # If there's significant autocorrelation, volatility clustering exists
            autocorr = squared_returns.autocorr(lag=1)
            
            return autocorr > 0.1  # Threshold for volatility clustering
            
        except Exception:
            return False
    
    def _create_market_summary(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of market conditions."""
        summary = {
            "market_regime": "unknown",
            "overall_sentiment": "neutral",
            "volatility_environment": "normal",
            "momentum_environment": "mixed"
        }
        
        try:
            # Market regime
            macro_data = market_data.get("macro_indicators", {})
            regime_data = macro_data.get("market_regime", {})
            summary["market_regime"] = regime_data.get("current_regime", "unknown")
            
            # Overall sentiment
            sentiment_data = market_data.get("sentiment_data", {})
            overall_sentiment = sentiment_data.get("overall_sentiment", {})
            if overall_sentiment:
                avg_sentiment = sum(overall_sentiment.values()) / len(overall_sentiment)
                if avg_sentiment > 0.2:
                    summary["overall_sentiment"] = "positive"
                elif avg_sentiment < -0.2:
                    summary["overall_sentiment"] = "negative"
                else:
                    summary["overall_sentiment"] = "neutral"
            
            # Volatility environment
            volatility_data = market_data.get("volatility_data", {})
            if volatility_data:
                avg_vol = sum(
                    data.get("historical_volatility", 0.15) 
                    for data in volatility_data.values()
                ) / len(volatility_data)
                
                if avg_vol > 0.3:
                    summary["volatility_environment"] = "high"
                elif avg_vol > 0.2:
                    summary["volatility_environment"] = "elevated"
                else:
                    summary["volatility_environment"] = "normal"
            
            # Momentum environment
            momentum_data = market_data.get("momentum_data", {})
            if momentum_data:
                momentum_scores = []
                for ticker_data in momentum_data.values():
                    score = self._calculate_momentum_score(ticker_data)
                    momentum_scores.append(score)
                
                if momentum_scores:
                    avg_momentum = sum(momentum_scores) / len(momentum_scores)
                    if avg_momentum > 0.3:
                        summary["momentum_environment"] = "strong_positive"
                    elif avg_momentum > 0.1:
                        summary["momentum_environment"] = "positive"
                    elif avg_momentum < -0.3:
                        summary["momentum_environment"] = "strong_negative"
                    elif avg_momentum < -0.1:
                        summary["momentum_environment"] = "negative"
                    else:
                        summary["momentum_environment"] = "mixed"
            
        except Exception as e:
            logger.error(f"Error creating market summary: {str(e)}")
        
        return summary
    
    async def _generate_allocation_insights(
        self, 
        base_allocation: Dict[str, float], 
        final_allocation: Dict[str, float], 
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate insights about the allocation changes."""
        try:
            insights = {
                "allocation_changes": {},
                "key_adjustments": [],
                "risk_considerations": [],
                "market_drivers": []
            }
            
            # Calculate allocation changes
            for ticker in base_allocation.keys():
                base_weight = base_allocation.get(ticker, 0)
                final_weight = final_allocation.get(ticker, 0)
                change = final_weight - base_weight
                change_pct = (change / base_weight * 100) if base_weight > 0 else 0
                
                insights["allocation_changes"][ticker] = {
                    "base_weight": base_weight,
                    "final_weight": final_weight,
                    "absolute_change": change,
                    "percentage_change": change_pct
                }
                
                # Identify significant changes
                if abs(change_pct) > 10:  # More than 10% change
                    direction = "increased" if change > 0 else "decreased"
                    insights["key_adjustments"].append(
                        f"{ticker} allocation {direction} by {abs(change_pct):.1f}%"
                    )
            
            # Add market-driven insights
            market_summary = market_data.get("summary", {})
            market_regime = market_summary.get("market_regime", "unknown")
            sentiment = market_summary.get("overall_sentiment", "neutral")
            volatility = market_summary.get("volatility_environment", "normal")
            
            insights["market_drivers"].append(f"Market regime: {market_regime}")
            insights["market_drivers"].append(f"Sentiment: {sentiment}")
            insights["market_drivers"].append(f"Volatility: {volatility}")
            
            # Add risk considerations
            if volatility in ["high", "elevated"]:
                insights["risk_considerations"].append(
                    "High volatility environment detected - allocations adjusted for risk management"
                )
            
            if sentiment == "negative":
                insights["risk_considerations"].append(
                    "Negative market sentiment - defensive positioning increased"
                )
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating allocation insights: {str(e)}")
            return {"error": str(e)}
