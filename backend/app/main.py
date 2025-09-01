from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple, Any
import os
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns, objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, AssetClass
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from app.services.ai_advisor_service import AIAdvisorService
import cvxpy as cp
import time
import math
import json
import logging
import random
import requests
import pandas_datareader as pdr
from pandas_datareader import data as web
from app.services.crypto_service import CryptoService
from logging.config import dictConfig
from app.middleware.security import RateLimitMiddleware, APIKeyMiddleware
from app.utils.trading import get_trading_credentials, get_account_data

# Configure logging
log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {"handlers": ["default"], "level": "INFO"},
}

dictConfig(log_config)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate required environment variables
required_env_vars = [
    "ALPACA_PAPER_API_KEY",
    "ALPACA_PAPER_SECRET_KEY",
    "TRADING_MODE",
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize FastAPI app
app = FastAPI(
    title="SmartPortfolio API",
    description="API for portfolio management and trading",
    version="1.0.0",
)

# Add security middlewares
rate_limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
app.add_middleware(RateLimitMiddleware, requests_per_minute=rate_limit)
app.add_middleware(APIKeyMiddleware)

# CORS configuration
origins = os.getenv("CORS_ORIGINS", "").split(",")
if not origins or origins == [""]:
    # Default CORS configuration for development
    origins = [
        "http://localhost:5173",
        "http://localhost:3000",
        "https://organica-ai-solutions.github.io",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthCheck(BaseModel):
    """Response model for health check endpoint."""
    status: str
    version: str
    environment: str

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": os.getenv("ENV", "development")
    }

def download_market_data(unique_tickers, start_date, end_date, max_retries=3):
    """Download market data with retries."""
    yahoo_data = pd.DataFrame()
    retry_delay = 1
    yahoo_success = False
    data_source = "yahoo_finance"  # Initialize data source
    
    for attempt in range(max_retries):
        try:
            print(f"Downloading market data from Yahoo Finance (attempt {attempt+1}/{max_retries})...")
            try:
                yahoo_data = yf.download(unique_tickers, start=start_date, end=end_date, progress=False)["Adj Close"]
            except KeyError as e:
                if str(e) == "'Adj Close'":
                    print("Adj Close not available, falling back to Close prices")
                    yahoo_data = yf.download(unique_tickers, start=start_date, end=end_date, progress=False)["Close"]
                else:
                    raise e
            
            if yahoo_data.empty:
                print("Batch download failed, trying individual downloads...")
                yahoo_data = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
                
                for ticker in unique_tickers:
                    try:
                        ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                        if not ticker_data.empty:
                            if "Adj Close" in ticker_data.columns:
                                yahoo_data[ticker] = ticker_data["Adj Close"]
                            else:
                                yahoo_data[ticker] = ticker_data["Close"]
                            print(f"Downloaded data for {ticker}")
                        else:
                            print(f"No data available for {ticker}")
                    except Exception as ticker_e:
                        print(f"Error downloading {ticker}: {str(ticker_e)}")
            
            if not yahoo_data.empty and len(yahoo_data.columns) > 0:
                yahoo_success = True
                return yahoo_data, data_source
            
            if attempt < max_retries - 1:
                print(f"No data downloaded. Retrying {attempt + 1}/{max_retries}...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("All Yahoo Finance attempts failed")
                
        except Exception as e:
            print(f"Error downloading data from Yahoo Finance: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying {attempt + 1}/{max_retries}...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("All Yahoo Finance attempts failed")
    
    return yahoo_data, data_source

@app.get("/readiness")
async def readiness_check():
    """Readiness check endpoint that verifies all external services."""
    services_status = {
        "alpaca": "unknown",
        "database": "not_configured",
        "api": "healthy"
    }
    
    try:
        # Get trading credentials with proper headers
        creds = get_trading_credentials()
        
        # Test Alpaca connection using the SDK
        try:
            # Initialize trading client with paper trading enabled
            trading_client = TradingClient(
                api_key=creds["api_key"],
                secret_key=creds["secret_key"],
                paper=creds["is_paper"],
                url_override=creds["url_override"]
            )
            
            # Try to get account info
            account = trading_client.get_account()
            if account:
                services_status["alpaca"] = "connected"
            else:
                services_status["alpaca"] = "error: no account data received"
                raise HTTPException(
                    status_code=503,
                    detail={
                        "status": "not_ready",
                        "services": services_status,
                        "message": "Alpaca API connection failed: no account data received"
                    }
                )
        except Exception as e:
            services_status["alpaca"] = f"error: {str(e)}"
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "services": services_status,
                    "message": f"Alpaca API connection failed: {str(e)}"
                }
            )
        
        # All checks passed
        return {
            "status": "ready",
            "services": services_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in readiness check: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "services": services_status,
                "message": f"Error in readiness check: {str(e)}"
            }
        )

@app.get("/account")
async def get_account():
    """Get account information."""
    try:
        # Get trading credentials based on mode
        creds = get_trading_credentials()
        logger.info(f"Using trading mode: {'paper' if creds['is_paper'] else 'live'}")
        
        if not creds["api_key"] or not creds["secret_key"]:
            logger.error("Missing API credentials")
            raise HTTPException(
                status_code=500,
                detail="Trading credentials not properly configured"
            )
        
        # Initialize trading client
        try:
            trading_client = TradingClient(
                api_key=creds["api_key"],
                secret_key=creds["secret_key"],
                paper=creds["is_paper"],
                url_override=creds["url_override"]
            )
        except Exception as client_error:
            logger.error(f"Failed to initialize trading client: {str(client_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize trading client: {str(client_error)}"
            )
        
        # Get account data with retries
        try:
            account_data = get_account_data(trading_client)
        except HTTPException as he:
            logger.error(f"HTTP error getting account data: {str(he)}")
            raise he
        except Exception as e:
            logger.error(f"Error getting account data: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error retrieving account data: {str(e)}"
            )
        
        # Add trading mode and environment info to response
        account_data.update({
            "trading_mode": "paper" if creds["is_paper"] else "live",
            "environment": os.getenv("ENV", "development"),
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info("Successfully retrieved account data")
        return account_data
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in get_account: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error getting account information: {str(e)}"
        )

# Initialize services
crypto_service = CryptoService()


@app.get("/env-check")
async def env_check():
    """Check environment variables (returns masked values for security)."""
    from backend.env_check import check_env_vars

    # Only accessible in development mode or with proper authorization
    if os.getenv("ENV", "development") == "development":
        return check_env_vars()
    else:
        return {"status": "Only available in development mode"}


class Portfolio(BaseModel):
    id: Optional[int] = None
    user_id: Optional[int] = None
    name: Optional[str] = None
    tickers: List[str]
    start_date: str
    risk_tolerance: str = "medium"  # 'low', 'medium', 'high'


class PortfolioAllocation(BaseModel):
    allocations: Dict[str, float]
    api_key: str
    secret_key: str
    use_paper: Optional[bool] = True
    investment_amount: Optional[float] = None
    current_positions: Optional[Dict[str, float]] = None
    target_positions: Optional[Dict[str, float]] = None
    rebalance_threshold: Optional[float] = 0.05


class TickerPreferences(BaseModel):
    risk_tolerance: str = "medium"
    investment_horizon: str = "long_term"
    sectors: Optional[List[str]] = None
    market_cap: Optional[str] = None
    risk_level: Optional[str] = None
    investment_style: Optional[str] = None


class RebalanceRequest(BaseModel):
    allocations: Dict[str, float]
    investment_amount: float
    current_positions: Dict[str, float]
    target_positions: Dict[str, float]
    rebalance_threshold: float
    api_key: str
    secret_key: str
    paper: bool = True  # Default to paper trading


def calculate_portfolio_metrics(data: pd.DataFrame, weights: Dict[str, float]) -> Dict:
    """Calculate additional portfolio metrics."""
    try:
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Convert weights dict to array
        weight_arr = np.array([weights[ticker] for ticker in data.columns])
        
        # Get market data aligned with our date range
        try:
            market_data = yf.download("^GSPC", start=data.index[0], end=data.index[-1])
            if "Adj Close" in market_data.columns:
                market_series = market_data["Adj Close"]
            else:
                market_series = market_data["Close"]
            market_returns = market_series.pct_change()
        except Exception as e:
            print(
                f"Error getting market data: {str(e)}. Using first asset as market proxy."
            )
            # Use the first asset as a proxy for the market
            market_returns = returns.iloc[:, 0]
        
        # Align market returns with asset returns
        aligned_data = pd.concat([returns, market_returns], axis=1).dropna()
        asset_returns = aligned_data[returns.columns]
        market_returns = aligned_data.iloc[:, -1]  # Last column is market returns
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(asset_returns.mean() * weight_arr) * 252
        portfolio_vol = np.sqrt(
            np.dot(weight_arr.T, np.dot(asset_returns.cov() * 252, weight_arr))
        )
        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        # Handle potential NaN or infinite values
        if np.isnan(portfolio_return) or np.isinf(portfolio_return):
            portfolio_return = 0.0
        if np.isnan(portfolio_vol) or np.isinf(portfolio_vol):
            portfolio_vol = 0.0
        if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
            sharpe_ratio = 0.0
        
        # Calculate individual asset metrics
        asset_returns_annual = asset_returns.mean() * 252
        asset_vols = asset_returns.std() * np.sqrt(252)
        
        # Calculate betas
        betas = {}
        for ticker in data.columns:
            covariance = np.cov(asset_returns[ticker], market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance > 0 else 0
            # Handle potential NaN or infinite values
            if np.isnan(beta) or np.isinf(beta):
                beta = 0.0
            betas[ticker] = beta
        
        # Calculate portfolio returns for VaR and CVaR
        portfolio_returns = asset_returns.dot(weight_arr)
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Handle potential NaN or infinite values
        if np.isnan(var_95) or np.isinf(var_95):
            var_95 = 0.0
        if np.isnan(cvar_95) or np.isinf(cvar_95):
            cvar_95 = 0.0
        
        # Create result dictionary with safe values
        result = {
            "portfolio_metrics": {
                "expected_annual_return": float(portfolio_return),
                "annual_volatility": float(portfolio_vol),
                "sharpe_ratio": float(sharpe_ratio),
                "value_at_risk_95": float(var_95),
                "conditional_var_95": float(cvar_95),
            },
            "asset_metrics": {},
        }
        
        # Add asset metrics with safe values
        for ticker in data.columns:
            annual_return = float(asset_returns_annual[ticker])
            annual_vol = float(asset_vols[ticker])
            beta = float(betas[ticker])
            
            # Handle potential NaN or infinite values
            if np.isnan(annual_return) or np.isinf(annual_return):
                annual_return = 0.0
            if np.isnan(annual_vol) or np.isinf(annual_vol):
                annual_vol = 0.0
            
            result["asset_metrics"][ticker] = {
                "annual_return": annual_return,
                "annual_volatility": annual_vol,
                "beta": beta,
                "weight": float(weights[ticker]),
            }
        
        return result
    except Exception as e:
        print(f"Error in calculate_portfolio_metrics: {str(e)}")
        # Return safe default values
        return {
            "portfolio_metrics": {
                "expected_annual_return": 0.0,
                "annual_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "value_at_risk_95": 0.0,
                "conditional_var_95": 0.0,
            },
            "asset_metrics": {
                ticker: {
                    "annual_return": 0.0,
                    "annual_volatility": 0.0,
                    "beta": 0.0,
                    "weight": float(weights.get(ticker, 0.0)),
            }
                for ticker in data.columns
            },
        }


def is_crypto(ticker: str) -> bool:
    """Check if the ticker is a cryptocurrency."""
    return ticker.endswith("-USD") or ticker.endswith("USDT")


def get_asset_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.Series:
    """Get asset data with special handling for crypto assets."""
    try:
        # For crypto assets, we don't need to adjust for market hours
        if is_crypto(ticker):
            data = yf.download(ticker, start=start_date, end=end_date)["Adj Close"]
        else:
            # For stocks, we need to ensure we're only looking at market hours
            data = yf.download(ticker, start=start_date, end=end_date)["Adj Close"]
            # Filter out weekend data for stocks
            data = data[data.index.dayofweek < 5]
            # Filter out data outside of market hours (9:30 AM - 4:00 PM EST)
            data = data[
                (data.index.time >= pd.Timestamp("09:30").time())
                & (data.index.time <= pd.Timestamp("16:00").time())
            ]
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        return data
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error downloading data for {ticker}: {str(e)}"
        )


def optimize_portfolio(
    mu: np.ndarray, S: np.ndarray, risk_tolerance: str, tickers: List[str]
) -> dict:
    """Optimize portfolio based on risk tolerance."""
    try:
        # Create efficient frontier object
        ef = EfficientFrontier(mu, S)
        
        # Optimize based on risk tolerance
        if risk_tolerance == "low":
            ef.min_volatility()
        elif risk_tolerance == "high":
            ef.max_sharpe()
        else:  # medium
            # Use a balanced approach
            # Get the minimum volatility as a reference
            min_vol_ef = EfficientFrontier(mu, S)
            min_vol_ef.min_volatility()
            min_vol = min_vol_ef.portfolio_performance()[1]
            
            # Get the max sharpe ratio volatility as a reference
            max_sharpe_ef = EfficientFrontier(mu, S)
            max_sharpe_ef.max_sharpe()
            max_sharpe_vol = max_sharpe_ef.portfolio_performance()[1]
            
            # Target a volatility between min and max
            target = min_vol + (max_sharpe_vol - min_vol) * 0.5
            ef = EfficientFrontier(mu, S)  # Create a new EF object
            ef.efficient_risk(target)  # Use target without the keyword
        
        # Get optimized weights
        weights = ef.clean_weights()
        
        # Calculate performance metrics
        perf = ef.portfolio_performance()
        expected_return, volatility, sharpe_ratio = perf
        
        # Create a simple discrete allocation
        discrete_allocation = {"shares": {}, "leftover": 0.0}
        
        # Try to calculate actual discrete allocation if possible
        try:
            latest_prices = {}
            for ticker in tickers:
                try:
                    ticker_data = yf.Ticker(ticker)
                    latest_prices[ticker] = ticker_data.history(period="1d")[
                        "Close"
                    ].iloc[-1]
                except Exception as price_error:
                    print(f"Error getting price for {ticker}: {str(price_error)}")
                    latest_prices[ticker] = 100.0  # Placeholder value
            
            # Calculate discrete allocation with $10,000 portfolio value
            if latest_prices:
                da = DiscreteAllocation(
                    weights, pd.Series(latest_prices), total_portfolio_value=10000
                )
                allocation, leftover = da.greedy_portfolio()
                discrete_allocation = {"shares": allocation, "leftover": leftover}
        except Exception as alloc_error:
            print(f"Error calculating discrete allocation: {str(alloc_error)}")
            # Keep the default empty discrete allocation
            
            return {
            "weights": weights,
                "metrics": {
                    "expected_return": float(expected_return),
                "volatility": float(volatility),
                    "sharpe_ratio": float(sharpe_ratio),
            },
                "discrete_allocation": discrete_allocation,
        }
    except Exception as e:
        print(f"Error in optimize_portfolio: {str(e)}")
        # Return equal weights as fallback
        equal_weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
        return {
            "weights": equal_weights,
            "metrics": {"expected_return": 0.0, "volatility": 0.0, "sharpe_ratio": 0.0},
            "discrete_allocation": {"shares": {}, "leftover": 10000.0},
        }


def get_risk_adjusted_equal_weights(
    tickers: List[str], mu: np.ndarray, S: np.ndarray
) -> Dict[str, float]:
    """Calculate risk-adjusted equal weights based on asset volatilities."""
    try:
        # Calculate individual asset volatilities
        vols = np.sqrt(np.diag(S))
        # Inverse volatility weighting
        inv_vols = 1 / vols
        weights = inv_vols / np.sum(inv_vols)
        return dict(zip(tickers, weights))
    except Exception:
        # Fallback to simple equal weights
        return {ticker: 1.0 / len(tickers) for ticker in tickers}


def calculate_portfolio_beta(
    mu: np.ndarray, S: np.ndarray, weights: Dict[str, float]
) -> float:
    """Calculate portfolio beta using CAPM."""
    try:
        weights_array = np.array(list(weights.values()))
        portfolio_var = np.sqrt(weights_array.T @ S @ weights_array)
        market_var = np.sqrt(S[0, 0])  # Assuming first asset is market proxy
        return portfolio_var / market_var if market_var > 0 else 1.0
    except Exception:
        return 1.0


def get_sector_constraints(
    tickers: List[str],
) -> Dict[str, Tuple[List[int], Tuple[float, float]]]:
    """Get sector constraints for portfolio optimization."""
    try:
        # Define basic sector classifications
        tech_tickers = ["AAPL", "GOOGL", "MSFT", "META", "NVDA"]
        finance_tickers = ["JPM", "BAC", "GS", "MS", "V"]
        crypto_tickers = [t for t in tickers if t.endswith("-USD")]
        
        constraints = {}
        
        # Add tech sector constraints
        tech_indices = [i for i, t in enumerate(tickers) if t in tech_tickers]
        if tech_indices:
            constraints["tech"] = (tech_indices, (0.0, 0.4))  # Max 40% in tech
            
        # Add finance sector constraints
        finance_indices = [i for i, t in enumerate(tickers) if t in finance_tickers]
        if finance_indices:
            constraints["finance"] = (
                finance_indices,
                (0.0, 0.35),
            )  # Max 35% in finance
            
        # Add crypto constraints
        crypto_indices = [i for i, t in enumerate(tickers) if t in crypto_tickers]
        if crypto_indices:
            constraints["crypto"] = (crypto_indices, (0.0, 0.30))  # Max 30% in crypto
            
        return constraints
    except Exception as e:
        print(f"Error creating sector constraints: {str(e)}")
        return {}


def create_portfolio_result(
    weights: np.ndarray, mu: np.ndarray, S: np.ndarray, tickers: List[str]
) -> dict:
    """Helper function to create portfolio result dictionary."""
    port_return = float(mu.T @ weights)
    port_risk = float(np.sqrt(weights.T @ S @ weights))
    sharpe = port_return / port_risk if port_risk > 0 else 0
    
    # Convert weights to dictionary
    weights_dict = {ticker: float(weight) for ticker, weight in zip(tickers, weights)}
    
    return {
        "weights": weights_dict,
        "metrics": {
            "expected_return": port_return,
            "volatility": port_risk,
            "sharpe_ratio": sharpe,
        },
    }


def get_historical_performance(data: pd.DataFrame, weights: Dict[str, float]) -> Dict:
    """Calculate historical performance metrics for the portfolio."""
    try:
        # Ensure we have data to work with
        if data.empty or len(data.columns) == 0:
            return {
                "dates": [],
                "portfolio_values": [],
                "drawdowns": [],
                "rolling_volatility": [],
                "rolling_sharpe": [],
            }
            
        # Calculate daily returns
        returns = data.pct_change().dropna()
        
        # If we don't have enough data points, return empty results
        if len(returns) < 5:  # Need at least 5 days for meaningful analysis
            return {
                "dates": [],
                "portfolio_values": [],
                "drawdowns": [],
                "rolling_volatility": [],
                "rolling_sharpe": [],
            }
        
        # Convert weights dict to array matching the order of columns in returns
        weight_arr = np.array([weights.get(ticker, 0) for ticker in returns.columns])
        
        # Normalize weights to sum to 1
        if np.sum(weight_arr) > 0:
            weight_arr = weight_arr / np.sum(weight_arr)
        
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weight_arr)
        
        # Calculate cumulative returns (starting with $10,000)
        initial_investment = 10000
        portfolio_values = initial_investment * (1 + portfolio_returns).cumprod()
        
        # Calculate drawdowns
        peak = portfolio_values.cummax()
        drawdowns = (portfolio_values - peak) / peak
        
        # Calculate rolling metrics (30-day window)
        window = min(30, len(portfolio_returns))
        rolling_vol = portfolio_returns.rolling(window=window).std() * np.sqrt(252)
        risk_free_rate = 0.02 / 252  # Assuming 2% annual risk-free rate
        rolling_sharpe = (
            (portfolio_returns.rolling(window=window).mean() - risk_free_rate)
            / portfolio_returns.rolling(window=window).std()
            * np.sqrt(252)
        )
        
        # Replace NaN and infinite values
        portfolio_values = portfolio_values.replace([np.inf, -np.inf], np.nan).fillna(
            initial_investment
        )
        drawdowns = drawdowns.replace([np.inf, -np.inf], np.nan).fillna(0)
        rolling_vol = rolling_vol.replace([np.inf, -np.inf], np.nan).fillna(0)
        rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Convert to lists for JSON serialization
        dates = returns.index.strftime("%Y-%m-%d").tolist()
        portfolio_values_list = portfolio_values.tolist()
        drawdowns_list = drawdowns.tolist()
        rolling_vol_list = rolling_vol.tolist()
        rolling_sharpe_list = rolling_sharpe.tolist()
        
        # Final check for any remaining non-finite values
        portfolio_values_list = [
            0 if not np.isfinite(x) else x for x in portfolio_values_list
        ]
        drawdowns_list = [0 if not np.isfinite(x) else x for x in drawdowns_list]
        rolling_vol_list = [0 if not np.isfinite(x) else x for x in rolling_vol_list]
        rolling_sharpe_list = [
            0 if not np.isfinite(x) else x for x in rolling_sharpe_list
        ]
        
        return {
            "dates": dates,
            "portfolio_values": portfolio_values_list,
            "drawdowns": drawdowns_list,
            "rolling_volatility": rolling_vol_list,
            "rolling_sharpe": rolling_sharpe_list,
        }
    except Exception as e:
        print(f"Error calculating historical performance: {str(e)}")
        # Return empty data with proper structure
        return {
            "dates": [],
            "portfolio_values": [],
            "drawdowns": [],
            "rolling_volatility": [],
            "rolling_sharpe": [],
        }


def clean_numeric_values(obj):
    """Clean numeric values to ensure they are JSON serializable."""
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, dict):
        return {k: clean_numeric_values(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_numeric_values(x) for x in obj]
    return obj


@app.post("/analyze-portfolio")
async def analyze_portfolio(request: Portfolio):
    """Analyze a portfolio using historical data and optimization techniques."""
    try:
        print(f"Analyzing portfolio for tickers: {request.tickers}")
        
        # Initialize variables
        yahoo_success = False
        data_source = "multiple_sources"
        combined_data = pd.DataFrame()
        
        # Get unique tickers (handle potential duplicates)
        unique_tickers = list(set(request.tickers))
        
        # Set date range
            end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        
        # Retry parameters
        max_retries = 3
        retry_delay = 1  # Initial delay in seconds
        
        # Try to download data from Yahoo Finance with retries
        for attempt in range(max_retries):
            try:
                print(f"Attempting Yahoo Finance download (attempt {attempt + 1}/{max_retries})")
                yahoo_data = yf.download(
                    unique_tickers,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    group_by="ticker",
                )
                
                if isinstance(yahoo_data, pd.DataFrame) and not yahoo_data.empty:
                    # Single ticker case
                    if "Adj Close" in yahoo_data.columns:
                        yahoo_data = yahoo_data["Adj Close"]
        else:
                        yahoo_data = yahoo_data["Close"]
                    yahoo_data = pd.DataFrame(yahoo_data)  # Convert Series to DataFrame
                    yahoo_success = True
                    break
                elif isinstance(yahoo_data, pd.DataFrame) and yahoo_data.empty:
                    print("Batch download failed, trying individual downloads...")
                    yahoo_data = pd.DataFrame(
                        index=pd.date_range(start=start_date, end=end_date)
                    )
                    
                    for ticker in unique_tickers:
                        try:
                            ticker_data = yf.download(
                                ticker, start=start_date, end=end_date, progress=False
                            )
                            if not ticker_data.empty:
                                if "Adj Close" in ticker_data.columns:
                                    yahoo_data[ticker] = ticker_data["Adj Close"]
                                else:
                                    yahoo_data[ticker] = ticker_data["Close"]
                                print(f"Downloaded data for {ticker}")
                            else:
                                print(f"No data available for {ticker}")
                        except Exception as ticker_e:
                            print(f"Error downloading {ticker}: {str(ticker_e)}")
                    
                    if not yahoo_data.empty and len(yahoo_data.columns) > 0:
                        yahoo_success = True
                        data_source = "yahoo_finance"
                        break
                
                if attempt < max_retries - 1:
                    print(f"No data downloaded. Retrying {attempt + 1}/{max_retries}...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("All Yahoo Finance attempts failed")
                    
            except Exception as e:
                print(f"Error downloading data from Yahoo Finance: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying {attempt + 1}/{max_retries}...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("All Yahoo Finance attempts failed")
        
        if yahoo_success and not yahoo_data.empty:
            if combined_data.empty:
                combined_data = yahoo_data
            else:
                combined_data = combined_data.join(yahoo_data, how="outer")
        
        # If we still don't have data, try alternative sources
        if combined_data.empty or len(combined_data.columns) == 0:
            print("Yahoo Finance failed to provide data, trying alternative sources")
            # Try using the alternative data sources defined elsewhere
            alt_data = get_alternative_stock_data(unique_tickers, start_date, end_date)
            
            if not alt_data.empty and len(alt_data.columns) > 0:
                combined_data = alt_data
                data_source = "alternative_sources"
                print("Successfully got data from alternative sources")
            else:
                print("Alternative sources also failed, falling back to simple analysis")
                return await analyze_portfolio_simple(request)
        
        # Check if we have any data to work with
        if combined_data.empty or len(combined_data.columns) == 0:
            print("No data available after all attempts, falling back to simple analysis")
            return await analyze_portfolio_simple(request)
        
        # Map back to original tickers from expanded list
        original_ticker_map = {}
        for original in request.tickers:
            # Find the first expanded ticker that has data
            found = False
            
            # Direct match first
            if original in combined_data.columns:
                original_ticker_map[original] = original
                found = True
            
            # Then look for prefix matches
            if not found:
                for column in combined_data.columns:
                    # More flexible matching - match by prefix or full ticker
                    if column.startswith(original.split("-")[0]) or column == original:
                        original_ticker_map[original] = column
                        found = True
                        break
        
        # If we're missing any original tickers, fall back to the simple endpoint
        if len(original_ticker_map) != len(request.tickers):
            print("Missing data for some tickers, falling back to simple analysis")
            return await analyze_portfolio_simple(request)
        
        # Create data with original ticker names
        original_data = pd.DataFrame(index=combined_data.index)
        for original, column in original_ticker_map.items():
            original_data[original] = combined_data[column]
        
        # Fill missing values
        original_data = original_data.fillna(method="ffill").fillna(method="bfill")
        
        # Add metadata
        response_metadata = {
            "data_source": data_source,
            "is_real_data": True,
            "tickers_used": original_ticker_map,
            "data_points": len(original_data),
            "date_range": {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
            },
        }
        
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(original_data)
        S = risk_models.sample_cov(original_data)
        
        # Optimize portfolio based on risk tolerance
        result = optimize_portfolio(
            mu, S, request.risk_tolerance, list(original_data.columns)
        )
        
        # Calculate additional portfolio metrics
        metrics = calculate_portfolio_metrics(original_data, result["weights"])
        
        # Get historical performance
        historical_performance = get_historical_performance(
            original_data, result["weights"]
        )
        
        # Combine results
        response = {
            "allocations": result["weights"],
            "metrics": metrics["portfolio_metrics"],
            "asset_metrics": metrics["asset_metrics"],
            "discrete_allocation": result["discrete_allocation"],
            "historical_performance": historical_performance,
            "metadata": response_metadata,
        }
        
        # Clean numeric values to ensure JSON serialization works
        response = clean_numeric_values(response)
        
        return response
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in analyze_portfolio: {str(e)}")
        # Fall back to simple analysis on any error
        try:
            print("Falling back to simple analysis due to error")
            return await analyze_portfolio_simple(request)
        except Exception as simple_e:
            print(f"Simple analysis also failed: {str(simple_e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze-portfolio-simple")
async def analyze_portfolio_simple(request: Portfolio):
    """Simplified portfolio analysis endpoint for testing."""
    try:
        print(f"Analyzing portfolio (simple) for tickers: {request.tickers}")
        
        # Helper function to generate realistic mock metrics based on ticker
        def generate_ticker_metrics(ticker):
            # Generate pseudo-random but consistent values based on ticker name
            ticker_sum = sum(ord(c) for c in ticker)
            seed = ticker_sum / 1000
            
            # Vary returns between 5% and 25% based on ticker
            return_factor = 0.05 + (seed % 0.20)
            # Vary volatility between 10% and 40% based on ticker
            volatility_factor = 0.10 + (seed % 0.30)
            # Vary beta between 0.6 and 1.8 based on ticker
            beta_factor = 0.6 + (seed % 1.2)
            # Vary alpha between -3% and 5% based on ticker
            alpha_factor = -0.03 + (seed % 0.08)
            # Vary max drawdown between -10% and -40% based on ticker
            max_drawdown = -0.10 - (seed % 0.30)
            
            return {
                "annual_return": return_factor,
                "annual_volatility": volatility_factor,
                "beta": beta_factor,
                "alpha": alpha_factor,
                "volatility": volatility_factor,
                "var_95": -volatility_factor * 1.65 / math.sqrt(252),  # Simplified VaR calculation
                "max_drawdown": max_drawdown,
                "correlation": 0.3 + (seed % 0.5),  # Correlation between 0.3 and 0.8
            }
        
        # Generate mock metrics for each ticker
        asset_metrics = {ticker: generate_ticker_metrics(ticker) for ticker in request.tickers}
        
        # Generate mock portfolio weights based on risk tolerance
        total_weight = 1.0
                weights = []
        remaining_tickers = len(request.tickers)
        
        for ticker in request.tickers[:-1]:
            # Allocate a portion of the remaining weight
            weight = total_weight * (1 - request.risk_tolerance) / remaining_tickers
                    weights.append(weight)
            total_weight -= weight
            remaining_tickers -= 1
        
        # Allocate remaining weight to last ticker
        weights.append(total_weight)
        
        # Create allocations dictionary
        allocations = dict(zip(request.tickers, weights))
        
        # Calculate portfolio metrics
        portfolio_return = sum(
            allocations[ticker] * asset_metrics[ticker]["annual_return"]
            for ticker in request.tickers
        )
        
        portfolio_volatility = sum(
            allocations[ticker] * asset_metrics[ticker]["volatility"]
            for ticker in request.tickers
        )
        
        # Generate mock historical performance data
        dates = pd.date_range(end=datetime.now(), periods=252, freq="B")  # Business days for a year
        performance_data = []
        cumulative_return = 1.0
        
        for date in dates:
            # Generate daily return based on portfolio characteristics
            daily_return = random.gauss(
                portfolio_return / 252,  # Daily mean return
                portfolio_volatility / math.sqrt(252)  # Daily volatility
            )
            cumulative_return *= (1 + daily_return)
            
            performance_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": cumulative_return
            })
        
        # Calculate discrete allocation for $100,000 portfolio
        portfolio_value = 100000
        discrete_allocation = {}
        remaining_cash = portfolio_value
        
        for ticker in request.tickers:
            allocation = allocations[ticker]
            ticker_value = portfolio_value * allocation
            discrete_allocation[ticker] = round(ticker_value, 2)
            remaining_cash -= discrete_allocation[ticker]
        
        # Add any remaining cash due to rounding
        if remaining_cash > 0:
            discrete_allocation["cash"] = round(remaining_cash, 2)
        
        response = {
            "allocations": allocations,
            "metrics": {
                "expected_annual_return": portfolio_return,
                "annual_volatility": portfolio_volatility,
                "sharpe_ratio": (portfolio_return - 0.02) / portfolio_volatility,  # Assuming 2% risk-free rate
                "max_drawdown": min(asset_metrics[ticker]["max_drawdown"] for ticker in request.tickers),
            },
            "asset_metrics": asset_metrics,
            "discrete_allocation": discrete_allocation,
            "historical_performance": performance_data,
            "metadata": {
                "data_source": "simulated",
                "is_real_data": False,
                "simulation_parameters": {
                    "timeframe": "1Y",
                    "data_points": len(performance_data),
                    "risk_free_rate": 0.02,
                }
            }
        }
        
        return response
        
    except Exception as e:
        print(f"Error in analyze_portfolio_simple: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze portfolio: {str(e)}"
        )


@app.post("/rebalance-portfolio")
async def rebalance_portfolio(allocation: RebalanceRequest):
    """Rebalance portfolio based on target allocations."""
    try:
        # Initialize trading client
        trading_client = TradingClient(
            api_key=allocation.api_key,
            secret_key=allocation.secret_key,
            paper=allocation.paper
        )
        
        # Get account data
        account_data = get_account_data(trading_client)
        
        # Extract account information
        buying_power = float(account_data["buying_power"])
        equity = float(account_data["equity"])
        
        # Validate buying power
        if buying_power < 1:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient buying power (${buying_power})"
            )
            
        # Separate tickers into stocks and crypto
        crypto_tickers = [ticker for ticker in allocation.allocations.keys() if "-USD" in ticker or "-USDT" in ticker]
        stock_tickers = [ticker for ticker in allocation.allocations.keys() if ticker not in crypto_tickers]
        
        logger.info(f"Processing {len(stock_tickers)} stocks and {len(crypto_tickers)} crypto assets")

        # Get current positions
        positions = trading_client.get_all_positions()
        current_positions = {p.symbol: float(p.market_value) for p in positions}

        # Calculate target positions
        target_positions = {
            symbol: equity * weight 
            for symbol, weight in allocation.allocations.items()
        }
        
        trades = []
        
        # Process stock orders
        for symbol in stock_tickers:
            try:
                current_value = current_positions.get(symbol, 0)
                target_value = target_positions[symbol]
                difference = target_value - current_value
                
                if abs(difference) > (allocation.rebalance_threshold or 0.05) * target_value:
                    # Get current price using StockHistoricalDataClient
                    request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                    latest_quote = trading_client.get_stock_latest_quote(request_params)
                    price = float(latest_quote[symbol].ask_price)  # Use ask price for buying
                    
                    # Calculate fractional shares (round to 6 decimal places for precision)
                    shares = round(abs(difference / price), 6)
                    if shares > 0:
                        side = OrderSide.BUY if difference > 0 else OrderSide.SELL
                        
                        # Use notional for fractional shares
                    order_data = MarketOrderRequest(
                        symbol=symbol,
                            notional=abs(difference),  # Use dollar amount instead of shares
                        side=side,
                        time_in_force=TimeInForce.DAY
                    )
                    
                        order = trading_client.submit_order(order_data)
                        trades.append({
                            "symbol": symbol,
                            "notional": abs(difference),
                            "estimated_shares": shares,
                            "side": side.value,
                            "type": "stock",
                            "order_id": order.id,
                            "status": order.status
                        })
                        logger.info(f"Placed {side.value} order for ${abs(difference):.2f} of {symbol} ({shares:.6f} shares)")
                    except Exception as e:
                logger.error(f"Error processing stock order for {symbol}: {str(e)}")
                trades.append({
                            "symbol": symbol,
                    "error": str(e),
                    "type": "stock",
                    "status": "failed"
                })
        
        # Process crypto orders
        for symbol in crypto_tickers:
            try:
                current_value = current_positions.get(symbol, 0)
                target_value = target_positions[symbol]
                difference = target_value - current_value
                
                if abs(difference) > (allocation.rebalance_threshold or 0.05) * target_value:
                    # Get current crypto price
                    request = CryptoBarsRequest(
                        symbol_or_symbols=symbol,
                        timeframe=TimeFrame.Minute,
                        start=datetime.now() - timedelta(minutes=5),
                        end=datetime.now()
                    )
                    bars = crypto_client.get_crypto_bars(request)
                    price = float(bars[symbol][-1].close)
                    
                    # Calculate quantity (round to 8 decimal places for crypto)
                    qty = round(abs(difference / price), 8)
                    if qty > 0:
                        side = OrderSide.BUY if difference > 0 else OrderSide.SELL
                        
                        # Use notional for crypto as well
                        order_data = MarketOrderRequest(
                            symbol=symbol,
                            notional=abs(difference),  # Use dollar amount instead of quantity
                            side=side,
                            time_in_force=TimeInForce.DAY
                        )
                        
                        order = trading_client.submit_order(order_data)
                        trades.append({
                            "symbol": symbol,
                            "notional": abs(difference),
                            "estimated_quantity": qty,
                            "side": side.value,
                            "type": "crypto",
                            "order_id": order.id,
                            "status": order.status
                        })
                        logger.info(f"Placed {side.value} order for ${abs(difference):.2f} of {symbol} ({qty:.8f} units)")
    except Exception as e:
                logger.error(f"Error processing crypto order for {symbol}: {str(e)}")
                trades.append({
                    "symbol": symbol,
                    "error": str(e),
                    "type": "crypto",
                    "status": "failed"
                })
        
        return {
            "status": "success",
            "trades": trades,
            "account": {
                "equity": equity,
                "buying_power": buying_power
            }
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error rebalancing portfolio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rebalance portfolio: {str(e)}"
        )


# Import the new services
from app.services.dynamic_allocation_service import DynamicAllocationService
from app.services.advanced_optimization_service import AdvancedOptimizationService, OptimizationMethod, FactorType
from app.services.risk_management_service import RiskManagementService, RiskRegime, DrawdownSeverity
from app.services.liquidity_management_service import LiquidityManagementService, MarketCondition, RebalanceFrequency
from app.services.tax_efficiency_service import TaxEfficiencyService, AccountType, TaxEfficiencyTier
from app.services.ml_intelligence_service import MLIntelligenceService, PredictionHorizon, ModelType, MarketRegime as MLMarketRegime

# Initialize services
dynamic_allocation_service = DynamicAllocationService()
advanced_optimization_service = AdvancedOptimizationService()
risk_management_service = RiskManagementService()
liquidity_management_service = LiquidityManagementService()
tax_efficiency_service = TaxEfficiencyService()

# Initialize ML Intelligence service
try:
    ml_intelligence_service = MLIntelligenceService()
    ML_ENABLED = True
    logger.info("ML Intelligence service initialized successfully")
except ImportError as e:
    logger.warning(f"ML Intelligence service not available: {str(e)}")
    ml_intelligence_service = None
    ML_ENABLED = False
except Exception as e:
    logger.error(f"Error initializing ML Intelligence service: {str(e)}")
    ml_intelligence_service = None
    ML_ENABLED = False

class DynamicAllocationRequest(BaseModel):
    tickers: List[str]
    base_allocation: Dict[str, float]
    risk_tolerance: str = "medium"
    lookback_days: int = 252

@app.post("/dynamic-allocation")
async def get_dynamic_allocation(request: DynamicAllocationRequest):
    """Get dynamic asset allocation based on current market conditions."""
    try:
        logger.info(f"Getting dynamic allocation for tickers: {request.tickers}")
        
        # Validate base allocation
        if not request.base_allocation or len(request.base_allocation) == 0:
            raise HTTPException(
                status_code=400,
                detail="Base allocation must be provided and non-empty"
            )
        
        # Validate that base allocation sums to approximately 1
        total_weight = sum(request.base_allocation.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow small rounding errors
            logger.warning(f"Base allocation sums to {total_weight}, normalizing to 1.0")
            # Normalize the allocation
            request.base_allocation = {
                ticker: weight / total_weight 
                for ticker, weight in request.base_allocation.items()
            }
        
        # Get dynamic allocation
        result = await dynamic_allocation_service.get_dynamic_allocation(
            tickers=request.tickers,
            base_allocation=request.base_allocation,
            risk_tolerance=request.risk_tolerance,
            lookback_days=request.lookback_days
        )
        
        logger.info("Dynamic allocation completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in dynamic allocation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get dynamic allocation: {str(e)}"
        )

@app.post("/market-analysis")
async def get_market_analysis():
    """Get comprehensive market analysis from all data sources."""
    try:
        logger.info("Getting comprehensive market analysis")
        
        # Get market data using the dynamic allocation service
        tickers = ["SPY", "QQQ", "BTC-USD", "GLD", "TLT"]  # Representative tickers
        market_data = await dynamic_allocation_service._get_comprehensive_market_data(
            tickers=tickers,
            lookback_days=252
        )
        
        logger.info("Market analysis completed successfully")
        return {
            "market_data": market_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in market analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get market analysis: {str(e)}"
        )

class OptimizedDynamicAllocationRequest(BaseModel):
    tickers: List[str]
    risk_tolerance: str = "medium"
    investment_amount: float = 10000
    use_dynamic_allocation: bool = True
    lookback_days: int = 252

@app.post("/analyze-portfolio-enhanced")
async def analyze_portfolio_enhanced(request: OptimizedDynamicAllocationRequest):
    """Enhanced portfolio analysis with dynamic allocation and comprehensive market data."""
    try:
        logger.info(f"Enhanced portfolio analysis for tickers: {request.tickers}")
        
        # First, get the traditional portfolio optimization
        portfolio_request = Portfolio(
            tickers=request.tickers,
            start_date=(datetime.now() - timedelta(days=request.lookback_days)).strftime('%Y-%m-%d'),
            risk_tolerance=request.risk_tolerance
        )
        
        # Get traditional analysis
        traditional_analysis = await analyze_portfolio(portfolio_request)
        
        if request.use_dynamic_allocation and "allocations" in traditional_analysis:
            # Apply dynamic allocation
            dynamic_result = await dynamic_allocation_service.get_dynamic_allocation(
                tickers=request.tickers,
                base_allocation=traditional_analysis["allocations"],
                risk_tolerance=request.risk_tolerance,
                lookback_days=request.lookback_days
            )
            
            # Combine results
            enhanced_analysis = {
                **traditional_analysis,
                "dynamic_allocation": dynamic_result,
                "enhanced_allocations": dynamic_result.get("final_allocation", traditional_analysis["allocations"]),
                "allocation_comparison": {
                    "traditional": traditional_analysis["allocations"],
                    "dynamic": dynamic_result.get("final_allocation", traditional_analysis["allocations"])
                }
            }
        else:
            enhanced_analysis = traditional_analysis
        
        logger.info("Enhanced portfolio analysis completed successfully")
        return enhanced_analysis
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in enhanced portfolio analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze portfolio with enhancements: {str(e)}"
        )

class AdvancedOptimizationRequest(BaseModel):
    tickers: List[str]
    method: str = "black_litterman"  # OptimizationMethod enum as string
    lookback_days: int = 252
    risk_tolerance: str = "medium"
    views: Optional[Dict[str, float]] = None
    factor_constraints: Optional[Dict[str, List[float]]] = None  # {factor: [min, max]}
    target_return: Optional[float] = None

@app.post("/optimize-portfolio-advanced")
async def optimize_portfolio_advanced(request: AdvancedOptimizationRequest):
    """Advanced portfolio optimization using sophisticated methods."""
    try:
        logger.info(f"Advanced optimization for tickers: {request.tickers} using method: {request.method}")
        
        # Validate method
        try:
            optimization_method = OptimizationMethod(request.method)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid optimization method: {request.method}. Valid methods: {list(OptimizationMethod)}"
            )
        
        # Convert factor constraints if provided
        factor_constraints = None
        if request.factor_constraints:
            factor_constraints = {}
            for factor_str, bounds in request.factor_constraints.items():
                try:
                    factor_type = FactorType(factor_str)
                    if len(bounds) == 2:
                        factor_constraints[factor_type] = (bounds[0], bounds[1])
                except ValueError:
                    logger.warning(f"Invalid factor type: {factor_str}")
        
        # Run advanced optimization
        result = await advanced_optimization_service.optimize_portfolio_advanced(
            tickers=request.tickers,
            method=optimization_method,
            lookback_days=request.lookback_days,
            risk_tolerance=request.risk_tolerance,
            views=request.views,
            factor_constraints=factor_constraints,
            target_return=request.target_return
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Optimization failed: {result['error']}"
            )
        
        logger.info("Advanced optimization completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in advanced optimization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to optimize portfolio: {str(e)}"
        )

class BlackLittermanRequest(BaseModel):
    tickers: List[str]
    views: Dict[str, float]  # {ticker: expected_return}
    view_confidence: Optional[float] = 0.025  # 2.5% uncertainty
    lookback_days: int = 252
    risk_tolerance: str = "medium"

@app.post("/black-litterman-optimization")
async def black_litterman_optimization(request: BlackLittermanRequest):
    """Black-Litterman portfolio optimization with investor views."""
    try:
        logger.info(f"Black-Litterman optimization with views: {request.views}")
        
        result = await advanced_optimization_service.optimize_portfolio_advanced(
            tickers=request.tickers,
            method=OptimizationMethod.BLACK_LITTERMAN,
            lookback_days=request.lookback_days,
            risk_tolerance=request.risk_tolerance,
            views=request.views,
            view_confidence=request.view_confidence
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Black-Litterman optimization failed: {result['error']}"
            )
        
        logger.info("Black-Litterman optimization completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in Black-Litterman optimization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run Black-Litterman optimization: {str(e)}"
        )

class FactorBasedRequest(BaseModel):
    tickers: List[str]
    factor_constraints: Dict[str, List[float]]  # {factor: [min, max]}
    lookback_days: int = 252
    target_factors: Optional[List[str]] = None  # Focus on specific factors

@app.post("/factor-based-optimization")
async def factor_based_optimization(request: FactorBasedRequest):
    """Factor-based portfolio optimization with factor constraints."""
    try:
        logger.info(f"Factor-based optimization with constraints: {request.factor_constraints}")
        
        # Convert factor constraints
        factor_constraints = {}
        for factor_str, bounds in request.factor_constraints.items():
            try:
                factor_type = FactorType(factor_str)
                if len(bounds) == 2:
                    factor_constraints[factor_type] = (bounds[0], bounds[1])
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid factor type: {factor_str}. Valid factors: {list(FactorType)}"
                )
        
        result = await advanced_optimization_service.optimize_portfolio_advanced(
            tickers=request.tickers,
            method=OptimizationMethod.FACTOR_BASED,
            lookback_days=request.lookback_days,
            factor_constraints=factor_constraints
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Factor-based optimization failed: {result['error']}"
            )
        
        logger.info("Factor-based optimization completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in factor-based optimization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run factor-based optimization: {str(e)}"
        )

@app.post("/risk-parity-optimization")
async def risk_parity_optimization(request: Portfolio):
    """Risk parity optimization for equal risk contribution."""
    try:
        logger.info(f"Risk parity optimization for tickers: {request.tickers}")
        
        result = await advanced_optimization_service.optimize_portfolio_advanced(
            tickers=request.tickers,
            method=OptimizationMethod.RISK_PARITY,
            lookback_days=252,
            risk_tolerance=request.risk_tolerance
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Risk parity optimization failed: {result['error']}"
            )
        
        logger.info("Risk parity optimization completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in risk parity optimization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run risk parity optimization: {str(e)}"
        )

@app.post("/minimum-variance-optimization")
async def minimum_variance_optimization(request: Portfolio):
    """Minimum variance portfolio optimization."""
    try:
        logger.info(f"Minimum variance optimization for tickers: {request.tickers}")
        
        result = await advanced_optimization_service.optimize_portfolio_advanced(
            tickers=request.tickers,
            method=OptimizationMethod.MIN_VOLATILITY,
            lookback_days=252,
            risk_tolerance=request.risk_tolerance
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Minimum variance optimization failed: {result['error']}"
            )
        
        logger.info("Minimum variance optimization completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in minimum variance optimization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run minimum variance optimization: {str(e)}"
        )

class OptimizationComparisonRequest(BaseModel):
    tickers: List[str]
    methods: List[str] = ["max_sharpe", "min_volatility", "black_litterman", "risk_parity"]
    lookback_days: int = 252
    risk_tolerance: str = "medium"
    views: Optional[Dict[str, float]] = None

@app.post("/compare-optimization-methods")
async def compare_optimization_methods(request: OptimizationComparisonRequest):
    """Compare multiple optimization methods side by side."""
    try:
        logger.info(f"Comparing optimization methods: {request.methods} for tickers: {request.tickers}")
        
        results = {}
        
        for method_str in request.methods:
            try:
                method = OptimizationMethod(method_str)
                
                result = await advanced_optimization_service.optimize_portfolio_advanced(
                    tickers=request.tickers,
                    method=method,
                    lookback_days=request.lookback_days,
                    risk_tolerance=request.risk_tolerance,
                    views=request.views if method == OptimizationMethod.BLACK_LITTERMAN else None
                )
                
                if "error" not in result:
                    results[method_str] = result
                else:
                    results[method_str] = {"error": result["error"]}
                    
            except ValueError:
                results[method_str] = {"error": f"Invalid method: {method_str}"}
            except Exception as e:
                results[method_str] = {"error": str(e)}
        
        # Add comparison summary
        comparison_summary = {
            "best_sharpe": None,
            "lowest_volatility": None,
            "most_diversified": None,
            "method_rankings": {}
        }
        
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        
        if valid_results:
            # Find best Sharpe ratio
            best_sharpe_method = max(
                valid_results.keys(), 
                key=lambda x: valid_results[x].get("sharpe_ratio", 0)
            )
            comparison_summary["best_sharpe"] = {
                "method": best_sharpe_method,
                "sharpe_ratio": valid_results[best_sharpe_method]["sharpe_ratio"]
            }
            
            # Find lowest volatility
            lowest_vol_method = min(
                valid_results.keys(), 
                key=lambda x: valid_results[x].get("volatility", float('inf'))
            )
            comparison_summary["lowest_volatility"] = {
                "method": lowest_vol_method,
                "volatility": valid_results[lowest_vol_method]["volatility"]
            }
            
            # Calculate diversification scores
            for method, result in valid_results.items():
                weights = result.get("weights", {})
                if weights:
                    concentration = sum(w**2 for w in weights.values())
                    diversification_score = 1.0 / concentration
                    comparison_summary["method_rankings"][method] = {
                        "diversification_score": diversification_score,
                        "concentration_ratio": concentration
                    }
        
        logger.info("Optimization method comparison completed successfully")
        return {
            "results": results,
            "comparison_summary": comparison_summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in optimization comparison: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare optimization methods: {str(e)}"
        )

class VolatilityPositionSizingRequest(BaseModel):
    portfolio_weights: Dict[str, float]
    tickers: List[str]
    lookback_days: int = 63
    target_portfolio_vol: float = 0.15

@app.post("/volatility-position-sizing")
async def volatility_position_sizing(request: VolatilityPositionSizingRequest):
    """Apply volatility-based position sizing to portfolio."""
    try:
        logger.info(f"Applying volatility position sizing for {len(request.tickers)} assets")
        
        result = await risk_management_service.apply_volatility_position_sizing(
            weights=request.portfolio_weights,
            tickers=request.tickers,
            lookback_days=request.lookback_days,
            target_portfolio_vol=request.target_portfolio_vol
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Volatility sizing failed: {result['error']}"
            )
        
        logger.info("Volatility position sizing completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in volatility position sizing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to apply volatility sizing: {str(e)}"
        )

class TailRiskHedgingRequest(BaseModel):
    portfolio_weights: Dict[str, float]
    risk_regime: str = "moderate_risk"
    hedge_budget: float = 0.05

@app.post("/tail-risk-hedging")
async def tail_risk_hedging(request: TailRiskHedgingRequest):
    """Implement tail risk hedging strategies."""
    try:
        logger.info(f"Implementing tail risk hedging for {request.risk_regime} regime")
        
        # Validate risk regime
        try:
            risk_regime_enum = RiskRegime(request.risk_regime)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid risk regime: {request.risk_regime}. Valid regimes: {list(RiskRegime)}"
            )
        
        result = await risk_management_service.implement_tail_risk_hedging(
            portfolio_weights=request.portfolio_weights,
            risk_regime=risk_regime_enum,
            hedge_budget=request.hedge_budget
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Tail hedging failed: {result['error']}"
            )
        
        logger.info("Tail risk hedging implemented successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in tail risk hedging: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to implement tail hedging: {str(e)}"
        )

class DrawdownControlRequest(BaseModel):
    current_weights: Dict[str, float]
    portfolio_value: float
    peak_value: float
    lookback_days: int = 252

@app.post("/drawdown-controls")
async def drawdown_controls(request: DrawdownControlRequest):
    """Apply drawdown control mechanisms."""
    try:
        logger.info(f"Applying drawdown controls - Current: ${request.portfolio_value:,.0f}, Peak: ${request.peak_value:,.0f}")
        
        result = await risk_management_service.apply_drawdown_controls(
            current_weights=request.current_weights,
            portfolio_value=request.portfolio_value,
            peak_value=request.peak_value,
            lookback_days=request.lookback_days
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Drawdown controls failed: {result['error']}"
            )
        
        logger.info("Drawdown controls applied successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in drawdown controls: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to apply drawdown controls: {str(e)}"
        )

class ComprehensiveRiskManagementRequest(BaseModel):
    portfolio_weights: Dict[str, float]
    tickers: List[str]
    portfolio_value: float = 100000
    peak_value: float = 100000
    target_vol: float = 0.15
    hedge_budget: float = 0.05
    lookback_days: int = 63

@app.post("/comprehensive-risk-management")
async def comprehensive_risk_management(request: ComprehensiveRiskManagementRequest):
    """Apply comprehensive risk management combining all strategies."""
    try:
        logger.info(f"Applying comprehensive risk management for portfolio value: ${request.portfolio_value:,.0f}")
        
        result = await risk_management_service.comprehensive_risk_management(
            portfolio_weights=request.portfolio_weights,
            tickers=request.tickers,
            portfolio_value=request.portfolio_value,
            peak_value=request.peak_value,
            target_vol=request.target_vol,
            hedge_budget=request.hedge_budget,
            lookback_days=request.lookback_days
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Risk management failed: {result['error']}"
            )
        
        logger.info("Comprehensive risk management completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in comprehensive risk management: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to apply risk management: {str(e)}"
        )

class OptimizedPortfolioWithRiskControlsRequest(BaseModel):
    tickers: List[str]
    optimization_method: str = "black_litterman"
    risk_tolerance: str = "medium"
    portfolio_value: float = 100000
    peak_value: float = 100000
    target_vol: float = 0.15
    hedge_budget: float = 0.05
    views: Optional[Dict[str, float]] = None
    factor_constraints: Optional[Dict[str, List[float]]] = None
    lookback_days: int = 63

@app.post("/optimize-with-risk-controls")
async def optimize_with_risk_controls(request: OptimizedPortfolioWithRiskControlsRequest):
    """Optimize portfolio and apply comprehensive risk controls."""
    try:
        logger.info(f"Running optimization with risk controls for {len(request.tickers)} assets")
        
        # Step 1: Run portfolio optimization
        try:
            optimization_method = OptimizationMethod(request.optimization_method)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid optimization method: {request.optimization_method}"
            )
        
        # Convert factor constraints if provided
        factor_constraints = None
        if request.factor_constraints:
            factor_constraints = {}
            for factor_str, bounds in request.factor_constraints.items():
                try:
                    factor_type = FactorType(factor_str)
                    if len(bounds) == 2:
                        factor_constraints[factor_type] = (bounds[0], bounds[1])
                except ValueError:
                    logger.warning(f"Invalid factor type: {factor_str}")
        
        optimization_result = await advanced_optimization_service.optimize_portfolio_advanced(
            tickers=request.tickers,
            method=optimization_method,
            lookback_days=request.lookback_days,
            risk_tolerance=request.risk_tolerance,
            views=request.views,
            factor_constraints=factor_constraints
        )
        
        if "error" in optimization_result:
            raise HTTPException(
                status_code=500,
                detail=f"Optimization failed: {optimization_result['error']}"
            )
        
        optimized_weights = optimization_result.get("weights", {})
        
        # Step 2: Apply comprehensive risk management
        risk_management_result = await risk_management_service.comprehensive_risk_management(
            portfolio_weights=optimized_weights,
            tickers=request.tickers,
            portfolio_value=request.portfolio_value,
            peak_value=request.peak_value,
            target_vol=request.target_vol,
            hedge_budget=request.hedge_budget,
            lookback_days=request.lookback_days
        )
        
        if "error" in risk_management_result:
            logger.warning(f"Risk management had issues: {risk_management_result['error']}")
            # Continue with optimized weights if risk management fails
            final_weights = optimized_weights
            risk_management_result = {"error": risk_management_result["error"]}
        else:
            final_weights = risk_management_result.get("final_weights", optimized_weights)
        
        # Step 3: Calculate performance comparison
        optimization_performance = {
            "expected_return": optimization_result.get("expected_return", 0),
            "volatility": optimization_result.get("volatility", 0),
            "sharpe_ratio": optimization_result.get("sharpe_ratio", 0)
        }
        
        # Calculate final portfolio metrics (simplified)
        weight_changes = {}
        for ticker in set(list(optimized_weights.keys()) + list(final_weights.keys())):
            original = optimized_weights.get(ticker, 0)
            final = final_weights.get(ticker, 0)
            if abs(final - original) > 0.001:  # Only show significant changes
                weight_changes[ticker] = {
                    "original": original,
                    "final": final,
                    "change": final - original
                }
        
        logger.info("Portfolio optimization with risk controls completed successfully")
        
        return {
            "final_portfolio": final_weights,
            "optimization_result": optimization_result,
            "risk_management": risk_management_result,
            "performance_comparison": {
                "optimization_only": optimization_performance,
                "with_risk_controls": {
                    "risk_regime": risk_management_result.get("risk_regime"),
                    "defensive_allocation": risk_management_result.get("summary", {}).get("defensive_allocation", 0),
                    "hedge_allocation": risk_management_result.get("summary", {}).get("hedge_allocation", 0)
                }
            },
            "weight_changes": weight_changes,
            "summary": {
                "optimization_method": request.optimization_method,
                "risk_controls_applied": "error" not in risk_management_result,
                "total_adjustments": len(weight_changes),
                "recommendations": risk_management_result.get("summary", {}).get("recommendations", [])
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in optimization with risk controls: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to optimize with risk controls: {str(e)}"
        )

@app.get("/risk-management-info")
async def get_risk_management_info():
    """Get information about available risk management features."""
    return {
        "features": {
            "volatility_position_sizing": {
                "description": "Adjusts position sizes based on asset volatility",
                "parameters": ["target_portfolio_vol", "lookback_days"],
                "benefits": ["Reduces concentration in volatile assets", "Controls portfolio-level risk"]
            },
            "tail_risk_hedging": {
                "description": "Implements hedging strategies during high-risk periods",
                "hedge_types": ["VIX protection", "Tail protection ETFs", "Safe haven assets", "Currency hedging"],
                "risk_regimes": list(RiskRegime)
            },
            "drawdown_controls": {
                "description": "Reduces exposure after losses exceed thresholds",
                "severity_levels": list(DrawdownSeverity),
                "actions": ["Risk reduction", "Cash allocation", "Recovery monitoring"]
            },
            "comprehensive_management": {
                "description": "Combines all risk management strategies",
                "workflow": ["Volatility sizing", "Tail hedging", "Drawdown controls", "Performance monitoring"]
            }
        },
        "risk_regimes": {
            regime.value: {
                "description": f"Market risk level: {regime.value.replace('_', ' ').title()}",
                "hedge_intensity": {
                    RiskRegime.LOW_RISK: "Minimal hedging",
                    RiskRegime.MODERATE_RISK: "Standard hedging",
                    RiskRegime.HIGH_RISK: "Elevated hedging",
                    RiskRegime.EXTREME_RISK: "Maximum hedging"
                }.get(regime, "Standard hedging")
            }
            for regime in RiskRegime
        },
        "default_parameters": {
            "target_portfolio_vol": 0.15,
            "hedge_budget": 0.05,
            "lookback_days": 63,
            "volatility_thresholds": {
                "low": "< 15% annual",
                "medium": "15-25% annual", 
                "high": "25-40% annual",
                "extreme": "> 40% annual"
            }
        }
    }

# Liquidity Management Endpoints

class CashBufferRequest(BaseModel):
    portfolio_value: float
    current_positions: Dict[str, float]
    volatility_forecast: Optional[float] = None
    stress_indicators: Optional[Dict[str, float]] = None

@app.post("/calculate-cash-buffer")
async def calculate_cash_buffer(request: CashBufferRequest):
    """Calculate optimal cash buffer based on volatility forecasts and market conditions."""
    try:
        logger.info(f"Calculating cash buffer for portfolio value: ${request.portfolio_value:,.0f}")
        
        result = await liquidity_management_service.calculate_optimal_cash_buffer(
            portfolio_value=request.portfolio_value,
            current_positions=request.current_positions,
            volatility_forecast=request.volatility_forecast,
            stress_indicators=request.stress_indicators
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"Cash buffer calculation failed: {result['error']}")
        
        logger.info("Cash buffer calculation completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error calculating cash buffer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate cash buffer: {str(e)}")

class RebalancingFrequencyRequest(BaseModel):
    current_positions: Dict[str, float]
    market_volatility: Optional[float] = None
    liquidity_constraints: Optional[Dict[str, float]] = None

@app.post("/determine-rebalancing-frequency")
async def determine_rebalancing_frequency(request: RebalancingFrequencyRequest):
    """Determine optimal rebalancing frequency based on market conditions."""
    try:
        logger.info("Determining optimal rebalancing frequency")
        
        result = await liquidity_management_service.determine_rebalancing_frequency(
            current_positions=request.current_positions,
            market_volatility=request.market_volatility,
            liquidity_constraints=request.liquidity_constraints
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"Frequency determination failed: {result['error']}")
        
        logger.info("Rebalancing frequency determination completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error determining rebalancing frequency: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to determine rebalancing frequency: {str(e)}")

class LiquidityScoringRequest(BaseModel):
    symbols: List[str]
    position_sizes: Optional[Dict[str, float]] = None
    time_horizon: Optional[int] = None

@app.post("/score-asset-liquidity")
async def score_asset_liquidity(request: LiquidityScoringRequest):
    """Score asset liquidity to avoid illiquid assets during market stress."""
    try:
        logger.info(f"Scoring liquidity for {len(request.symbols)} assets")
        
        result = await liquidity_management_service.score_asset_liquidity(
            symbols=request.symbols,
            position_sizes=request.position_sizes,
            time_horizon=request.time_horizon
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"Liquidity scoring failed: {result['error']}")
        
        logger.info("Asset liquidity scoring completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error scoring asset liquidity: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to score asset liquidity: {str(e)}")

class LiquidityAwareAllocationRequest(BaseModel):
    target_allocation: Dict[str, float]
    market_condition: Optional[str] = None
    liquidity_requirements: Optional[Dict[str, Any]] = None

@app.post("/liquidity-aware-allocation")
async def generate_liquidity_aware_allocation(request: LiquidityAwareAllocationRequest):
    """Generate allocation that considers liquidity constraints."""
    try:
        logger.info("Generating liquidity-aware allocation")
        
        # Convert market condition string to enum if provided
        market_condition = None
        if request.market_condition:
            try:
                market_condition = MarketCondition(request.market_condition)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid market condition: {request.market_condition}. Valid conditions: {list(MarketCondition)}"
                )
        
        result = await liquidity_management_service.generate_liquidity_aware_allocation(
            target_allocation=request.target_allocation,
            market_condition=market_condition,
            liquidity_requirements=request.liquidity_requirements
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"Liquidity-aware allocation failed: {result['error']}")
        
        logger.info("Liquidity-aware allocation generated successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error generating liquidity-aware allocation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate liquidity-aware allocation: {str(e)}")

# Tax Efficiency Endpoints

class TaxLossHarvestingRequest(BaseModel):
    portfolio_positions: Dict[str, Dict[str, Any]]  # symbol -> {quantity, current_price, cost_basis}
    account_type: str = "taxable"
    min_loss_threshold: float = 1000.0
    min_loss_percentage: float = 0.05

@app.post("/tax-loss-harvesting")
async def identify_tax_loss_harvesting(request: TaxLossHarvestingRequest):
    """Identify tax-loss harvesting opportunities."""
    try:
        logger.info("Identifying tax-loss harvesting opportunities")
        
        # Convert account type string to enum
        try:
            account_type = AccountType(request.account_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid account type: {request.account_type}. Valid types: {list(AccountType)}"
            )
        
        result = await tax_efficiency_service.identify_tax_loss_harvesting_opportunities(
            portfolio_positions=request.portfolio_positions,
            account_type=account_type,
            min_loss_threshold=request.min_loss_threshold,
            min_loss_percentage=request.min_loss_percentage
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"Tax-loss harvesting analysis failed: {result['error']}")
        
        logger.info("Tax-loss harvesting analysis completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in tax-loss harvesting analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze tax-loss harvesting: {str(e)}")

class AssetLocationRequest(BaseModel):
    target_allocation: Dict[str, float]
    available_accounts: Dict[str, float]  # account_type -> available_capacity
    current_positions: Optional[Dict[str, Dict[str, float]]] = None  # account_type -> {symbol: allocation}

@app.post("/optimize-asset-location")
async def optimize_asset_location(request: AssetLocationRequest):
    """Optimize asset location across account types for tax efficiency."""
    try:
        logger.info("Optimizing asset location for tax efficiency")
        
        # Convert account type strings to enums
        available_accounts = {}
        for account_str, capacity in request.available_accounts.items():
            try:
                account_type = AccountType(account_str)
                available_accounts[account_type] = capacity
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid account type: {account_str}. Valid types: {list(AccountType)}"
                )
        
        # Convert current positions if provided
        current_positions = None
        if request.current_positions:
            current_positions = {}
            for account_str, positions in request.current_positions.items():
                try:
                    account_type = AccountType(account_str)
                    current_positions[account_type] = positions
                except ValueError:
                    logger.warning(f"Invalid account type in current positions: {account_str}")
        
        result = await tax_efficiency_service.optimize_asset_location(
            target_allocation=request.target_allocation,
            available_accounts=available_accounts,
            current_positions=current_positions
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"Asset location optimization failed: {result['error']}")
        
        logger.info("Asset location optimization completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error optimizing asset location: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize asset location: {str(e)}")

class TaxAwareRebalancingRequest(BaseModel):
    current_positions: Dict[str, Dict[str, float]]  # account_type -> {symbol: quantity}
    target_allocation: Dict[str, float]
    current_prices: Dict[str, float]
    cost_basis_data: Dict[str, Dict[str, Any]]  # symbol -> cost basis info
    max_tax_impact: float = 0.02

@app.post("/tax-aware-rebalancing")
async def plan_tax_aware_rebalancing(request: TaxAwareRebalancingRequest):
    """Plan tax-aware rebalancing that considers tax implications."""
    try:
        logger.info("Planning tax-aware rebalancing")
        
        # Convert account type strings to enums
        current_positions = {}
        for account_str, positions in request.current_positions.items():
            try:
                account_type = AccountType(account_str)
                current_positions[account_type] = positions
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid account type: {account_str}. Valid types: {list(AccountType)}"
                )
        
        result = await tax_efficiency_service.plan_tax_aware_rebalancing(
            current_positions=current_positions,
            target_allocation=request.target_allocation,
            current_prices=request.current_prices,
            cost_basis_data=request.cost_basis_data,
            max_tax_impact=request.max_tax_impact
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"Tax-aware rebalancing failed: {result['error']}")
        
        logger.info("Tax-aware rebalancing plan completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error planning tax-aware rebalancing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to plan tax-aware rebalancing: {str(e)}")

class TaxAlphaRequest(BaseModel):
    portfolio_performance: Dict[str, Any]
    tax_management_actions: List[Dict[str, Any]]
    benchmark_tax_drag: float = 0.015

@app.post("/calculate-tax-alpha")
async def calculate_tax_alpha(request: TaxAlphaRequest):
    """Calculate tax alpha - the value added through tax management."""
    try:
        logger.info("Calculating tax alpha")
        
        result = await tax_efficiency_service.calculate_tax_alpha(
            portfolio_performance=request.portfolio_performance,
            tax_management_actions=request.tax_management_actions,
            benchmark_tax_drag=request.benchmark_tax_drag
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"Tax alpha calculation failed: {result['error']}")
        
        logger.info("Tax alpha calculation completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error calculating tax alpha: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate tax alpha: {str(e)}")

# Comprehensive Portfolio Management Endpoint

class ComprehensivePortfolioRequest(BaseModel):
    tickers: List[str]
    base_allocation: Dict[str, float]
    risk_tolerance: str = "medium"
    optimization_method: str = "black_litterman"
    portfolio_value: float = 100000
    peak_value: float = 100000
    available_accounts: Optional[Dict[str, float]] = None
    current_positions: Optional[Dict[str, Dict[str, float]]] = None
    views: Optional[Dict[str, float]] = None
    factor_constraints: Optional[Dict[str, List[float]]] = None
    enable_tax_optimization: bool = False
    enable_liquidity_management: bool = True
    enable_risk_management: bool = True

@app.post("/comprehensive-portfolio-management")
async def comprehensive_portfolio_management(request: ComprehensivePortfolioRequest):
    """Complete portfolio management with optimization, risk controls, liquidity management, and tax efficiency."""
    try:
        logger.info(f"Running comprehensive portfolio management for {len(request.tickers)} assets")
        
        # Step 1: Portfolio Optimization
        try:
            optimization_method = OptimizationMethod(request.optimization_method)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid optimization method: {request.optimization_method}"
            )
        
        optimization_result = await advanced_optimization_service.optimize_portfolio_advanced(
            tickers=request.tickers,
            method=optimization_method,
            lookback_days=63,
            risk_tolerance=request.risk_tolerance,
            views=request.views,
            factor_constraints=None  # Handle factor constraints conversion if needed
        )
        
        if "error" in optimization_result:
            raise HTTPException(status_code=500, detail=f"Optimization failed: {optimization_result['error']}")
        
        current_weights = optimization_result.get("weights", {})
        
        # Step 2: Risk Management (if enabled)
        risk_management_result = {}
        if request.enable_risk_management:
            risk_management_result = await risk_management_service.comprehensive_risk_management(
                portfolio_weights=current_weights,
                tickers=request.tickers,
                portfolio_value=request.portfolio_value,
                peak_value=request.peak_value,
                target_vol=0.15,
                hedge_budget=0.05,
                lookback_days=63
            )
            
            if "error" not in risk_management_result:
                current_weights = risk_management_result.get("final_weights", current_weights)
        
        # Step 3: Liquidity Management (if enabled)
        liquidity_result = {}
        if request.enable_liquidity_management:
            liquidity_result = await liquidity_management_service.generate_liquidity_aware_allocation(
                target_allocation=current_weights,
                market_condition=None,
                liquidity_requirements=None
            )
            
            if "error" not in liquidity_result:
                current_weights = liquidity_result.get("liquidity_adjusted_allocation", current_weights)
        
        # Step 4: Tax Optimization (if enabled and accounts provided)
        tax_optimization_result = {}
        if request.enable_tax_optimization and request.available_accounts:
            # Convert account strings to enums
            available_accounts = {}
            for account_str, capacity in request.available_accounts.items():
                try:
                    account_type = AccountType(account_str)
                    available_accounts[account_type] = capacity
                except ValueError:
                    logger.warning(f"Invalid account type: {account_str}")
            
            if available_accounts:
                tax_optimization_result = await tax_efficiency_service.optimize_asset_location(
                    target_allocation=current_weights,
                    available_accounts=available_accounts,
                    current_positions=None
                )
        
        # Step 5: Calculate comprehensive metrics
        final_allocation = current_weights
        
        # Calculate allocation changes
        allocation_changes = {}
        for symbol in set(list(request.base_allocation.keys()) + list(final_allocation.keys())):
            original = request.base_allocation.get(symbol, 0)
            final = final_allocation.get(symbol, 0)
            if abs(final - original) > 0.001:
                allocation_changes[symbol] = {
                    "original": original,
                    "final": final,
                    "change": final - original,
                    "change_pct": (final - original) / original if original > 0 else float('inf')
                }
        
        # Generate comprehensive summary
        summary = {
            "optimization_method": request.optimization_method,
            "risk_management_enabled": request.enable_risk_management,
            "liquidity_management_enabled": request.enable_liquidity_management,
            "tax_optimization_enabled": request.enable_tax_optimization,
            "total_adjustments": len(allocation_changes),
            "major_changes": len([c for c in allocation_changes.values() if abs(c["change"]) > 0.05]),
            "portfolio_value": request.portfolio_value
        }
        
        logger.info("Comprehensive portfolio management completed successfully")
        
        return {
            "final_allocation": final_allocation,
            "original_allocation": request.base_allocation,
            "allocation_changes": allocation_changes,
            "optimization_result": optimization_result,
            "risk_management": risk_management_result,
            "liquidity_management": liquidity_result,
            "tax_optimization": tax_optimization_result,
            "summary": summary,
            "recommendations": [
                "Portfolio optimized using " + request.optimization_method,
                f"Risk management {'applied' if request.enable_risk_management else 'disabled'}",
                f"Liquidity constraints {'considered' if request.enable_liquidity_management else 'ignored'}",
                f"Tax optimization {'applied' if request.enable_tax_optimization else 'disabled'}"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in comprehensive portfolio management: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to complete comprehensive portfolio management: {str(e)}"
        )

# ML Intelligence Endpoints

class MLTrainingRequest(BaseModel):
    symbols: List[str]
    horizons: Optional[List[str]] = None
    model_types: Optional[List[str]] = None
    lookback_days: int = 252
    retrain: bool = False

@app.post("/ml/train-models")
async def train_ml_models(request: MLTrainingRequest):
    """Train ML models for price prediction."""
    if not ML_ENABLED:
        raise HTTPException(status_code=503, detail="ML Intelligence service not available")
    
    try:
        logger.info(f"Training ML models for {len(request.symbols)} symbols")
        
        # Convert string enums to proper enums
        horizons = None
        if request.horizons:
            try:
                horizons = [PredictionHorizon(h) for h in request.horizons]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid horizon: {str(e)}")
        
        model_types = None
        if request.model_types:
            try:
                model_types = [ModelType(m) for m in request.model_types]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid model type: {str(e)}")
        
        result = await ml_intelligence_service.train_price_prediction_models(
            symbols=request.symbols,
            horizons=horizons,
            model_types=model_types,
            lookback_days=request.lookback_days,
            retrain=request.retrain
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"ML training failed: {result['error']}")
        
        logger.info("ML model training completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error training ML models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to train ML models: {str(e)}")

class MLPredictionRequest(BaseModel):
    symbols: List[str]
    horizons: Optional[List[str]] = None
    include_confidence: bool = True

@app.post("/ml/predict-movements")
async def predict_price_movements(request: MLPredictionRequest):
    """Predict short-term price movements using trained ML models."""
    if not ML_ENABLED:
        raise HTTPException(status_code=503, detail="ML Intelligence service not available")
    
    try:
        logger.info(f"Generating ML predictions for {len(request.symbols)} symbols")
        
        # Convert string enums to proper enums
        horizons = None
        if request.horizons:
            try:
                horizons = [PredictionHorizon(h) for h in request.horizons]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid horizon: {str(e)}")
        
        result = await ml_intelligence_service.predict_price_movements(
            symbols=request.symbols,
            horizons=horizons,
            include_confidence=request.include_confidence
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"ML prediction failed: {result['error']}")
        
        logger.info("ML price predictions completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error predicting price movements: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to predict price movements: {str(e)}")

class MarketRegimeRequest(BaseModel):
    lookback_days: int = 252
    retrain_model: bool = False

@app.post("/ml/identify-regimes")
async def identify_market_regimes(request: MarketRegimeRequest):
    """Use clustering algorithms to identify market regimes."""
    if not ML_ENABLED:
        raise HTTPException(status_code=503, detail="ML Intelligence service not available")
    
    try:
        logger.info("Identifying market regimes using ML clustering")
        
        result = await ml_intelligence_service.identify_market_regimes(
            lookback_days=request.lookback_days,
            retrain_model=request.retrain_model
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"Market regime analysis failed: {result['error']}")
        
        logger.info("Market regime identification completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error identifying market regimes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to identify market regimes: {str(e)}")

class RLOptimizationRequest(BaseModel):
    portfolio_weights: Dict[str, float]
    market_state: Dict[str, Any]
    learning_mode: bool = False
    episodes: int = 1000

@app.post("/ml/rl-optimization")
async def reinforcement_learning_optimization(request: RLOptimizationRequest):
    """Use reinforcement learning for dynamic portfolio optimization."""
    if not ML_ENABLED:
        raise HTTPException(status_code=503, detail="ML Intelligence service not available")
    
    try:
        logger.info("Running reinforcement learning portfolio optimization")
        
        result = await ml_intelligence_service.reinforcement_learning_optimization(
            portfolio_weights=request.portfolio_weights,
            market_state=request.market_state,
            learning_mode=request.learning_mode,
            episodes=request.episodes
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"RL optimization failed: {result['error']}")
        
        logger.info("RL optimization completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in RL optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to run RL optimization: {str(e)}")

class EnsembleMLRequest(BaseModel):
    symbols: List[str]
    prediction_horizon: str = "5_days"
    include_regime_analysis: bool = True
    include_rl_insights: bool = False

@app.post("/ml/ensemble-prediction")
async def ensemble_ml_prediction(request: EnsembleMLRequest):
    """Comprehensive ML prediction combining multiple approaches."""
    if not ML_ENABLED:
        raise HTTPException(status_code=503, detail="ML Intelligence service not available")
    
    try:
        logger.info(f"Running ensemble ML prediction for {len(request.symbols)} symbols")
        
        # Convert prediction horizon
        try:
            prediction_horizon = PredictionHorizon(request.prediction_horizon)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid prediction horizon: {request.prediction_horizon}")
        
        result = await ml_intelligence_service.ensemble_ml_prediction(
            symbols=request.symbols,
            prediction_horizon=prediction_horizon,
            include_regime_analysis=request.include_regime_analysis,
            include_rl_insights=request.include_rl_insights
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"Ensemble ML prediction failed: {result['error']}")
        
        logger.info("Ensemble ML prediction completed successfully")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in ensemble ML prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to run ensemble ML prediction: {str(e)}")

@app.get("/ml/model-status")
async def get_ml_model_status():
    """Get status of all trained ML models."""
    if not ML_ENABLED:
        raise HTTPException(status_code=503, detail="ML Intelligence service not available")
    
    try:
        status = ml_intelligence_service.get_model_status()
        return status
    except Exception as e:
        logger.error(f"Error getting ML model status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

# AI-Powered Comprehensive Portfolio Management

class AIPortfolioRequest(BaseModel):
    tickers: List[str]
    base_allocation: Dict[str, float]
    risk_tolerance: str = "medium"
    portfolio_value: float = 100000
    peak_value: float = 100000
    enable_ml_predictions: bool = True
    enable_regime_analysis: bool = True
    enable_rl_optimization: bool = False
    prediction_horizon: str = "5_days"
    train_models: bool = False
    available_accounts: Optional[Dict[str, float]] = None
    enable_tax_optimization: bool = False
    enable_liquidity_management: bool = True
    enable_risk_management: bool = True

@app.post("/ai-portfolio-management")
async def ai_powered_portfolio_management(request: AIPortfolioRequest):
    """Complete AI-powered portfolio management with ML intelligence."""
    try:
        logger.info(f"Running AI-powered portfolio management for {len(request.tickers)} assets")
        
        # Step 1: ML Model Training (if requested)
        ml_training_result = {}
        if ML_ENABLED and request.train_models:
            logger.info("Training ML models...")
            ml_training_result = await ml_intelligence_service.train_price_prediction_models(
                symbols=request.tickers,
                horizons=[PredictionHorizon(request.prediction_horizon)],
                retrain=True
            )
        
        # Step 2: ML Predictions and Analysis
        ml_insights = {}
        if ML_ENABLED and request.enable_ml_predictions:
            logger.info("Generating ML predictions...")
            try:
                prediction_horizon = PredictionHorizon(request.prediction_horizon)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid prediction horizon: {request.prediction_horizon}")
            
            ml_insights = await ml_intelligence_service.ensemble_ml_prediction(
                symbols=request.tickers,
                prediction_horizon=prediction_horizon,
                include_regime_analysis=request.enable_regime_analysis,
                include_rl_insights=request.enable_rl_optimization
            )
            
            # Adjust allocation based on ML insights
            if "final_recommendations" in ml_insights:
                individual_recs = ml_insights["final_recommendations"].get("individual_recommendations", {})
                
                # Apply ML-driven allocation adjustments
                ml_adjusted_allocation = request.base_allocation.copy()
                for symbol, rec in individual_recs.items():
                    if symbol in ml_adjusted_allocation:
                        confidence = rec.get("confidence", 0.5)
                        expected_return = rec.get("expected_return", 0.0)
                        
                        # Adjust allocation based on ML signal strength
                        if rec["action"] == "buy" and confidence > 0.7:
                            ml_adjusted_allocation[symbol] *= (1 + expected_return * confidence * 0.2)
                        elif rec["action"] == "sell" and confidence > 0.7:
                            ml_adjusted_allocation[symbol] *= (1 - abs(expected_return) * confidence * 0.2)
                
                # Normalize weights
                total_weight = sum(ml_adjusted_allocation.values())
                if total_weight > 0:
                    ml_adjusted_allocation = {k: v/total_weight for k, v in ml_adjusted_allocation.items()}
                
                # Use ML-adjusted allocation for subsequent steps
                current_allocation = ml_adjusted_allocation
            else:
                current_allocation = request.base_allocation
        else:
            current_allocation = request.base_allocation
        
        # Step 3: Advanced Portfolio Optimization
        logger.info("Running portfolio optimization...")
        try:
            optimization_method = OptimizationMethod.BLACK_LITTERMAN
        except ValueError:
            optimization_method = OptimizationMethod.MAX_SHARPE
        
        optimization_result = await advanced_optimization_service.optimize_portfolio_advanced(
            tickers=request.tickers,
            method=optimization_method,
            lookback_days=63,
            risk_tolerance=request.risk_tolerance,
            views=None
        )
        
        if "error" not in optimization_result:
            current_weights = optimization_result.get("weights", current_allocation)
        else:
            current_weights = current_allocation
        
        # Step 4: Risk Management (if enabled)
        risk_management_result = {}
        if request.enable_risk_management:
            logger.info("Applying risk management...")
            risk_management_result = await risk_management_service.comprehensive_risk_management(
                portfolio_weights=current_weights,
                tickers=request.tickers,
                portfolio_value=request.portfolio_value,
                peak_value=request.peak_value,
                target_vol=0.15,
                hedge_budget=0.05,
                lookback_days=63
            )
            
            if "error" not in risk_management_result:
                current_weights = risk_management_result.get("final_weights", current_weights)
        
        # Step 5: Liquidity Management (if enabled)
        liquidity_result = {}
        if request.enable_liquidity_management:
            logger.info("Applying liquidity management...")
            liquidity_result = await liquidity_management_service.generate_liquidity_aware_allocation(
                target_allocation=current_weights,
                market_condition=None,
                liquidity_requirements=None
            )
            
            if "error" not in liquidity_result:
                current_weights = liquidity_result.get("liquidity_adjusted_allocation", current_weights)
        
        # Step 6: Tax Optimization (if enabled and accounts provided)
        tax_optimization_result = {}
        if request.enable_tax_optimization and request.available_accounts:
            logger.info("Applying tax optimization...")
            # Convert account strings to enums
            available_accounts = {}
            for account_str, capacity in request.available_accounts.items():
                try:
                    account_type = AccountType(account_str)
                    available_accounts[account_type] = capacity
                except ValueError:
                    logger.warning(f"Invalid account type: {account_str}")
            
            if available_accounts:
                tax_optimization_result = await tax_efficiency_service.optimize_asset_location(
                    target_allocation=current_weights,
                    available_accounts=available_accounts,
                    current_positions=None
                )
        
        # Step 7: Generate Final Analysis and Recommendations
        final_allocation = current_weights
        
        # Calculate allocation changes
        allocation_changes = {}
        for symbol in set(list(request.base_allocation.keys()) + list(final_allocation.keys())):
            original = request.base_allocation.get(symbol, 0)
            final = final_allocation.get(symbol, 0)
            if abs(final - original) > 0.001:
                allocation_changes[symbol] = {
                    "original": original,
                    "final": final,
                    "change": final - original,
                    "change_pct": (final - original) / original if original > 0 else float('inf')
                }
        
        # Generate AI insights
        ai_insights = {
            "ml_model_confidence": ml_insights.get("portfolio_metrics", {}).get("average_confidence", 0.5) if ml_insights else 0.0,
            "regime_analysis": ml_insights.get("component_analyses", {}).get("regime_analysis", {}) if ml_insights else {},
            "optimization_improvement": optimization_result.get("metrics", {}).get("sharpe_ratio", 0) if optimization_result else 0,
            "risk_adjustment": "applied" if request.enable_risk_management else "disabled",
            "liquidity_optimization": "applied" if request.enable_liquidity_management else "disabled",
            "tax_efficiency": "applied" if request.enable_tax_optimization else "disabled"
        }
        
        # Generate comprehensive recommendations
        recommendations = []
        
        if ml_insights:
            portfolio_insights = ml_insights.get("portfolio_insights", {})
            if portfolio_insights:
                market_sentiment = portfolio_insights.get("market_sentiment", "neutral")
                recommendations.append(f"ML analysis indicates {market_sentiment} market sentiment")
                
                avg_confidence = portfolio_insights.get("avg_confidence", 0.5)
                if avg_confidence > 0.7:
                    recommendations.append("High confidence in ML predictions - consider implementing recommendations")
                elif avg_confidence < 0.4:
                    recommendations.append("Low confidence in ML predictions - maintain conservative approach")
        
        if allocation_changes:
            major_changes = [symbol for symbol, change in allocation_changes.items() if abs(change["change"]) > 0.05]
            if major_changes:
                recommendations.append(f"Major allocation changes recommended for: {', '.join(major_changes)}")
        
        # Performance projections
        performance_projection = {
            "expected_annual_return": sum(
                final_allocation.get(symbol, 0) * ml_insights.get("final_recommendations", {}).get("individual_recommendations", {}).get(symbol, {}).get("expected_return", 0.08)
                for symbol in request.tickers
            ) if ml_insights else 0.08,
            "projected_sharpe_ratio": optimization_result.get("metrics", {}).get("sharpe_ratio", 1.0) if optimization_result else 1.0,
            "risk_score": "moderate",
            "ml_confidence_score": ai_insights["ml_model_confidence"]
        }
        
        logger.info("AI-powered portfolio management completed successfully")
        
        return {
            "ai_insights": ai_insights,
            "final_allocation": final_allocation,
            "original_allocation": request.base_allocation,
            "allocation_changes": allocation_changes,
            "ml_training_result": ml_training_result,
            "ml_insights": ml_insights,
            "optimization_result": optimization_result,
            "risk_management": risk_management_result,
            "liquidity_management": liquidity_result,
            "tax_optimization": tax_optimization_result,
            "performance_projection": performance_projection,
            "recommendations": recommendations,
            "processing_summary": {
                "ml_enabled": ML_ENABLED,
                "models_trained": request.train_models,
                "ml_predictions_generated": request.enable_ml_predictions,
                "regime_analysis_performed": request.enable_regime_analysis,
                "rl_optimization_used": request.enable_rl_optimization,
                "optimization_applied": True,
                "risk_management_applied": request.enable_risk_management,
                "liquidity_management_applied": request.enable_liquidity_management,
                "tax_optimization_applied": request.enable_tax_optimization
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in AI-powered portfolio management: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to complete AI-powered portfolio management: {str(e)}"
        )

@app.get("/portfolio-management-capabilities")
async def get_portfolio_management_capabilities():
    """Get information about all available portfolio management features."""
    capabilities = {
        "optimization_methods": {
            method.value: f"Advanced {method.value.replace('_', ' ').title()} optimization"
            for method in OptimizationMethod
        },
        "risk_management": {
            "volatility_position_sizing": "Adjusts position sizes based on asset volatility",
            "tail_risk_hedging": "Implements hedging during high-risk periods",
            "drawdown_controls": "Reduces exposure after significant losses",
            "comprehensive_risk_management": "Integrates all risk management strategies"
        },
        "liquidity_management": {
            "cash_buffer_optimization": "Dynamic cash allocation based on market conditions",
            "rebalancing_frequency": "Adaptive rebalancing based on market volatility",
            "liquidity_scoring": "Asset liquidity assessment for stress scenarios",
            "liquidity_aware_allocation": "Portfolio allocation considering liquidity constraints"
        },
        "tax_efficiency": {
            "tax_loss_harvesting": "Systematic realization of losses for tax benefits",
            "asset_location_optimization": "Optimal placement of assets across account types",
            "tax_aware_rebalancing": "Rebalancing strategies that minimize tax impact",
            "tax_alpha_calculation": "Measurement of value added through tax management"
        },
        "account_types": [account.value for account in AccountType],
        "market_conditions": [condition.value for condition in MarketCondition],
        "rebalancing_frequencies": [freq.value for freq in RebalanceFrequency],
        "comprehensive_management": {
            "description": "Integrated portfolio management combining all features",
            "capabilities": [
                "Multi-objective optimization",
                "Dynamic risk management",
                "Liquidity-aware allocation",
                "Tax-efficient implementation"
            ]
        }
    }
    
    # Add ML capabilities if available
    if ML_ENABLED:
        capabilities["ml_intelligence"] = {
            "price_prediction": "ML models for short-term price movement forecasting",
            "regime_identification": "Clustering algorithms for market regime detection",
            "reinforcement_learning": "RL-based dynamic portfolio optimization",
            "ensemble_methods": "Combined ML approaches for robust predictions",
            "prediction_horizons": [horizon.value for horizon in PredictionHorizon],
            "model_types": [model.value for model in ModelType],
            "market_regimes": [regime.value for regime in MLMarketRegime]
        }
        capabilities["ai_powered_management"] = {
            "description": "Complete AI-driven portfolio management system",
            "features": [
                "ML-based price predictions",
                "Market regime analysis",
                "Reinforcement learning optimization",
                "Multi-objective optimization",
                "Dynamic risk management",
                "Liquidity-aware allocation",
                "Tax-efficient implementation"
            ]
        }
    else:
        capabilities["ml_intelligence"] = {
            "status": "Not available - ML libraries not installed",
            "install_requirements": ["scikit-learn", "tensorflow", "xgboost", "lightgbm"]
        }
    
    return capabilities

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000) 
