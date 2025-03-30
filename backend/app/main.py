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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000) 
