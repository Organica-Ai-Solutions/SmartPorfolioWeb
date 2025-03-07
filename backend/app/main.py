from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
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
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from app.services.ai_advisor_service import AIAdvisorService
import cvxpy as cp
import time
import math

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://organica-ai-solutions.github.io",  # GitHub Pages domain
        "http://localhost:5173",  # Local development
        "http://127.0.0.1:5173",  # Local development alternative
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Portfolio(BaseModel):
    id: Optional[int] = None
    user_id: Optional[int] = None
    name: Optional[str] = None
    tickers: List[str]
    start_date: str
    risk_tolerance: str = "medium"  # 'low', 'medium', 'high'

class PortfolioAllocation(BaseModel):
    allocations: Dict[str, float]
    alpaca_api_key: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    use_paper_trading: Optional[bool] = True

class TickerPreferences(BaseModel):
    risk_tolerance: str = "medium"
    investment_horizon: str = "long_term"
    sectors: Optional[List[str]] = None
    market_cap: Optional[str] = None
    risk_level: Optional[str] = None
    investment_style: Optional[str] = None

def calculate_portfolio_metrics(data: pd.DataFrame, weights: Dict[str, float]) -> Dict:
    """Calculate additional portfolio metrics."""
    try:
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Convert weights dict to array
        weight_arr = np.array([weights[ticker] for ticker in data.columns])
        
        # Get market data aligned with our date range
        try:
            market_data = yf.download('^GSPC', start=data.index[0], end=data.index[-1])
            if 'Adj Close' in market_data.columns:
                market_series = market_data['Adj Close']
            else:
                market_series = market_data['Close']
            market_returns = market_series.pct_change()
        except Exception as e:
            print(f"Error getting market data: {str(e)}. Using first asset as market proxy.")
            # Use the first asset as a proxy for the market
            market_returns = returns.iloc[:, 0]
        
        # Align market returns with asset returns
        aligned_data = pd.concat([returns, market_returns], axis=1).dropna()
        asset_returns = aligned_data[returns.columns]
        market_returns = aligned_data.iloc[:, -1]  # Last column is market returns
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(asset_returns.mean() * weight_arr) * 252
        portfolio_vol = np.sqrt(np.dot(weight_arr.T, np.dot(asset_returns.cov() * 252, weight_arr)))
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
            covariance = np.cov(asset_returns[ticker], market_returns)[0,1]
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
                "conditional_var_95": float(cvar_95)
            },
            "asset_metrics": {}
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
                    "weight": float(weights[ticker])
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
                "conditional_var_95": 0.0
            },
            "asset_metrics": {
                ticker: {
                    "annual_return": 0.0,
                    "annual_volatility": 0.0,
                    "beta": 0.0,
                    "weight": float(weights.get(ticker, 0.0))
                } for ticker in data.columns
            }
        }

def is_crypto(ticker: str) -> bool:
    """Check if the ticker is a cryptocurrency."""
    return ticker.endswith('-USD') or ticker.endswith('USDT')

def get_asset_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.Series:
    """Get asset data with special handling for crypto assets."""
    try:
        # For crypto assets, we don't need to adjust for market hours
        if is_crypto(ticker):
            data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
        else:
            # For stocks, we need to ensure we're only looking at market hours
            data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
            # Filter out weekend data for stocks
            data = data[data.index.dayofweek < 5]
            # Filter out data outside of market hours (9:30 AM - 4:00 PM EST)
            data = data[
                (data.index.time >= pd.Timestamp('09:30').time()) &
                (data.index.time <= pd.Timestamp('16:00').time())
            ]
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading data for {ticker}: {str(e)}")

def optimize_portfolio(mu: np.ndarray, S: np.ndarray, risk_tolerance: str, tickers: List[str]) -> dict:
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
        discrete_allocation = {
            "shares": {},
            "leftover": 0.0
        }
        
        # Try to calculate actual discrete allocation if possible
        try:
            latest_prices = {}
            for ticker in tickers:
                try:
                    ticker_data = yf.Ticker(ticker)
                    latest_prices[ticker] = ticker_data.history(period="1d")["Close"].iloc[-1]
                except Exception as price_error:
                    print(f"Error getting price for {ticker}: {str(price_error)}")
                    latest_prices[ticker] = 100.0  # Placeholder value
            
            # Calculate discrete allocation with $10,000 portfolio value
            if latest_prices:
                da = DiscreteAllocation(weights, pd.Series(latest_prices), total_portfolio_value=10000)
                allocation, leftover = da.greedy_portfolio()
                discrete_allocation = {
                    "shares": allocation,
                    "leftover": leftover
                }
        except Exception as alloc_error:
            print(f"Error calculating discrete allocation: {str(alloc_error)}")
            # Keep the default empty discrete allocation
            
            return {
            "weights": weights,
                "metrics": {
                    "expected_return": float(expected_return),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio)
            },
            "discrete_allocation": discrete_allocation
        }
    except Exception as e:
        print(f"Error in optimize_portfolio: {str(e)}")
        # Return equal weights as fallback
        equal_weights = {ticker: 1.0/len(tickers) for ticker in tickers}
        return {
            "weights": equal_weights,
            "metrics": {
                "expected_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0
            },
            "discrete_allocation": {
                "shares": {},
                "leftover": 10000.0
            }
        }

def get_risk_adjusted_equal_weights(tickers: List[str], mu: np.ndarray, S: np.ndarray) -> Dict[str, float]:
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
        return {ticker: 1.0/len(tickers) for ticker in tickers}

def calculate_portfolio_beta(mu: np.ndarray, S: np.ndarray, weights: Dict[str, float]) -> float:
    """Calculate portfolio beta using CAPM."""
    try:
        weights_array = np.array(list(weights.values()))
        portfolio_var = np.sqrt(weights_array.T @ S @ weights_array)
        market_var = np.sqrt(S[0,0])  # Assuming first asset is market proxy
        return portfolio_var / market_var if market_var > 0 else 1.0
    except Exception:
        return 1.0

def get_sector_constraints(tickers: List[str]) -> Dict[str, Tuple[List[int], Tuple[float, float]]]:
    """Get sector constraints for portfolio optimization."""
    try:
        # Define basic sector classifications
        tech_tickers = ['AAPL', 'GOOGL', 'MSFT', 'META', 'NVDA']
        finance_tickers = ['JPM', 'BAC', 'GS', 'MS', 'V']
        crypto_tickers = [t for t in tickers if t.endswith('-USD')]
        
        constraints = {}
        
        # Add tech sector constraints
        tech_indices = [i for i, t in enumerate(tickers) if t in tech_tickers]
        if tech_indices:
            constraints['tech'] = (tech_indices, (0.0, 0.4))  # Max 40% in tech
            
        # Add finance sector constraints
        finance_indices = [i for i, t in enumerate(tickers) if t in finance_tickers]
        if finance_indices:
            constraints['finance'] = (finance_indices, (0.0, 0.35))  # Max 35% in finance
            
        # Add crypto constraints
        crypto_indices = [i for i, t in enumerate(tickers) if t in crypto_tickers]
        if crypto_indices:
            constraints['crypto'] = (crypto_indices, (0.0, 0.30))  # Max 30% in crypto
            
        return constraints
    except Exception as e:
        print(f"Error creating sector constraints: {str(e)}")
        return {}

def create_portfolio_result(weights: np.ndarray, mu: np.ndarray, S: np.ndarray, tickers: List[str]) -> dict:
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
            "sharpe_ratio": sharpe
        }
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
                "rolling_sharpe": []
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
                "rolling_sharpe": []
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
        rolling_sharpe = (portfolio_returns.rolling(window=window).mean() - risk_free_rate) / \
                        portfolio_returns.rolling(window=window).std() * np.sqrt(252)
        
        # Replace NaN and infinite values
        portfolio_values = portfolio_values.replace([np.inf, -np.inf], np.nan).fillna(initial_investment)
        drawdowns = drawdowns.replace([np.inf, -np.inf], np.nan).fillna(0)
        rolling_vol = rolling_vol.replace([np.inf, -np.inf], np.nan).fillna(0)
        rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Convert to lists for JSON serialization
        dates = returns.index.strftime('%Y-%m-%d').tolist()
        portfolio_values_list = portfolio_values.tolist()
        drawdowns_list = drawdowns.tolist()
        rolling_vol_list = rolling_vol.tolist()
        rolling_sharpe_list = rolling_sharpe.tolist()
        
        # Final check for any remaining non-finite values
        portfolio_values_list = [0 if not np.isfinite(x) else x for x in portfolio_values_list]
        drawdowns_list = [0 if not np.isfinite(x) else x for x in drawdowns_list]
        rolling_vol_list = [0 if not np.isfinite(x) else x for x in rolling_vol_list]
        rolling_sharpe_list = [0 if not np.isfinite(x) else x for x in rolling_sharpe_list]
        
        return {
            "dates": dates,
            "portfolio_values": portfolio_values_list,
            "drawdowns": drawdowns_list,
            "rolling_volatility": rolling_vol_list,
            "rolling_sharpe": rolling_sharpe_list
        }
    except Exception as e:
        print(f"Error calculating historical performance: {str(e)}")
        # Return empty data with proper structure
        return {
            "dates": [],
            "portfolio_values": [],
            "drawdowns": [],
            "rolling_volatility": [],
            "rolling_sharpe": []
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
    try:
        print(f"Analyzing portfolio for tickers: {request.tickers}")
        
        # Validate input
        if not request.tickers:
            raise HTTPException(status_code=400, detail="No tickers provided")
        
        # Parse dates
        try:
            start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
            end_date = datetime.now()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Download data with retries
        max_retries = 3
        retry_delay = 1  # seconds
        data = pd.DataFrame()
        
        for attempt in range(max_retries):
            try:
                print("Downloading market data...")
                try:
                    data = yf.download(request.tickers, start=start_date, end=end_date)['Adj Close']
                except KeyError as e:
                    if str(e) == "'Adj Close'":
                        print("Adj Close not available, falling back to Close prices")
                        data = yf.download(request.tickers, start=start_date, end=end_date)['Close']
                    else:
                        raise e
                
                # Check if we got any data
        if data.empty:
                    raise ValueError("No data returned from Yahoo Finance")
                
                # If it's a Series (single ticker), convert to DataFrame
                if isinstance(data, pd.Series):
                    data = pd.DataFrame(data)
                    data.columns = [request.tickers[0]]
                
                # Check for missing tickers
                missing_tickers = [ticker for ticker in request.tickers if ticker not in data.columns]
                if missing_tickers:
                    print(f"Missing data for tickers: {missing_tickers}")
                    
                    # Try to download missing tickers individually
                    for ticker in missing_tickers:
                        print(f"Downloading data for {ticker}...")
                        try:
                            # Try Adj Close first, fall back to Close
                            try:
                                ticker_data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
                            except KeyError:
                                ticker_data = yf.download(ticker, start=start_date, end=end_date)['Close']
                                
                            if not ticker_data.empty:
                                data[ticker] = ticker_data
                        except Exception as e:
                            print(f"Failed to download {ticker}: {str(e)}")
                
                # If we still don't have data for all tickers, but have some data, proceed with what we have
                if not data.empty:
                    available_tickers = list(data.columns)
                    if available_tickers:
                        print(f"Proceeding with available tickers: {available_tickers}")
                        break
                
                # If we have no data at all, retry
                if attempt < max_retries - 1:
                    print(f"Retry {attempt + 1}/{max_retries} after {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise ValueError("Failed to download data after multiple attempts")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error downloading data: {str(e)}. Retrying {attempt + 1}/{max_retries}...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise ValueError(f"Failed to download data after {max_retries} attempts: {str(e)}")
        
        # Check if we have any data to work with
        if data.empty or len(data.columns) == 0:
            raise HTTPException(status_code=400, detail="Could not retrieve data for any of the provided tickers")
        
        # Fill missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        
        # Optimize portfolio based on risk tolerance
        result = optimize_portfolio(mu, S, request.risk_tolerance, list(data.columns))
        
        # Calculate additional portfolio metrics
        metrics = calculate_portfolio_metrics(data, result['weights'])
        
        # Get historical performance
        historical_performance = get_historical_performance(data, result['weights'])
        
        # Combine results
        response = {
            "allocations": result['weights'],
            "metrics": metrics['portfolio_metrics'],
            "asset_metrics": metrics['asset_metrics'],
            "discrete_allocation": result['discrete_allocation'],
            "historical_performance": historical_performance
        }
        
        # Clean numeric values to ensure JSON serialization works
        response = clean_numeric_values(response)
        
        return response
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in analyze_portfolio: {str(e)}")
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
                "sharpe_ratio": (return_factor - 0.02) / volatility_factor if volatility_factor > 0 else 0,
                "sortino_ratio": ((return_factor - 0.02) / volatility_factor) * 1.2 if volatility_factor > 0 else 0,
            }
        
        # Generate varied weights based on return/risk characteristics
        def generate_optimized_weights(tickers, asset_metrics):
            raw_scores = {}
            for ticker in tickers:
                metrics = asset_metrics[ticker]
                # Score based on a simplified risk-adjusted return formula
                score = (metrics["annual_return"] / max(0.01, metrics["volatility"])) + (0.5 - abs(metrics["beta"] - 1))
                raw_scores[ticker] = max(0.1, score)  # Ensure minimal weight
            
            # Normalize to sum to 1
            total_score = sum(raw_scores.values())
            return {ticker: score / total_score for ticker, score in raw_scores.items()}
        
        # Generate asset metrics
        asset_metrics = {ticker: generate_ticker_metrics(ticker) for ticker in request.tickers}
        
        # Generate optimized weights
        optimized_weights = generate_optimized_weights(request.tickers, asset_metrics)
        
        # Update asset metrics with weights
        for ticker, weight in optimized_weights.items():
            asset_metrics[ticker]["weight"] = weight
        
        # Calculate portfolio metrics
        portfolio_return = sum(metrics["annual_return"] * optimized_weights[ticker] for ticker, metrics in asset_metrics.items())
        
        # Calculate weighted average volatility (simplified; should use covariance matrix)
        portfolio_volatility = sum(metrics["volatility"] * optimized_weights[ticker] for ticker, metrics in asset_metrics.items())
        
        # Calculate other portfolio metrics
        portfolio_beta = sum(metrics["beta"] * optimized_weights[ticker] for ticker, metrics in asset_metrics.items())
        portfolio_sharpe = (portfolio_return - 0.02) / portfolio_volatility if portfolio_volatility > 0 else 0
        portfolio_sortino = portfolio_sharpe * 1.2  # Simplified
        portfolio_var = -portfolio_volatility * 1.65 / math.sqrt(252)
        portfolio_max_dd = min(metrics["max_drawdown"] for metrics in asset_metrics.values()) * 0.8  # Simplified
        
        # Generate sample dates
        sample_dates = ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"]
        # Sample portfolio values
        portfolio_values = [10000, 10200, 10400, 10300, 10500]
        # Generate S&P 500 benchmark values with slightly different performance
        sp500_values = [10000, 10150, 10300, 10250, 10380]
        # Calculate relative performance (percentage difference)
        relative_performance = [(p/s - 1) * 100 for p, s in zip(portfolio_values, sp500_values)]
        
        return {
            "allocations": optimized_weights,
            "metrics": {
                "expected_return": portfolio_return,
                "volatility": portfolio_volatility,
                "sharpe_ratio": portfolio_sharpe,
                "sortino_ratio": portfolio_sortino,
                "beta": portfolio_beta,
                "max_drawdown": portfolio_max_dd,
                "var_95": portfolio_var,
                "cvar_95": portfolio_var * 1.2  # Simplified
            },
            "asset_metrics": asset_metrics,
            "discrete_allocation": {
                "shares": {ticker: int(10000 * optimized_weights[ticker] / 100) for ticker in request.tickers},
                "leftover": 1000.0
            },
            "historical_performance": {
                "dates": sample_dates,
                "portfolio_values": portfolio_values,
                "drawdowns": [0, 0, 0, -0.01, 0],
                "rolling_volatility": [0.1, 0.12, 0.11, 0.13, 0.12],
                "rolling_sharpe": [0.5, 0.6, 0.55, 0.5, 0.6]
            },
            "market_comparison": {
                "dates": sample_dates,
                "market_values": sp500_values,
                "relative_performance": relative_performance
            }
        }
    except Exception as e:
        print(f"Error in analyze_portfolio_simple: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rebalance-portfolio")
async def rebalance_portfolio(allocation: PortfolioAllocation):
    try:
        # Validate allocations exist and are not empty
        if not allocation.allocations or not isinstance(allocation.allocations, dict):
            raise HTTPException(status_code=400, detail="Invalid allocation data: allocations must be a non-empty dictionary")

        # Convert allocations to float and validate
        formatted_allocations = {}
        for ticker, weight in allocation.allocations.items():
            try:
                weight_float = float(weight)
                if not (0 <= weight_float <= 1):
                    raise ValueError(f"Weight for {ticker} must be between 0 and 1")
                formatted_allocations[ticker] = weight_float
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail=f"Invalid weight value for {ticker}")

        # Validate total allocation
        total_allocation = sum(formatted_allocations.values())
        if not (0.99 <= total_allocation <= 1.01):
            raise HTTPException(
                status_code=400, 
                detail=f"Total allocation must sum to 1 (got {total_allocation:.4f})"
            )

        print(f"Received allocations: {formatted_allocations}")
        print(f"Total allocation: {total_allocation}")

        # Get API keys (prioritize keys from request, fall back to environment variables)
        api_key = allocation.alpaca_api_key or os.getenv("ALPACA_API_KEY")
        secret_key = allocation.alpaca_secret_key or os.getenv("ALPACA_SECRET_KEY")
        use_paper = allocation.use_paper_trading if allocation.use_paper_trading is not None else True
        
        # Validate API keys
        if not api_key or not secret_key:
            raise HTTPException(
                status_code=400,
                detail="Alpaca API keys are required. Please provide them in the request or set them as environment variables."
            )

        # Initialize Alpaca client
        try:
            trading_client = TradingClient(api_key, secret_key, paper=use_paper)

            # Test the connection by getting account info
        account = trading_client.get_account()
        equity = float(account.equity)
        print(f"Account equity: ${equity}")
        except Exception as e:
            raise HTTPException(
                status_code=401,
                detail=f"Failed to connect to Alpaca API: {str(e)}"
            )

        # Get current positions
        positions = {p.symbol: float(p.qty) for p in trading_client.get_all_positions()}
        print(f"Current positions: {positions}")

        # Calculate target position values
        target_positions = {
            symbol: equity * weight 
            for symbol, weight in formatted_allocations.items()
        }
        print(f"Target positions: {target_positions}")

        # Get current prices
        current_prices = {}
        for symbol in formatted_allocations.keys():
            try:
                ticker = yf.Ticker(symbol)
                current_price = ticker.history(period='1d')['Close'].iloc[-1]
                current_prices[symbol] = float(current_price)
                print(f"Current price for {symbol}: ${current_price:.2f}")
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error getting price for {symbol}: {str(e)}"
                )

        # Calculate required trades
        orders = []
        for symbol, target_value in target_positions.items():
            current_shares = positions.get(symbol, 0)
            target_shares = int(target_value / current_prices[symbol])
            
            if abs(target_shares - current_shares) > 0:
                side = OrderSide.BUY if target_shares > current_shares else OrderSide.SELL
                qty = abs(target_shares - current_shares)
                
                if qty > 0:  # Only create order if quantity is positive
                    order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        time_in_force=TimeInForce.DAY
                    )
                    
                    try:
                        order = trading_client.submit_order(order_data=order_data)
                        print(f"Order submitted for {symbol}: {side.value} {qty} shares")
                        orders.append({
                            "symbol": symbol,
                            "qty": qty,
                            "side": side.value,
                            "status": "executed",
                            "order_id": order.id
                        })
                    except Exception as e:
                        print(f"Error submitting order for {symbol}: {str(e)}")
                        orders.append({
                            "symbol": symbol,
                            "qty": qty,
                            "side": side.value,
                            "status": "failed",
                            "error": str(e)
                        })

        return {
            "message": "Portfolio rebalancing completed",
            "orders": orders,
            "account_balance": {
                "equity": equity,
                "cash": float(account.cash),
                "buying_power": float(account.buying_power)
            }
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error during rebalancing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during rebalancing: {str(e)}")

@app.post("/rebalance-portfolio-simple")
async def rebalance_portfolio_simple(allocation: PortfolioAllocation):
    """Simplified rebalance endpoint for testing."""
    try:
        # Validate allocations exist and are not empty
        if not allocation.allocations or not isinstance(allocation.allocations, dict):
            raise HTTPException(status_code=400, detail="Invalid allocation data: allocations must be a non-empty dictionary")

        # Convert allocations to float and validate
        formatted_allocations = {}
        for ticker, weight in allocation.allocations.items():
            try:
                weight_float = float(weight)
                if not (0 <= weight_float <= 1):
                    raise ValueError(f"Weight for {ticker} must be between 0 and 1")
                formatted_allocations[ticker] = weight_float
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail=f"Invalid weight value for {ticker}")

        # Validate total allocation
        total_allocation = sum(formatted_allocations.values())
        if not (0.99 <= total_allocation <= 1.01):
            raise HTTPException(
                status_code=400, 
                detail=f"Total allocation must sum to 1 (got {total_allocation:.4f})"
            )

        print(f"Received allocations: {formatted_allocations}")
        print(f"Total allocation: {total_allocation}")
        
        # Check if API keys are provided
        if not allocation.alpaca_api_key or not allocation.alpaca_secret_key:
            raise HTTPException(
                status_code=400,
                detail="Alpaca API keys are required. Please provide them in the request."
            )
        
        # Return mock response
        return {
            "message": "Portfolio rebalancing completed (simulated)",
            "orders": [
                {
                    "symbol": ticker,
                    "qty": int(10000 * weight / 100),  # Mock quantity
                    "side": "buy" if weight > 0.2 else "sell",
                    "status": "executed",
                    "order_id": f"mock-order-{ticker}"
                } for ticker, weight in formatted_allocations.items()
            ],
            "account_balance": {
                "equity": 10000.0,
                "cash": 2000.0,
                "buying_power": 4000.0
            }
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error during rebalancing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during rebalancing: {str(e)}")

@app.post("/get-ticker-suggestions")
async def get_ticker_suggestions(preferences: TickerPreferences):
    try:
        ai_advisor = AIAdvisorService()
        suggestions = await ai_advisor.get_portfolio_advice({
            "preferences": preferences.dict()
        })
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

async def get_portfolio_metrics(portfolio_data: Dict) -> Dict:
    """Get advanced portfolio metrics with AI insights."""
    try:
        # Convert timestamps to strings in historical data
        if 'historical_data' in portfolio_data and 'returns' in portfolio_data['historical_data']:
            returns_dict = {}
            for ticker, returns in portfolio_data['historical_data']['returns'].items():
                if isinstance(returns, pd.Series):
                    returns_dict[ticker] = {
                        date.strftime('%Y-%m-%d'): float(value)
                        for date, value in returns.items()
                        if not (np.isnan(value) or np.isinf(value))
                    }
                elif isinstance(returns, dict):
                    returns_dict[ticker] = {
                        str(date) if isinstance(date, pd.Timestamp) else str(date): float(value)
                        for date, value in returns.items()
                        if not (np.isnan(value) or np.isinf(value))
                    }
                else:
                    returns_dict[ticker] = returns
            portfolio_data['historical_data']['returns'] = returns_dict

        # Get AI insights
        ai_advisor = AIAdvisorService()
        metrics = await ai_advisor.get_portfolio_metrics(portfolio_data)
        
        # Clean up any potential inf/nan values in the response
        def clean_value(value):
            if isinstance(value, (float, np.float64)):
                if np.isnan(value) or np.isinf(value):
                    return 0.0
                return float(value)
            elif isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [clean_value(v) for v in value]
            return value
        
        cleaned_metrics = clean_value(metrics)
        
        return cleaned_metrics
        
    except Exception as e:
        print(f"Error getting portfolio metrics: {str(e)}")
        return {
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "market_regime": "normal",
            "recommendations": [
                "Unable to generate recommendations at this time",
                "Please try again later"
            ]
        }

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to check if the API is working."""
    return {"status": "ok", "message": "API is working"}

@app.post("/ai-portfolio-analysis")
async def ai_portfolio_analysis(request: Portfolio):
    """Get AI-powered analysis of a portfolio."""
    try:
        # First, perform regular portfolio analysis
        portfolio_data = await analyze_portfolio_simple(request)
        
        # Then, get AI insights
        ai_advisor = AIAdvisorService()
        ai_insights = await ai_advisor.get_portfolio_metrics(portfolio_data)
        
        # Combine the results
        result = {
            **portfolio_data,
            "ai_insights": ai_insights
        }
        
        return result
    except Exception as e:
        print(f"Error in AI portfolio analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in AI portfolio analysis: {str(e)}")

@app.post("/ai-rebalance-explanation")
async def ai_rebalance_explanation(allocation: PortfolioAllocation):
    """Get AI-powered explanation of a portfolio rebalance operation."""
    try:
        # First, perform regular rebalance operation
        rebalance_result = await rebalance_portfolio_simple(allocation)
        
        # Then, get AI insights
        ai_advisor = AIAdvisorService()
        
        # Prepare data for AI analysis
        analysis_data = {
            "allocations": allocation.allocations,
            "rebalance_result": rebalance_result,
            "market_data": {
                "current_regime": "normal",  # This would ideally come from market analysis
                "volatility": "medium",
                "trend": "neutral"
            }
        }
        
        # Get AI advice
        ai_insights = await ai_advisor.get_portfolio_advice(analysis_data)
        
        # Combine the results
        result = {
            **rebalance_result,
            "ai_insights": ai_insights
        }
        
        return result
    except Exception as e:
        print(f"Error in AI rebalance explanation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in AI rebalance explanation: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 