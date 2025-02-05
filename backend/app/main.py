from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns, objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation
from dotenv import load_dotenv
from alpaca_trade_api import REST
from scipy.optimize import minimize
from app.services.ai_advisor_service import AIAdvisorService

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
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

class TickerPreferences(BaseModel):
    risk_tolerance: str = "medium"
    investment_horizon: str = "long_term"
    sectors: Optional[List[str]] = None
    market_cap: Optional[str] = None

def calculate_portfolio_metrics(data: pd.DataFrame, weights: Dict[str, float]) -> Dict:
    """Calculate additional portfolio metrics."""
    returns = data.pct_change()
    
    # Convert weights dict to array
    weight_arr = np.array([weights[ticker] for ticker in data.columns])
    
    # Calculate portfolio metrics
    portfolio_return = np.sum(returns.mean() * weight_arr) * 252
    portfolio_vol = np.sqrt(np.dot(weight_arr.T, np.dot(returns.cov() * 252, weight_arr)))
    sharpe_ratio = portfolio_return / portfolio_vol
    
    # Calculate individual asset metrics
    asset_returns = returns.mean() * 252
    asset_vols = returns.std() * np.sqrt(252)
    
    # Calculate beta
    market_data = yf.download('^GSPC', start=data.index[0], end=data.index[-1])['Adj Close']
    market_returns = market_data.pct_change()
    betas = {}
    for ticker in data.columns:
        covariance = np.cov(returns[ticker].dropna(), market_returns.dropna())[0,1]
        market_variance = np.var(market_returns.dropna())
        betas[ticker] = covariance / market_variance
    
    # Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR)
    portfolio_returns = returns.dot(weight_arr)
    var_95 = np.percentile(portfolio_returns, 5)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    
    return {
        "portfolio_metrics": {
            "expected_annual_return": float(portfolio_return),
            "annual_volatility": float(portfolio_vol),
            "sharpe_ratio": float(sharpe_ratio),
            "value_at_risk_95": float(var_95),
            "conditional_var_95": float(cvar_95)
        },
        "asset_metrics": {
            ticker: {
                "annual_return": float(asset_returns[ticker]),
                "annual_volatility": float(asset_vols[ticker]),
                "beta": float(betas[ticker]),
                "weight": float(weights[ticker])
            } for ticker in data.columns
        }
    }

def is_crypto(ticker: str) -> bool:
    """Check if the ticker is a cryptocurrency."""
    return ticker.endswith('-USD') or ticker.endswith('USDT')

def get_asset_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.Series:
    try:
        if is_crypto(ticker):
            data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
        else:
            data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading data for {ticker}: {str(e)}")

def optimize_portfolio(mu: np.ndarray, S: np.ndarray, risk_tolerance: str, tickers: List[str]) -> dict:
    """Optimize portfolio weights based on risk tolerance."""
    try:
        n = len(tickers)
        if n < 2:
            raise ValueError("Need at least 2 assets for optimization")
            
        # Validate inputs
        if np.any(np.isnan(mu)) or np.any(np.isnan(S)):
            raise ValueError("Invalid values in input data")
            
        # Add small constant to diagonal of covariance matrix for numerical stability
        S = S + np.eye(n) * 1e-6
        
        # Define risk weights based on tolerance
        risk_weights = {
            "low": {"crypto_max": 0.1, "stock_min": 0.6},
            "medium": {"crypto_max": 0.2, "stock_min": 0.4},
            "high": {"crypto_max": 0.4, "stock_min": 0.2}
        }
        
        if risk_tolerance not in risk_weights:
            raise ValueError(f"Invalid risk tolerance: {risk_tolerance}")
            
        weights = risk_weights[risk_tolerance]
        
        # Create masks for crypto and stock assets
        crypto_mask = np.array([1 if is_crypto(ticker) else 0 for ticker in tickers])
        stock_mask = np.array([1 if not is_crypto(ticker) else 0 for ticker in tickers])
        
        def objective(w):
            try:
                portfolio_return = np.dot(w, mu)
                portfolio_risk = np.sqrt(np.dot(w.T, np.dot(S, w)))
                
                # Handle division by zero
                if portfolio_risk < 1e-8:
                    return -portfolio_return * 1e8  # Large penalty for zero risk
                return -portfolio_return / portfolio_risk
            except Exception:
                return np.inf  # Return large value for invalid solutions
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # weights sum to 1
            {'type': 'ineq', 'fun': lambda w: weights["crypto_max"] - np.dot(w, crypto_mask)},  # max crypto allocation
            {'type': 'ineq', 'fun': lambda w: np.dot(w, stock_mask) - weights["stock_min"]},  # min stock allocation
            {'type': 'ineq', 'fun': lambda w: w}  # non-negative weights
        ]
        
        # Initial guess: equal weights
        w0 = np.array([1/n] * n)
        
        # Optimize with multiple attempts
        for attempt in range(3):
            try:
                result = minimize(
                    objective,
                    w0,
                    method='SLSQP',
                    constraints=constraints,
                    bounds=[(0, 1) for _ in range(n)],
                    options={'ftol': 1e-8, 'maxiter': 1000}
                )
                
                if result.success:
                    break
                    
                # If failed, try different initial weights
                w0 = np.random.dirichlet(np.ones(n))
                
            except Exception:
                if attempt == 2:
                    raise
                continue
        
        if not result.success:
            raise ValueError("Portfolio optimization failed after multiple attempts")
        
        # Clean up small weights
        weights = result.x
        weights[weights < 1e-4] = 0  # Set very small weights to zero
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)  # Renormalize
        
        # Calculate metrics
        portfolio_return = np.dot(weights, mu)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        return {
            'weights': dict(zip(tickers, weights)),
            'metrics': {
                'expected_return': float(portfolio_return),
                'volatility': float(portfolio_risk),
                'sharpe_ratio': float(sharpe_ratio)
            }
        }
        
    except Exception as e:
        raise ValueError(f"Error in portfolio optimization: {str(e)}")

@app.post("/analyze-portfolio")
async def analyze_portfolio(request: Portfolio):
    try:
        # Validate input
        if not request.tickers:
            raise HTTPException(status_code=400, detail="No tickers provided")
        if len(request.tickers) < 2:
            raise HTTPException(status_code=400, detail="Please provide at least 2 tickers")
        if not request.start_date:
            raise HTTPException(status_code=400, detail="No start date provided")
        if request.risk_tolerance not in ["low", "medium", "high"]:
            raise HTTPException(status_code=400, detail="Invalid risk tolerance. Must be 'low', 'medium', or 'high'")
            
        # Convert start_date to datetime with error handling
        try:
            start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
            if start_date > datetime.now():
                raise HTTPException(status_code=400, detail="Start date cannot be in the future")
            if start_date < datetime.now() - timedelta(days=3650):  # 10 years
                raise HTTPException(status_code=400, detail="Start date cannot be more than 10 years in the past")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
            
        end_date = datetime.now()
        
        # Download data for all assets with progress tracking
        data = {}
        min_start_date = None
        max_end_date = None
        
        # Validate each ticker and download data
        invalid_tickers = []
        for ticker in request.tickers:
            try:
                print(f"Downloading data for {ticker}...")  # Debug log
                series = get_asset_data(ticker, start_date, end_date)
                if not series.empty:
                    data[ticker] = series
                    # Update common date range
                    if min_start_date is None or series.index[0] > min_start_date:
                        min_start_date = series.index[0]
                    if max_end_date is None or series.index[-1] < max_end_date:
                        max_end_date = series.index[-1]
                else:
                    invalid_tickers.append(ticker)
            except Exception as e:
                print(f"Error downloading {ticker}: {str(e)}")  # Debug log
                invalid_tickers.append(ticker)
                
        if not data:
            raise HTTPException(status_code=400, detail="Could not fetch data for any of the provided tickers")
        if invalid_tickers:
            raise HTTPException(status_code=400, detail=f"Invalid or unavailable tickers: {', '.join(invalid_tickers)}")
        
        # Create DataFrame with aligned data
        df = pd.DataFrame(data)
        print(f"Data shape: {df.shape}")  # Debug log
        
        # Ensure we have enough data points
        if len(df) < 30:  # Minimum 30 days of data
            raise HTTPException(status_code=400, detail="Insufficient historical data. Need at least 30 days.")
        
        # Slice to common date range and handle missing data
        df = df.loc[min_start_date:max_end_date]
        df = df.resample('B').ffill(limit=5).bfill().interpolate()
        df = df.dropna()
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No valid data available after processing")

        # Calculate returns and covariance with error handling
        returns = df.pct_change().dropna()
        if len(returns) < 2:
            raise HTTPException(status_code=400, detail="Insufficient data for returns calculation")
            
        try:
            print("Calculating returns and covariance...")  # Debug log
            mu = returns.mean().values * 252  # Annualized returns
            S = returns.cov().values * 252  # Annualized covariance
            
            # Check for invalid values
            if np.any(np.isnan(mu)) or np.any(np.isnan(S)):
                raise HTTPException(status_code=400, detail="Invalid values in returns calculation")
                
            # Optimize portfolio
            print("Optimizing portfolio...")  # Debug log
            result = optimize_portfolio(mu, S, request.risk_tolerance, list(data.keys()))
            
            # Get latest prices
            latest_prices = df.iloc[-1]
            
            # Prepare response
            allocation = []
            for ticker, weight in result['weights'].items():
                allocation.append({
                    "ticker": ticker,
                    "weight": weight,
                    "type": "crypto" if is_crypto(ticker) else "stock",
                    "latest_price": float(latest_prices[ticker])
                })
            
            return {
                "allocation": allocation,
                "metrics": result['metrics']
            }
            
        except Exception as e:
            print(f"Error in optimization: {str(e)}")  # Debug log
            raise HTTPException(status_code=400, detail=f"Error in portfolio optimization: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")  # Debug log
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rebalance-portfolio")
async def rebalance_portfolio(allocation: PortfolioAllocation):
    try:
        # Initialize Alpaca API client
        alpaca_api_key = os.getenv("ALPACA_API_KEY")
        alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not alpaca_api_key or not alpaca_secret_key:
            raise HTTPException(
                status_code=400,
                detail="Alpaca API credentials not configured"
            )

        alpaca = REST(
            alpaca_api_key,
            alpaca_secret_key,
            base_url='https://paper-api.alpaca.markets'
        )

        # Get account information
        account = alpaca.get_account()
        equity = float(account.equity)

        # Get current positions
        positions = {p.symbol: p for p in alpaca.list_positions()}

        # Process each allocation
        orders = []
        for symbol, target_allocation in allocation.allocations.items():
            # Calculate target value
            target_value = equity * target_allocation
            
            # Get current position if exists
            current_position = positions.get(symbol)
            
            if current_position:
                current_value = float(current_position.market_value)
                diff_value = target_value - current_value
                
                if abs(diff_value) > 1:  # Only trade if difference is significant
                    # Get current price
                    latest_trade = alpaca.get_latest_trade(symbol)
                    price = float(latest_trade.price)
                    
                    # Calculate quantity to trade
                    qty = abs(int(diff_value / price))
                    
                    if qty > 0:
                        side = 'buy' if diff_value > 0 else 'sell'
                        orders.append({
                            "symbol": symbol,
                            "qty": qty,
                            "side": side
                        })
            else:
                # New position
                latest_trade = alpaca.get_latest_trade(symbol)
                price = float(latest_trade.price)
                qty = int(target_value / price)
                
                if qty > 0:
                    orders.append({
                        "symbol": symbol,
                        "qty": qty,
                        "side": "buy"
                    })

        # Execute orders
        executed_orders = []
        for order in orders:
            try:
                executed_order = alpaca.submit_order(
                    symbol=order["symbol"],
                    qty=order["qty"],
                    side=order["side"],
                    type='market',
                    time_in_force='gtc'
                )
                executed_orders.append({
                    **order,
                    "status": "executed",
                    "order_id": executed_order.id
                })
            except Exception as e:
                executed_orders.append({
                    **order,
                    "status": "failed",
                    "error": str(e)
                })

        return {
            "message": "Portfolio rebalancing completed",
            "orders": executed_orders,
            "account_balance": {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 