from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
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
        market_data = yf.download('^GSPC', start=data.index[0], end=data.index[-1])['Adj Close']
        market_returns = market_data.pct_change()
        
        # Align market returns with asset returns
        aligned_data = pd.concat([returns, market_returns], axis=1).dropna()
        asset_returns = aligned_data[returns.columns]
        market_returns = aligned_data[market_returns.name]
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(asset_returns.mean() * weight_arr) * 252
        portfolio_vol = np.sqrt(np.dot(weight_arr.T, np.dot(asset_returns.cov() * 252, weight_arr)))
        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        # Calculate individual asset metrics
        asset_returns_annual = asset_returns.mean() * 252
        asset_vols = asset_returns.std() * np.sqrt(252)
        
        # Calculate betas
        betas = {}
        for ticker in data.columns:
            covariance = np.cov(asset_returns[ticker], market_returns)[0,1]
            market_variance = np.var(market_returns)
            betas[ticker] = covariance / market_variance if market_variance > 0 else 0
        
        # Calculate portfolio returns for VaR and CVaR
        portfolio_returns = asset_returns.dot(weight_arr)
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
                    "annual_return": float(asset_returns_annual[ticker]),
                    "annual_volatility": float(asset_vols[ticker]),
                    "beta": float(betas[ticker]),
                    "weight": float(weights[ticker])
                } for ticker in data.columns
            }
        }
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
            
        # Validate inputs and ensure dimensions match
        if mu.shape[0] != n or S.shape[0] != n or S.shape[1] != n:
            raise ValueError(f"Dimension mismatch: mu shape {mu.shape}, S shape {S.shape}, expected shape ({n},) and ({n},{n})")
            
        # Validate no NaN or inf values
        if np.any(np.isnan(mu)) or np.any(np.isnan(S)) or np.any(np.isinf(mu)) or np.any(np.isinf(S)):
            raise ValueError("Invalid values (NaN or inf) in input data")
            
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
            except Exception as e:
                print(f"Error in objective function: {str(e)}")
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
        
        # Print debug information
        print(f"Initial weights shape: {w0.shape}")
        print(f"Crypto mask shape: {crypto_mask.shape}")
        print(f"Stock mask shape: {stock_mask.shape}")
        
        # Optimize with multiple attempts
        best_result = None
        min_objective = np.inf
        
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
                
                if result.success and result.fun < min_objective:
                    best_result = result
                    min_objective = result.fun
                
                # If not successful, try different initial weights
                w0 = np.random.dirichlet(np.ones(n))
                
            except Exception as e:
                print(f"Optimization attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:
                    raise
                continue
        
        if best_result is None:
            raise ValueError("Portfolio optimization failed after multiple attempts")
        
        # Clean up small weights
        weights = best_result.x
        weights[weights < 1e-4] = 0  # Set very small weights to zero
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)  # Renormalize
        
        # Calculate metrics
        portfolio_return = np.dot(weights, mu)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        # Print final results
        print(f"Optimization successful. Portfolio return: {portfolio_return:.4f}, risk: {portfolio_risk:.4f}, Sharpe: {sharpe_ratio:.4f}")
        
        return {
            'weights': dict(zip(tickers, weights)),
            'metrics': {
                'expected_return': float(portfolio_return),
                'volatility': float(portfolio_risk),
                'sharpe_ratio': float(sharpe_ratio)
            }
        }
        
    except Exception as e:
        print(f"Error in portfolio optimization: {str(e)}")
        raise ValueError(f"Error in portfolio optimization: {str(e)}")

def get_historical_performance(data: pd.DataFrame, weights: Dict[str, float]) -> Dict:
    """Calculate historical performance of the portfolio."""
    try:
        # Calculate daily returns for each asset
        returns = data.pct_change()
        
        # Calculate weighted returns
        weight_array = np.array([weights[ticker] for ticker in data.columns])
        portfolio_returns = returns.dot(weight_array)
        
        # Calculate cumulative returns
        portfolio_values = (1 + portfolio_returns).cumprod()
        
        # Calculate drawdown
        peak = portfolio_values.expanding(min_periods=1).max()
        drawdowns = (portfolio_values - peak) / peak
        
        # Replace inf and nan values
        portfolio_returns = portfolio_returns.replace([np.inf, -np.inf], np.nan).fillna(0)
        portfolio_values = portfolio_values.replace([np.inf, -np.inf], np.nan).fillna(1)
        drawdowns = drawdowns.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Convert timestamps to string format and ensure all values are Python native types
        dates = [d.strftime('%Y-%m-%d') for d in data.index]
        values = [float(x) for x in portfolio_values.values]
        drawdown_values = [float(x) for x in drawdowns.values]
        return_values = [float(x) for x in portfolio_returns.values]
        
        return {
            "dates": dates,
            "portfolio_values": values,
            "drawdowns": drawdown_values,
            "returns": return_values
        }
    except Exception as e:
        print(f"Error in get_historical_performance: {str(e)}")
        return {
            "dates": [],
            "portfolio_values": [],
            "drawdowns": [],
            "returns": []
        }

@app.post("/analyze-portfolio")
async def analyze_portfolio(request: Portfolio):
    try:
        # Validate request
        if len(request.tickers) < 2:
            raise HTTPException(status_code=400, detail="At least 2 tickers are required")

        print(f"Analyzing portfolio for tickers: {request.tickers}")
        
        # Download and align data
        data = pd.DataFrame()
        for ticker in request.tickers:
            print(f"Downloading data for {ticker}...")
            stock_data = yf.download(ticker, start=request.start_date)['Adj Close']
            data[ticker] = stock_data

        if data.empty:
            raise HTTPException(status_code=400, detail="No data available for the selected tickers")

        print(f"Data shape after alignment: {data.shape}")
        
        # Calculate returns and covariance
        print("Calculating returns and covariance...")
        returns = data.pct_change().dropna()
        print(f"Returns shape: {returns.shape}")
        
        mu = expected_returns.mean_historical_return(data)
        print(f"Expected returns shape: {mu.shape}")
        
        S = risk_models.sample_cov(data)
        print(f"Covariance matrix shape: {S.shape}")
        
        # Optimize portfolio
        print(f"Number of tickers: {len(request.tickers)}")
        print("Optimizing portfolio...")
        
        # Initialize weights
        initial_weights = np.array([1/len(request.tickers)] * len(request.tickers))
        print(f"Initial weights shape: {initial_weights.shape}")
        
        # Create masks for different asset types
        crypto_mask = np.array([is_crypto(ticker) for ticker in request.tickers])
        print(f"Crypto mask shape: {crypto_mask.shape}")
        
        stock_mask = ~crypto_mask
        print(f"Stock mask shape: {stock_mask.shape}")
        
        # Optimize based on risk tolerance
        ef = EfficientFrontier(mu, S)
        
        if request.risk_tolerance == "low":
            weights = ef.min_volatility()
        elif request.risk_tolerance == "high":
            weights = ef.max_sharpe()
        else:  # medium
            weights = ef.efficient_risk(target_volatility=0.2)
            
        cleaned_weights = ef.clean_weights()
        perf = ef.portfolio_performance()
        print(f"Optimization successful. Portfolio return: {perf[0]:.4f}, risk: {perf[1]:.4f}, Sharpe: {perf[2]:.4f}")
        
        # Format allocations as a list of dictionaries
        allocations = [
            {"ticker": ticker, "weight": float(weight)}
            for ticker, weight in cleaned_weights.items()
        ]
        
        # Get historical performance
        portfolio_values = (1 + returns.dot(pd.Series(cleaned_weights))).cumprod()
        dates = returns.index.strftime('%Y-%m-%d').tolist()
        values = portfolio_values.values.tolist()
        
        # Calculate drawdowns
        peak = portfolio_values.expanding(min_periods=1).max()
        drawdowns = (portfolio_values - peak) / peak
        
        # Get additional metrics for each asset
        asset_metrics = {}
        for ticker in request.tickers:
            ticker_returns = returns[ticker]
            asset_metrics[ticker] = {
                "beta": float(ticker_returns.cov(returns.dot(pd.Series(cleaned_weights)))) / float(returns.dot(pd.Series(cleaned_weights)).var()),
                "alpha": float(mu[ticker] - 0.02),  # Using 2% risk-free rate
                "correlation": float(ticker_returns.corr(returns.dot(pd.Series(cleaned_weights))))
            }
        
        # Get AI insights
        ai_service = AIAdvisorService()
        ai_insights = await ai_service.get_portfolio_metrics({
            "weights": cleaned_weights,
            "historical_data": {
                "returns": returns.to_dict()
            }
        })
        
        return {
            "allocation": allocations,
            "metrics": {
                "expected_return": float(perf[0]),
                "volatility": float(perf[1]),
                "sharpe_ratio": float(perf[2])
            },
            "historical_performance": {
                "dates": dates,
                "portfolio_values": values,
                "drawdowns": drawdowns.values.tolist()
            },
            "asset_metrics": asset_metrics,
            "ai_insights": ai_insights
        }
        
    except Exception as e:
        print(f"Error in portfolio analysis: {str(e)}")
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

        # Initialize Alpaca client
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        trading_client = TradingClient(api_key, secret_key, paper=True)

        # Get account information
        account = trading_client.get_account()
        equity = float(account.equity)
        print(f"Account equity: ${equity}")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 