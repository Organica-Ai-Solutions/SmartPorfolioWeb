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
import cvxpy as cp

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
    """Optimize portfolio based on risk tolerance using PyPortfolioOpt."""
    try:
        print(f"Starting optimization for {len(tickers)} tickers with {risk_tolerance} risk tolerance")
        
        # Initialize EfficientFrontier
        ef = EfficientFrontier(mu, S)
        
        # Set weight constraints based on number of assets
        n_assets = len(tickers)
        if n_assets == 2:
            min_weight = 0.01  # 1% minimum
            max_weight = 0.99  # 99% maximum
        else:
            min_weight = 0.05  # 5% minimum
            max_weight = 0.60  # 60% maximum
            
        ef.add_constraint(lambda x: x >= min_weight)
        ef.add_constraint(lambda x: x <= max_weight)
        
        try:
            # Calculate minimum achievable volatility
            min_vol_portfolio = EfficientFrontier(mu, S)
            min_vol_portfolio.add_constraint(lambda x: x >= min_weight)
            min_vol_portfolio.add_constraint(lambda x: x <= max_weight)
            min_vol_portfolio.min_volatility()
            min_vol = min_vol_portfolio.portfolio_performance()[1]
            print(f"Minimum achievable volatility: {min_vol}")
            
            # Optimize based on risk tolerance
            if risk_tolerance == "low":
                weights = ef.min_volatility()
            elif risk_tolerance == "high":
                weights = ef.max_sharpe(risk_free_rate=0.02)
            else:  # medium
                # Set target volatility 30% higher than minimum
                target_vol = min_vol * 1.3
                print(f"Setting target volatility to: {target_vol}")
                try:
                    weights = ef.efficient_risk(target_volatility=target_vol)
                except Exception as e:
                    print(f"Error in efficient_risk: {str(e)}")
                    # Fallback to maximum Sharpe ratio if efficient_risk fails
                    weights = ef.max_sharpe(risk_free_rate=0.02)
            
            # Clean and format weights
            cleaned_weights = ef.clean_weights()
            weights_dict = dict(zip(tickers, cleaned_weights.values()))
            
            # Calculate portfolio metrics
            perf = ef.portfolio_performance()
            expected_return, portfolio_risk, sharpe = perf
            
            return {
                "weights": weights_dict,
                "metrics": {
                    "expected_return": float(expected_return),
                    "volatility": float(portfolio_risk),
                    "sharpe_ratio": float(sharpe)
                }
            }
            
        except Exception as optimization_error:
            print(f"Optimization error: {str(optimization_error)}")
            # Fallback to equal weights if optimization fails
            equal_weights = {ticker: 1.0/n_assets for ticker in tickers}
            weights_array = np.array(list(equal_weights.values()))
            
            # Calculate metrics for equal weights
            exp_return = float(mu.T @ weights_array)
            vol = float(np.sqrt(weights_array.T @ S @ weights_array))
            sharpe = (exp_return - 0.02) / vol if vol > 0 else 0
            
            return {
                "weights": equal_weights,
                "metrics": {
                    "expected_return": exp_return,
                    "volatility": vol,
                    "sharpe_ratio": sharpe
                }
            }
            
    except Exception as e:
        print(f"Error in portfolio optimization: {str(e)}")
        # Return equal weights as ultimate fallback
        equal_weights = {ticker: 1.0/len(tickers) for ticker in tickers}
        return {
            "weights": equal_weights,
            "metrics": {
                "expected_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0
            }
        }

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
        # Validate request
        if len(request.tickers) < 2:
            raise HTTPException(status_code=400, detail="At least 2 tickers are required")

        print(f"Analyzing portfolio for tickers: {request.tickers}")
        
        # Download market data - use crypto-specific index for crypto-heavy portfolios
        print("Downloading market data...")
        crypto_count = sum(1 for ticker in request.tickers if is_crypto(ticker))
        if crypto_count > len(request.tickers) / 2:
            # If more than half are crypto, use BTC as the market benchmark
            spy_data = yf.download('BTC-USD', start=request.start_date)['Adj Close']
            market_symbol = 'BTC-USD'
        else:
            # Otherwise use S&P 500
            spy_data = yf.download('^GSPC', start=request.start_date)['Adj Close']
            market_symbol = '^GSPC'
            
        market_data = pd.DataFrame()
        market_data[market_symbol] = spy_data
        
        # Download and validate data for each ticker
        data = pd.DataFrame()
        for ticker in request.tickers:
            print(f"Downloading data for {ticker}...")
            stock_data = yf.download(ticker, start=request.start_date)['Adj Close']
            if stock_data.empty:
                raise HTTPException(status_code=400, detail=f"No data available for {ticker}")
            data[ticker] = stock_data

        if data.empty:
            raise HTTPException(status_code=400, detail="No data available for the selected tickers")

        # Calculate returns and covariance
        returns = data.pct_change().dropna()
        market_returns = market_data.pct_change().dropna()
        
        if len(returns) < 2:
            raise HTTPException(status_code=400, detail="Insufficient data for analysis")
        
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
        
        # Optimize portfolio
        optimization_result = optimize_portfolio(mu, S, request.risk_tolerance, request.tickers)
        cleaned_weights = optimization_result["weights"]
        
        # Convert weights to series for calculations
        weights_series = pd.Series(cleaned_weights)
        
        # Calculate portfolio returns and metrics
        portfolio_returns = returns.dot(weights_series)
        
        # Calculate market beta
        market_beta = portfolio_returns.cov(market_returns[market_symbol]) / market_returns[market_symbol].var()
        
        # Calculate rolling metrics
        rolling_window = min(30, len(returns) - 1)
        rolling_vol = portfolio_returns.rolling(window=rolling_window).std() * np.sqrt(252)
        rolling_returns = portfolio_returns.rolling(window=rolling_window).mean() * 252
        rolling_sharpe = (rolling_returns / rolling_vol).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Calculate cumulative returns and drawdowns
        portfolio_values = (1 + portfolio_returns).cumprod()
        market_values = (1 + market_returns[market_symbol]).cumprod()
        
        # Clean up any inf/nan values
        portfolio_values = portfolio_values.replace([np.inf, -np.inf], np.nan).ffill().fillna(1)
        market_values = market_values.replace([np.inf, -np.inf], np.nan).ffill().fillna(1)
        relative_perf = (portfolio_values / market_values).replace([np.inf, -np.inf], np.nan).fillna(1)
        
        # Calculate drawdowns
        peak = portfolio_values.expanding(min_periods=1).max()
        drawdowns = ((portfolio_values - peak) / peak).replace([np.inf, -np.inf], np.nan).fillna(0)
        max_drawdown = float(drawdowns.min())
        
        # Calculate risk metrics
        returns_no_nan = portfolio_returns.replace([np.inf, -np.inf], np.nan).dropna()
        var_95 = float(np.percentile(returns_no_nan, 5))
        cvar_95 = float(returns_no_nan[returns_no_nan <= var_95].mean())
        
        # Calculate Sortino ratio
        negative_returns = portfolio_returns[portfolio_returns < 0]
        sortino_ratio = float((portfolio_returns.mean() * 252) / (negative_returns.std() * np.sqrt(252))) if len(negative_returns) > 0 else 0
        
        # Format dates and prepare response
        dates = returns.index.strftime('%Y-%m-%d').tolist()
        
        # Validate historical performance data
        if len(dates) == 0 or len(portfolio_values) == 0:
            raise HTTPException(status_code=400, detail="Invalid or missing historical performance data")
            
        # Ensure all data arrays have the same length
        min_length = min(len(dates), len(portfolio_values), len(drawdowns), len(rolling_vol), len(rolling_sharpe))
        dates = dates[:min_length]
        portfolio_values = portfolio_values.values[:min_length]
        drawdowns = drawdowns.values[:min_length]
        rolling_vol = rolling_vol.values[:min_length]
        rolling_sharpe = rolling_sharpe.values[:min_length]
        market_values = market_values.values[:min_length]
        relative_perf = relative_perf.values[:min_length]

        response = {
            "allocation": [
                {"ticker": ticker, "weight": float(weight)}
                for ticker, weight in cleaned_weights.items()
            ],
            "metrics": {
                "expected_return": float(optimization_result["metrics"]["expected_return"]),
                "volatility": float(optimization_result["metrics"]["volatility"]),
                "sharpe_ratio": float(optimization_result["metrics"]["sharpe_ratio"]),
                "sortino_ratio": float(sortino_ratio),
                "beta": float(market_beta),
                "max_drawdown": float(max_drawdown),
                "var_95": float(var_95),
                "cvar_95": float(cvar_95)
            },
            "historical_performance": {
                "dates": dates,
                "portfolio_values": [float(x) for x in portfolio_values],
                "drawdowns": [float(x) for x in drawdowns],
                "rolling_volatility": [float(x) for x in rolling_vol],
                "rolling_sharpe": [float(x) for x in rolling_sharpe]
            },
            "market_comparison": {
                "dates": dates,
                "market_values": [float(x) for x in market_values],
                "relative_performance": [float(x) for x in relative_perf]
            }
        }
        
        # Add AI insights
        try:
            ai_advisor = AIAdvisorService()
            portfolio_data = {
                "weights": cleaned_weights,
                "historical_data": {
                    "returns": returns.to_dict(),
                    "dates": dates
                },
                "metrics": response["metrics"],
                "market_comparison": response["market_comparison"]
            }
            
            ai_insights = await ai_advisor.get_portfolio_metrics(portfolio_data)
            response["ai_insights"] = ai_insights
        except Exception as ai_error:
            print(f"Error getting AI insights: {str(ai_error)}")
            # Provide default AI insights if there's an error
            response["ai_insights"] = {
                "explanations": {
                    "summary": {
                        "en": "Portfolio analysis completed successfully. Consider reviewing the metrics for detailed insights.",
                        "es": "Análisis de portafolio completado con éxito. Considere revisar las métricas para obtener información detallada."
                    },
                    "risk_analysis": {
                        "en": "Risk metrics indicate the portfolio's current risk profile.",
                        "es": "Las métricas de riesgo indican el perfil de riesgo actual del portafolio."
                    },
                    "diversification_analysis": {
                        "en": "Review asset allocation for optimal diversification.",
                        "es": "Revise la asignación de activos para una diversificación óptima."
                    },
                    "market_context": {
                        "en": "Consider current market conditions when making investment decisions.",
                        "es": "Considere las condiciones actuales del mercado al tomar decisiones de inversión."
                    },
                    "stress_test_interpretation": {
                        "en": "Evaluate portfolio resilience under various market scenarios.",
                        "es": "Evalúe la resiliencia del portafolio bajo varios escenarios de mercado."
                    }
                }
            }
        
        # Clean all numeric values before returning
        return clean_numeric_values(response)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in analyze_portfolio: {str(e)}")
        # Return a valid response with default values
        return clean_numeric_values({
            "allocation": [
                {"ticker": ticker, "weight": 1.0/len(request.tickers)}
                for ticker in request.tickers
            ],
            "metrics": {
                "expected_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "beta": 0.0,
                "max_drawdown": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0
            },
            "historical_performance": {
                "dates": [],
                "portfolio_values": [],
                "drawdowns": [],
                "rolling_volatility": [],
                "rolling_sharpe": []
            },
            "market_comparison": {
                "dates": [],
                "market_values": [],
                "relative_performance": []
            }
        })

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