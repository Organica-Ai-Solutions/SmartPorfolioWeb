import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from enum import Enum
import asyncio
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# PyPortfolioOpt imports
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.objective_functions import L2_reg
import cvxpy as cp

logger = logging.getLogger(__name__)

class OptimizationMethod(str, Enum):
    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    FACTOR_BASED = "factor_based"
    MULTI_OBJECTIVE = "multi_objective"

class FactorType(str, Enum):
    VALUE = "value"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    SIZE = "size"
    PROFITABILITY = "profitability"
    INVESTMENT = "investment"
    LOW_VOLATILITY = "low_volatility"

class AdvancedOptimizationService:
    """Service for advanced portfolio optimization strategies."""
    
    def __init__(self):
        # Factor definitions for different asset classes
        self.stock_factors = {
            FactorType.VALUE: ["PE_ratio", "PB_ratio", "EV_EBITDA"],
            FactorType.MOMENTUM: ["price_momentum_12m", "earnings_momentum", "revenue_momentum"],
            FactorType.QUALITY: ["ROE", "ROA", "debt_to_equity", "current_ratio"],
            FactorType.SIZE: ["market_cap"],
            FactorType.PROFITABILITY: ["gross_margin", "operating_margin", "net_margin"],
            FactorType.INVESTMENT: ["capex_to_sales", "asset_turnover"],
            FactorType.LOW_VOLATILITY: ["volatility_12m", "beta", "max_drawdown"]
        }
        
        # Market cap and sector data cache
        self._stock_info_cache = {}
        self._cache_timestamp = None
        self._cache_expiry = timedelta(hours=6)  # Cache for 6 hours
    
    async def optimize_portfolio_advanced(
        self,
        tickers: List[str],
        method: OptimizationMethod = OptimizationMethod.BLACK_LITTERMAN,
        lookback_days: int = 252,
        risk_tolerance: str = "medium",
        views: Optional[Dict[str, float]] = None,
        factor_constraints: Optional[Dict[FactorType, Tuple[float, float]]] = None,
        target_return: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Advanced portfolio optimization using various sophisticated methods.
        
        Args:
            tickers: List of asset tickers
            method: Optimization method to use
            lookback_days: Historical data period
            risk_tolerance: Risk tolerance level
            views: Dict of ticker -> expected return views for Black-Litterman
            factor_constraints: Dict of factor -> (min, max) constraints
            target_return: Target portfolio return
            **kwargs: Additional method-specific parameters
        """
        try:
            logger.info(f"Starting advanced optimization with method: {method}")
            
            # Get historical data
            data = await self._get_historical_data(tickers, lookback_days)
            
            if data.empty:
                raise ValueError("No historical data available for optimization")
            
            # Calculate base statistics
            mu = expected_returns.mean_historical_return(data)
            S = risk_models.sample_cov(data)
            
            # Get factor data if needed
            factor_data = None
            if method == OptimizationMethod.FACTOR_BASED or factor_constraints:
                factor_data = await self._get_factor_data(tickers)
            
            # Run the specific optimization method
            if method == OptimizationMethod.BLACK_LITTERMAN:
                result = await self._black_litterman_optimization(
                    data, mu, S, tickers, views, risk_tolerance, **kwargs
                )
            elif method == OptimizationMethod.MIN_VOLATILITY:
                result = await self._minimum_variance_optimization(
                    mu, S, tickers, factor_constraints, factor_data, **kwargs
                )
            elif method == OptimizationMethod.FACTOR_BASED:
                result = await self._factor_based_optimization(
                    data, mu, S, tickers, factor_data, factor_constraints, **kwargs
                )
            elif method == OptimizationMethod.RISK_PARITY:
                result = await self._risk_parity_optimization(
                    mu, S, tickers, **kwargs
                )
            elif method == OptimizationMethod.MULTI_OBJECTIVE:
                result = await self._multi_objective_optimization(
                    mu, S, tickers, factor_data, risk_tolerance, **kwargs
                )
            else:  # MAX_SHARPE
                result = await self._max_sharpe_optimization(
                    mu, S, tickers, factor_constraints, factor_data, **kwargs
                )
            
            # Add optimization metadata
            result.update({
                "optimization_method": method,
                "lookback_days": lookback_days,
                "risk_tolerance": risk_tolerance,
                "optimization_timestamp": datetime.now().isoformat(),
                "data_period": {
                    "start": data.index[0].strftime("%Y-%m-%d"),
                    "end": data.index[-1].strftime("%Y-%m-%d"),
                    "data_points": len(data)
                }
            })
            
            logger.info(f"Advanced optimization completed successfully using {method}")
            return result
            
        except Exception as e:
            logger.error(f"Error in advanced optimization: {str(e)}")
            return {
                "error": str(e),
                "method": method,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _black_litterman_optimization(
        self,
        data: pd.DataFrame,
        mu: pd.Series,
        S: pd.DataFrame,
        tickers: List[str],
        views: Optional[Dict[str, float]] = None,
        risk_tolerance: str = "medium",
        **kwargs
    ) -> Dict[str, Any]:
        """Implement Black-Litterman model optimization."""
        try:
            logger.info("Running Black-Litterman optimization")
            
            # Market capitalization weights (proxy for market equilibrium)
            market_caps = await self._get_market_caps(tickers)
            market_weights = self._calculate_market_cap_weights(market_caps)
            
            # Risk aversion parameter based on risk tolerance
            risk_aversion = {
                "low": 2.0,
                "medium": 3.0,
                "high": 5.0
            }.get(risk_tolerance, 3.0)
            
            # Create Black-Litterman model
            bl = BlackLittermanModel(
                S, 
                pi="market",  # Use market-implied returns
                market_caps=market_caps,
                risk_aversion=risk_aversion
            )
            
            # Add views if provided
            if views:
                # Convert views to the format expected by BlackLittermanModel
                tickers_with_views = list(views.keys())
                view_matrix = np.zeros((len(views), len(tickers)))
                view_returns = []
                
                for i, (ticker, expected_return) in enumerate(views.items()):
                    if ticker in tickers:
                        ticker_idx = tickers.index(ticker)
                        view_matrix[i, ticker_idx] = 1.0
                        view_returns.append(expected_return)
                
                if len(view_returns) > 0:
                    # View uncertainty (can be adjusted based on confidence)
                    view_uncertainty = np.diag([0.025] * len(view_returns))  # 2.5% uncertainty
                    
                    bl.bl_views(
                        P=view_matrix,
                        Q=view_returns,
                        omega=view_uncertainty
                    )
            
            # Get Black-Litterman expected returns and covariance
            bl_mu = bl.bl_returns()
            bl_S = bl.bl_cov()
            
            # Optimize portfolio
            ef = EfficientFrontier(bl_mu, bl_S)
            
            # Add regularization to avoid extreme weights
            ef.add_objective(L2_reg, gamma=0.1)
            
            # Optimize based on method
            if kwargs.get("target_return"):
                weights = ef.efficient_return(kwargs["target_return"])
            else:
                weights = ef.max_sharpe()
            
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance()
            
            return {
                "weights": cleaned_weights,
                "expected_return": float(performance[0]),
                "volatility": float(performance[1]),
                "sharpe_ratio": float(performance[2]),
                "market_weights": market_weights,
                "bl_expected_returns": bl_mu.to_dict(),
                "views_applied": views or {},
                "risk_aversion": risk_aversion,
                "method_details": {
                    "prior_returns": mu.to_dict(),
                    "posterior_returns": bl_mu.to_dict(),
                    "view_impact": {
                        ticker: float(bl_mu[ticker] - mu[ticker]) 
                        for ticker in tickers
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {str(e)}")
            raise
    
    async def _minimum_variance_optimization(
        self,
        mu: pd.Series,
        S: pd.DataFrame,
        tickers: List[str],
        factor_constraints: Optional[Dict[FactorType, Tuple[float, float]]] = None,
        factor_data: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Implement minimum variance optimization with optional factor constraints."""
        try:
            logger.info("Running minimum variance optimization")
            
            ef = EfficientFrontier(mu, S)
            
            # Add factor constraints if provided
            if factor_constraints and factor_data:
                self._add_factor_constraints(ef, factor_constraints, factor_data, tickers)
            
            # Add weight constraints to avoid extreme positions
            ef.add_constraint(lambda w: w >= 0.01)  # Min 1% position
            ef.add_constraint(lambda w: w <= 0.4)   # Max 40% position
            
            # Minimize volatility
            weights = ef.min_volatility()
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance()
            
            # Calculate additional risk metrics
            portfolio_weights = np.array([cleaned_weights[ticker] for ticker in tickers])
            portfolio_variance = portfolio_weights.T @ S.values @ portfolio_weights
            risk_contributions = self._calculate_risk_contributions(portfolio_weights, S.values)
            
            return {
                "weights": cleaned_weights,
                "expected_return": float(performance[0]),
                "volatility": float(performance[1]),
                "sharpe_ratio": float(performance[2]),
                "portfolio_variance": float(portfolio_variance),
                "risk_contributions": {
                    ticker: float(risk_contributions[i]) 
                    for i, ticker in enumerate(tickers)
                },
                "method_details": {
                    "optimization_objective": "minimum_variance",
                    "factor_constraints_applied": factor_constraints is not None,
                    "diversification_ratio": self._calculate_diversification_ratio(portfolio_weights, S.values)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in minimum variance optimization: {str(e)}")
            raise
    
    async def _factor_based_optimization(
        self,
        data: pd.DataFrame,
        mu: pd.Series,
        S: pd.DataFrame,
        tickers: List[str],
        factor_data: Dict,
        factor_constraints: Optional[Dict[FactorType, Tuple[float, float]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Implement factor-based optimization with multi-factor model."""
        try:
            logger.info("Running factor-based optimization")
            
            # Build factor model
            factor_exposures, factor_returns = self._build_factor_model(data, tickers, factor_data)
            
            # Use factor model for expected returns
            factor_mu = self._calculate_factor_expected_returns(factor_exposures, factor_returns)
            
            ef = EfficientFrontier(factor_mu, S)
            
            # Add factor constraints
            if factor_constraints:
                self._add_factor_constraints(ef, factor_constraints, factor_data, tickers)
            
            # Add diversification constraints
            ef.add_constraint(lambda w: w >= 0.02)  # Min 2% position
            ef.add_constraint(lambda w: w <= 0.25)  # Max 25% position
            
            # Optimize for factor-adjusted Sharpe ratio
            weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance()
            
            # Calculate factor exposures of the optimized portfolio
            portfolio_weights = np.array([cleaned_weights[ticker] for ticker in tickers])
            portfolio_factor_exposures = self._calculate_portfolio_factor_exposures(
                portfolio_weights, factor_exposures, tickers
            )
            
            return {
                "weights": cleaned_weights,
                "expected_return": float(performance[0]),
                "volatility": float(performance[1]),
                "sharpe_ratio": float(performance[2]),
                "factor_exposures": portfolio_factor_exposures,
                "factor_expected_returns": factor_mu.to_dict(),
                "method_details": {
                    "optimization_objective": "factor_based_max_sharpe",
                    "factors_used": list(factor_exposures.columns),
                    "factor_model_r2": self._calculate_factor_model_r2(factor_exposures, factor_returns, data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in factor-based optimization: {str(e)}")
            raise
    
    async def _risk_parity_optimization(
        self,
        mu: pd.Series,
        S: pd.DataFrame,
        tickers: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Implement risk parity optimization (equal risk contribution)."""
        try:
            logger.info("Running risk parity optimization")
            
            # Risk parity objective function
            def risk_parity_objective(weights, cov_matrix):
                portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
                risk_contrib = (weights * (cov_matrix @ weights)) / portfolio_risk
                target_risk = portfolio_risk / len(weights)  # Equal risk contribution
                return np.sum((risk_contrib - target_risk) ** 2)
            
            # Initial guess (equal weights)
            n_assets = len(tickers)
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
            ]
            
            # Bounds (long-only with reasonable limits)
            bounds = [(0.01, 0.5) for _ in range(n_assets)]
            
            # Optimize
            result = minimize(
                risk_parity_objective,
                x0,
                args=(S.values,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                logger.warning("Risk parity optimization did not converge")
            
            # Create weights dictionary
            optimal_weights = result.x
            weights_dict = {ticker: float(w) for ticker, w in zip(tickers, optimal_weights)}
            
            # Calculate performance metrics
            portfolio_return = float(optimal_weights.T @ mu.values)
            portfolio_vol = float(np.sqrt(optimal_weights.T @ S.values @ optimal_weights))
            sharpe_ratio = float(portfolio_return / portfolio_vol if portfolio_vol > 0 else 0)
            
            # Calculate risk contributions
            risk_contributions = self._calculate_risk_contributions(optimal_weights, S.values)
            
            return {
                "weights": weights_dict,
                "expected_return": portfolio_return,
                "volatility": portfolio_vol,
                "sharpe_ratio": sharpe_ratio,
                "risk_contributions": {
                    ticker: float(risk_contributions[i]) 
                    for i, ticker in enumerate(tickers)
                },
                "method_details": {
                    "optimization_objective": "risk_parity",
                    "optimization_success": result.success,
                    "risk_concentration": float(np.std(risk_contributions)),
                    "effective_number_of_bets": 1.0 / np.sum(risk_contributions ** 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {str(e)}")
            raise
    
    async def _multi_objective_optimization(
        self,
        mu: pd.Series,
        S: pd.DataFrame,
        tickers: List[str],
        factor_data: Optional[Dict],
        risk_tolerance: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Implement multi-objective optimization combining return, risk, and factor exposures."""
        try:
            logger.info("Running multi-objective optimization")
            
            # Risk tolerance weights
            risk_weights = {
                "low": {"return": 0.3, "risk": 0.5, "diversification": 0.2},
                "medium": {"return": 0.4, "risk": 0.4, "diversification": 0.2},
                "high": {"return": 0.5, "risk": 0.3, "diversification": 0.2}
            }.get(risk_tolerance, {"return": 0.4, "risk": 0.4, "diversification": 0.2})
            
            # Multi-objective function
            def multi_objective(weights, mu_vals, cov_matrix, risk_weights):
                # Return objective (maximize)
                portfolio_return = weights.T @ mu_vals
                
                # Risk objective (minimize)
                portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
                
                # Diversification objective (maximize)
                diversification = 1.0 / np.sum(weights ** 2)  # Inverse concentration
                
                # Normalize objectives
                return_norm = portfolio_return / np.abs(mu_vals).max()
                risk_norm = portfolio_risk / np.sqrt(np.diag(cov_matrix)).max()
                div_norm = diversification / len(weights)
                
                # Combined objective (to minimize)
                objective = (
                    -risk_weights["return"] * return_norm +
                    risk_weights["risk"] * risk_norm +
                    -risk_weights["diversification"] * div_norm
                )
                
                return objective
            
            # Initial guess
            n_assets = len(tickers)
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            ]
            
            # Bounds
            bounds = [(0.01, 0.4) for _ in range(n_assets)]
            
            # Optimize
            result = minimize(
                multi_objective,
                x0,
                args=(mu.values, S.values, risk_weights),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            # Create weights dictionary
            optimal_weights = result.x
            weights_dict = {ticker: float(w) for ticker, w in zip(tickers, optimal_weights)}
            
            # Calculate performance metrics
            portfolio_return = float(optimal_weights.T @ mu.values)
            portfolio_vol = float(np.sqrt(optimal_weights.T @ S.values @ optimal_weights))
            sharpe_ratio = float(portfolio_return / portfolio_vol if portfolio_vol > 0 else 0)
            
            return {
                "weights": weights_dict,
                "expected_return": portfolio_return,
                "volatility": portfolio_vol,
                "sharpe_ratio": sharpe_ratio,
                "method_details": {
                    "optimization_objective": "multi_objective",
                    "risk_tolerance_weights": risk_weights,
                    "optimization_success": result.success,
                    "diversification_index": float(1.0 / np.sum(optimal_weights ** 2)),
                    "concentration_ratio": float(np.sum(optimal_weights ** 2))
                }
            }
            
        except Exception as e:
            logger.error(f"Error in multi-objective optimization: {str(e)}")
            raise
    
    async def _max_sharpe_optimization(
        self,
        mu: pd.Series,
        S: pd.DataFrame,
        tickers: List[str],
        factor_constraints: Optional[Dict[FactorType, Tuple[float, float]]] = None,
        factor_data: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Enhanced max Sharpe optimization with factor constraints."""
        try:
            logger.info("Running enhanced max Sharpe optimization")
            
            ef = EfficientFrontier(mu, S)
            
            # Add factor constraints if provided
            if factor_constraints and factor_data:
                self._add_factor_constraints(ef, factor_constraints, factor_data, tickers)
            
            # Add basic constraints
            ef.add_constraint(lambda w: w >= 0.005)  # Min 0.5% position
            ef.add_constraint(lambda w: w <= 0.4)    # Max 40% position
            
            # Optimize
            weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance()
            
            return {
                "weights": cleaned_weights,
                "expected_return": float(performance[0]),
                "volatility": float(performance[1]),
                "sharpe_ratio": float(performance[2]),
                "method_details": {
                    "optimization_objective": "max_sharpe_enhanced",
                    "factor_constraints_applied": factor_constraints is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Error in max Sharpe optimization: {str(e)}")
            raise
    
    async def _get_historical_data(self, tickers: List[str], lookback_days: int) -> pd.DataFrame:
        """Get historical price data for optimization."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Download data with error handling
            data = pd.DataFrame()
            for ticker in tickers:
                try:
                    ticker_data = yf.download(
                        ticker, 
                        start=start_date, 
                        end=end_date, 
                        progress=False
                    )
                    if not ticker_data.empty and 'Adj Close' in ticker_data.columns:
                        data[ticker] = ticker_data['Adj Close']
                    elif not ticker_data.empty and 'Close' in ticker_data.columns:
                        data[ticker] = ticker_data['Close']
                except Exception as e:
                    logger.warning(f"Could not download data for {ticker}: {str(e)}")
            
            # Forward fill missing data
            data = data.fillna(method='ffill').dropna()
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return pd.DataFrame()
    
    async def _get_market_caps(self, tickers: List[str]) -> Dict[str, float]:
        """Get market capitalizations for tickers."""
        try:
            market_caps = {}
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    market_cap = info.get('marketCap')
                    if market_cap:
                        market_caps[ticker] = float(market_cap)
                    else:
                        # Use a default market cap for assets without this data
                        market_caps[ticker] = 1e9  # $1B default
                        
                except Exception as e:
                    logger.warning(f"Could not get market cap for {ticker}: {str(e)}")
                    market_caps[ticker] = 1e9  # Default
            
            return market_caps
            
        except Exception as e:
            logger.error(f"Error getting market caps: {str(e)}")
            return {ticker: 1e9 for ticker in tickers}
    
    def _calculate_market_cap_weights(self, market_caps: Dict[str, float]) -> Dict[str, float]:
        """Calculate market capitalization weights."""
        total_market_cap = sum(market_caps.values())
        if total_market_cap == 0:
            # Equal weights if no market cap data
            return {ticker: 1.0/len(market_caps) for ticker in market_caps.keys()}
        
        return {
            ticker: market_cap / total_market_cap 
            for ticker, market_cap in market_caps.items()
        }
    
    async def _get_factor_data(self, tickers: List[str]) -> Dict[str, Any]:
        """Get factor data for tickers (simplified implementation)."""
        try:
            factor_data = {}
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    # Extract factor-related data
                    factor_data[ticker] = {
                        # Value factors
                        "PE_ratio": info.get('trailingPE', 20.0),
                        "PB_ratio": info.get('priceToBook', 3.0),
                        "EV_EBITDA": info.get('enterpriseToEbitda', 15.0),
                        
                        # Quality factors
                        "ROE": info.get('returnOnEquity', 0.15),
                        "ROA": info.get('returnOnAssets', 0.08),
                        "debt_to_equity": info.get('debtToEquity', 50.0),
                        "current_ratio": info.get('currentRatio', 1.5),
                        
                        # Size factor
                        "market_cap": info.get('marketCap', 1e9),
                        
                        # Profitability factors
                        "gross_margin": info.get('grossMargins', 0.3),
                        "operating_margin": info.get('operatingMargins', 0.15),
                        "net_margin": info.get('profitMargins', 0.1),
                        
                        # Beta (volatility related)
                        "beta": info.get('beta', 1.0)
                    }
                    
                except Exception as e:
                    logger.warning(f"Could not get factor data for {ticker}: {str(e)}")
                    # Use default values
                    factor_data[ticker] = {
                        "PE_ratio": 20.0, "PB_ratio": 3.0, "EV_EBITDA": 15.0,
                        "ROE": 0.15, "ROA": 0.08, "debt_to_equity": 50.0,
                        "current_ratio": 1.5, "market_cap": 1e9,
                        "gross_margin": 0.3, "operating_margin": 0.15,
                        "net_margin": 0.1, "beta": 1.0
                    }
            
            return factor_data
            
        except Exception as e:
            logger.error(f"Error getting factor data: {str(e)}")
            return {}
    
    def _add_factor_constraints(
        self, 
        ef: EfficientFrontier, 
        factor_constraints: Dict[FactorType, Tuple[float, float]], 
        factor_data: Dict, 
        tickers: List[str]
    ):
        """Add factor-based constraints to the optimization."""
        try:
            for factor_type, (min_exposure, max_exposure) in factor_constraints.items():
                if factor_type == FactorType.VALUE:
                    # Value constraint: limit portfolio average P/E ratio
                    def value_constraint(weights):
                        pe_ratios = [factor_data[ticker]["PE_ratio"] for ticker in tickers]
                        portfolio_pe = sum(w * pe for w, pe in zip(weights, pe_ratios))
                        return max_exposure - portfolio_pe  # <= max_exposure
                    
                    ef.add_constraint(value_constraint)
                    
                elif factor_type == FactorType.QUALITY:
                    # Quality constraint: minimum portfolio ROE
                    def quality_constraint(weights):
                        roes = [factor_data[ticker]["ROE"] for ticker in tickers]
                        portfolio_roe = sum(w * roe for w, roe in zip(weights, roes))
                        return portfolio_roe - min_exposure  # >= min_exposure
                    
                    ef.add_constraint(quality_constraint)
                    
                elif factor_type == FactorType.SIZE:
                    # Size constraint: limit exposure to small/large caps
                    def size_constraint_min(weights):
                        market_caps = [factor_data[ticker]["market_cap"] for ticker in tickers]
                        small_cap_weight = sum(
                            w for w, mc in zip(weights, market_caps) 
                            if mc < 2e9  # Small cap threshold
                        )
                        return small_cap_weight - min_exposure
                    
                    def size_constraint_max(weights):
                        market_caps = [factor_data[ticker]["market_cap"] for ticker in tickers]
                        small_cap_weight = sum(
                            w for w, mc in zip(weights, market_caps) 
                            if mc < 2e9
                        )
                        return max_exposure - small_cap_weight
                    
                    ef.add_constraint(size_constraint_min)
                    ef.add_constraint(size_constraint_max)
                    
        except Exception as e:
            logger.warning(f"Error adding factor constraints: {str(e)}")
    
    def _build_factor_model(
        self, 
        data: pd.DataFrame, 
        tickers: List[str], 
        factor_data: Dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build a simple factor model from available data."""
        try:
            # Calculate returns
            returns = data.pct_change().dropna()
            
            # Create factor exposures matrix
            factor_exposures = pd.DataFrame(index=tickers)
            
            for ticker in tickers:
                if ticker in factor_data:
                    # Normalize factors
                    factor_exposures.loc[ticker, 'Value'] = 1.0 / factor_data[ticker]["PE_ratio"]
                    factor_exposures.loc[ticker, 'Quality'] = factor_data[ticker]["ROE"]
                    factor_exposures.loc[ticker, 'Size'] = np.log(factor_data[ticker]["market_cap"])
                    factor_exposures.loc[ticker, 'LowVol'] = 1.0 / factor_data[ticker]["beta"]
            
            # Calculate factor returns using simple regression
            factor_returns = pd.DataFrame(index=returns.index)
            
            for factor in factor_exposures.columns:
                factor_return_series = pd.Series(index=returns.index, dtype=float)
                
                for date in returns.index:
                    # Simple cross-sectional regression
                    try:
                        y = returns.loc[date, tickers].values
                        X = factor_exposures[factor].values.reshape(-1, 1)
                        
                        # Handle NaN values
                        valid_mask = ~(np.isnan(y) | np.isnan(X.flatten()))
                        if np.sum(valid_mask) > 2:
                            reg = LinearRegression().fit(X[valid_mask], y[valid_mask])
                            factor_return_series[date] = reg.coef_[0]
                        else:
                            factor_return_series[date] = 0.0
                    except:
                        factor_return_series[date] = 0.0
                
                factor_returns[factor] = factor_return_series
            
            return factor_exposures, factor_returns
            
        except Exception as e:
            logger.error(f"Error building factor model: {str(e)}")
            # Return empty DataFrames
            return pd.DataFrame(), pd.DataFrame()
    
    def _calculate_factor_expected_returns(
        self, 
        factor_exposures: pd.DataFrame, 
        factor_returns: pd.DataFrame
    ) -> pd.Series:
        """Calculate expected returns using factor model."""
        try:
            # Factor risk premia (mean factor returns)
            factor_premia = factor_returns.mean()
            
            # Expected returns = factor exposures × factor premia
            expected_returns = factor_exposures @ factor_premia
            
            return expected_returns
            
        except Exception as e:
            logger.error(f"Error calculating factor expected returns: {str(e)}")
            return pd.Series()
    
    def _calculate_portfolio_factor_exposures(
        self, 
        weights: np.ndarray, 
        factor_exposures: pd.DataFrame, 
        tickers: List[str]
    ) -> Dict[str, float]:
        """Calculate portfolio-level factor exposures."""
        try:
            portfolio_exposures = {}
            
            for factor in factor_exposures.columns:
                exposure = sum(
                    weights[i] * factor_exposures.loc[ticker, factor]
                    for i, ticker in enumerate(tickers)
                    if ticker in factor_exposures.index
                )
                portfolio_exposures[factor] = float(exposure)
            
            return portfolio_exposures
            
        except Exception as e:
            logger.error(f"Error calculating portfolio factor exposures: {str(e)}")
            return {}
    
    def _calculate_factor_model_r2(
        self, 
        factor_exposures: pd.DataFrame, 
        factor_returns: pd.DataFrame, 
        data: pd.DataFrame
    ) -> float:
        """Calculate R-squared of the factor model."""
        try:
            returns = data.pct_change().dropna()
            
            # Calculate explained variance for each asset
            r_squared_values = []
            
            for ticker in factor_exposures.index:
                if ticker in returns.columns:
                    y = returns[ticker].dropna()
                    
                    # Align dates
                    common_dates = y.index.intersection(factor_returns.index)
                    if len(common_dates) > 10:
                        y_aligned = y[common_dates]
                        X = factor_returns[common_dates].values
                        
                        # Calculate R-squared
                        try:
                            reg = LinearRegression().fit(X, y_aligned)
                            r_squared = reg.score(X, y_aligned)
                            r_squared_values.append(max(0, r_squared))
                        except:
                            pass
            
            return float(np.mean(r_squared_values)) if r_squared_values else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating factor model R-squared: {str(e)}")
            return 0.0
    
    def _calculate_risk_contributions(self, weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Calculate risk contributions for each asset."""
        try:
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol
            return risk_contrib / np.sum(risk_contrib)  # Normalize to sum to 1
            
        except Exception as e:
            logger.error(f"Error calculating risk contributions: {str(e)}")
            return np.ones(len(weights)) / len(weights)
    
    def _calculate_diversification_ratio(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate diversification ratio."""
        try:
            # Weighted average volatility
            individual_vols = np.sqrt(np.diag(cov_matrix))
            weighted_avg_vol = weights.T @ individual_vols
            
            # Portfolio volatility
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            
            # Diversification ratio
            return float(weighted_avg_vol / portfolio_vol) if portfolio_vol > 0 else 1.0
            
        except Exception as e:
            logger.error(f"Error calculating diversification ratio: {str(e)}")
            return 1.0
