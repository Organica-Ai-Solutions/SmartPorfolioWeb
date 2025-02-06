import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import yfinance as yf
from sqlalchemy.orm import Session
from app.models import Portfolio, PortfolioSnapshot, RiskAlert, Notification
from pypfopt import risk_models, expected_returns, objective_functions
from pypfopt.efficient_frontier import EfficientFrontier
import scipy.stats

class PortfolioService:
    def __init__(self, db: Session):
        self.db = db

    def create_portfolio_snapshot(
        self,
        portfolio_id: int,
        allocations: Dict[str, float],
        metrics: Dict,
        asset_metrics: Dict,
        total_value: float,
        cash_position: float
    ) -> PortfolioSnapshot:
        """Create a new portfolio snapshot."""
        snapshot = PortfolioSnapshot(
            portfolio_id=portfolio_id,
            allocations=allocations,
            metrics=metrics,
            asset_metrics=asset_metrics,
            total_value=total_value,
            cash_position=cash_position
        )
        self.db.add(snapshot)
        self.db.commit()
        self.db.refresh(snapshot)
        return snapshot

    def calculate_risk_metrics(self, data: pd.DataFrame, weights: Dict[str, float]) -> Dict:
        """Calculate comprehensive risk metrics for the portfolio."""
        # Handle missing or invalid data
        data = data.fillna(method='ffill').fillna(method='bfill')
        returns = data.pct_change().fillna(0)
        weight_arr = np.array([weights[ticker] for ticker in data.columns])
        
        # Calculate portfolio returns with proper handling of outliers
        portfolio_returns = returns.dot(weight_arr)
        portfolio_returns = portfolio_returns.clip(portfolio_returns.quantile(0.001), portfolio_returns.quantile(0.999))
        
        # Calculate drawdown with proper handling
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Calculate VaR and CVaR at different confidence levels with better statistical handling
        confidence_levels = [0.99, 0.95, 0.90]
        var_metrics = {}
        cvar_metrics = {}
        for conf in confidence_levels:
            # Use Gaussian and historical VaR/CVaR
            gaussian_var = float(portfolio_returns.mean() - portfolio_returns.std() * np.sqrt(2) * scipy.stats.norm.ppf(conf))
            historical_var = float(np.percentile(portfolio_returns, (1 - conf) * 100))
            # Use the more conservative estimate
            var = min(gaussian_var, historical_var)
            var_metrics[f"var_{int(conf * 100)}"] = var
            
            # Calculate CVaR (Expected Shortfall)
            cvar = float(portfolio_returns[portfolio_returns <= var].mean())
            cvar_metrics[f"cvar_{int(conf * 100)}"] = cvar
        
        # Calculate rolling volatility with adjustable window
        min_window = min(30, len(returns) - 1)
        rolling_vol = returns.rolling(window=min_window, min_periods=1).std() * np.sqrt(252)
        current_vol = rolling_vol.iloc[-1]
        
        # Calculate correlation matrix with shrinkage
        correlation_matrix = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
        
        # Calculate tracking error vs appropriate benchmark
        try:
            # Determine if portfolio is crypto-heavy
            crypto_count = sum(1 for ticker in data.columns if ticker.endswith('-USD'))
            benchmark = '^GSPC' if crypto_count < len(data.columns) / 2 else 'BTC-USD'
            benchmark_data = yf.download(benchmark, start=data.index[0], end=data.index[-1])['Adj Close']
            benchmark_returns = benchmark_data.pct_change().fillna(0)
            tracking_error = np.std(portfolio_returns - benchmark_returns) * np.sqrt(252)
        except Exception:
            tracking_error = current_vol  # Fallback to volatility if benchmark data unavailable
        
        # Calculate additional risk metrics
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else current_vol
        
        # Calculate Sortino ratio
        risk_free_rate = 0.02  # Adjustable risk-free rate
        excess_return = portfolio_returns.mean() * 252 - risk_free_rate
        sortino_ratio = excess_return / downside_vol if downside_vol > 0 else 0
        
        return {
            "max_drawdown": float(max_drawdown),
            "current_volatility": float(current_vol),
            "tracking_error": float(tracking_error),
            "correlation_matrix": correlation_matrix.to_dict(),
            "downside_volatility": float(downside_vol),
            "sortino_ratio": float(sortino_ratio),
            **var_metrics,
            **cvar_metrics
        }

    def check_risk_alerts(
        self,
        portfolio_id: int,
        risk_metrics: Dict,
        thresholds: Optional[Dict] = None,
        tickers: List[str] = None
    ) -> List[RiskAlert]:
        """Check for risk alerts based on current metrics and thresholds."""
        # Determine if portfolio is crypto-heavy
        is_crypto_heavy = False
        if tickers:
            crypto_count = sum(1 for ticker in tickers if ticker.endswith('-USD') or ticker.endswith('USDT'))
            is_crypto_heavy = crypto_count > len(tickers) / 2

        # Adjust thresholds based on portfolio composition
        if thresholds is None:
            if is_crypto_heavy:
                thresholds = {
                    "max_drawdown": -0.30,  # 30% maximum drawdown for crypto
                    "volatility": 0.50,     # 50% annualized volatility for crypto
                    "var_95": -0.05,        # 5% daily VaR at 95% confidence for crypto
                    "concentration": 0.35    # 35% maximum allocation to single asset
                }
            else:
                thresholds = {
                    "max_drawdown": -0.15,  # 15% maximum drawdown for stocks
                    "volatility": 0.25,     # 25% annualized volatility for stocks
                    "var_95": -0.02,        # 2% daily VaR at 95% confidence for stocks
                    "concentration": 0.35    # 35% maximum allocation to single asset
                }
        
        alerts = []
        
        # Check drawdown
        if risk_metrics["max_drawdown"] < thresholds["max_drawdown"]:
            alert = RiskAlert(
                portfolio_id=portfolio_id,
                alert_type="drawdown",
                severity="high",
                message=f"Portfolio drawdown ({risk_metrics['max_drawdown']:.2%}) exceeds threshold ({thresholds['max_drawdown']:.2%})",
                metrics={"max_drawdown": risk_metrics["max_drawdown"]}
            )
            alerts.append(alert)
        
        # Check volatility
        if risk_metrics["current_volatility"] > thresholds["volatility"]:
            alert = RiskAlert(
                portfolio_id=portfolio_id,
                alert_type="volatility",
                severity="medium",
                message=f"Portfolio volatility ({risk_metrics['current_volatility']:.2%}) exceeds threshold ({thresholds['volatility']:.2%})",
                metrics={"volatility": risk_metrics["current_volatility"]}
            )
            alerts.append(alert)
        
        # Check VaR
        if risk_metrics["var_95"] < thresholds["var_95"]:
            alert = RiskAlert(
                portfolio_id=portfolio_id,
                alert_type="var",
                severity="medium",
                message=f"Portfolio VaR (95%) ({risk_metrics['var_95']:.2%}) exceeds threshold ({thresholds['var_95']:.2%})",
                metrics={"var_95": risk_metrics["var_95"]}
            )
            alerts.append(alert)
        
        # Save alerts to database
        for alert in alerts:
            self.db.add(alert)
            
            # Create notification for each alert
            notification = Notification(
                user_id=self.db.query(Portfolio).get(portfolio_id).user_id,
                type="risk_alert",
                title=f"Risk Alert: {alert.alert_type.title()}",
                message=alert.message
            )
            self.db.add(notification)
        
        self.db.commit()
        return alerts

    def optimize_portfolio(
        self,
        tickers: List[str],
        risk_tolerance: str,
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """Optimize portfolio based on risk tolerance and constraints."""
        if end_date is None:
            end_date = datetime.now()
        
        # Download data
        data = pd.DataFrame()
        for ticker in tickers:
            stock_data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
            data[ticker] = stock_data
        
        # Calculate expected returns and covariance
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
        
        # Initialize Efficient Frontier
        ef = EfficientFrontier(mu, S)
        
        # Add constraints based on risk tolerance
        sector_constraints = self.get_sector_constraints(tickers)
        ef.add_sector_constraints(sector_constraints)
        
        # Add minimum and maximum weight constraints
        if risk_tolerance == "low":
            ef.add_constraint(lambda x: x <= 0.25)  # Max 25% in any asset
            weights = ef.min_volatility()
        elif risk_tolerance == "medium":
            ef.add_constraint(lambda x: x <= 0.35)  # Max 35% in any asset
            weights = ef.maximum_sharpe()
        else:  # high
            ef.add_constraint(lambda x: x <= 0.45)  # Max 45% in any asset
            weights = ef.maximize_quadratic_utility(risk_aversion=0.5)
        
        # Clean weights
        cleaned_weights = ef.clean_weights()
        
        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics(data, cleaned_weights)
        
        return {
            "weights": cleaned_weights,
            "risk_metrics": risk_metrics,
            "performance_metrics": ef.portfolio_performance(verbose=True)
        }

    def get_sector_constraints(self, tickers: List[str]) -> Dict:
        """Get sector constraints for portfolio optimization."""
        sector_mappings = {}
        sector_bounds = {
            "Technology": (0, 0.40),
            "Financial": (0, 0.35),
            "Healthcare": (0, 0.30),
            "Consumer": (0, 0.30),
            "Industrial": (0, 0.30),
            "Energy": (0, 0.25),
            "Materials": (0, 0.25),
            "Utilities": (0, 0.20),
            "Real Estate": (0, 0.20),
        }
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                sector = stock.info.get("sector", "Other")
                sector_mappings[ticker] = sector
            except:
                sector_mappings[ticker] = "Other"
        
        return {
            "sectors": sector_mappings,
            "bounds": sector_bounds
        } 