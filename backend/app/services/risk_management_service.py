import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RiskRegime(str, Enum):
    LOW_RISK = "low_risk"
    MODERATE_RISK = "moderate_risk"
    HIGH_RISK = "high_risk"
    EXTREME_RISK = "extreme_risk"

class HedgeType(str, Enum):
    VIX_HEDGE = "vix_hedge"
    TAIL_PROTECTION = "tail_protection"
    CURRENCY_HEDGE = "currency_hedge"
    CORRELATION_HEDGE = "correlation_hedge"

class DrawdownSeverity(str, Enum):
    MINOR = "minor"         # 0-5%
    MODERATE = "moderate"   # 5-10%
    MAJOR = "major"         # 10-20%
    SEVERE = "severe"       # 20%+

class RiskManagementService:
    """Advanced risk management service with volatility sizing, tail hedging, and drawdown controls."""
    
    def __init__(self):
        # Risk thresholds
        self.volatility_thresholds = {
            "low": 0.15,      # Below 15% annual volatility
            "medium": 0.25,   # 15-25% annual volatility
            "high": 0.40,     # 25-40% annual volatility
            "extreme": 1.0    # Above 40% annual volatility
        }
        
        # Drawdown thresholds
        self.drawdown_thresholds = {
            DrawdownSeverity.MINOR: 0.05,
            DrawdownSeverity.MODERATE: 0.10,
            DrawdownSeverity.MAJOR: 0.20,
            DrawdownSeverity.SEVERE: 0.30
        }
        
        # Position sizing parameters
        self.volatility_scaling = {
            "low": 1.0,       # No scaling for low vol
            "medium": 0.8,    # 20% reduction for medium vol
            "high": 0.6,      # 40% reduction for high vol
            "extreme": 0.3    # 70% reduction for extreme vol
        }
        
        # Cache for risk calculations
        self._risk_cache = {}
        self._cache_timestamp = None
        self._cache_expiry = timedelta(hours=1)
    
    async def apply_volatility_position_sizing(
        self,
        weights: Dict[str, float],
        tickers: List[str],
        lookback_days: int = 63,
        target_portfolio_vol: float = 0.15
    ) -> Dict[str, Any]:
        """
        Apply volatility-based position sizing to portfolio weights.
        More volatile assets get smaller allocations.
        """
        try:
            logger.info("Applying volatility-based position sizing")
            
            # Get volatility data for each asset
            volatilities = await self._calculate_asset_volatilities(tickers, lookback_days)
            
            if not volatilities:
                logger.warning("No volatility data available, returning original weights")
                return {
                    "adjusted_weights": weights,
                    "volatility_data": {},
                    "adjustments": {},
                    "target_portfolio_vol": target_portfolio_vol
                }
            
            # Calculate volatility-adjusted weights
            adjusted_weights = {}
            adjustments = {}
            
            total_adjusted_weight = 0.0
            
            for ticker in tickers:
                if ticker in weights and ticker in volatilities:
                    original_weight = weights[ticker]
                    asset_vol = volatilities[ticker]
                    
                    # Determine volatility category
                    vol_category = self._categorize_volatility(asset_vol)
                    
                    # Apply volatility scaling
                    vol_adjustment = self.volatility_scaling[vol_category]
                    
                    # Calculate inverse volatility weight
                    # Assets with higher volatility get lower weights
                    inv_vol_weight = (1.0 / asset_vol) if asset_vol > 0 else 0.0
                    
                    # Combine original weight with volatility adjustment
                    adjusted_weight = original_weight * vol_adjustment * inv_vol_weight
                    
                    adjusted_weights[ticker] = adjusted_weight
                    total_adjusted_weight += adjusted_weight
                    
                    adjustments[ticker] = {
                        "original_weight": original_weight,
                        "volatility": asset_vol,
                        "volatility_category": vol_category,
                        "volatility_scaling": vol_adjustment,
                        "inverse_vol_weight": inv_vol_weight,
                        "raw_adjusted_weight": adjusted_weight
                    }
            
            # Normalize weights to sum to 1
            if total_adjusted_weight > 0:
                for ticker in adjusted_weights:
                    adjusted_weights[ticker] /= total_adjusted_weight
                    adjustments[ticker]["final_weight"] = adjusted_weights[ticker]
                    adjustments[ticker]["weight_change"] = (
                        adjusted_weights[ticker] - adjustments[ticker]["original_weight"]
                    )
            
            # Calculate portfolio-level metrics
            portfolio_vol = self._calculate_portfolio_volatility(adjusted_weights, volatilities, tickers)
            
            # Scale portfolio to target volatility if needed
            if portfolio_vol > 0 and target_portfolio_vol > 0:
                vol_scaling_factor = target_portfolio_vol / portfolio_vol
                if vol_scaling_factor < 1.0:  # Only scale down, not up
                    cash_allocation = 1.0 - vol_scaling_factor
                    for ticker in adjusted_weights:
                        adjusted_weights[ticker] *= vol_scaling_factor
                    
                    # Add cash position if significant scaling
                    if cash_allocation > 0.01:  # More than 1% cash
                        adjusted_weights["CASH"] = cash_allocation
            
            logger.info("Volatility-based position sizing completed")
            
            return {
                "adjusted_weights": adjusted_weights,
                "volatility_data": volatilities,
                "adjustments": adjustments,
                "portfolio_volatility": portfolio_vol,
                "target_portfolio_vol": target_portfolio_vol,
                "scaling_applied": portfolio_vol > target_portfolio_vol,
                "risk_reduction": max(0, (portfolio_vol - target_portfolio_vol) / portfolio_vol) if portfolio_vol > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error in volatility position sizing: {str(e)}")
            return {
                "adjusted_weights": weights,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def implement_tail_risk_hedging(
        self,
        portfolio_weights: Dict[str, float],
        risk_regime: RiskRegime,
        hedge_budget: float = 0.05
    ) -> Dict[str, Any]:
        """
        Implement tail risk hedging strategies during high-risk periods.
        """
        try:
            logger.info(f"Implementing tail risk hedging for {risk_regime} regime")
            
            hedge_allocations = {}
            hedge_strategies = []
            
            if risk_regime in [RiskRegime.HIGH_RISK, RiskRegime.EXTREME_RISK]:
                
                # Determine hedge intensity based on risk regime
                hedge_intensity = {
                    RiskRegime.HIGH_RISK: 0.6,      # Use 60% of hedge budget
                    RiskRegime.EXTREME_RISK: 1.0    # Use full hedge budget
                }.get(risk_regime, 0.5)
                
                effective_hedge_budget = hedge_budget * hedge_intensity
                
                # Strategy 1: VIX/Volatility Protection
                vix_allocation = effective_hedge_budget * 0.4  # 40% of hedge budget
                if vix_allocation > 0.01:  # More than 1%
                    hedge_allocations["VXX"] = vix_allocation  # VIX ETF
                    hedge_strategies.append({
                        "type": HedgeType.VIX_HEDGE,
                        "allocation": vix_allocation,
                        "description": "Volatility spike protection via VIX exposure"
                    })
                
                # Strategy 2: Put Protection / Tail Risk ETF
                tail_allocation = effective_hedge_budget * 0.3  # 30% of hedge budget
                if tail_allocation > 0.01:
                    hedge_allocations["TAIL"] = tail_allocation  # Tail risk ETF
                    hedge_strategies.append({
                        "type": HedgeType.TAIL_PROTECTION,
                        "allocation": tail_allocation,
                        "description": "Direct tail risk protection"
                    })
                
                # Strategy 3: Safe Haven Assets
                safe_haven_allocation = effective_hedge_budget * 0.2  # 20% of hedge budget
                if safe_haven_allocation > 0.01:
                    hedge_allocations["GLD"] = safe_haven_allocation * 0.6  # Gold
                    hedge_allocations["TLT"] = safe_haven_allocation * 0.4  # Long Treasury
                    hedge_strategies.append({
                        "type": HedgeType.CORRELATION_HEDGE,
                        "allocation": safe_haven_allocation,
                        "description": "Safe haven allocation (Gold + Long Treasuries)"
                    })
                
                # Strategy 4: Currency Hedge (if international exposure)
                currency_allocation = effective_hedge_budget * 0.1  # 10% of hedge budget
                if currency_allocation > 0.01 and self._has_international_exposure(portfolio_weights):
                    hedge_allocations["UUP"] = currency_allocation  # USD strengthening
                    hedge_strategies.append({
                        "type": HedgeType.CURRENCY_HEDGE,
                        "allocation": currency_allocation,
                        "description": "USD strengthening hedge for international exposure"
                    })
            
            # Adjust main portfolio weights to accommodate hedges
            total_hedge_allocation = sum(hedge_allocations.values())
            adjusted_portfolio = {}
            
            if total_hedge_allocation > 0:
                # Scale down main portfolio
                scaling_factor = (1.0 - total_hedge_allocation)
                for ticker, weight in portfolio_weights.items():
                    adjusted_portfolio[ticker] = weight * scaling_factor
                
                # Add hedge positions
                adjusted_portfolio.update(hedge_allocations)
            else:
                adjusted_portfolio = portfolio_weights.copy()
            
            logger.info(f"Tail risk hedging implemented with {len(hedge_strategies)} strategies")
            
            return {
                "hedged_portfolio": adjusted_portfolio,
                "hedge_strategies": hedge_strategies,
                "total_hedge_allocation": total_hedge_allocation,
                "risk_regime": risk_regime,
                "hedge_budget_used": total_hedge_allocation,
                "hedge_effectiveness": await self._estimate_hedge_effectiveness(hedge_strategies),
                "implementation_details": {
                    "hedge_intensity": hedge_intensity,
                    "scaling_factor": 1.0 - total_hedge_allocation,
                    "strategies_count": len(hedge_strategies)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in tail risk hedging: {str(e)}")
            return {
                "hedged_portfolio": portfolio_weights,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def apply_drawdown_controls(
        self,
        current_weights: Dict[str, float],
        portfolio_value: float,
        peak_value: float,
        lookback_days: int = 252
    ) -> Dict[str, Any]:
        """
        Apply drawdown control mechanisms that reduce exposure after losses exceed thresholds.
        """
        try:
            logger.info("Applying drawdown control mechanisms")
            
            # Calculate current drawdown
            current_drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0.0
            drawdown_severity = self._classify_drawdown_severity(current_drawdown)
            
            # Determine risk reduction based on drawdown severity
            risk_reduction_factors = {
                DrawdownSeverity.MINOR: 1.0,      # No reduction
                DrawdownSeverity.MODERATE: 0.9,   # 10% reduction
                DrawdownSeverity.MAJOR: 0.7,      # 30% reduction
                DrawdownSeverity.SEVERE: 0.5      # 50% reduction
            }
            
            risk_reduction_factor = risk_reduction_factors[drawdown_severity]
            
            # Calculate additional metrics
            volatility_adjustment = await self._calculate_volatility_adjustment(lookback_days)
            correlation_adjustment = await self._calculate_correlation_adjustment(current_weights, lookback_days)
            
            # Apply composite adjustment
            composite_adjustment = min(
                risk_reduction_factor,
                volatility_adjustment,
                correlation_adjustment
            )
            
            # Adjust portfolio weights
            adjusted_weights = {}
            cash_allocation = 0.0
            
            if composite_adjustment < 1.0:
                # Scale down risky assets
                for ticker, weight in current_weights.items():
                    if ticker not in ["CASH", "SHY", "BIL"]:  # Exclude cash-like instruments
                        adjusted_weights[ticker] = weight * composite_adjustment
                    else:
                        adjusted_weights[ticker] = weight
                
                # Calculate cash allocation from risk reduction
                total_reduction = sum(current_weights.values()) - sum(adjusted_weights.values())
                
                # Allocate to safe assets
                if total_reduction > 0.01:  # More than 1%
                    cash_allocation = total_reduction * 0.6  # 60% to cash
                    treasury_allocation = total_reduction * 0.4  # 40% to short treasuries
                    
                    adjusted_weights["CASH"] = adjusted_weights.get("CASH", 0) + cash_allocation
                    adjusted_weights["SHY"] = adjusted_weights.get("SHY", 0) + treasury_allocation
            else:
                adjusted_weights = current_weights.copy()
            
            # Calculate recovery signals
            recovery_signals = await self._calculate_recovery_signals(lookback_days)
            
            # Determine if we should start reducing defensive positions
            recovery_threshold = 0.7  # Start reducing defense when 70% confident in recovery
            if recovery_signals.get("confidence", 0) > recovery_threshold and current_drawdown < 0.05:
                # Gradually reduce defensive positions
                recovery_factor = min(1.2, 1.0 + (recovery_signals["confidence"] - recovery_threshold))
                
                for ticker in adjusted_weights:
                    if ticker not in ["CASH", "SHY", "BIL"]:
                        adjusted_weights[ticker] *= recovery_factor
                
                # Normalize weights
                total_weight = sum(adjusted_weights.values())
                if total_weight > 1.0:
                    for ticker in adjusted_weights:
                        adjusted_weights[ticker] /= total_weight
            
            logger.info(f"Drawdown controls applied: {drawdown_severity} severity, {composite_adjustment:.2%} adjustment")
            
            return {
                "adjusted_weights": adjusted_weights,
                "drawdown_metrics": {
                    "current_drawdown": current_drawdown,
                    "drawdown_severity": drawdown_severity,
                    "peak_value": peak_value,
                    "current_value": portfolio_value
                },
                "adjustments": {
                    "risk_reduction_factor": risk_reduction_factor,
                    "volatility_adjustment": volatility_adjustment,
                    "correlation_adjustment": correlation_adjustment,
                    "composite_adjustment": composite_adjustment,
                    "cash_allocation_added": cash_allocation
                },
                "recovery_signals": recovery_signals,
                "control_effectiveness": {
                    "risk_reduction": 1.0 - composite_adjustment,
                    "defensive_allocation": cash_allocation + adjusted_weights.get("SHY", 0),
                    "recovery_confidence": recovery_signals.get("confidence", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in drawdown controls: {str(e)}")
            return {
                "adjusted_weights": current_weights,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def comprehensive_risk_management(
        self,
        portfolio_weights: Dict[str, float],
        tickers: List[str],
        portfolio_value: float = 100000,
        peak_value: float = 100000,
        target_vol: float = 0.15,
        hedge_budget: float = 0.05,
        lookback_days: int = 63
    ) -> Dict[str, Any]:
        """
        Apply comprehensive risk management combining all strategies.
        """
        try:
            logger.info("Applying comprehensive risk management")
            
            # Step 1: Detect current risk regime
            risk_regime = await self._detect_risk_regime(lookback_days)
            
            # Step 2: Apply volatility-based position sizing
            vol_sizing_result = await self.apply_volatility_position_sizing(
                portfolio_weights, tickers, lookback_days, target_vol
            )
            
            current_weights = vol_sizing_result["adjusted_weights"]
            
            # Step 3: Apply tail risk hedging if needed
            hedge_result = await self.implement_tail_risk_hedging(
                current_weights, risk_regime, hedge_budget
            )
            
            current_weights = hedge_result["hedged_portfolio"]
            
            # Step 4: Apply drawdown controls
            drawdown_result = await self.apply_drawdown_controls(
                current_weights, portfolio_value, peak_value, lookback_days
            )
            
            final_weights = drawdown_result["adjusted_weights"]
            
            # Step 5: Calculate overall risk metrics
            risk_metrics = await self._calculate_comprehensive_risk_metrics(
                original_weights=portfolio_weights,
                final_weights=final_weights,
                tickers=tickers,
                lookback_days=lookback_days
            )
            
            # Step 6: Generate risk management summary
            summary = self._generate_risk_management_summary(
                vol_sizing_result,
                hedge_result,
                drawdown_result,
                risk_metrics,
                risk_regime
            )
            
            logger.info("Comprehensive risk management completed")
            
            return {
                "final_weights": final_weights,
                "risk_regime": risk_regime,
                "volatility_sizing": vol_sizing_result,
                "tail_hedging": hedge_result,
                "drawdown_controls": drawdown_result,
                "risk_metrics": risk_metrics,
                "summary": summary,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive risk management: {str(e)}")
            return {
                "final_weights": portfolio_weights,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # Helper methods
    
    async def _calculate_asset_volatilities(self, tickers: List[str], lookback_days: int) -> Dict[str, float]:
        """Calculate annualized volatilities for assets."""
        try:
            volatilities = {}
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 10)  # Extra buffer
            
            for ticker in tickers:
                try:
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if not data.empty and len(data) > 10:
                        if 'Adj Close' in data.columns:
                            prices = data['Adj Close']
                        else:
                            prices = data['Close']
                        
                        returns = prices.pct_change().dropna()
                        if len(returns) > 5:
                            # Annualized volatility
                            vol = returns.std() * np.sqrt(252)
                            volatilities[ticker] = float(vol)
                
                except Exception as e:
                    logger.warning(f"Could not calculate volatility for {ticker}: {str(e)}")
                    # Use default volatility based on asset type
                    volatilities[ticker] = self._get_default_volatility(ticker)
            
            return volatilities
            
        except Exception as e:
            logger.error(f"Error calculating asset volatilities: {str(e)}")
            return {}
    
    def _categorize_volatility(self, volatility: float) -> str:
        """Categorize asset volatility into buckets."""
        if volatility <= self.volatility_thresholds["low"]:
            return "low"
        elif volatility <= self.volatility_thresholds["medium"]:
            return "medium"
        elif volatility <= self.volatility_thresholds["high"]:
            return "high"
        else:
            return "extreme"
    
    def _calculate_portfolio_volatility(
        self, 
        weights: Dict[str, float], 
        volatilities: Dict[str, float], 
        tickers: List[str]
    ) -> float:
        """Calculate portfolio volatility (simplified)."""
        try:
            portfolio_vol = 0.0
            for ticker in tickers:
                if ticker in weights and ticker in volatilities:
                    weight = weights[ticker]
                    vol = volatilities[ticker]
                    portfolio_vol += (weight ** 2) * (vol ** 2)
            
            return np.sqrt(portfolio_vol)
        except:
            return 0.0
    
    def _classify_drawdown_severity(self, drawdown: float) -> DrawdownSeverity:
        """Classify drawdown severity."""
        abs_drawdown = abs(drawdown)
        
        if abs_drawdown >= self.drawdown_thresholds[DrawdownSeverity.SEVERE]:
            return DrawdownSeverity.SEVERE
        elif abs_drawdown >= self.drawdown_thresholds[DrawdownSeverity.MAJOR]:
            return DrawdownSeverity.MAJOR
        elif abs_drawdown >= self.drawdown_thresholds[DrawdownSeverity.MODERATE]:
            return DrawdownSeverity.MODERATE
        else:
            return DrawdownSeverity.MINOR
    
    def _has_international_exposure(self, weights: Dict[str, float]) -> bool:
        """Check if portfolio has international exposure."""
        international_tickers = ['EFA', 'EEM', 'VEA', 'VWO', 'IEFA', 'IEMG']
        return any(ticker in weights for ticker in international_tickers)
    
    def _get_default_volatility(self, ticker: str) -> float:
        """Get default volatility based on asset type."""
        if ticker in ['BTC-USD', 'ETH-USD']:
            return 0.8  # Crypto
        elif ticker in ['SPY', 'QQQ', 'IWM']:
            return 0.18  # Equity ETFs
        elif ticker in ['TLT', 'SHY', 'BIL']:
            return 0.05  # Bonds
        elif ticker in ['GLD', 'SLV']:
            return 0.2  # Commodities
        else:
            return 0.25  # Default for stocks
    
    async def _detect_risk_regime(self, lookback_days: int) -> RiskRegime:
        """Detect current market risk regime."""
        try:
            # Simple VIX-based regime detection
            vix_data = yf.download("^VIX", period=f"{lookback_days}d", progress=False)
            
            if not vix_data.empty:
                current_vix = float(vix_data['Close'].iloc[-1])
                
                if current_vix < 15:
                    return RiskRegime.LOW_RISK
                elif current_vix < 25:
                    return RiskRegime.MODERATE_RISK
                elif current_vix < 35:
                    return RiskRegime.HIGH_RISK
                else:
                    return RiskRegime.EXTREME_RISK
            
        except Exception as e:
            logger.warning(f"Could not detect risk regime: {str(e)}")
        
        # Default to moderate risk
        return RiskRegime.MODERATE_RISK
    
    async def _estimate_hedge_effectiveness(self, hedge_strategies: List[Dict]) -> float:
        """Estimate effectiveness of hedge strategies."""
        if not hedge_strategies:
            return 0.0
        
        # Simple effectiveness scoring
        effectiveness_scores = {
            HedgeType.VIX_HEDGE: 0.7,
            HedgeType.TAIL_PROTECTION: 0.8,
            HedgeType.CORRELATION_HEDGE: 0.6,
            HedgeType.CURRENCY_HEDGE: 0.5
        }
        
        total_allocation = sum(strategy.get("allocation", 0) for strategy in hedge_strategies)
        weighted_effectiveness = 0.0
        
        for strategy in hedge_strategies:
            hedge_type = strategy.get("type")
            allocation = strategy.get("allocation", 0)
            
            if hedge_type in effectiveness_scores and total_allocation > 0:
                weight = allocation / total_allocation
                effectiveness = effectiveness_scores[hedge_type]
                weighted_effectiveness += weight * effectiveness
        
        return weighted_effectiveness
    
    async def _calculate_volatility_adjustment(self, lookback_days: int) -> float:
        """Calculate volatility-based adjustment factor."""
        try:
            # Use SPY as market proxy
            spy_data = yf.download("SPY", period=f"{lookback_days}d", progress=False)
            
            if not spy_data.empty and len(spy_data) > 20:
                returns = spy_data['Adj Close'].pct_change().dropna()
                current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
                
                # Adjust based on current volatility vs normal
                normal_vol = 0.16  # Historical SPY volatility
                vol_ratio = current_vol / normal_vol
                
                # Higher volatility = more risk reduction
                if vol_ratio > 2.0:
                    return 0.5  # 50% risk reduction
                elif vol_ratio > 1.5:
                    return 0.7  # 30% risk reduction
                elif vol_ratio > 1.2:
                    return 0.85  # 15% risk reduction
                else:
                    return 1.0  # No adjustment
            
        except Exception as e:
            logger.warning(f"Could not calculate volatility adjustment: {str(e)}")
        
        return 1.0
    
    async def _calculate_correlation_adjustment(self, weights: Dict[str, float], lookback_days: int) -> float:
        """Calculate correlation-based adjustment factor."""
        try:
            # If correlations are too high, reduce exposure
            major_tickers = [ticker for ticker in weights.keys() if weights[ticker] > 0.1]
            
            if len(major_tickers) < 2:
                return 1.0
            
            # Download data for correlation calculation
            data = pd.DataFrame()
            for ticker in major_tickers[:5]:  # Limit to avoid API limits
                try:
                    ticker_data = yf.download(ticker, period=f"{lookback_days}d", progress=False)
                    if not ticker_data.empty:
                        data[ticker] = ticker_data['Adj Close']
                except:
                    pass
            
            if len(data.columns) > 1:
                returns = data.pct_change().dropna()
                corr_matrix = returns.corr()
                
                # Calculate average correlation
                avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()
                
                # Higher correlation = more risk reduction
                if avg_corr > 0.8:
                    return 0.6  # 40% risk reduction
                elif avg_corr > 0.6:
                    return 0.8  # 20% risk reduction
                else:
                    return 1.0  # No adjustment
            
        except Exception as e:
            logger.warning(f"Could not calculate correlation adjustment: {str(e)}")
        
        return 1.0
    
    async def _calculate_recovery_signals(self, lookback_days: int) -> Dict[str, Any]:
        """Calculate market recovery signals."""
        try:
            # Simple recovery signals based on market momentum
            spy_data = yf.download("SPY", period=f"{lookback_days}d", progress=False)
            
            if not spy_data.empty and len(spy_data) > 20:
                prices = spy_data['Adj Close']
                
                # Calculate momentum indicators
                sma_20 = prices.rolling(20).mean()
                sma_50 = prices.rolling(50).mean() if len(prices) > 50 else sma_20
                
                current_price = prices.iloc[-1]
                
                # Recovery signals
                above_sma20 = current_price > sma_20.iloc[-1]
                above_sma50 = current_price > sma_50.iloc[-1]
                price_momentum = (current_price / prices.iloc[-20] - 1) if len(prices) > 20 else 0
                
                # Calculate confidence score
                confidence = 0.0
                if above_sma20:
                    confidence += 0.3
                if above_sma50:
                    confidence += 0.3
                if price_momentum > 0.02:  # 2% gain in 20 days
                    confidence += 0.4
                
                return {
                    "confidence": confidence,
                    "above_sma20": above_sma20,
                    "above_sma50": above_sma50,
                    "price_momentum": price_momentum,
                    "recovery_strength": "strong" if confidence > 0.7 else "moderate" if confidence > 0.4 else "weak"
                }
            
        except Exception as e:
            logger.warning(f"Could not calculate recovery signals: {str(e)}")
        
        return {
            "confidence": 0.5,
            "recovery_strength": "unknown"
        }
    
    async def _calculate_comprehensive_risk_metrics(
        self,
        original_weights: Dict[str, float],
        final_weights: Dict[str, float],
        tickers: List[str],
        lookback_days: int
    ) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        try:
            # Calculate weight changes
            weight_changes = {}
            for ticker in set(list(original_weights.keys()) + list(final_weights.keys())):
                original = original_weights.get(ticker, 0)
                final = final_weights.get(ticker, 0)
                weight_changes[ticker] = final - original
            
            # Calculate concentration metrics
            original_concentration = sum(w**2 for w in original_weights.values())
            final_concentration = sum(w**2 for w in final_weights.values())
            
            # Calculate defensive allocation
            defensive_assets = ["CASH", "SHY", "BIL", "TLT", "GLD"]
            original_defensive = sum(original_weights.get(asset, 0) for asset in defensive_assets)
            final_defensive = sum(final_weights.get(asset, 0) for asset in defensive_assets)
            
            return {
                "weight_changes": weight_changes,
                "concentration_change": final_concentration - original_concentration,
                "defensive_allocation_change": final_defensive - original_defensive,
                "total_adjustments": len([k for k, v in weight_changes.items() if abs(v) > 0.01]),
                "max_position_change": max([abs(v) for v in weight_changes.values()]) if weight_changes else 0,
                "risk_reduction_score": max(0, (original_concentration - final_concentration) / original_concentration) if original_concentration > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}
    
    def _generate_risk_management_summary(
        self,
        vol_sizing: Dict,
        hedge_result: Dict,
        drawdown_result: Dict,
        risk_metrics: Dict,
        risk_regime: RiskRegime
    ) -> Dict[str, Any]:
        """Generate comprehensive risk management summary."""
        
        actions_taken = []
        
        # Volatility sizing actions
        if vol_sizing.get("scaling_applied", False):
            actions_taken.append(f"Applied volatility scaling - reduced portfolio risk by {vol_sizing.get('risk_reduction', 0):.1%}")
        
        # Hedging actions
        hedge_strategies = hedge_result.get("hedge_strategies", [])
        if hedge_strategies:
            actions_taken.append(f"Implemented {len(hedge_strategies)} tail hedging strategies")
        
        # Drawdown control actions
        drawdown_severity = drawdown_result.get("drawdown_metrics", {}).get("drawdown_severity")
        if drawdown_severity and drawdown_severity != DrawdownSeverity.MINOR:
            actions_taken.append(f"Applied {drawdown_severity} drawdown controls")
        
        # Overall risk assessment
        risk_score = {
            RiskRegime.LOW_RISK: 1,
            RiskRegime.MODERATE_RISK: 2,
            RiskRegime.HIGH_RISK: 3,
            RiskRegime.EXTREME_RISK: 4
        }.get(risk_regime, 2)
        
        effectiveness_score = risk_metrics.get("risk_reduction_score", 0) * 100
        
        return {
            "risk_regime": risk_regime,
            "risk_score": risk_score,
            "actions_taken": actions_taken,
            "effectiveness_score": effectiveness_score,
            "defensive_allocation": drawdown_result.get("control_effectiveness", {}).get("defensive_allocation", 0),
            "hedge_allocation": hedge_result.get("total_hedge_allocation", 0),
            "recommendations": self._generate_recommendations(risk_regime, vol_sizing, hedge_result, drawdown_result)
        }
    
    def _generate_recommendations(
        self,
        risk_regime: RiskRegime,
        vol_sizing: Dict,
        hedge_result: Dict,
        drawdown_result: Dict
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if risk_regime in [RiskRegime.HIGH_RISK, RiskRegime.EXTREME_RISK]:
            recommendations.append("Consider increasing hedge budget given elevated risk environment")
            recommendations.append("Monitor tail hedging positions for effectiveness")
        
        portfolio_vol = vol_sizing.get("portfolio_volatility", 0)
        if portfolio_vol > 0.2:
            recommendations.append("Portfolio volatility elevated - consider further position sizing adjustments")
        
        drawdown = drawdown_result.get("drawdown_metrics", {}).get("current_drawdown", 0)
        if drawdown > 0.1:
            recommendations.append("Significant drawdown detected - maintain defensive positions until recovery signals strengthen")
        
        recovery_confidence = drawdown_result.get("recovery_signals", {}).get("confidence", 0)
        if recovery_confidence > 0.7 and drawdown < 0.05:
            recommendations.append("Strong recovery signals - consider gradually reducing defensive allocations")
        
        return recommendations
