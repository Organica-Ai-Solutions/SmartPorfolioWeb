import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass
import asyncio
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MarketCondition(str, Enum):
    CALM = "calm"
    VOLATILE = "volatile"
    STRESSED = "stressed"
    CRISIS = "crisis"

class LiquidityTier(str, Enum):
    TIER_1 = "tier_1"  # Most liquid (cash, major ETFs)
    TIER_2 = "tier_2"  # Highly liquid (large cap stocks)
    TIER_3 = "tier_3"  # Moderately liquid (mid cap, bonds)
    TIER_4 = "tier_4"  # Lower liquidity (small cap, emerging markets)
    TIER_5 = "tier_5"  # Illiquid (alternatives, micro cap)

class RebalanceFrequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    OPPORTUNISTIC = "opportunistic"

@dataclass
class LiquidityMetrics:
    symbol: str
    daily_volume: float
    avg_bid_ask_spread: float
    price_impact: float
    volatility: float
    liquidity_tier: LiquidityTier
    liquidity_score: float
    days_to_liquidate: float  # Days to liquidate position without major impact

@dataclass
class CashBufferPolicy:
    base_cash_percentage: float = 0.05  # 5% base cash
    volatility_multiplier: float = 2.0   # Scale with volatility
    stress_multiplier: float = 3.0       # Additional buffer during stress
    max_cash_percentage: float = 0.25    # Maximum 25% cash
    min_cash_percentage: float = 0.02    # Minimum 2% cash

class LiquidityManagementService:
    """Advanced liquidity management with cash buffers, rebalancing frequency, and liquidity scoring."""
    
    def __init__(self):
        # Market condition thresholds
        self.volatility_thresholds = {
            MarketCondition.CALM: 0.15,      # VIX < 15
            MarketCondition.VOLATILE: 0.25,  # VIX 15-25
            MarketCondition.STRESSED: 0.35,  # VIX 25-35
            MarketCondition.CRISIS: float('inf')  # VIX > 35
        }
        
        # Liquidity tier definitions
        self.tier_definitions = {
            LiquidityTier.TIER_1: {
                "min_daily_volume": 10_000_000,  # $10M daily volume
                "max_bid_ask_spread": 0.001,     # 0.1% spread
                "examples": ["SPY", "QQQ", "IWM", "CASH"]
            },
            LiquidityTier.TIER_2: {
                "min_daily_volume": 1_000_000,   # $1M daily volume
                "max_bid_ask_spread": 0.005,     # 0.5% spread
                "examples": ["AAPL", "MSFT", "GOOGL"]
            },
            LiquidityTier.TIER_3: {
                "min_daily_volume": 100_000,     # $100K daily volume
                "max_bid_ask_spread": 0.01,      # 1% spread
                "examples": ["Medium cap stocks", "Investment grade bonds"]
            },
            LiquidityTier.TIER_4: {
                "min_daily_volume": 10_000,      # $10K daily volume
                "max_bid_ask_spread": 0.02,      # 2% spread
                "examples": ["Small cap stocks", "Emerging market ETFs"]
            },
            LiquidityTier.TIER_5: {
                "min_daily_volume": 0,           # Any volume
                "max_bid_ask_spread": float('inf'),  # Any spread
                "examples": ["Micro cap", "Private equity", "Real estate"]
            }
        }
        
        # Default cash buffer policy
        self.cash_policy = CashBufferPolicy()
        
        # Rebalancing frequency mapping
        self.rebalance_mapping = {
            MarketCondition.CALM: RebalanceFrequency.MONTHLY,
            MarketCondition.VOLATILE: RebalanceFrequency.WEEKLY,
            MarketCondition.STRESSED: RebalanceFrequency.DAILY,
            MarketCondition.CRISIS: RebalanceFrequency.OPPORTUNISTIC
        }
        
        # Cache for liquidity metrics
        self._liquidity_cache = {}
        self._cache_timestamp = None
        self._cache_expiry = timedelta(hours=4)
    
    async def calculate_optimal_cash_buffer(
        self,
        portfolio_value: float,
        current_positions: Dict[str, float],
        volatility_forecast: Optional[float] = None,
        stress_indicators: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate optimal cash buffer based on volatility forecasts and market conditions.
        """
        try:
            logger.info("Calculating optimal cash buffer")
            
            # Get current market condition
            market_condition = await self._assess_market_condition()
            
            # Get volatility forecast if not provided
            if volatility_forecast is None:
                volatility_forecast = await self._forecast_portfolio_volatility(current_positions)
            
            # Calculate base cash requirement
            base_cash_pct = self.cash_policy.base_cash_percentage
            
            # Volatility adjustment
            vol_adjustment = 0.0
            if volatility_forecast > 0.20:  # Above 20% volatility
                excess_vol = volatility_forecast - 0.20
                vol_adjustment = excess_vol * self.cash_policy.volatility_multiplier
            
            # Stress adjustment
            stress_adjustment = 0.0
            if stress_indicators:
                stress_score = self._calculate_stress_score(stress_indicators)
                if stress_score > 0.5:  # Moderate stress
                    stress_adjustment = (stress_score - 0.5) * self.cash_policy.stress_multiplier
            
            # Market condition adjustment
            condition_adjustment = {
                MarketCondition.CALM: 0.0,
                MarketCondition.VOLATILE: 0.02,    # +2%
                MarketCondition.STRESSED: 0.05,   # +5%
                MarketCondition.CRISIS: 0.10      # +10%
            }.get(market_condition, 0.0)
            
            # Calculate total cash requirement
            total_cash_pct = min(
                base_cash_pct + vol_adjustment + stress_adjustment + condition_adjustment,
                self.cash_policy.max_cash_percentage
            )
            total_cash_pct = max(total_cash_pct, self.cash_policy.min_cash_percentage)
            
            # Calculate required cash amount
            required_cash = portfolio_value * total_cash_pct
            current_cash = current_positions.get("CASH", 0.0) * portfolio_value
            cash_adjustment = required_cash - current_cash
            
            # Determine funding source/target for cash adjustment
            funding_plan = await self._plan_cash_adjustment(
                cash_adjustment, current_positions, portfolio_value
            )
            
            logger.info(f"Optimal cash buffer calculated: {total_cash_pct:.1%}")
            
            return {
                "target_cash_percentage": total_cash_pct,
                "target_cash_amount": required_cash,
                "current_cash_amount": current_cash,
                "cash_adjustment_needed": cash_adjustment,
                "market_condition": market_condition,
                "volatility_forecast": volatility_forecast,
                "adjustments": {
                    "base_cash": base_cash_pct,
                    "volatility_adjustment": vol_adjustment,
                    "stress_adjustment": stress_adjustment,
                    "condition_adjustment": condition_adjustment
                },
                "funding_plan": funding_plan,
                "rationale": self._generate_cash_buffer_rationale(
                    market_condition, volatility_forecast, total_cash_pct
                )
            }
            
        except Exception as e:
            logger.error(f"Error calculating cash buffer: {str(e)}")
            return {
                "target_cash_percentage": self.cash_policy.base_cash_percentage,
                "error": str(e)
            }
    
    async def determine_rebalancing_frequency(
        self,
        current_positions: Dict[str, float],
        market_volatility: Optional[float] = None,
        liquidity_constraints: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Determine optimal rebalancing frequency based on market conditions.
        """
        try:
            logger.info("Determining optimal rebalancing frequency")
            
            # Assess market condition
            market_condition = await self._assess_market_condition()
            
            # Get base frequency from market condition
            base_frequency = self.rebalance_mapping[market_condition]
            
            # Calculate drift metrics
            drift_metrics = await self._calculate_portfolio_drift(current_positions)
            
            # Liquidity assessment
            liquidity_assessment = await self._assess_portfolio_liquidity(current_positions)
            
            # Transaction cost analysis
            cost_analysis = await self._estimate_rebalancing_costs(current_positions)
            
            # Volatility clustering detection
            vol_clustering = await self._detect_volatility_clustering()
            
            # Adjust frequency based on multiple factors
            frequency_score = self._calculate_frequency_score(
                market_condition=market_condition,
                drift_magnitude=drift_metrics["max_drift"],
                avg_liquidity=liquidity_assessment["avg_liquidity_score"],
                transaction_costs=cost_analysis["total_cost_bps"],
                volatility_clustering=vol_clustering["is_clustering"]
            )
            
            # Determine final frequency
            final_frequency = self._score_to_frequency(frequency_score)
            
            # Calculate next rebalancing date
            next_rebalance = self._calculate_next_rebalance_date(final_frequency)
            
            logger.info(f"Optimal rebalancing frequency: {final_frequency}")
            
            return {
                "recommended_frequency": final_frequency,
                "market_condition": market_condition,
                "next_rebalance_date": next_rebalance.isoformat(),
                "frequency_score": frequency_score,
                "analysis": {
                    "drift_metrics": drift_metrics,
                    "liquidity_assessment": liquidity_assessment,
                    "cost_analysis": cost_analysis,
                    "volatility_clustering": vol_clustering
                },
                "rationale": self._generate_frequency_rationale(
                    final_frequency, market_condition, drift_metrics, cost_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Error determining rebalancing frequency: {str(e)}")
            return {
                "recommended_frequency": RebalanceFrequency.MONTHLY,
                "error": str(e)
            }
    
    async def score_asset_liquidity(
        self,
        symbols: List[str],
        position_sizes: Optional[Dict[str, float]] = None,
        time_horizon: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Score asset liquidity to avoid illiquid assets during market stress.
        """
        try:
            logger.info(f"Scoring liquidity for {len(symbols)} assets")
            
            liquidity_metrics = {}
            
            for symbol in symbols:
                metrics = await self._calculate_asset_liquidity_metrics(
                    symbol, position_sizes.get(symbol) if position_sizes else None
                )
                liquidity_metrics[symbol] = metrics
            
            # Portfolio-level liquidity analysis
            portfolio_liquidity = self._analyze_portfolio_liquidity(
                liquidity_metrics, position_sizes
            )
            
            # Stress test liquidity
            stress_scenarios = await self._stress_test_liquidity(
                liquidity_metrics, position_sizes
            )
            
            # Generate recommendations
            recommendations = self._generate_liquidity_recommendations(
                liquidity_metrics, portfolio_liquidity, stress_scenarios
            )
            
            logger.info("Asset liquidity scoring completed")
            
            return {
                "asset_liquidity_metrics": {
                    symbol: {
                        "liquidity_tier": metrics.liquidity_tier.value,
                        "liquidity_score": metrics.liquidity_score,
                        "daily_volume": metrics.daily_volume,
                        "bid_ask_spread": metrics.avg_bid_ask_spread,
                        "days_to_liquidate": metrics.days_to_liquidate,
                        "price_impact": metrics.price_impact
                    }
                    for symbol, metrics in liquidity_metrics.items()
                },
                "portfolio_liquidity": portfolio_liquidity,
                "stress_scenarios": stress_scenarios,
                "recommendations": recommendations,
                "overall_liquidity_score": portfolio_liquidity["weighted_liquidity_score"]
            }
            
        except Exception as e:
            logger.error(f"Error scoring asset liquidity: {str(e)}")
            return {"error": str(e)}
    
    async def generate_liquidity_aware_allocation(
        self,
        target_allocation: Dict[str, float],
        market_condition: Optional[MarketCondition] = None,
        liquidity_requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate allocation that considers liquidity constraints.
        """
        try:
            logger.info("Generating liquidity-aware allocation")
            
            if market_condition is None:
                market_condition = await self._assess_market_condition()
            
            # Score liquidity of target assets
            symbols = list(target_allocation.keys())
            liquidity_analysis = await self.score_asset_liquidity(symbols, target_allocation)
            
            # Apply liquidity constraints based on market condition
            adjusted_allocation = self._apply_liquidity_constraints(
                target_allocation,
                liquidity_analysis["asset_liquidity_metrics"],
                market_condition
            )
            
            # Ensure cash buffer requirements
            cash_buffer_result = await self.calculate_optimal_cash_buffer(
                portfolio_value=100000,  # Normalized
                current_positions=adjusted_allocation
            )
            
            # Final adjustment for cash buffer
            final_allocation = self._incorporate_cash_buffer(
                adjusted_allocation,
                cash_buffer_result["target_cash_percentage"]
            )
            
            # Calculate impact metrics
            impact_metrics = self._calculate_allocation_impact(
                target_allocation, final_allocation, liquidity_analysis
            )
            
            logger.info("Liquidity-aware allocation generated")
            
            return {
                "original_allocation": target_allocation,
                "liquidity_adjusted_allocation": final_allocation,
                "market_condition": market_condition,
                "liquidity_analysis": liquidity_analysis,
                "cash_buffer_info": cash_buffer_result,
                "impact_metrics": impact_metrics,
                "adjustments_made": impact_metrics["total_adjustments"] > 0
            }
            
        except Exception as e:
            logger.error(f"Error generating liquidity-aware allocation: {str(e)}")
            return {
                "liquidity_adjusted_allocation": target_allocation,
                "error": str(e)
            }
    
    # Helper methods
    
    async def _assess_market_condition(self) -> MarketCondition:
        """Assess current market condition based on volatility indicators."""
        try:
            # Get VIX data
            vix_data = yf.download("^VIX", period="5d", progress=False)
            
            if not vix_data.empty:
                current_vix = float(vix_data['Close'].iloc[-1])
                
                if current_vix < 15:
                    return MarketCondition.CALM
                elif current_vix < 25:
                    return MarketCondition.VOLATILE
                elif current_vix < 35:
                    return MarketCondition.STRESSED
                else:
                    return MarketCondition.CRISIS
            
        except Exception as e:
            logger.warning(f"Could not assess market condition: {str(e)}")
        
        return MarketCondition.VOLATILE  # Default to volatile
    
    async def _forecast_portfolio_volatility(self, positions: Dict[str, float]) -> float:
        """Forecast portfolio volatility."""
        try:
            # Simple volatility forecast using recent returns
            vol_estimates = []
            
            for symbol, weight in positions.items():
                if symbol == "CASH":
                    continue
                
                try:
                    data = yf.download(symbol, period="30d", progress=False)
                    if not data.empty:
                        returns = data['Adj Close'].pct_change().dropna()
                        vol = returns.std() * np.sqrt(252) * weight  # Annualized
                        vol_estimates.append(vol)
                except:
                    pass
            
            if vol_estimates:
                return np.sqrt(sum(v**2 for v in vol_estimates))  # Simplified portfolio vol
            else:
                return 0.15  # Default 15% volatility
                
        except Exception as e:
            logger.error(f"Error forecasting volatility: {str(e)}")
            return 0.15
    
    def _calculate_stress_score(self, stress_indicators: Dict[str, float]) -> float:
        """Calculate composite stress score from various indicators."""
        try:
            # Weighted stress score
            weights = {
                "credit_spreads": 0.3,
                "currency_volatility": 0.2,
                "correlation_increase": 0.3,
                "liquidity_decline": 0.2
            }
            
            stress_score = 0.0
            total_weight = 0.0
            
            for indicator, value in stress_indicators.items():
                if indicator in weights:
                    # Normalize indicators to 0-1 scale
                    normalized_value = min(1.0, max(0.0, value))
                    stress_score += normalized_value * weights[indicator]
                    total_weight += weights[indicator]
            
            return stress_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating stress score: {str(e)}")
            return 0.0
    
    async def _plan_cash_adjustment(
        self,
        cash_adjustment: float,
        current_positions: Dict[str, float],
        portfolio_value: float
    ) -> Dict[str, Any]:
        """Plan how to adjust cash position."""
        try:
            if abs(cash_adjustment) < portfolio_value * 0.01:  # Less than 1%
                return {"action": "no_action", "reason": "adjustment_too_small"}
            
            if cash_adjustment > 0:  # Need more cash
                # Find most liquid positions to sell
                liquidity_scores = {}
                for symbol, weight in current_positions.items():
                    if symbol != "CASH" and weight > 0:
                        # Get liquidity score (simplified)
                        score = await self._get_simple_liquidity_score(symbol)
                        liquidity_scores[symbol] = {"weight": weight, "liquidity": score}
                
                # Sort by liquidity (highest first)
                sorted_positions = sorted(
                    liquidity_scores.items(),
                    key=lambda x: x[1]["liquidity"],
                    reverse=True
                )
                
                funding_plan = []
                remaining_needed = abs(cash_adjustment)
                
                for symbol, data in sorted_positions:
                    if remaining_needed <= 0:
                        break
                    
                    position_value = data["weight"] * portfolio_value
                    reduction = min(remaining_needed, position_value * 0.5)  # Max 50% reduction
                    
                    if reduction > portfolio_value * 0.005:  # Min 0.5% of portfolio
                        funding_plan.append({
                            "symbol": symbol,
                            "action": "reduce",
                            "amount": reduction,
                            "percentage": reduction / portfolio_value
                        })
                        remaining_needed -= reduction
                
                return {
                    "action": "raise_cash",
                    "amount_needed": abs(cash_adjustment),
                    "funding_plan": funding_plan,
                    "remaining_shortfall": remaining_needed
                }
            
            else:  # Too much cash, need to invest
                return {
                    "action": "deploy_cash",
                    "amount_to_deploy": abs(cash_adjustment),
                    "suggestion": "increase_allocations_proportionally"
                }
                
        except Exception as e:
            logger.error(f"Error planning cash adjustment: {str(e)}")
            return {"action": "manual_review", "error": str(e)}
    
    async def _get_simple_liquidity_score(self, symbol: str) -> float:
        """Get simplified liquidity score for a symbol."""
        try:
            # Use cached value if available
            cache_key = f"liquidity_{symbol}"
            if (cache_key in self._liquidity_cache and 
                self._cache_timestamp and 
                datetime.now() - self._cache_timestamp < self._cache_expiry):
                return self._liquidity_cache[cache_key]
            
            # Get recent volume data
            data = yf.download(symbol, period="30d", progress=False)
            
            if not data.empty:
                avg_volume = data['Volume'].mean()
                avg_price = data['Close'].mean()
                dollar_volume = avg_volume * avg_price
                
                # Simple scoring based on dollar volume
                if dollar_volume > 10_000_000:  # $10M+
                    score = 1.0
                elif dollar_volume > 1_000_000:  # $1M+
                    score = 0.8
                elif dollar_volume > 100_000:  # $100K+
                    score = 0.6
                elif dollar_volume > 10_000:  # $10K+
                    score = 0.4
                else:
                    score = 0.2
                
                # Cache the result
                self._liquidity_cache[cache_key] = score
                self._cache_timestamp = datetime.now()
                
                return score
            
            return 0.5  # Default moderate liquidity
            
        except Exception as e:
            logger.warning(f"Error getting liquidity score for {symbol}: {str(e)}")
            return 0.5
    
    def _generate_cash_buffer_rationale(
        self,
        market_condition: MarketCondition,
        volatility_forecast: float,
        target_cash_pct: float
    ) -> List[str]:
        """Generate rationale for cash buffer recommendation."""
        rationale = []
        
        if market_condition == MarketCondition.CRISIS:
            rationale.append("Crisis conditions detected - maintaining high cash buffer for flexibility")
        elif market_condition == MarketCondition.STRESSED:
            rationale.append("Market stress detected - increased cash buffer recommended")
        
        if volatility_forecast > 0.25:
            rationale.append(f"High volatility forecast ({volatility_forecast:.1%}) - additional cash buffer needed")
        
        if target_cash_pct > 0.15:
            rationale.append("Elevated cash allocation to preserve capital and provide rebalancing flexibility")
        elif target_cash_pct < 0.05:
            rationale.append("Low cash allocation appropriate given stable market conditions")
        
        return rationale
    
    async def _calculate_portfolio_drift(self, positions: Dict[str, float]) -> Dict[str, Any]:
        """Calculate how much portfolio has drifted from targets."""
        # This would normally compare to target allocation
        # For now, we'll simulate drift detection
        
        try:
            # Simulate drift calculation
            max_drift = 0.0
            avg_drift = 0.0
            drift_by_asset = {}
            
            for symbol, weight in positions.items():
                if symbol != "CASH":
                    # Simulate some drift (in reality, compare to target)
                    simulated_drift = np.random.uniform(0, 0.05)  # 0-5% drift
                    drift_by_asset[symbol] = simulated_drift
                    max_drift = max(max_drift, simulated_drift)
            
            avg_drift = np.mean(list(drift_by_asset.values())) if drift_by_asset else 0.0
            
            return {
                "max_drift": max_drift,
                "avg_drift": avg_drift,
                "drift_by_asset": drift_by_asset,
                "needs_rebalancing": max_drift > 0.03  # 3% threshold
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio drift: {str(e)}")
            return {
                "max_drift": 0.02,
                "avg_drift": 0.01,
                "drift_by_asset": {},
                "needs_rebalancing": False
            }
    
    async def _assess_portfolio_liquidity(self, positions: Dict[str, float]) -> Dict[str, Any]:
        """Assess overall portfolio liquidity."""
        try:
            liquidity_scores = []
            total_weight = 0.0
            
            for symbol, weight in positions.items():
                if weight > 0:
                    if symbol == "CASH":
                        score = 1.0  # Cash is perfectly liquid
                    else:
                        score = await self._get_simple_liquidity_score(symbol)
                    
                    liquidity_scores.append(score * weight)
                    total_weight += weight
            
            weighted_liquidity = sum(liquidity_scores) / total_weight if total_weight > 0 else 0.5
            
            return {
                "avg_liquidity_score": weighted_liquidity,
                "liquidity_tier": self._score_to_tier(weighted_liquidity),
                "illiquid_concentration": sum(
                    weight for symbol, weight in positions.items()
                    if await self._get_simple_liquidity_score(symbol) < 0.5
                )
            }
            
        except Exception as e:
            logger.error(f"Error assessing portfolio liquidity: {str(e)}")
            return {
                "avg_liquidity_score": 0.5,
                "liquidity_tier": LiquidityTier.TIER_3,
                "illiquid_concentration": 0.0
            }
    
    def _score_to_tier(self, score: float) -> LiquidityTier:
        """Convert liquidity score to tier."""
        if score >= 0.9:
            return LiquidityTier.TIER_1
        elif score >= 0.7:
            return LiquidityTier.TIER_2
        elif score >= 0.5:
            return LiquidityTier.TIER_3
        elif score >= 0.3:
            return LiquidityTier.TIER_4
        else:
            return LiquidityTier.TIER_5
    
    async def _estimate_rebalancing_costs(self, positions: Dict[str, float]) -> Dict[str, Any]:
        """Estimate transaction costs for rebalancing."""
        try:
            # Simplified cost estimation
            total_cost_bps = 0.0
            cost_by_asset = {}
            
            for symbol, weight in positions.items():
                if symbol != "CASH" and weight > 0:
                    # Estimate cost based on liquidity tier
                    liquidity_score = await self._get_simple_liquidity_score(symbol)
                    
                    if liquidity_score >= 0.9:
                        cost_bps = 5  # 5 bps for highly liquid
                    elif liquidity_score >= 0.7:
                        cost_bps = 10  # 10 bps for liquid
                    elif liquidity_score >= 0.5:
                        cost_bps = 20  # 20 bps for moderate
                    else:
                        cost_bps = 50  # 50 bps for illiquid
                    
                    weighted_cost = cost_bps * weight
                    cost_by_asset[symbol] = cost_bps
                    total_cost_bps += weighted_cost
            
            return {
                "total_cost_bps": total_cost_bps,
                "cost_by_asset": cost_by_asset,
                "cost_percentage": total_cost_bps / 10000,  # Convert bps to percentage
                "high_cost_threshold": total_cost_bps > 25  # 25 bps threshold
            }
            
        except Exception as e:
            logger.error(f"Error estimating rebalancing costs: {str(e)}")
            return {
                "total_cost_bps": 15.0,
                "cost_by_asset": {},
                "cost_percentage": 0.0015,
                "high_cost_threshold": False
            }
    
    async def _detect_volatility_clustering(self) -> Dict[str, Any]:
        """Detect if we're in a period of volatility clustering."""
        try:
            # Get SPY data as market proxy
            data = yf.download("SPY", period="60d", progress=False)
            
            if not data.empty:
                returns = data['Adj Close'].pct_change().dropna()
                vol_series = returns.rolling(5).std()  # 5-day rolling vol
                
                current_vol = vol_series.iloc[-1]
                avg_vol = vol_series.mean()
                
                # Clustering if current vol > 1.5x average
                is_clustering = current_vol > avg_vol * 1.5
                
                return {
                    "is_clustering": is_clustering,
                    "current_vol": float(current_vol),
                    "average_vol": float(avg_vol),
                    "vol_ratio": float(current_vol / avg_vol)
                }
            
        except Exception as e:
            logger.warning(f"Error detecting volatility clustering: {str(e)}")
        
        return {
            "is_clustering": False,
            "current_vol": 0.01,
            "average_vol": 0.01,
            "vol_ratio": 1.0
        }
    
    def _calculate_frequency_score(
        self,
        market_condition: MarketCondition,
        drift_magnitude: float,
        avg_liquidity: float,
        transaction_costs: float,
        volatility_clustering: bool
    ) -> float:
        """Calculate rebalancing frequency score (higher = more frequent)."""
        try:
            score = 0.0
            
            # Market condition component
            condition_scores = {
                MarketCondition.CALM: 0.2,
                MarketCondition.VOLATILE: 0.5,
                MarketCondition.STRESSED: 0.8,
                MarketCondition.CRISIS: 1.0
            }
            score += condition_scores.get(market_condition, 0.5) * 0.3
            
            # Drift component
            if drift_magnitude > 0.05:  # >5% drift
                score += 0.3
            elif drift_magnitude > 0.03:  # >3% drift
                score += 0.2
            elif drift_magnitude > 0.01:  # >1% drift
                score += 0.1
            
            # Liquidity component (higher liquidity = can rebalance more often)
            score += avg_liquidity * 0.2
            
            # Cost component (higher costs = rebalance less often)
            if transaction_costs > 30:  # >30 bps
                score -= 0.2
            elif transaction_costs > 20:  # >20 bps
                score -= 0.1
            
            # Volatility clustering
            if volatility_clustering:
                score += 0.2
            
            return max(0.0, min(1.0, score))  # Clamp to 0-1
            
        except Exception as e:
            logger.error(f"Error calculating frequency score: {str(e)}")
            return 0.5
    
    def _score_to_frequency(self, score: float) -> RebalanceFrequency:
        """Convert frequency score to rebalancing frequency."""
        if score >= 0.8:
            return RebalanceFrequency.DAILY
        elif score >= 0.6:
            return RebalanceFrequency.WEEKLY
        elif score >= 0.4:
            return RebalanceFrequency.MONTHLY
        else:
            return RebalanceFrequency.QUARTERLY
    
    def _calculate_next_rebalance_date(self, frequency: RebalanceFrequency) -> datetime:
        """Calculate next rebalancing date."""
        now = datetime.now()
        
        if frequency == RebalanceFrequency.DAILY:
            return now + timedelta(days=1)
        elif frequency == RebalanceFrequency.WEEKLY:
            return now + timedelta(weeks=1)
        elif frequency == RebalanceFrequency.MONTHLY:
            return now + timedelta(days=30)
        elif frequency == RebalanceFrequency.QUARTERLY:
            return now + timedelta(days=90)
        else:  # OPPORTUNISTIC
            return now + timedelta(days=7)  # Check weekly for opportunities
    
    def _generate_frequency_rationale(
        self,
        frequency: RebalanceFrequency,
        market_condition: MarketCondition,
        drift_metrics: Dict,
        cost_analysis: Dict
    ) -> List[str]:
        """Generate rationale for rebalancing frequency."""
        rationale = []
        
        if frequency == RebalanceFrequency.DAILY:
            rationale.append("Daily rebalancing recommended due to high market volatility or significant drift")
        elif frequency == RebalanceFrequency.WEEKLY:
            rationale.append("Weekly rebalancing appropriate for current market conditions")
        elif frequency == RebalanceFrequency.MONTHLY:
            rationale.append("Monthly rebalancing suitable given moderate drift and transaction costs")
        else:
            rationale.append("Quarterly rebalancing sufficient in current stable environment")
        
        if market_condition in [MarketCondition.STRESSED, MarketCondition.CRISIS]:
            rationale.append(f"Market stress ({market_condition}) requires more frequent monitoring")
        
        if drift_metrics["max_drift"] > 0.05:
            rationale.append("Significant portfolio drift detected - more frequent rebalancing needed")
        
        if cost_analysis["total_cost_bps"] > 25:
            rationale.append("High transaction costs favor less frequent rebalancing")
        
        return rationale
    
    async def _calculate_asset_liquidity_metrics(
        self,
        symbol: str,
        position_size: Optional[float] = None
    ) -> LiquidityMetrics:
        """Calculate comprehensive liquidity metrics for an asset."""
        try:
            # Get market data
            data = yf.download(symbol, period="60d", progress=False)
            
            if data.empty:
                return LiquidityMetrics(
                    symbol=symbol,
                    daily_volume=0,
                    avg_bid_ask_spread=0.05,  # 5% default spread
                    price_impact=0.02,
                    volatility=0.25,
                    liquidity_tier=LiquidityTier.TIER_5,
                    liquidity_score=0.1,
                    days_to_liquidate=10
                )
            
            # Calculate metrics
            avg_volume = data['Volume'].mean()
            avg_price = data['Close'].mean()
            daily_dollar_volume = avg_volume * avg_price
            
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            # Estimate bid-ask spread (simplified)
            high_low_spread = ((data['High'] - data['Low']) / data['Close']).mean()
            estimated_spread = min(0.05, high_low_spread * 0.5)  # Conservative estimate
            
            # Calculate liquidity score
            liquidity_score = self._calculate_liquidity_score(
                daily_dollar_volume, estimated_spread, volatility
            )
            
            # Determine liquidity tier
            liquidity_tier = self._determine_liquidity_tier(
                daily_dollar_volume, estimated_spread
            )
            
            # Estimate price impact
            price_impact = self._estimate_price_impact(
                daily_dollar_volume, position_size, volatility
            )
            
            # Days to liquidate
            if position_size and daily_dollar_volume > 0:
                days_to_liquidate = (position_size * avg_price) / (daily_dollar_volume * 0.1)  # 10% participation
            else:
                days_to_liquidate = 1.0  # Default
            
            return LiquidityMetrics(
                symbol=symbol,
                daily_volume=daily_dollar_volume,
                avg_bid_ask_spread=estimated_spread,
                price_impact=price_impact,
                volatility=volatility,
                liquidity_tier=liquidity_tier,
                liquidity_score=liquidity_score,
                days_to_liquidate=days_to_liquidate
            )
            
        except Exception as e:
            logger.error(f"Error calculating liquidity metrics for {symbol}: {str(e)}")
            return LiquidityMetrics(
                symbol=symbol,
                daily_volume=100000,
                avg_bid_ask_spread=0.01,
                price_impact=0.005,
                volatility=0.20,
                liquidity_tier=LiquidityTier.TIER_3,
                liquidity_score=0.5,
                days_to_liquidate=2.0
            )
    
    def _calculate_liquidity_score(
        self,
        daily_dollar_volume: float,
        bid_ask_spread: float,
        volatility: float
    ) -> float:
        """Calculate composite liquidity score."""
        try:
            # Volume component (0-1)
            if daily_dollar_volume >= 10_000_000:
                vol_score = 1.0
            elif daily_dollar_volume >= 1_000_000:
                vol_score = 0.8
            elif daily_dollar_volume >= 100_000:
                vol_score = 0.6
            elif daily_dollar_volume >= 10_000:
                vol_score = 0.4
            else:
                vol_score = 0.2
            
            # Spread component (0-1, lower spread = higher score)
            spread_score = max(0, 1.0 - bid_ask_spread * 100)  # 1% spread = 0 score
            
            # Volatility component (0-1, lower vol = higher score for liquidity)
            vol_component = max(0, 1.0 - volatility * 2)  # 50% vol = 0 score
            
            # Weighted composite
            composite_score = (
                vol_score * 0.5 +
                spread_score * 0.3 +
                vol_component * 0.2
            )
            
            return max(0.0, min(1.0, composite_score))
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {str(e)}")
            return 0.5
    
    def _determine_liquidity_tier(
        self,
        daily_dollar_volume: float,
        bid_ask_spread: float
    ) -> LiquidityTier:
        """Determine liquidity tier based on volume and spread."""
        if (daily_dollar_volume >= 10_000_000 and bid_ask_spread <= 0.001):
            return LiquidityTier.TIER_1
        elif (daily_dollar_volume >= 1_000_000 and bid_ask_spread <= 0.005):
            return LiquidityTier.TIER_2
        elif (daily_dollar_volume >= 100_000 and bid_ask_spread <= 0.01):
            return LiquidityTier.TIER_3
        elif (daily_dollar_volume >= 10_000 and bid_ask_spread <= 0.02):
            return LiquidityTier.TIER_4
        else:
            return LiquidityTier.TIER_5
    
    def _estimate_price_impact(
        self,
        daily_dollar_volume: float,
        position_size: Optional[float],
        volatility: float
    ) -> float:
        """Estimate price impact of trading a position."""
        if not position_size or daily_dollar_volume == 0:
            return 0.001  # 0.1% default
        
        # Simplified square-root impact model
        participation_rate = position_size / daily_dollar_volume
        impact = np.sqrt(participation_rate) * volatility * 0.5
        
        return min(0.05, impact)  # Cap at 5%
    
    def _analyze_portfolio_liquidity(
        self,
        liquidity_metrics: Dict[str, LiquidityMetrics],
        position_sizes: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Analyze portfolio-level liquidity."""
        try:
            if not liquidity_metrics:
                return {"weighted_liquidity_score": 0.5}
            
            total_weight = 0.0
            weighted_score = 0.0
            tier_distribution = {tier: 0.0 for tier in LiquidityTier}
            
            for symbol, metrics in liquidity_metrics.items():
                weight = position_sizes.get(symbol, 1.0 / len(liquidity_metrics)) if position_sizes else 1.0 / len(liquidity_metrics)
                
                weighted_score += metrics.liquidity_score * weight
                tier_distribution[metrics.liquidity_tier] += weight
                total_weight += weight
            
            if total_weight > 0:
                weighted_score /= total_weight
                # Normalize tier distribution
                for tier in tier_distribution:
                    tier_distribution[tier] /= total_weight
            
            # Calculate concentration in illiquid assets
            illiquid_concentration = (
                tier_distribution[LiquidityTier.TIER_4] +
                tier_distribution[LiquidityTier.TIER_5]
            )
            
            return {
                "weighted_liquidity_score": weighted_score,
                "tier_distribution": {tier.value: weight for tier, weight in tier_distribution.items()},
                "illiquid_concentration": illiquid_concentration,
                "overall_tier": self._score_to_tier(weighted_score),
                "liquidity_quality": "High" if weighted_score > 0.8 else "Medium" if weighted_score > 0.5 else "Low"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio liquidity: {str(e)}")
            return {"weighted_liquidity_score": 0.5}
    
    async def _stress_test_liquidity(
        self,
        liquidity_metrics: Dict[str, LiquidityMetrics],
        position_sizes: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Stress test portfolio liquidity under various scenarios."""
        try:
            scenarios = {
                "mild_stress": {"volume_decline": 0.3, "spread_increase": 2.0},
                "moderate_stress": {"volume_decline": 0.5, "spread_increase": 3.0},
                "severe_stress": {"volume_decline": 0.7, "spread_increase": 5.0}
            }
            
            stress_results = {}
            
            for scenario_name, params in scenarios.items():
                scenario_score = 0.0
                total_weight = 0.0
                max_days_to_liquidate = 0.0
                
                for symbol, metrics in liquidity_metrics.items():
                    weight = position_sizes.get(symbol, 1.0 / len(liquidity_metrics)) if position_sizes else 1.0 / len(liquidity_metrics)
                    
                    # Adjust metrics for stress
                    stressed_volume = metrics.daily_volume * (1 - params["volume_decline"])
                    stressed_spread = metrics.avg_bid_ask_spread * params["spread_increase"]
                    
                    # Recalculate score
                    stressed_score = self._calculate_liquidity_score(
                        stressed_volume, stressed_spread, metrics.volatility
                    )
                    
                    # Recalculate days to liquidate
                    if position_sizes and position_sizes.get(symbol, 0) > 0:
                        stressed_days = metrics.days_to_liquidate * params["spread_increase"]
                    else:
                        stressed_days = 1.0
                    
                    scenario_score += stressed_score * weight
                    max_days_to_liquidate = max(max_days_to_liquidate, stressed_days)
                    total_weight += weight
                
                if total_weight > 0:
                    scenario_score /= total_weight
                
                stress_results[scenario_name] = {
                    "liquidity_score": scenario_score,
                    "max_liquidation_days": max_days_to_liquidate,
                    "scenario_parameters": params
                }
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error in liquidity stress test: {str(e)}")
            return {}
    
    def _generate_liquidity_recommendations(
        self,
        liquidity_metrics: Dict[str, LiquidityMetrics],
        portfolio_liquidity: Dict[str, Any],
        stress_scenarios: Dict[str, Any]
    ) -> List[str]:
        """Generate liquidity-based recommendations."""
        recommendations = []
        
        overall_score = portfolio_liquidity.get("weighted_liquidity_score", 0.5)
        illiquid_concentration = portfolio_liquidity.get("illiquid_concentration", 0.0)
        
        if overall_score < 0.5:
            recommendations.append("Consider increasing allocation to more liquid assets")
        
        if illiquid_concentration > 0.3:
            recommendations.append("High concentration in illiquid assets - consider reducing for better flexibility")
        
        # Check for assets with poor liquidity
        for symbol, metrics in liquidity_metrics.items():
            if metrics.liquidity_score < 0.3:
                recommendations.append(f"Consider reducing exposure to {symbol} due to poor liquidity")
            
            if metrics.days_to_liquidate > 10:
                recommendations.append(f"{symbol} may take {metrics.days_to_liquidate:.1f} days to liquidate - monitor position size")
        
        # Stress test recommendations
        if stress_scenarios:
            severe_stress = stress_scenarios.get("severe_stress", {})
            if severe_stress.get("liquidity_score", 1.0) < 0.3:
                recommendations.append("Portfolio may face significant liquidity constraints under stress - consider increasing cash buffer")
        
        return recommendations
    
    def _apply_liquidity_constraints(
        self,
        target_allocation: Dict[str, float],
        liquidity_metrics: Dict[str, Dict],
        market_condition: MarketCondition
    ) -> Dict[str, float]:
        """Apply liquidity constraints to target allocation."""
        try:
            adjusted_allocation = target_allocation.copy()
            
            # Define maximum allocation limits based on liquidity tier and market condition
            max_allocations = {
                MarketCondition.CALM: {
                    "tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.4, "tier_5": 0.2
                },
                MarketCondition.VOLATILE: {
                    "tier_1": 1.0, "tier_2": 0.7, "tier_3": 0.5, "tier_4": 0.3, "tier_5": 0.1
                },
                MarketCondition.STRESSED: {
                    "tier_1": 1.0, "tier_2": 0.6, "tier_3": 0.4, "tier_4": 0.2, "tier_5": 0.05
                },
                MarketCondition.CRISIS: {
                    "tier_1": 1.0, "tier_2": 0.5, "tier_3": 0.3, "tier_4": 0.1, "tier_5": 0.02
                }
            }
            
            limits = max_allocations[market_condition]
            total_reduction = 0.0
            
            for symbol, allocation in target_allocation.items():
                if symbol in liquidity_metrics:
                    tier = liquidity_metrics[symbol]["liquidity_tier"]
                    max_allowed = limits.get(tier, 0.1)
                    
                    if allocation > max_allowed:
                        reduction = allocation - max_allowed
                        adjusted_allocation[symbol] = max_allowed
                        total_reduction += reduction
            
            # Redistribute excess to more liquid assets
            if total_reduction > 0:
                liquid_assets = [
                    symbol for symbol, metrics in liquidity_metrics.items()
                    if metrics["liquidity_tier"] in ["tier_1", "tier_2"]
                ]
                
                if liquid_assets:
                    redistribution_per_asset = total_reduction / len(liquid_assets)
                    for symbol in liquid_assets:
                        adjusted_allocation[symbol] = adjusted_allocation.get(symbol, 0) + redistribution_per_asset
            
            return adjusted_allocation
            
        except Exception as e:
            logger.error(f"Error applying liquidity constraints: {str(e)}")
            return target_allocation
    
    def _incorporate_cash_buffer(
        self,
        allocation: Dict[str, float],
        target_cash_pct: float
    ) -> Dict[str, float]:
        """Incorporate required cash buffer into allocation."""
        try:
            final_allocation = allocation.copy()
            current_cash = allocation.get("CASH", 0.0)
            
            if current_cash < target_cash_pct:
                # Need more cash
                cash_needed = target_cash_pct - current_cash
                scaling_factor = (1.0 - target_cash_pct) / (1.0 - current_cash)
                
                # Scale down all non-cash positions
                for symbol in final_allocation:
                    if symbol != "CASH":
                        final_allocation[symbol] *= scaling_factor
                
                final_allocation["CASH"] = target_cash_pct
            
            return final_allocation
            
        except Exception as e:
            logger.error(f"Error incorporating cash buffer: {str(e)}")
            return allocation
    
    def _calculate_allocation_impact(
        self,
        original: Dict[str, float],
        adjusted: Dict[str, float],
        liquidity_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate impact of liquidity adjustments."""
        try:
            total_adjustments = 0.0
            adjustments_by_asset = {}
            
            all_symbols = set(list(original.keys()) + list(adjusted.keys()))
            
            for symbol in all_symbols:
                orig_weight = original.get(symbol, 0.0)
                adj_weight = adjusted.get(symbol, 0.0)
                change = adj_weight - orig_weight
                
                if abs(change) > 0.001:  # More than 0.1% change
                    adjustments_by_asset[symbol] = {
                        "original": orig_weight,
                        "adjusted": adj_weight,
                        "change": change,
                        "change_pct": change / orig_weight if orig_weight > 0 else float('inf')
                    }
                    total_adjustments += abs(change)
            
            return {
                "total_adjustments": total_adjustments,
                "adjustments_by_asset": adjustments_by_asset,
                "num_assets_adjusted": len(adjustments_by_asset),
                "liquidity_score_change": self._calculate_score_change(original, adjusted, liquidity_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error calculating allocation impact: {str(e)}")
            return {"total_adjustments": 0.0}
    
    def _calculate_score_change(
        self,
        original: Dict[str, float],
        adjusted: Dict[str, float],
        liquidity_analysis: Dict[str, Any]
    ) -> float:
        """Calculate change in overall liquidity score."""
        try:
            asset_metrics = liquidity_analysis.get("asset_liquidity_metrics", {})
            
            def calculate_portfolio_score(allocation):
                total_score = 0.0
                total_weight = 0.0
                
                for symbol, weight in allocation.items():
                    if symbol in asset_metrics:
                        score = asset_metrics[symbol]["liquidity_score"]
                        total_score += score * weight
                        total_weight += weight
                    elif symbol == "CASH":
                        total_score += 1.0 * weight  # Cash has perfect liquidity
                        total_weight += weight
                
                return total_score / total_weight if total_weight > 0 else 0.5
            
            original_score = calculate_portfolio_score(original)
            adjusted_score = calculate_portfolio_score(adjusted)
            
            return adjusted_score - original_score
            
        except Exception as e:
            logger.error(f"Error calculating score change: {str(e)}")
            return 0.0
