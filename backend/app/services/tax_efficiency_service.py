import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta, date
from enum import Enum
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass
import asyncio
from decimal import Decimal, ROUND_HALF_UP
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AccountType(str, Enum):
    TAXABLE = "taxable"
    TRADITIONAL_IRA = "traditional_ira"
    ROTH_IRA = "roth_ira"
    HSA = "hsa"
    FOUR_OH_ONE_K = "401k"
    ROTH_401K = "roth_401k"
    TAXABLE_TRUST = "taxable_trust"

class TaxTreatment(str, Enum):
    ORDINARY_INCOME = "ordinary_income"
    QUALIFIED_DIVIDENDS = "qualified_dividends"
    LONG_TERM_CAPITAL_GAINS = "long_term_capital_gains"
    SHORT_TERM_CAPITAL_GAINS = "short_term_capital_gains"
    TAX_FREE = "tax_free"
    TAX_DEFERRED = "tax_deferred"

class TaxEfficiencyTier(str, Enum):
    TAX_EFFICIENT = "tax_efficient"      # Index funds, tax-managed funds
    MODERATELY_EFFICIENT = "moderately_efficient"  # Large cap stocks
    TAX_INEFFICIENT = "tax_inefficient"  # Active funds, REITs, bonds

@dataclass
class TaxLot:
    symbol: str
    quantity: float
    purchase_price: float
    purchase_date: date
    account_type: AccountType
    cost_basis: float = None
    
    def __post_init__(self):
        if self.cost_basis is None:
            self.cost_basis = self.quantity * self.purchase_price

@dataclass
class TaxHarvestingOpportunity:
    symbol: str
    current_price: float
    unrealized_loss: float
    tax_savings: float
    replacement_symbol: Optional[str]
    wash_sale_risk: bool
    days_until_available: int
    lot_details: List[TaxLot]

@dataclass
class AssetLocation:
    symbol: str
    tax_efficiency_tier: TaxEfficiencyTier
    dividend_yield: float
    turnover_ratio: float
    tax_drag: float  # Estimated annual tax drag
    preferred_account_types: List[AccountType]

class TaxEfficiencyService:
    """Advanced tax efficiency service with tax-loss harvesting, asset location, and tax-aware rebalancing."""
    
    def __init__(self):
        # Tax rates (these would normally be configurable per user)
        self.tax_rates = {
            "ordinary_income": 0.37,  # Top federal rate
            "long_term_capital_gains": 0.20,  # Top federal rate
            "qualified_dividends": 0.20,
            "state_tax": 0.13,  # CA top rate as example
            "net_investment_income": 0.038  # Medicare surtax
        }
        
        # Account characteristics
        self.account_characteristics = {
            AccountType.TAXABLE: {
                "tax_deferred": False,
                "rmd_required": False,
                "contribution_limit": None,
                "early_withdrawal_penalty": False
            },
            AccountType.TRADITIONAL_IRA: {
                "tax_deferred": True,
                "rmd_required": True,
                "contribution_limit": 6500,  # 2023 limit
                "early_withdrawal_penalty": True
            },
            AccountType.ROTH_IRA: {
                "tax_deferred": False,
                "rmd_required": False,
                "contribution_limit": 6500,
                "early_withdrawal_penalty": False  # Contributions only
            },
            AccountType.FOUR_OH_ONE_K: {
                "tax_deferred": True,
                "rmd_required": True,
                "contribution_limit": 22500,
                "early_withdrawal_penalty": True
            }
        }
        
        # Asset tax efficiency classifications
        self.asset_tax_efficiency = {
            # Tax efficient
            "SPY": TaxEfficiencyTier.TAX_EFFICIENT,
            "VTI": TaxEfficiencyTier.TAX_EFFICIENT,
            "VXUS": TaxEfficiencyTier.TAX_EFFICIENT,
            "BND": TaxEfficiencyTier.MODERATELY_EFFICIENT,
            
            # Tax inefficient
            "REIT": TaxEfficiencyTier.TAX_INEFFICIENT,
            "BOND": TaxEfficiencyTier.TAX_INEFFICIENT,
            "GOLD": TaxEfficiencyTier.MODERATELY_EFFICIENT
        }
        
        # Wash sale tracking
        self.wash_sale_period = 30  # 30 days before and after
        self.recent_sales = {}  # Track recent sales for wash sale rules
        
        # Tax lot tracking
        self.tax_lots = {}  # account_id -> List[TaxLot]
    
    async def identify_tax_loss_harvesting_opportunities(
        self,
        portfolio_positions: Dict[str, Dict],  # symbol -> {quantity, current_price, cost_basis, purchase_dates}
        account_type: AccountType = AccountType.TAXABLE,
        min_loss_threshold: float = 1000.0,
        min_loss_percentage: float = 0.05
    ) -> Dict[str, Any]:
        """
        Identify tax-loss harvesting opportunities.
        """
        try:
            logger.info(f"Identifying tax-loss harvesting opportunities for {account_type}")
            
            if account_type != AccountType.TAXABLE:
                return {
                    "opportunities": [],
                    "total_harvestable_losses": 0.0,
                    "estimated_tax_savings": 0.0,
                    "message": "Tax-loss harvesting only applies to taxable accounts"
                }
            
            opportunities = []
            total_harvestable_losses = 0.0
            total_tax_savings = 0.0
            
            for symbol, position_data in portfolio_positions.items():
                # Calculate unrealized loss
                current_value = position_data["quantity"] * position_data["current_price"]
                total_cost_basis = position_data.get("cost_basis", current_value)
                unrealized_loss = total_cost_basis - current_value
                
                # Check if it meets harvesting criteria
                if (unrealized_loss > min_loss_threshold and 
                    unrealized_loss / total_cost_basis > min_loss_percentage):
                    
                    # Check for wash sale risk
                    wash_sale_risk = await self._check_wash_sale_risk(symbol)
                    
                    # Find replacement asset
                    replacement = await self._find_replacement_asset(symbol)
                    
                    # Calculate tax savings
                    tax_savings = self._calculate_tax_savings(unrealized_loss)
                    
                    # Get tax lot details
                    tax_lots = await self._get_tax_lots_for_symbol(symbol, account_type)
                    
                    opportunity = TaxHarvestingOpportunity(
                        symbol=symbol,
                        current_price=position_data["current_price"],
                        unrealized_loss=unrealized_loss,
                        tax_savings=tax_savings,
                        replacement_symbol=replacement,
                        wash_sale_risk=wash_sale_risk["has_risk"],
                        days_until_available=wash_sale_risk["days_until_clear"],
                        lot_details=tax_lots
                    )
                    
                    opportunities.append(opportunity)
                    total_harvestable_losses += unrealized_loss
                    total_tax_savings += tax_savings
            
            # Sort by tax savings (highest first)
            opportunities.sort(key=lambda x: x.tax_savings, reverse=True)
            
            # Generate harvesting strategy
            strategy = self._generate_harvesting_strategy(opportunities)
            
            logger.info(f"Found {len(opportunities)} tax-loss harvesting opportunities")
            
            return {
                "opportunities": [
                    {
                        "symbol": opp.symbol,
                        "unrealized_loss": opp.unrealized_loss,
                        "tax_savings": opp.tax_savings,
                        "replacement_symbol": opp.replacement_symbol,
                        "wash_sale_risk": opp.wash_sale_risk,
                        "days_until_available": opp.days_until_available,
                        "recommendation": "HARVEST" if not opp.wash_sale_risk else "WAIT"
                    }
                    for opp in opportunities
                ],
                "total_harvestable_losses": total_harvestable_losses,
                "estimated_tax_savings": total_tax_savings,
                "immediate_harvestable": sum(
                    opp.tax_savings for opp in opportunities if not opp.wash_sale_risk
                ),
                "strategy": strategy,
                "tax_rates_used": self.tax_rates
            }
            
        except Exception as e:
            logger.error(f"Error identifying tax-loss harvesting opportunities: {str(e)}")
            return {
                "opportunities": [],
                "total_harvestable_losses": 0.0,
                "estimated_tax_savings": 0.0,
                "error": str(e)
            }
    
    async def optimize_asset_location(
        self,
        target_allocation: Dict[str, float],
        available_accounts: Dict[AccountType, float],  # account_type -> available_capacity
        current_positions: Optional[Dict[AccountType, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Optimize asset location across account types for tax efficiency.
        """
        try:
            logger.info("Optimizing asset location for tax efficiency")
            
            # Analyze tax characteristics of each asset
            asset_analysis = {}
            for symbol in target_allocation.keys():
                analysis = await self._analyze_asset_tax_characteristics(symbol)
                asset_analysis[symbol] = analysis
            
            # Calculate total portfolio value
            total_value = sum(available_accounts.values())
            
            # Generate optimal location strategy
            location_strategy = self._optimize_location_allocation(
                target_allocation, available_accounts, asset_analysis, total_value
            )
            
            # Calculate tax efficiency improvements
            current_tax_drag = self._calculate_current_tax_drag(
                current_positions, asset_analysis
            ) if current_positions else 0.0
            
            optimized_tax_drag = self._calculate_optimized_tax_drag(
                location_strategy, asset_analysis
            )
            
            tax_savings = (current_tax_drag - optimized_tax_drag) * total_value
            
            # Generate transition plan
            transition_plan = self._generate_transition_plan(
                current_positions, location_strategy, available_accounts
            ) if current_positions else []
            
            logger.info("Asset location optimization completed")
            
            return {
                "optimized_allocation": location_strategy,
                "asset_analysis": {
                    symbol: {
                        "tax_efficiency_tier": analysis.tax_efficiency_tier.value,
                        "dividend_yield": analysis.dividend_yield,
                        "estimated_tax_drag": analysis.tax_drag,
                        "preferred_accounts": [acc.value for acc in analysis.preferred_account_types]
                    }
                    for symbol, analysis in asset_analysis.items()
                },
                "tax_efficiency_metrics": {
                    "current_tax_drag": current_tax_drag,
                    "optimized_tax_drag": optimized_tax_drag,
                    "annual_tax_savings": tax_savings,
                    "efficiency_improvement": (current_tax_drag - optimized_tax_drag) / current_tax_drag if current_tax_drag > 0 else 0
                },
                "transition_plan": transition_plan,
                "recommendations": self._generate_location_recommendations(
                    location_strategy, asset_analysis, available_accounts
                )
            }
            
        except Exception as e:
            logger.error(f"Error optimizing asset location: {str(e)}")
            return {"error": str(e)}
    
    async def plan_tax_aware_rebalancing(
        self,
        current_positions: Dict[AccountType, Dict[str, float]],  # account -> {symbol: quantity}
        target_allocation: Dict[str, float],
        current_prices: Dict[str, float],
        cost_basis_data: Dict[str, Dict],  # symbol -> cost basis info
        max_tax_impact: float = 0.02  # Max 2% of portfolio value in taxes
    ) -> Dict[str, Any]:
        """
        Plan tax-aware rebalancing that considers tax implications.
        """
        try:
            logger.info("Planning tax-aware rebalancing")
            
            # Calculate current portfolio value and allocation
            portfolio_analysis = self._analyze_current_portfolio(
                current_positions, current_prices
            )
            
            # Identify required trades for rebalancing
            required_trades = self._calculate_required_trades(
                portfolio_analysis, target_allocation, current_prices
            )
            
            # Analyze tax implications of each trade
            tax_analysis = {}
            total_tax_impact = 0.0
            
            for trade in required_trades:
                tax_impact = await self._analyze_trade_tax_impact(
                    trade, cost_basis_data.get(trade["symbol"], {}), current_prices[trade["symbol"]]
                )
                tax_analysis[f"{trade['symbol']}_{trade['action']}"] = tax_impact
                total_tax_impact += tax_impact["tax_owed"]
            
            # Optimize rebalancing approach
            if total_tax_impact > max_tax_impact * portfolio_analysis["total_value"]:
                # Use tax-efficient rebalancing strategies
                optimized_plan = self._optimize_tax_efficient_rebalancing(
                    required_trades, tax_analysis, portfolio_analysis, max_tax_impact
                )
            else:
                # Proceed with normal rebalancing
                optimized_plan = {
                    "approach": "standard_rebalancing",
                    "trades": required_trades,
                    "total_tax_impact": total_tax_impact
                }
            
            # Generate implementation timeline
            implementation_timeline = self._generate_implementation_timeline(
                optimized_plan, tax_analysis
            )
            
            # Calculate efficiency metrics
            efficiency_metrics = self._calculate_rebalancing_efficiency(
                optimized_plan, portfolio_analysis, total_tax_impact
            )
            
            logger.info("Tax-aware rebalancing plan completed")
            
            return {
                "current_portfolio": portfolio_analysis,
                "target_allocation": target_allocation,
                "required_trades": required_trades,
                "tax_analysis": tax_analysis,
                "optimized_plan": optimized_plan,
                "implementation_timeline": implementation_timeline,
                "efficiency_metrics": efficiency_metrics,
                "recommendations": self._generate_rebalancing_recommendations(
                    optimized_plan, efficiency_metrics, total_tax_impact
                )
            }
            
        except Exception as e:
            logger.error(f"Error planning tax-aware rebalancing: {str(e)}")
            return {"error": str(e)}
    
    async def calculate_tax_alpha(
        self,
        portfolio_performance: Dict[str, Any],
        tax_management_actions: List[Dict[str, Any]],
        benchmark_tax_drag: float = 0.015  # 1.5% typical tax drag
    ) -> Dict[str, Any]:
        """
        Calculate tax alpha - the value added through tax management.
        """
        try:
            logger.info("Calculating tax alpha")
            
            # Calculate gross returns
            gross_return = portfolio_performance.get("gross_return", 0.0)
            
            # Calculate tax costs
            tax_costs = 0.0
            for action in tax_management_actions:
                if action["type"] == "tax_loss_harvest":
                    tax_costs -= action.get("tax_savings", 0.0)  # Negative cost (savings)
                elif action["type"] == "realization":
                    tax_costs += action.get("tax_owed", 0.0)
                elif action["type"] == "dividend":
                    tax_costs += action.get("dividend_tax", 0.0)
            
            # Calculate net return after taxes
            net_return = gross_return - tax_costs
            
            # Calculate benchmark return after typical tax drag
            benchmark_net_return = gross_return - (gross_return * benchmark_tax_drag)
            
            # Tax alpha is the outperformance vs benchmark after taxes
            tax_alpha = net_return - benchmark_net_return
            
            # Additional metrics
            tax_efficiency_ratio = net_return / gross_return if gross_return != 0 else 1.0
            
            return {
                "tax_alpha": tax_alpha,
                "tax_alpha_bps": tax_alpha * 10000,  # In basis points
                "gross_return": gross_return,
                "net_return": net_return,
                "total_tax_costs": tax_costs,
                "tax_efficiency_ratio": tax_efficiency_ratio,
                "benchmark_comparison": {
                    "benchmark_net_return": benchmark_net_return,
                    "outperformance": tax_alpha,
                    "relative_tax_efficiency": (tax_efficiency_ratio - (1 - benchmark_tax_drag)) / (1 - benchmark_tax_drag)
                },
                "tax_management_summary": {
                    "total_actions": len(tax_management_actions),
                    "loss_harvesting_savings": sum(
                        -action.get("tax_savings", 0) for action in tax_management_actions 
                        if action["type"] == "tax_loss_harvest"
                    ),
                    "realization_costs": sum(
                        action.get("tax_owed", 0) for action in tax_management_actions 
                        if action["type"] == "realization"
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating tax alpha: {str(e)}")
            return {"tax_alpha": 0.0, "error": str(e)}
    
    # Helper methods
    
    async def _check_wash_sale_risk(self, symbol: str) -> Dict[str, Any]:
        """Check if selling would trigger wash sale rules."""
        try:
            # Check recent sales history
            recent_sale = self.recent_sales.get(symbol)
            
            if recent_sale:
                days_since_sale = (datetime.now().date() - recent_sale["date"]).days
                if days_since_sale < self.wash_sale_period:
                    return {
                        "has_risk": True,
                        "days_until_clear": self.wash_sale_period - days_since_sale,
                        "blocking_sale_date": recent_sale["date"]
                    }
            
            # Check for substantially identical securities
            # This would normally involve more sophisticated matching
            similar_holdings = await self._find_similar_securities(symbol)
            
            return {
                "has_risk": len(similar_holdings) > 0,
                "days_until_clear": 0,
                "similar_securities": similar_holdings
            }
            
        except Exception as e:
            logger.error(f"Error checking wash sale risk: {str(e)}")
            return {"has_risk": False, "days_until_clear": 0}
    
    async def _find_replacement_asset(self, symbol: str) -> Optional[str]:
        """Find suitable replacement asset for tax-loss harvesting."""
        try:
            # Simple replacement logic - in practice, this would be more sophisticated
            replacements = {
                "SPY": "IVV",      # S&P 500 alternatives
                "IVV": "SPY",
                "VTI": "ITOT",     # Total market alternatives
                "ITOT": "VTI",
                "QQQ": "QQQM",     # NASDAQ alternatives
                "QQQM": "QQQ"
            }
            
            return replacements.get(symbol)
            
        except Exception as e:
            logger.error(f"Error finding replacement asset: {str(e)}")
            return None
    
    def _calculate_tax_savings(self, loss_amount: float) -> float:
        """Calculate tax savings from realizing a loss."""
        try:
            # Assume loss offsets capital gains first (at capital gains rate)
            # Then ordinary income (up to $3,000 limit)
            
            capital_gains_offset = min(loss_amount, 100000)  # Assume some CG to offset
            ordinary_income_offset = min(max(0, loss_amount - capital_gains_offset), 3000)
            
            cg_savings = capital_gains_offset * self.tax_rates["long_term_capital_gains"]
            ordinary_savings = ordinary_income_offset * self.tax_rates["ordinary_income"]
            
            # Add state tax savings
            state_savings = loss_amount * self.tax_rates["state_tax"]
            
            return cg_savings + ordinary_savings + state_savings
            
        except Exception as e:
            logger.error(f"Error calculating tax savings: {str(e)}")
            return 0.0
    
    async def _get_tax_lots_for_symbol(self, symbol: str, account_type: AccountType) -> List[TaxLot]:
        """Get tax lots for a specific symbol."""
        try:
            # This would normally fetch from a database
            # For now, create sample tax lots
            
            lots = []
            num_lots = np.random.randint(1, 4)  # 1-3 lots
            
            for i in range(num_lots):
                purchase_date = datetime.now().date() - timedelta(days=np.random.randint(30, 730))
                quantity = np.random.uniform(10, 100)
                purchase_price = np.random.uniform(50, 200)
                
                lot = TaxLot(
                    symbol=symbol,
                    quantity=quantity,
                    purchase_price=purchase_price,
                    purchase_date=purchase_date,
                    account_type=account_type
                )
                lots.append(lot)
            
            return lots
            
        except Exception as e:
            logger.error(f"Error getting tax lots: {str(e)}")
            return []
    
    def _generate_harvesting_strategy(self, opportunities: List[TaxHarvestingOpportunity]) -> Dict[str, Any]:
        """Generate overall tax-loss harvesting strategy."""
        try:
            immediate_opportunities = [opp for opp in opportunities if not opp.wash_sale_risk]
            delayed_opportunities = [opp for opp in opportunities if opp.wash_sale_risk]
            
            strategy = {
                "immediate_actions": len(immediate_opportunities),
                "delayed_actions": len(delayed_opportunities),
                "total_immediate_savings": sum(opp.tax_savings for opp in immediate_opportunities),
                "approach": "systematic_harvesting"
            }
            
            if len(immediate_opportunities) > 5:
                strategy["recommendation"] = "Implement in batches to avoid market timing concentration"
            elif len(immediate_opportunities) > 0:
                strategy["recommendation"] = "Execute immediately for tax savings"
            else:
                strategy["recommendation"] = "Wait for wash sale periods to clear"
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating harvesting strategy: {str(e)}")
            return {"approach": "manual_review"}
    
    async def _analyze_asset_tax_characteristics(self, symbol: str) -> AssetLocation:
        """Analyze tax characteristics of an asset."""
        try:
            # Get asset information
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Estimate dividend yield
            dividend_yield = info.get("dividendYield", 0.0) or 0.0
            
            # Estimate turnover (simplified)
            if symbol.startswith(("SPY", "VTI", "IVV")):
                turnover_ratio = 0.03  # Low turnover index funds
            elif symbol.startswith(("QQQ", "XLK")):
                turnover_ratio = 0.10  # Moderate turnover
            else:
                turnover_ratio = 0.50  # High turnover assumption
            
            # Determine tax efficiency tier
            tax_efficiency_tier = self._classify_tax_efficiency(symbol, dividend_yield, turnover_ratio)
            
            # Calculate estimated tax drag
            tax_drag = self._estimate_tax_drag(dividend_yield, turnover_ratio)
            
            # Determine preferred account types
            preferred_accounts = self._determine_preferred_accounts(tax_efficiency_tier, dividend_yield)
            
            return AssetLocation(
                symbol=symbol,
                tax_efficiency_tier=tax_efficiency_tier,
                dividend_yield=dividend_yield,
                turnover_ratio=turnover_ratio,
                tax_drag=tax_drag,
                preferred_account_types=preferred_accounts
            )
            
        except Exception as e:
            logger.error(f"Error analyzing tax characteristics for {symbol}: {str(e)}")
            return AssetLocation(
                symbol=symbol,
                tax_efficiency_tier=TaxEfficiencyTier.MODERATELY_EFFICIENT,
                dividend_yield=0.02,
                turnover_ratio=0.20,
                tax_drag=0.015,
                preferred_account_types=[AccountType.TAXABLE]
            )
    
    def _classify_tax_efficiency(self, symbol: str, dividend_yield: float, turnover_ratio: float) -> TaxEfficiencyTier:
        """Classify asset tax efficiency."""
        if symbol in self.asset_tax_efficiency:
            return self.asset_tax_efficiency[symbol]
        
        # General classification rules
        if turnover_ratio < 0.05 and dividend_yield < 0.02:
            return TaxEfficiencyTier.TAX_EFFICIENT
        elif turnover_ratio > 0.30 or dividend_yield > 0.04:
            return TaxEfficiencyTier.TAX_INEFFICIENT
        else:
            return TaxEfficiencyTier.MODERATELY_EFFICIENT
    
    def _estimate_tax_drag(self, dividend_yield: float, turnover_ratio: float) -> float:
        """Estimate annual tax drag."""
        # Dividend tax drag
        dividend_tax = dividend_yield * self.tax_rates["qualified_dividends"]
        
        # Capital gains tax drag (assume 50% of turnover realizes gains)
        cg_tax = turnover_ratio * 0.5 * 0.10 * self.tax_rates["long_term_capital_gains"]  # 10% avg gain
        
        return dividend_tax + cg_tax
    
    def _determine_preferred_accounts(self, tier: TaxEfficiencyTier, dividend_yield: float) -> List[AccountType]:
        """Determine preferred account types for an asset."""
        if tier == TaxEfficiencyTier.TAX_EFFICIENT:
            return [AccountType.TAXABLE, AccountType.ROTH_IRA]
        elif tier == TaxEfficiencyTier.TAX_INEFFICIENT or dividend_yield > 0.03:
            return [AccountType.TRADITIONAL_IRA, AccountType.FOUR_OH_ONE_K, AccountType.HSA]
        else:
            return [AccountType.TAXABLE, AccountType.ROTH_IRA, AccountType.TRADITIONAL_IRA]
    
    def _optimize_location_allocation(
        self,
        target_allocation: Dict[str, float],
        available_accounts: Dict[AccountType, float],
        asset_analysis: Dict[str, AssetLocation],
        total_value: float
    ) -> Dict[AccountType, Dict[str, float]]:
        """Optimize asset location across accounts."""
        try:
            # Initialize result
            location_strategy = {account: {} for account in available_accounts.keys()}
            
            # Sort assets by tax inefficiency (most tax-inefficient first)
            sorted_assets = sorted(
                target_allocation.items(),
                key=lambda x: asset_analysis[x[0]].tax_drag,
                reverse=True
            )
            
            # Remaining capacity in each account
            remaining_capacity = available_accounts.copy()
            
            # Allocate assets starting with most tax-inefficient
            for symbol, target_pct in sorted_assets:
                target_value = target_pct * total_value
                analysis = asset_analysis[symbol]
                
                # Try to place in preferred accounts first
                allocated_value = 0.0
                
                for preferred_account in analysis.preferred_account_types:
                    if preferred_account in remaining_capacity:
                        available_space = remaining_capacity[preferred_account]
                        allocation = min(target_value - allocated_value, available_space)
                        
                        if allocation > 0:
                            allocation_pct = allocation / total_value
                            location_strategy[preferred_account][symbol] = allocation_pct
                            remaining_capacity[preferred_account] -= allocation
                            allocated_value += allocation
                        
                        if allocated_value >= target_value:
                            break
                
                # If not fully allocated, use remaining accounts
                if allocated_value < target_value:
                    remaining_to_allocate = target_value - allocated_value
                    
                    # Allocate proportionally to remaining account capacity
                    total_remaining_capacity = sum(remaining_capacity.values())
                    
                    if total_remaining_capacity > 0:
                        for account, capacity in remaining_capacity.items():
                            if capacity > 0:
                                proportion = capacity / total_remaining_capacity
                                allocation = min(remaining_to_allocate * proportion, capacity)
                                allocation_pct = allocation / total_value
                                
                                if allocation_pct > 0:
                                    location_strategy[account][symbol] = location_strategy[account].get(symbol, 0) + allocation_pct
                                    remaining_capacity[account] -= allocation
            
            return location_strategy
            
        except Exception as e:
            logger.error(f"Error optimizing location allocation: {str(e)}")
            return {account: {} for account in available_accounts.keys()}
    
    def _calculate_current_tax_drag(
        self,
        current_positions: Dict[AccountType, Dict[str, float]],
        asset_analysis: Dict[str, AssetLocation]
    ) -> float:
        """Calculate current portfolio tax drag."""
        try:
            total_drag = 0.0
            total_taxable_value = 0.0
            
            for account_type, positions in current_positions.items():
                account_is_taxable = (account_type == AccountType.TAXABLE)
                
                for symbol, allocation in positions.items():
                    if symbol in asset_analysis:
                        if account_is_taxable:
                            # Tax drag applies in taxable accounts
                            total_drag += allocation * asset_analysis[symbol].tax_drag
                            total_taxable_value += allocation
                        # No tax drag in tax-advantaged accounts
            
            return total_drag / total_taxable_value if total_taxable_value > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating current tax drag: {str(e)}")
            return 0.0
    
    def _calculate_optimized_tax_drag(
        self,
        location_strategy: Dict[AccountType, Dict[str, float]],
        asset_analysis: Dict[str, AssetLocation]
    ) -> float:
        """Calculate tax drag under optimized allocation."""
        try:
            total_drag = 0.0
            total_taxable_value = 0.0
            
            for account_type, positions in location_strategy.items():
                account_is_taxable = (account_type == AccountType.TAXABLE)
                
                for symbol, allocation in positions.items():
                    if symbol in asset_analysis:
                        if account_is_taxable:
                            total_drag += allocation * asset_analysis[symbol].tax_drag
                            total_taxable_value += allocation
            
            return total_drag / total_taxable_value if total_taxable_value > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating optimized tax drag: {str(e)}")
            return 0.0
    
    def _generate_transition_plan(
        self,
        current_positions: Dict[AccountType, Dict[str, float]],
        target_strategy: Dict[AccountType, Dict[str, float]],
        account_capacities: Dict[AccountType, float]
    ) -> List[Dict[str, Any]]:
        """Generate plan to transition to optimal asset location."""
        try:
            transition_actions = []
            
            # For each account and asset, determine required changes
            all_accounts = set(list(current_positions.keys()) + list(target_strategy.keys()))
            
            for account in all_accounts:
                current = current_positions.get(account, {})
                target = target_strategy.get(account, {})
                
                # Get all symbols
                all_symbols = set(list(current.keys()) + list(target.keys()))
                
                for symbol in all_symbols:
                    current_allocation = current.get(symbol, 0.0)
                    target_allocation = target.get(symbol, 0.0)
                    change = target_allocation - current_allocation
                    
                    if abs(change) > 0.01:  # More than 1% change
                        action = {
                            "account": account.value,
                            "symbol": symbol,
                            "current_allocation": current_allocation,
                            "target_allocation": target_allocation,
                            "change": change,
                            "action_type": "increase" if change > 0 else "decrease",
                            "priority": "high" if abs(change) > 0.05 else "medium"
                        }
                        transition_actions.append(action)
            
            # Sort by priority and impact
            transition_actions.sort(key=lambda x: (x["priority"] == "high", abs(x["change"])), reverse=True)
            
            return transition_actions
            
        except Exception as e:
            logger.error(f"Error generating transition plan: {str(e)}")
            return []
    
    def _generate_location_recommendations(
        self,
        location_strategy: Dict[AccountType, Dict[str, float]],
        asset_analysis: Dict[str, AssetLocation],
        available_accounts: Dict[AccountType, float]
    ) -> List[str]:
        """Generate asset location recommendations."""
        recommendations = []
        
        # General recommendations
        taxable_allocation = location_strategy.get(AccountType.TAXABLE, {})
        tax_advantaged_allocation = {}
        
        for account_type, allocation in location_strategy.items():
            if account_type != AccountType.TAXABLE:
                for symbol, pct in allocation.items():
                    tax_advantaged_allocation[symbol] = tax_advantaged_allocation.get(symbol, 0) + pct
        
        # Check for tax-inefficient assets in taxable accounts
        for symbol, allocation in taxable_allocation.items():
            if symbol in asset_analysis:
                analysis = asset_analysis[symbol]
                if analysis.tax_efficiency_tier == TaxEfficiencyTier.TAX_INEFFICIENT and allocation > 0.05:
                    recommendations.append(f"Consider moving {symbol} to tax-advantaged account due to high tax drag")
        
        # Check for tax-efficient assets in tax-advantaged accounts
        for symbol, allocation in tax_advantaged_allocation.items():
            if symbol in asset_analysis:
                analysis = asset_analysis[symbol]
                if analysis.tax_efficiency_tier == TaxEfficiencyTier.TAX_EFFICIENT and allocation > 0.10:
                    recommendations.append(f"Consider moving {symbol} to taxable account to preserve tax-advantaged space")
        
        # Account capacity recommendations
        total_tax_advantaged_capacity = sum(
            capacity for account_type, capacity in available_accounts.items()
            if account_type != AccountType.TAXABLE
        )
        
        if total_tax_advantaged_capacity > 0.5:  # More than 50% in tax-advantaged
            recommendations.append("Significant tax-advantaged capacity available - prioritize tax-inefficient assets")
        
        return recommendations
    
    def _analyze_current_portfolio(
        self,
        current_positions: Dict[AccountType, Dict[str, float]],
        current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze current portfolio composition."""
        try:
            total_value = 0.0
            total_allocation = {}
            account_values = {}
            
            for account_type, positions in current_positions.items():
                account_value = 0.0
                for symbol, quantity in positions.items():
                    if symbol in current_prices:
                        position_value = quantity * current_prices[symbol]
                        account_value += position_value
                        total_value += position_value
                        total_allocation[symbol] = total_allocation.get(symbol, 0) + position_value
                
                account_values[account_type] = account_value
            
            # Convert to percentages
            if total_value > 0:
                total_allocation = {symbol: value / total_value for symbol, value in total_allocation.items()}
                account_percentages = {account: value / total_value for account, value in account_values.items()}
            else:
                account_percentages = {}
            
            return {
                "total_value": total_value,
                "total_allocation": total_allocation,
                "account_values": account_values,
                "account_percentages": account_percentages
            }
            
        except Exception as e:
            logger.error(f"Error analyzing current portfolio: {str(e)}")
            return {"total_value": 0.0}
    
    def _calculate_required_trades(
        self,
        current_portfolio: Dict[str, Any],
        target_allocation: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Calculate trades required for rebalancing."""
        try:
            trades = []
            total_value = current_portfolio["total_value"]
            current_allocation = current_portfolio["total_allocation"]
            
            for symbol, target_pct in target_allocation.items():
                current_pct = current_allocation.get(symbol, 0.0)
                difference = target_pct - current_pct
                
                if abs(difference) > 0.01:  # More than 1% difference
                    trade_value = difference * total_value
                    trade_quantity = trade_value / current_prices.get(symbol, 100)
                    
                    trade = {
                        "symbol": symbol,
                        "action": "buy" if difference > 0 else "sell",
                        "current_allocation": current_pct,
                        "target_allocation": target_pct,
                        "difference": difference,
                        "trade_value": abs(trade_value),
                        "trade_quantity": abs(trade_quantity),
                        "current_price": current_prices.get(symbol, 100)
                    }
                    trades.append(trade)
            
            return trades
            
        except Exception as e:
            logger.error(f"Error calculating required trades: {str(e)}")
            return []
    
    async def _analyze_trade_tax_impact(
        self,
        trade: Dict[str, Any],
        cost_basis_data: Dict[str, Any],
        current_price: float
    ) -> Dict[str, Any]:
        """Analyze tax impact of a specific trade."""
        try:
            if trade["action"] == "buy":
                return {
                    "tax_owed": 0.0,
                    "gain_loss": 0.0,
                    "holding_period": "new_position"
                }
            
            # For sell orders, calculate gain/loss
            avg_cost_basis = cost_basis_data.get("avg_cost_basis", current_price)
            gain_loss_per_share = current_price - avg_cost_basis
            total_gain_loss = gain_loss_per_share * trade["trade_quantity"]
            
            # Determine holding period (simplified)
            holding_period = cost_basis_data.get("avg_holding_period_days", 400)
            is_long_term = holding_period > 365
            
            # Calculate tax owed
            if total_gain_loss > 0:  # Gain
                tax_rate = (self.tax_rates["long_term_capital_gains"] if is_long_term 
                           else self.tax_rates["ordinary_income"])
                tax_owed = total_gain_loss * tax_rate
            else:  # Loss
                tax_owed = 0.0  # Losses offset other gains
            
            return {
                "tax_owed": tax_owed,
                "gain_loss": total_gain_loss,
                "gain_loss_per_share": gain_loss_per_share,
                "holding_period": "long_term" if is_long_term else "short_term",
                "tax_rate_applied": tax_rate if total_gain_loss > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trade tax impact: {str(e)}")
            return {"tax_owed": 0.0, "gain_loss": 0.0}
    
    def _optimize_tax_efficient_rebalancing(
        self,
        required_trades: List[Dict[str, Any]],
        tax_analysis: Dict[str, Dict[str, Any]],
        portfolio_analysis: Dict[str, Any],
        max_tax_impact: float
    ) -> Dict[str, Any]:
        """Optimize rebalancing to minimize tax impact."""
        try:
            total_value = portfolio_analysis["total_value"]
            max_tax_dollars = max_tax_impact * total_value
            
            # Separate trades by tax impact
            buy_trades = [t for t in required_trades if t["action"] == "buy"]
            sell_trades = [t for t in required_trades if t["action"] == "sell"]
            
            # Sort sell trades by tax efficiency (losses first, then smallest gains)
            sell_trades.sort(key=lambda t: tax_analysis[f"{t['symbol']}_sell"]["gain_loss"])
            
            optimized_plan = {
                "approach": "tax_efficient_rebalancing",
                "phases": []
            }
            
            # Phase 1: Tax-loss harvesting
            loss_trades = [t for t in sell_trades 
                          if tax_analysis[f"{t['symbol']}_sell"]["gain_loss"] < 0]
            if loss_trades:
                optimized_plan["phases"].append({
                    "phase": 1,
                    "description": "Harvest tax losses",
                    "trades": loss_trades,
                    "tax_impact": sum(tax_analysis[f"{t['symbol']}_sell"]["tax_owed"] for t in loss_trades)
                })
            
            # Phase 2: Execute necessary buys and low-tax sells
            remaining_tax_budget = max_tax_dollars
            phase_2_trades = buy_trades.copy()
            
            for trade in sell_trades:
                trade_tax = tax_analysis[f"{trade['symbol']}_sell"]["tax_owed"]
                if trade_tax <= remaining_tax_budget:
                    phase_2_trades.append(trade)
                    remaining_tax_budget -= trade_tax
            
            if phase_2_trades:
                optimized_plan["phases"].append({
                    "phase": 2,
                    "description": "Execute low-tax-impact trades",
                    "trades": phase_2_trades,
                    "tax_impact": sum(
                        tax_analysis[f"{t['symbol']}_sell"]["tax_owed"] 
                        for t in phase_2_trades if t["action"] == "sell"
                    )
                })
            
            # Phase 3: Defer high-tax trades
            deferred_trades = [t for t in sell_trades if t not in phase_2_trades]
            if deferred_trades:
                optimized_plan["phases"].append({
                    "phase": 3,
                    "description": "Defer high-tax-impact trades",
                    "trades": deferred_trades,
                    "recommendation": "Consider deferring to next tax year or offsetting with losses"
                })
            
            return optimized_plan
            
        except Exception as e:
            logger.error(f"Error optimizing tax-efficient rebalancing: {str(e)}")
            return {"approach": "standard_rebalancing", "trades": required_trades}
    
    def _generate_implementation_timeline(
        self,
        optimized_plan: Dict[str, Any],
        tax_analysis: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate implementation timeline for rebalancing plan."""
        timeline = []
        
        if "phases" in optimized_plan:
            for phase in optimized_plan["phases"]:
                if phase["phase"] <= 2:  # Execute phases 1 and 2
                    timeline.append({
                        "date": datetime.now().date(),
                        "phase": phase["phase"],
                        "description": phase["description"],
                        "trades": len(phase.get("trades", [])),
                        "estimated_tax_impact": phase.get("tax_impact", 0)
                    })
                else:  # Defer phase 3
                    timeline.append({
                        "date": (datetime.now() + timedelta(days=90)).date(),
                        "phase": phase["phase"],
                        "description": "Review deferred trades",
                        "trades": len(phase.get("trades", [])),
                        "estimated_tax_impact": sum(
                            tax_analysis[f"{t['symbol']}_sell"]["tax_owed"] 
                            for t in phase.get("trades", [])
                        )
                    })
        
        return timeline
    
    def _calculate_rebalancing_efficiency(
        self,
        optimized_plan: Dict[str, Any],
        portfolio_analysis: Dict[str, Any],
        total_tax_impact: float
    ) -> Dict[str, Any]:
        """Calculate efficiency metrics for rebalancing plan."""
        total_value = portfolio_analysis["total_value"]
        
        return {
            "tax_efficiency_ratio": 1.0 - (total_tax_impact / total_value),
            "total_tax_impact_pct": total_tax_impact / total_value,
            "approach": optimized_plan.get("approach", "standard"),
            "phases_planned": len(optimized_plan.get("phases", [])),
            "immediate_execution_ratio": self._calculate_immediate_execution_ratio(optimized_plan)
        }
    
    def _calculate_immediate_execution_ratio(self, optimized_plan: Dict[str, Any]) -> float:
        """Calculate ratio of trades that can be executed immediately."""
        if "phases" not in optimized_plan:
            return 1.0
        
        total_trades = sum(len(phase.get("trades", [])) for phase in optimized_plan["phases"])
        immediate_trades = sum(
            len(phase.get("trades", []))
            for phase in optimized_plan["phases"]
            if phase["phase"] <= 2
        )
        
        return immediate_trades / total_trades if total_trades > 0 else 1.0
    
    def _generate_rebalancing_recommendations(
        self,
        optimized_plan: Dict[str, Any],
        efficiency_metrics: Dict[str, Any],
        total_tax_impact: float
    ) -> List[str]:
        """Generate rebalancing recommendations."""
        recommendations = []
        
        if efficiency_metrics["tax_efficiency_ratio"] < 0.95:
            recommendations.append("High tax impact detected - consider phased rebalancing approach")
        
        if optimized_plan.get("approach") == "tax_efficient_rebalancing":
            recommendations.append("Tax-efficient rebalancing strategy recommended")
        
        immediate_ratio = efficiency_metrics.get("immediate_execution_ratio", 1.0)
        if immediate_ratio < 0.8:
            recommendations.append("Consider deferring some trades to minimize tax impact")
        
        if total_tax_impact > 10000:  # $10k+ in taxes
            recommendations.append("Significant tax impact - consider tax-loss harvesting opportunities")
        
        return recommendations
    
    async def _find_similar_securities(self, symbol: str) -> List[str]:
        """Find securities that might trigger wash sale rules."""
        # Simplified implementation
        similar_securities = {
            "SPY": ["IVV", "VOO"],
            "IVV": ["SPY", "VOO"],
            "VOO": ["SPY", "IVV"],
            "QQQ": ["QQQM"],
            "QQQM": ["QQQ"]
        }
        
        return similar_securities.get(symbol, [])
