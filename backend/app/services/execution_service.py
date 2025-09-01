import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta, time
from enum import Enum
import pandas as pd
import numpy as np
import yfinance as yf
import asyncio
import time as time_module
from dataclasses import dataclass
import uuid
from collections import deque
import math

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

logger = logging.getLogger(__name__)

class ExecutionAlgorithm(str, Enum):
    VWAP = "vwap"
    TWAP = "twap"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ARRIVAL_PRICE = "arrival_price"
    POV = "percentage_of_volume"  # Participation rate
    ICEBERG = "iceberg"
    SMART_ROUTING = "smart_routing"

class OrderStatus(str, Enum):
    PENDING = "pending"
    WORKING = "working"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class VenueType(str, Enum):
    PRIMARY_EXCHANGE = "primary_exchange"
    DARK_POOL = "dark_pool"
    ECN = "ecn"
    MARKET_MAKER = "market_maker"
    RETAIL_WHOLESALER = "retail_wholesaler"

@dataclass
class ExecutionVenue:
    name: str
    venue_type: VenueType
    fee_structure: Dict[str, float]  # maker_fee, taker_fee, etc.
    liquidity_score: float  # 0-1 scale
    speed_score: float  # 0-1 scale
    min_size: int
    max_size: int
    supports_hidden: bool = False
    supports_iceberg: bool = False

@dataclass
class OrderSlice:
    slice_id: str
    parent_order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: Optional[float]
    order_type: OrderType
    venue: str
    scheduled_time: datetime
    max_participation_rate: float = 0.1  # Max 10% of volume
    urgency_factor: float = 1.0  # 1.0 = normal, >1.0 = more urgent

@dataclass
class ExecutionOrder:
    order_id: str
    symbol: str
    side: OrderSide
    total_quantity: float
    algorithm: ExecutionAlgorithm
    time_horizon: timedelta
    max_participation_rate: float = 0.15
    price_limit: Optional[float] = None
    urgency: float = 1.0
    created_at: datetime
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    slices: List[OrderSlice] = None
    venue_allocations: Dict[str, float] = None

class ExecutionService:
    """Advanced execution service with VWAP/TWAP algorithms and smart order routing."""
    
    def __init__(self):
        from app.utils.trading import get_alpaca_credentials
        
        # Initialize Alpaca clients
        api_key, secret_key, base_url = get_alpaca_credentials()
        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        
        # Execution venues (simplified for demonstration)
        self.venues = {
            "NASDAQ": ExecutionVenue(
                name="NASDAQ",
                venue_type=VenueType.PRIMARY_EXCHANGE,
                fee_structure={"maker_fee": 0.0020, "taker_fee": 0.0030},
                liquidity_score=0.9,
                speed_score=0.8,
                min_size=1,
                max_size=1000000,
                supports_hidden=True,
                supports_iceberg=True
            ),
            "NYSE": ExecutionVenue(
                name="NYSE",
                venue_type=VenueType.PRIMARY_EXCHANGE,
                fee_structure={"maker_fee": 0.0015, "taker_fee": 0.0025},
                liquidity_score=0.95,
                speed_score=0.7,
                min_size=1,
                max_size=1000000,
                supports_hidden=True,
                supports_iceberg=True
            ),
            "IEX": ExecutionVenue(
                name="IEX",
                venue_type=VenueType.ECN,
                fee_structure={"maker_fee": 0.0000, "taker_fee": 0.0009},
                liquidity_score=0.7,
                speed_score=0.9,
                min_size=1,
                max_size=500000,
                supports_hidden=True,
                supports_iceberg=False
            ),
            "DARK_POOL_1": ExecutionVenue(
                name="DARK_POOL_1",
                venue_type=VenueType.DARK_POOL,
                fee_structure={"maker_fee": 0.0010, "taker_fee": 0.0010},
                liquidity_score=0.6,
                speed_score=0.6,
                min_size=100,
                max_size=100000,
                supports_hidden=True,
                supports_iceberg=True
            )
        }
        
        # Active orders tracking
        self.active_orders: Dict[str, ExecutionOrder] = {}
        self.order_history: List[ExecutionOrder] = []
        
        # Market data cache
        self._market_data_cache = {}
        self._volume_profiles = {}
        
        # VWAP/TWAP parameters
        self.default_slice_interval = timedelta(minutes=5)
        self.max_participation_rate = 0.20  # Max 20% of volume
        self.default_urgency = 1.0
    
    async def execute_order_with_algorithm(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        algorithm: ExecutionAlgorithm,
        time_horizon: Optional[timedelta] = None,
        price_limit: Optional[float] = None,
        max_participation_rate: float = 0.15,
        urgency: float = 1.0
    ) -> Dict[str, Any]:
        """
        Execute an order using specified algorithm.
        """
        try:
            logger.info(f"Executing {algorithm} order: {side} {quantity} {symbol}")
            
            # Generate order ID
            order_id = str(uuid.uuid4())
            
            # Set default time horizon based on algorithm
            if time_horizon is None:
                time_horizon = self._get_default_time_horizon(algorithm, quantity, symbol)
            
            # Create execution order
            execution_order = ExecutionOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                total_quantity=quantity,
                algorithm=algorithm,
                time_horizon=time_horizon,
                max_participation_rate=max_participation_rate,
                price_limit=price_limit,
                urgency=urgency,
                created_at=datetime.now(),
                slices=[],
                venue_allocations={}
            )
            
            # Add to active orders
            self.active_orders[order_id] = execution_order
            
            # Execute based on algorithm
            if algorithm == ExecutionAlgorithm.VWAP:
                result = await self._execute_vwap(execution_order)
            elif algorithm == ExecutionAlgorithm.TWAP:
                result = await self._execute_twap(execution_order)
            elif algorithm == ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL:
                result = await self._execute_implementation_shortfall(execution_order)
            elif algorithm == ExecutionAlgorithm.POV:
                result = await self._execute_percentage_of_volume(execution_order)
            elif algorithm == ExecutionAlgorithm.ICEBERG:
                result = await self._execute_iceberg(execution_order)
            elif algorithm == ExecutionAlgorithm.SMART_ROUTING:
                result = await self._execute_smart_routing(execution_order)
            else:
                result = await self._execute_arrival_price(execution_order)
            
            # Update order status
            if result.get("success", False):
                execution_order.status = OrderStatus.WORKING
            else:
                execution_order.status = OrderStatus.REJECTED
            
            logger.info(f"Order {order_id} execution initiated with {algorithm}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing order with algorithm: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "order_id": order_id if 'order_id' in locals() else None
            }
    
    async def _execute_vwap(self, order: ExecutionOrder) -> Dict[str, Any]:
        """
        Execute VWAP (Volume Weighted Average Price) algorithm.
        Splits order based on historical volume patterns.
        """
        try:
            logger.info(f"Executing VWAP algorithm for {order.symbol}")
            
            # Get historical volume profile
            volume_profile = await self._get_volume_profile(order.symbol, lookback_days=20)
            
            if not volume_profile:
                logger.warning(f"No volume profile available for {order.symbol}, using TWAP fallback")
                return await self._execute_twap(order)
            
            # Calculate VWAP schedule
            slices = self._calculate_vwap_schedule(
                order=order,
                volume_profile=volume_profile
            )
            
            # Apply smart order routing to each slice
            routed_slices = []
            for slice_order in slices:
                venues = await self._route_order_slice(slice_order)
                slice_order.venue = venues[0] if venues else "NASDAQ"  # Default venue
                routed_slices.append(slice_order)
            
            order.slices = routed_slices
            
            # Execute first slice immediately
            if routed_slices:
                first_slice_result = await self._execute_slice(routed_slices[0])
                
                # Schedule remaining slices
                await self._schedule_remaining_slices(routed_slices[1:])
            
            return {
                "success": True,
                "order_id": order.order_id,
                "algorithm": "VWAP",
                "total_slices": len(routed_slices),
                "schedule": [
                    {
                        "slice_id": s.slice_id,
                        "quantity": s.quantity,
                        "scheduled_time": s.scheduled_time.isoformat(),
                        "venue": s.venue
                    }
                    for s in routed_slices
                ],
                "estimated_completion": (order.created_at + order.time_horizon).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in VWAP execution: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _execute_twap(self, order: ExecutionOrder) -> Dict[str, Any]:
        """
        Execute TWAP (Time Weighted Average Price) algorithm.
        Evenly distributes order over time.
        """
        try:
            logger.info(f"Executing TWAP algorithm for {order.symbol}")
            
            # Calculate number of slices based on time horizon
            slice_interval = self.default_slice_interval
            total_slices = max(1, int(order.time_horizon.total_seconds() / slice_interval.total_seconds()))
            
            # Ensure we don't have too many tiny slices
            total_slices = min(total_slices, 50)  # Max 50 slices
            slice_size = order.total_quantity / total_slices
            
            # Create evenly distributed slices
            slices = []
            current_time = order.created_at
            
            for i in range(total_slices):
                slice_id = f"{order.order_id}_slice_{i+1}"
                
                # Add some randomization to avoid predictable patterns
                time_variance = timedelta(
                    seconds=np.random.uniform(-30, 30)  # ±30 seconds
                )
                
                slice_order = OrderSlice(
                    slice_id=slice_id,
                    parent_order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=slice_size,
                    price=order.price_limit,
                    order_type=OrderType.LIMIT if order.price_limit else OrderType.MARKET,
                    venue="NASDAQ",  # Will be updated by routing
                    scheduled_time=current_time + time_variance,
                    max_participation_rate=order.max_participation_rate
                )
                
                slices.append(slice_order)
                current_time += slice_interval
            
            # Apply smart routing
            routed_slices = []
            for slice_order in slices:
                venues = await self._route_order_slice(slice_order)
                slice_order.venue = venues[0] if venues else "NASDAQ"
                routed_slices.append(slice_order)
            
            order.slices = routed_slices
            
            # Execute first slice
            if routed_slices:
                await self._execute_slice(routed_slices[0])
                await self._schedule_remaining_slices(routed_slices[1:])
            
            return {
                "success": True,
                "order_id": order.order_id,
                "algorithm": "TWAP",
                "total_slices": len(routed_slices),
                "slice_size": slice_size,
                "slice_interval_minutes": slice_interval.total_seconds() / 60,
                "schedule": [
                    {
                        "slice_id": s.slice_id,
                        "quantity": s.quantity,
                        "scheduled_time": s.scheduled_time.isoformat(),
                        "venue": s.venue
                    }
                    for s in routed_slices
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in TWAP execution: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _execute_implementation_shortfall(self, order: ExecutionOrder) -> Dict[str, Any]:
        """
        Implementation Shortfall algorithm - minimizes market impact + timing risk.
        """
        try:
            logger.info(f"Executing Implementation Shortfall for {order.symbol}")
            
            # Get market impact model parameters
            impact_params = await self._estimate_market_impact(order.symbol, order.total_quantity)
            
            # Calculate optimal execution schedule
            # This is a simplified version - real IS algorithms are quite complex
            
            # Front-load execution if market impact is low
            if impact_params["linear_impact"] < 0.001:  # Less than 0.1% impact per share
                # Execute 50% immediately, rest over time
                immediate_portion = 0.5
            else:
                # Execute more gradually if high impact
                immediate_portion = 0.2
            
            slices = []
            
            # Immediate execution slice
            if immediate_portion > 0:
                immediate_slice = OrderSlice(
                    slice_id=f"{order.order_id}_immediate",
                    parent_order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.total_quantity * immediate_portion,
                    price=None,  # Market order for immediate execution
                    order_type=OrderType.MARKET,
                    venue="NASDAQ",
                    scheduled_time=order.created_at,
                    urgency_factor=2.0  # High urgency
                )
                slices.append(immediate_slice)
            
            # Remaining quantity over time
            remaining_quantity = order.total_quantity * (1 - immediate_portion)
            remaining_slices = max(1, int(order.time_horizon.total_seconds() / 300))  # 5-min intervals
            
            for i in range(remaining_slices):
                slice_time = order.created_at + timedelta(seconds=(i + 1) * 300)
                
                slice_order = OrderSlice(
                    slice_id=f"{order.order_id}_is_{i+1}",
                    parent_order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=remaining_quantity / remaining_slices,
                    price=order.price_limit,
                    order_type=OrderType.LIMIT if order.price_limit else OrderType.MARKET,
                    venue="NASDAQ",
                    scheduled_time=slice_time,
                    max_participation_rate=order.max_participation_rate * 0.8  # More conservative
                )
                slices.append(slice_order)
            
            # Apply routing
            for slice_order in slices:
                venues = await self._route_order_slice(slice_order)
                slice_order.venue = venues[0] if venues else "NASDAQ"
            
            order.slices = slices
            
            # Execute immediate slice
            if slices:
                await self._execute_slice(slices[0])
                await self._schedule_remaining_slices(slices[1:])
            
            return {
                "success": True,
                "order_id": order.order_id,
                "algorithm": "Implementation Shortfall",
                "immediate_execution": immediate_portion,
                "market_impact_estimate": impact_params,
                "total_slices": len(slices)
            }
            
        except Exception as e:
            logger.error(f"Error in Implementation Shortfall: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _execute_percentage_of_volume(self, order: ExecutionOrder) -> Dict[str, Any]:
        """
        Percentage of Volume (POV) algorithm - maintains participation rate.
        """
        try:
            logger.info(f"Executing POV algorithm for {order.symbol}")
            
            # This would require real-time volume monitoring
            # For now, we'll simulate based on expected volumes
            
            target_participation = order.max_participation_rate
            
            # Create adaptive slices that adjust based on volume
            slices = []
            slice_interval = timedelta(minutes=2)  # Check every 2 minutes
            total_intervals = int(order.time_horizon.total_seconds() / slice_interval.total_seconds())
            
            for i in range(total_intervals):
                slice_time = order.created_at + timedelta(seconds=i * slice_interval.total_seconds())
                
                # Dynamic sizing based on expected volume (simplified)
                base_size = order.total_quantity / total_intervals
                
                slice_order = OrderSlice(
                    slice_id=f"{order.order_id}_pov_{i+1}",
                    parent_order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=base_size,
                    price=order.price_limit,
                    order_type=OrderType.LIMIT if order.price_limit else OrderType.MARKET,
                    venue="NASDAQ",
                    scheduled_time=slice_time,
                    max_participation_rate=target_participation
                )
                slices.append(slice_order)
            
            order.slices = slices
            
            return {
                "success": True,
                "order_id": order.order_id,
                "algorithm": "Percentage of Volume",
                "target_participation_rate": target_participation,
                "monitoring_interval_minutes": slice_interval.total_seconds() / 60
            }
            
        except Exception as e:
            logger.error(f"Error in POV execution: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _execute_iceberg(self, order: ExecutionOrder) -> Dict[str, Any]:
        """
        Iceberg algorithm - shows only small portions to minimize market impact.
        """
        try:
            logger.info(f"Executing Iceberg algorithm for {order.symbol}")
            
            # Calculate iceberg slice size (typically 5-10% of average volume)
            volume_data = await self._get_average_volume(order.symbol)
            avg_volume = volume_data.get("avg_volume", 100000)
            
            # Iceberg slice size: 5% of daily volume or 10% of order, whichever is smaller
            iceberg_size = min(
                order.total_quantity * 0.1,
                avg_volume * 0.05
            )
            
            # Ensure minimum viable size
            iceberg_size = max(iceberg_size, 100)
            
            # Calculate number of icebergs needed
            num_icebergs = math.ceil(order.total_quantity / iceberg_size)
            
            slices = []
            for i in range(num_icebergs):
                remaining_qty = order.total_quantity - (i * iceberg_size)
                slice_qty = min(iceberg_size, remaining_qty)
                
                # Stagger timing slightly
                slice_time = order.created_at + timedelta(minutes=i * 3)
                
                slice_order = OrderSlice(
                    slice_id=f"{order.order_id}_iceberg_{i+1}",
                    parent_order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=slice_qty,
                    price=order.price_limit,
                    order_type=OrderType.LIMIT if order.price_limit else OrderType.MARKET,
                    venue="DARK_POOL_1",  # Prefer dark pools for iceberg
                    scheduled_time=slice_time
                )
                slices.append(slice_order)
            
            order.slices = slices
            
            # Execute first iceberg
            if slices:
                await self._execute_slice(slices[0])
            
            return {
                "success": True,
                "order_id": order.order_id,
                "algorithm": "Iceberg",
                "iceberg_size": iceberg_size,
                "num_icebergs": num_icebergs,
                "hidden_quantity": order.total_quantity - iceberg_size
            }
            
        except Exception as e:
            logger.error(f"Error in Iceberg execution: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _execute_smart_routing(self, order: ExecutionOrder) -> Dict[str, Any]:
        """
        Smart Order Routing - finds best execution venues.
        """
        try:
            logger.info(f"Executing Smart Routing for {order.symbol}")
            
            # Analyze all available venues
            venue_analysis = await self._analyze_venues_for_order(order)
            
            # Split order across multiple venues for best execution
            venue_allocations = self._optimize_venue_allocation(order, venue_analysis)
            
            slices = []
            for venue_name, allocation in venue_allocations.items():
                if allocation > 0:
                    slice_order = OrderSlice(
                        slice_id=f"{order.order_id}_sr_{venue_name}",
                        parent_order_id=order.order_id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.total_quantity * allocation,
                        price=order.price_limit,
                        order_type=OrderType.LIMIT if order.price_limit else OrderType.MARKET,
                        venue=venue_name,
                        scheduled_time=order.created_at
                    )
                    slices.append(slice_order)
            
            order.slices = slices
            order.venue_allocations = venue_allocations
            
            # Execute all slices simultaneously
            for slice_order in slices:
                await self._execute_slice(slice_order)
            
            return {
                "success": True,
                "order_id": order.order_id,
                "algorithm": "Smart Routing",
                "venue_allocations": venue_allocations,
                "venue_analysis": venue_analysis,
                "total_venues": len(venue_allocations)
            }
            
        except Exception as e:
            logger.error(f"Error in Smart Routing: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _execute_arrival_price(self, order: ExecutionOrder) -> Dict[str, Any]:
        """
        Arrival Price algorithm - simple immediate execution.
        """
        try:
            logger.info(f"Executing Arrival Price for {order.symbol}")
            
            # Single immediate execution
            slice_order = OrderSlice(
                slice_id=f"{order.order_id}_arrival",
                parent_order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.total_quantity,
                price=order.price_limit,
                order_type=OrderType.LIMIT if order.price_limit else OrderType.MARKET,
                venue="NASDAQ",
                scheduled_time=order.created_at
            )
            
            # Apply routing
            venues = await self._route_order_slice(slice_order)
            slice_order.venue = venues[0] if venues else "NASDAQ"
            
            order.slices = [slice_order]
            
            # Execute immediately
            result = await self._execute_slice(slice_order)
            
            return {
                "success": True,
                "order_id": order.order_id,
                "algorithm": "Arrival Price",
                "execution_venue": slice_order.venue,
                "immediate_execution": True
            }
            
        except Exception as e:
            logger.error(f"Error in Arrival Price: {str(e)}")
            return {"success": False, "error": str(e)}
    
    # Helper methods
    
    def _get_default_time_horizon(self, algorithm: ExecutionAlgorithm, quantity: float, symbol: str) -> timedelta:
        """Get default time horizon based on algorithm and order size."""
        if algorithm == ExecutionAlgorithm.ARRIVAL_PRICE:
            return timedelta(minutes=1)
        elif algorithm == ExecutionAlgorithm.ICEBERG:
            # Longer for iceberg to hide order
            return timedelta(hours=2)
        elif algorithm in [ExecutionAlgorithm.VWAP, ExecutionAlgorithm.TWAP]:
            # Scale with order size
            if quantity > 10000:
                return timedelta(hours=4)
            elif quantity > 1000:
                return timedelta(hours=2)
            else:
                return timedelta(hours=1)
        else:
            return timedelta(hours=1)
    
    async def _get_volume_profile(self, symbol: str, lookback_days: int = 20) -> Dict[str, Any]:
        """Get historical volume profile for VWAP calculation."""
        try:
            # This would normally use high-frequency data
            # For now, we'll use daily data and estimate intraday patterns
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Use yfinance for volume data
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                return None
            
            avg_volume = data['Volume'].mean()
            
            # Simplified intraday volume profile (real systems use minute-by-minute data)
            # Market open: high volume, mid-day: lower, close: high
            intraday_profile = {
                "09:30": 0.15,  # Market open
                "10:00": 0.12,
                "11:00": 0.08,
                "12:00": 0.06,  # Lunch time low
                "13:00": 0.06,
                "14:00": 0.08,
                "15:00": 0.10,
                "15:30": 0.15,  # Close high
                "16:00": 0.20   # Market close
            }
            
            return {
                "avg_daily_volume": avg_volume,
                "intraday_profile": intraday_profile,
                "lookback_days": lookback_days
            }
            
        except Exception as e:
            logger.error(f"Error getting volume profile: {str(e)}")
            return None
    
    def _calculate_vwap_schedule(self, order: ExecutionOrder, volume_profile: Dict) -> List[OrderSlice]:
        """Calculate VWAP execution schedule based on volume profile."""
        try:
            intraday_profile = volume_profile["intraday_profile"]
            total_profile_volume = sum(intraday_profile.values())
            
            slices = []
            current_time = order.created_at
            
            # Distribute order quantity based on historical volume patterns
            for time_str, volume_pct in intraday_profile.items():
                # Parse time
                hour, minute = map(int, time_str.split(':'))
                slice_time = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                # If time has passed today, schedule for next trading day
                if slice_time <= current_time:
                    slice_time += timedelta(days=1)
                
                # Calculate slice size based on volume
                volume_weight = volume_pct / total_profile_volume
                slice_quantity = order.total_quantity * volume_weight
                
                if slice_quantity > 0:
                    slice_order = OrderSlice(
                        slice_id=f"{order.order_id}_vwap_{time_str.replace(':', '')}",
                        parent_order_id=order.order_id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=slice_quantity,
                        price=order.price_limit,
                        order_type=OrderType.LIMIT if order.price_limit else OrderType.MARKET,
                        venue="NASDAQ",
                        scheduled_time=slice_time,
                        max_participation_rate=order.max_participation_rate
                    )
                    slices.append(slice_order)
            
            return slices
            
        except Exception as e:
            logger.error(f"Error calculating VWAP schedule: {str(e)}")
            return []
    
    async def _route_order_slice(self, slice_order: OrderSlice) -> List[str]:
        """Smart order routing for a single slice."""
        try:
            # Analyze venues for this specific slice
            venue_scores = {}
            
            for venue_name, venue in self.venues.items():
                score = 0.0
                
                # Size constraints
                if slice_order.quantity < venue.min_size or slice_order.quantity > venue.max_size:
                    continue
                
                # Liquidity score (higher is better)
                score += venue.liquidity_score * 0.4
                
                # Speed score (higher is better)  
                score += venue.speed_score * 0.2
                
                # Fee score (lower fees are better)
                if slice_order.order_type == OrderType.MARKET:
                    fee = venue.fee_structure.get("taker_fee", 0.003)
                else:
                    fee = venue.fee_structure.get("maker_fee", 0.002)
                
                fee_score = max(0, 1.0 - fee * 1000)  # Normalize fees
                score += fee_score * 0.3
                
                # Venue type preference based on order characteristics
                if slice_order.quantity > 1000 and venue.venue_type == VenueType.DARK_POOL:
                    score += 0.1  # Prefer dark pools for large orders
                
                venue_scores[venue_name] = score
            
            # Sort venues by score
            sorted_venues = sorted(venue_scores.items(), key=lambda x: x[1], reverse=True)
            
            return [venue[0] for venue in sorted_venues[:3]]  # Return top 3 venues
            
        except Exception as e:
            logger.error(f"Error routing order slice: {str(e)}")
            return ["NASDAQ"]  # Default fallback
    
    async def _execute_slice(self, slice_order: OrderSlice) -> Dict[str, Any]:
        """Execute a single order slice."""
        try:
            logger.info(f"Executing slice {slice_order.slice_id} on {slice_order.venue}")
            
            # In a real implementation, this would route to the specific venue
            # For now, we'll use Alpaca as the execution venue
            
            if slice_order.order_type == OrderType.MARKET:
                order_request = MarketOrderRequest(
                    symbol=slice_order.symbol,
                    qty=slice_order.quantity,
                    side=slice_order.side,
                    time_in_force=TimeInForce.DAY
                )
            else:
                # For limit orders, we need a price
                current_price = await self._get_current_price(slice_order.symbol)
                if slice_order.price is None:
                    # Set a reasonable limit price
                    if slice_order.side == OrderSide.BUY:
                        limit_price = current_price * 1.002  # 0.2% above current
                    else:
                        limit_price = current_price * 0.998  # 0.2% below current
                else:
                    limit_price = slice_order.price
                
                order_request = LimitOrderRequest(
                    symbol=slice_order.symbol,
                    qty=slice_order.quantity,
                    side=slice_order.side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price
                )
            
            # Submit order (commented out for testing)
            # order_response = self.trading_client.submit_order(order_request)
            
            # Simulate successful execution for testing
            order_response = {
                "id": f"simulated_{slice_order.slice_id}",
                "status": "accepted",
                "filled_qty": slice_order.quantity,
                "filled_avg_price": await self._get_current_price(slice_order.symbol)
            }
            
            logger.info(f"Slice {slice_order.slice_id} executed successfully")
            
            return {
                "success": True,
                "slice_id": slice_order.slice_id,
                "execution_id": order_response.get("id"),
                "venue": slice_order.venue,
                "filled_quantity": slice_order.quantity,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing slice: {str(e)}")
            return {
                "success": False,
                "slice_id": slice_order.slice_id,
                "error": str(e)
            }
    
    async def _schedule_remaining_slices(self, slices: List[OrderSlice]):
        """Schedule remaining slices for later execution."""
        for slice_order in slices:
            # In a real system, this would use a job scheduler
            # For now, we'll just log the scheduling
            logger.info(f"Scheduled slice {slice_order.slice_id} for {slice_order.scheduled_time}")
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol."""
        try:
            # In a real system, would use live market data
            # For testing, use yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            
            if not data.empty:
                return float(data['Close'].iloc[-1])
            else:
                # Fallback to a reasonable price
                return 100.0
                
        except Exception as e:
            logger.warning(f"Could not get current price for {symbol}: {str(e)}")
            return 100.0  # Default price
    
    async def _get_average_volume(self, symbol: str) -> Dict[str, float]:
        """Get average trading volume for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="30d")
            
            if not data.empty:
                avg_volume = data['Volume'].mean()
                return {"avg_volume": float(avg_volume)}
            else:
                return {"avg_volume": 100000.0}  # Default
                
        except Exception as e:
            logger.warning(f"Could not get volume data for {symbol}: {str(e)}")
            return {"avg_volume": 100000.0}
    
    async def _estimate_market_impact(self, symbol: str, quantity: float) -> Dict[str, float]:
        """Estimate market impact for an order."""
        try:
            volume_data = await self._get_average_volume(symbol)
            avg_volume = volume_data["avg_volume"]
            
            # Simplified market impact model
            participation_rate = quantity / avg_volume
            
            # Linear impact: roughly 0.1% per 1% of volume
            linear_impact = participation_rate * 0.001
            
            # Square-root impact for larger orders
            sqrt_impact = np.sqrt(participation_rate) * 0.002
            
            return {
                "linear_impact": linear_impact,
                "sqrt_impact": sqrt_impact,
                "participation_rate": participation_rate,
                "estimated_total_impact": linear_impact + sqrt_impact
            }
            
        except Exception as e:
            logger.error(f"Error estimating market impact: {str(e)}")
            return {
                "linear_impact": 0.001,
                "sqrt_impact": 0.002,
                "participation_rate": 0.1,
                "estimated_total_impact": 0.003
            }
    
    async def _analyze_venues_for_order(self, order: ExecutionOrder) -> Dict[str, Dict]:
        """Analyze all venues for order execution."""
        venue_analysis = {}
        
        for venue_name, venue in self.venues.items():
            analysis = {
                "liquidity_score": venue.liquidity_score,
                "speed_score": venue.speed_score,
                "fees": venue.fee_structure,
                "suitable_for_size": venue.min_size <= order.total_quantity <= venue.max_size,
                "supports_hidden": venue.supports_hidden,
                "venue_type": venue.venue_type.value
            }
            
            # Calculate expected cost
            if order.price_limit:
                fee = venue.fee_structure.get("maker_fee", 0.002)
            else:
                fee = venue.fee_structure.get("taker_fee", 0.003)
            
            analysis["expected_fee_rate"] = fee
            venue_analysis[venue_name] = analysis
        
        return venue_analysis
    
    def _optimize_venue_allocation(self, order: ExecutionOrder, venue_analysis: Dict) -> Dict[str, float]:
        """Optimize allocation across venues."""
        try:
            # Simple allocation optimization
            suitable_venues = {
                name: analysis for name, analysis in venue_analysis.items()
                if analysis["suitable_for_size"]
            }
            
            if not suitable_venues:
                return {"NASDAQ": 1.0}  # Fallback
            
            # Score-based allocation
            total_score = 0
            venue_scores = {}
            
            for venue_name, analysis in suitable_venues.items():
                # Composite score
                score = (
                    analysis["liquidity_score"] * 0.4 +
                    analysis["speed_score"] * 0.3 +
                    (1.0 - analysis["expected_fee_rate"] * 100) * 0.3  # Lower fees = higher score
                )
                venue_scores[venue_name] = max(0, score)
                total_score += venue_scores[venue_name]
            
            # Normalize to allocations
            if total_score > 0:
                allocations = {
                    venue: score / total_score 
                    for venue, score in venue_scores.items()
                }
                
                # Ensure minimum viable allocations
                final_allocations = {}
                for venue, allocation in allocations.items():
                    if allocation > 0.1:  # Only allocate if >10%
                        final_allocations[venue] = allocation
                
                # Renormalize
                total_allocation = sum(final_allocations.values())
                if total_allocation > 0:
                    return {
                        venue: alloc / total_allocation 
                        for venue, alloc in final_allocations.items()
                    }
            
            return {"NASDAQ": 1.0}  # Fallback
            
        except Exception as e:
            logger.error(f"Error optimizing venue allocation: {str(e)}")
            return {"NASDAQ": 1.0}
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of an execution order."""
        if order_id not in self.active_orders:
            return {"error": "Order not found"}
        
        order = self.active_orders[order_id]
        
        return {
            "order_id": order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "total_quantity": order.total_quantity,
            "filled_quantity": order.filled_quantity,
            "status": order.status.value,
            "algorithm": order.algorithm.value,
            "created_at": order.created_at.isoformat(),
            "slices_executed": len([s for s in order.slices if s.scheduled_time <= datetime.now()]),
            "total_slices": len(order.slices),
            "venue_allocations": order.venue_allocations or {}
        }
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an execution order."""
        if order_id not in self.active_orders:
            return {"error": "Order not found"}
        
        order = self.active_orders[order_id]
        order.status = OrderStatus.CANCELLED
        
        # Move to history
        self.order_history.append(order)
        del self.active_orders[order_id]
        
        return {
            "success": True,
            "order_id": order_id,
            "status": "cancelled",
            "cancelled_at": datetime.now().isoformat()
        }
