import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import httpx
import asyncio
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class OptionsService:
    """Service for analyzing options market data to gauge market expectations and volatility."""
    
    def __init__(self):
        self.polygon_api_key = os.getenv("POLYGON_API_KEY", "")
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY", "")
        
        # Check if we have API keys
        if not any([self.polygon_api_key, self.finnhub_api_key]):
            logger.warning("No options API keys configured. Options analysis will use fallback methods.")
    
    async def get_options_data(self, tickers: List[str]) -> Dict[str, Any]:
        """Get options market data and analysis for a list of tickers."""
        try:
            # Create tasks for parallel execution
            tasks = []
            
            # Add tasks based on available API keys
            if self.polygon_api_key:
                tasks.append(self._get_polygon_options_data(tickers))
            
            if self.finnhub_api_key:
                tasks.append(self._get_finnhub_options_data(tickers))
            
            # Always add the fallback options analysis
            tasks.append(self._get_yfinance_options_data(tickers))
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results, filtering out exceptions
            combined_data = {
                "implied_volatility": {},
                "put_call_ratio": {},
                "options_volume": {},
                "volatility_smile": {},
                "volatility_term_structure": {},
                "market_expectations": {},
                "sources": [],
                "timestamp": datetime.now().isoformat()
            }
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in options analysis: {str(result)}")
                    continue
                
                # Update combined data with valid results
                if "source" in result:
                    combined_data["sources"].append(result["source"])
                
                # Merge options data
                for key in ["implied_volatility", "put_call_ratio", "options_volume", 
                           "volatility_smile", "volatility_term_structure", "market_expectations"]:
                    if key in result:
                        combined_data[key].update(result[key])
            
            # Ensure we have data for all tickers
            for ticker in tickers:
                if ticker not in combined_data["implied_volatility"]:
                    combined_data["implied_volatility"][ticker] = None
                if ticker not in combined_data["put_call_ratio"]:
                    combined_data["put_call_ratio"][ticker] = None
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error analyzing options data: {str(e)}")
            return {
                "implied_volatility": {ticker: None for ticker in tickers},
                "put_call_ratio": {ticker: None for ticker in tickers},
                "sources": ["error"],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_polygon_options_data(self, tickers: List[str]) -> Dict[str, Any]:
        """Get options data from Polygon.io API."""
        try:
            options_data = {
                "implied_volatility": {},
                "put_call_ratio": {},
                "options_volume": {},
                "volatility_smile": {},
                "market_expectations": {}
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                for ticker in tickers:
                    # Get current date
                    today = datetime.now()
                    
                    # Get options expirations
                    expirations_response = await client.get(
                        f"https://api.polygon.io/v3/reference/options/contracts",
                        params={
                            "underlying_ticker": ticker,
                            "limit": 1000,
                            "apiKey": self.polygon_api_key
                        }
                    )
                    
                    if expirations_response.status_code != 200:
                        logger.warning(f"Polygon API request failed: {expirations_response.text}")
                        continue
                    
                    expirations_data = expirations_response.json()
                    contracts = expirations_data.get("results", [])
                    
                    if not contracts:
                        logger.warning(f"No options contracts found for {ticker}")
                        continue
                    
                    # Extract unique expirations
                    expirations = set()
                    for contract in contracts:
                        exp_date = contract.get("expiration_date")
                        if exp_date:
                            expirations.add(exp_date)
                    
                    # Sort expirations and get the closest ones
                    expirations = sorted(list(expirations))
                    near_expirations = expirations[:3] if len(expirations) >= 3 else expirations
                    
                    # Initialize data structures
                    calls_volume = 0
                    puts_volume = 0
                    iv_values = []
                    strikes = []
                    call_ivs = []
                    put_ivs = []
                    
                    # Process each expiration
                    for expiration in near_expirations:
                        # Get options chain for this expiration
                        chain_response = await client.get(
                            f"https://api.polygon.io/v3/snapshot/options/{ticker}",
                            params={
                                "expiration_date": expiration,
                                "apiKey": self.polygon_api_key
                            }
                        )
                        
                        if chain_response.status_code != 200:
                            continue
                        
                        chain_data = chain_response.json()
                        snapshots = chain_data.get("results", [])
                        
                        for snapshot in snapshots:
                            details = snapshot.get("details", {})
                            contract_type = details.get("contract_type")
                            strike_price = details.get("strike_price")
                            
                            # Get volume and implied volatility
                            day_data = snapshot.get("day", {})
                            volume = day_data.get("volume", 0)
                            
                            greeks = snapshot.get("greeks", {})
                            iv = greeks.get("implied_volatility")
                            
                            if iv is not None:
                                iv_values.append(iv)
                            
                            if contract_type == "call":
                                calls_volume += volume
                                if strike_price and iv:
                                    strikes.append(strike_price)
                                    call_ivs.append(iv)
                            elif contract_type == "put":
                                puts_volume += volume
                                if strike_price and iv:
                                    put_ivs.append(iv)
                    
                    # Calculate average implied volatility
                    avg_iv = np.mean(iv_values) if iv_values else None
                    
                    # Calculate put/call ratio
                    pc_ratio = puts_volume / calls_volume if calls_volume > 0 else None
                    
                    # Store results
                    options_data["implied_volatility"][ticker] = avg_iv
                    options_data["put_call_ratio"][ticker] = pc_ratio
                    options_data["options_volume"][ticker] = calls_volume + puts_volume
                    
                    # Create volatility smile data
                    if strikes and call_ivs:
                        # Sort by strike price
                        strike_iv_pairs = sorted(zip(strikes, call_ivs), key=lambda x: x[0])
                        options_data["volatility_smile"][ticker] = {
                            "strikes": [pair[0] for pair in strike_iv_pairs],
                            "call_ivs": [pair[1] for pair in strike_iv_pairs]
                        }
                    
                    # Interpret market expectations
                    market_expectation = self._interpret_options_data(avg_iv, pc_ratio)
                    options_data["market_expectations"][ticker] = market_expectation
            
            # Create overall result
            result = {
                "source": "polygon",
                **options_data,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting Polygon options data: {str(e)}")
            raise
    
    async def _get_finnhub_options_data(self, tickers: List[str]) -> Dict[str, Any]:
        """Get options data from Finnhub API."""
        try:
            options_data = {
                "implied_volatility": {},
                "put_call_ratio": {},
                "options_volume": {},
                "market_expectations": {}
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                for ticker in tickers:
                    # Get options chain
                    response = await client.get(
                        "https://finnhub.io/api/v1/stock/option-chain",
                        params={
                            "symbol": ticker,
                            "token": self.finnhub_api_key
                        }
                    )
                    
                    if response.status_code != 200:
                        logger.warning(f"Finnhub API request failed: {response.text}")
                        continue
                    
                    data = response.json()
                    
                    if not data or "data" not in data:
                        continue
                    
                    # Process options data
                    calls_volume = 0
                    puts_volume = 0
                    iv_values = []
                    
                    for option_data in data.get("data", []):
                        options = option_data.get("options", {})
                        
                        # Process calls
                        for call in options.get("CALL", []):
                            volume = call.get("volume", 0)
                            calls_volume += volume
                            
                            iv = call.get("impliedVolatility")
                            if iv is not None:
                                iv_values.append(iv)
                        
                        # Process puts
                        for put in options.get("PUT", []):
                            volume = put.get("volume", 0)
                            puts_volume += volume
                            
                            iv = put.get("impliedVolatility")
                            if iv is not None:
                                iv_values.append(iv)
                    
                    # Calculate average implied volatility
                    avg_iv = np.mean(iv_values) if iv_values else None
                    
                    # Calculate put/call ratio
                    pc_ratio = puts_volume / calls_volume if calls_volume > 0 else None
                    
                    # Store results
                    options_data["implied_volatility"][ticker] = avg_iv
                    options_data["put_call_ratio"][ticker] = pc_ratio
                    options_data["options_volume"][ticker] = calls_volume + puts_volume
                    
                    # Interpret market expectations
                    market_expectation = self._interpret_options_data(avg_iv, pc_ratio)
                    options_data["market_expectations"][ticker] = market_expectation
            
            # Create overall result
            result = {
                "source": "finnhub",
                **options_data,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting Finnhub options data: {str(e)}")
            raise
    
    async def _get_yfinance_options_data(self, tickers: List[str]) -> Dict[str, Any]:
        """Fallback method to get options data using yfinance."""
        try:
            options_data = {
                "implied_volatility": {},
                "put_call_ratio": {},
                "options_volume": {},
                "volatility_smile": {},
                "volatility_term_structure": {},
                "market_expectations": {}
            }
            
            for ticker in tickers:
                try:
                    # Get stock data
                    stock = yf.Ticker(ticker)
                    
                    # Get expirations
                    expirations = stock.options
                    
                    if not expirations:
                        logger.warning(f"No options expirations found for {ticker}")
                        continue
                    
                    # Use the first 3 expirations or fewer if not available
                    near_expirations = expirations[:3] if len(expirations) >= 3 else expirations
                    
                    # Initialize data structures
                    calls_volume = 0
                    puts_volume = 0
                    iv_values = []
                    term_structure = {}
                    
                    # Store strike prices and IVs for volatility smile
                    strikes = []
                    call_ivs = []
                    
                    # Process each expiration
                    for expiration in near_expirations:
                        # Get options chain for this expiration
                        opt = stock.option_chain(expiration)
                        
                        # Process calls
                        for _, call in opt.calls.iterrows():
                            volume = call.get("volume", 0)
                            if not pd.isna(volume):
                                calls_volume += volume
                            
                            iv = call.get("impliedVolatility")
                            if iv is not None and not pd.isna(iv):
                                iv_values.append(iv)
                                
                                # Store for volatility smile
                                strike = call.get("strike")
                                if strike is not None and not pd.isna(strike):
                                    strikes.append(strike)
                                    call_ivs.append(iv)
                        
                        # Process puts
                        for _, put in opt.puts.iterrows():
                            volume = put.get("volume", 0)
                            if not pd.isna(volume):
                                puts_volume += volume
                            
                            iv = put.get("impliedVolatility")
                            if iv is not None and not pd.isna(iv):
                                iv_values.append(iv)
                        
                        # Calculate average IV for this expiration
                        exp_ivs = [iv for iv in iv_values if not pd.isna(iv)]
                        if exp_ivs:
                            term_structure[expiration] = np.mean(exp_ivs)
                    
                    # Calculate average implied volatility
                    valid_ivs = [iv for iv in iv_values if not pd.isna(iv)]
                    avg_iv = np.mean(valid_ivs) if valid_ivs else None
                    
                    # Calculate put/call ratio
                    pc_ratio = puts_volume / calls_volume if calls_volume > 0 else None
                    
                    # Store results
                    options_data["implied_volatility"][ticker] = avg_iv
                    options_data["put_call_ratio"][ticker] = pc_ratio
                    options_data["options_volume"][ticker] = calls_volume + puts_volume
                    
                    # Create volatility smile data
                    if strikes and call_ivs:
                        # Sort by strike price
                        strike_iv_pairs = sorted(zip(strikes, call_ivs), key=lambda x: x[0])
                        options_data["volatility_smile"][ticker] = {
                            "strikes": [pair[0] for pair in strike_iv_pairs],
                            "call_ivs": [pair[1] for pair in strike_iv_pairs]
                        }
                    
                    # Store term structure
                    if term_structure:
                        options_data["volatility_term_structure"][ticker] = term_structure
                    
                    # Interpret market expectations
                    market_expectation = self._interpret_options_data(avg_iv, pc_ratio)
                    options_data["market_expectations"][ticker] = market_expectation
                    
                except Exception as ticker_error:
                    logger.error(f"Error processing options for {ticker}: {str(ticker_error)}")
            
            # Create overall result
            result = {
                "source": "yfinance",
                **options_data,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting yfinance options data: {str(e)}")
            raise
    
    def _interpret_options_data(self, implied_volatility: Optional[float], put_call_ratio: Optional[float]) -> Dict[str, Any]:
        """Interpret options data to gauge market expectations."""
        result = {
            "volatility_expectation": "unknown",
            "sentiment": "unknown",
            "expected_move": None
        }
        
        # Interpret implied volatility
        if implied_volatility is not None:
            if implied_volatility > 0.5:  # 50% IV is very high
                result["volatility_expectation"] = "very_high"
                result["expected_move"] = f"{implied_volatility * 100:.1f}% in either direction over the next year"
            elif implied_volatility > 0.3:
                result["volatility_expectation"] = "high"
                result["expected_move"] = f"{implied_volatility * 100:.1f}% in either direction over the next year"
            elif implied_volatility > 0.2:
                result["volatility_expectation"] = "moderate"
                result["expected_move"] = f"{implied_volatility * 100:.1f}% in either direction over the next year"
            else:
                result["volatility_expectation"] = "low"
                result["expected_move"] = f"{implied_volatility * 100:.1f}% in either direction over the next year"
        
        # Interpret put/call ratio
        if put_call_ratio is not None:
            if put_call_ratio > 1.5:
                result["sentiment"] = "very_bearish"
            elif put_call_ratio > 1.0:
                result["sentiment"] = "bearish"
            elif put_call_ratio > 0.7:
                result["sentiment"] = "neutral"
            elif put_call_ratio > 0.5:
                result["sentiment"] = "bullish"
            else:
                result["sentiment"] = "very_bullish"
        
        return result
