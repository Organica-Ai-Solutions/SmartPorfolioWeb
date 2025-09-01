import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import httpx
import asyncio
from fastapi import HTTPException
import json
import fredapi
from enum import Enum

logger = logging.getLogger(__name__)

class MarketRegime(str, Enum):
    EXPANSION = "expansion"
    PEAK = "peak"
    CONTRACTION = "contraction"
    TROUGH = "trough"
    RECOVERY = "recovery"
    STAGFLATION = "stagflation"
    REFLATION = "reflation"
    DISINFLATION = "disinflation"

class MacroService:
    """Service for analyzing macroeconomic indicators."""
    
    def __init__(self):
        self.fred_api_key = os.getenv("FRED_API_KEY", "")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY", "")
        self.world_bank_api = "https://api.worldbank.org/v2"
        
        # Initialize FRED API if key is available
        self.fred = None
        if self.fred_api_key:
            try:
                self.fred = fredapi.Fred(api_key=self.fred_api_key)
            except Exception as e:
                logger.error(f"Failed to initialize FRED API: {str(e)}")
        
        # Check if we have any API keys
        if not any([self.fred_api_key, self.alpha_vantage_key]):
            logger.warning("No macro API keys configured. Macro analysis will use fallback methods.")
    
    async def get_macro_indicators(self) -> Dict[str, Any]:
        """Get comprehensive macroeconomic indicators."""
        try:
            # Create tasks for parallel execution
            tasks = []
            
            # Add tasks based on available API keys
            if self.fred:
                tasks.append(self._get_fred_indicators())
            
            if self.alpha_vantage_key:
                tasks.append(self._get_alpha_vantage_indicators())
            
            # Always add the fallback indicators
            tasks.append(self._get_fallback_indicators())
            
            # Add World Bank data (no API key required)
            tasks.append(self._get_world_bank_indicators())
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results, filtering out exceptions
            combined_data = {
                "inflation": {},
                "employment": {},
                "gdp": {},
                "interest_rates": {},
                "housing": {},
                "consumer": {},
                "industrial": {},
                "market_regime": {},
                "sources": [],
                "timestamp": datetime.now().isoformat()
            }
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in macro analysis: {str(result)}")
                    continue
                
                # Update combined data with valid results
                if "source" in result:
                    combined_data["sources"].append(result["source"])
                
                # Merge macro data
                for key in ["inflation", "employment", "gdp", "interest_rates", 
                           "housing", "consumer", "industrial", "market_regime"]:
                    if key in result:
                        combined_data[key].update(result[key])
            
            # Determine market regime if not already set
            if not combined_data["market_regime"].get("current_regime"):
                combined_data["market_regime"]["current_regime"] = self._determine_market_regime(combined_data)
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error analyzing macro indicators: {str(e)}")
            return {
                "inflation": {"cpi_yoy": None},
                "employment": {"unemployment_rate": None},
                "gdp": {"gdp_growth": None},
                "interest_rates": {"fed_funds_rate": None},
                "market_regime": {"current_regime": "unknown"},
                "sources": ["error"],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_fred_indicators(self) -> Dict[str, Any]:
        """Get macroeconomic indicators from FRED."""
        try:
            # Define FRED series IDs for various indicators
            series_ids = {
                "inflation": {
                    "cpi_yoy": "CPIAUCSL",  # CPI All Urban Consumers
                    "core_cpi_yoy": "CPILFESL",  # CPI Less Food & Energy
                    "ppi": "PPIACO",  # Producer Price Index
                    "pce_price_index": "PCEPI"  # Personal Consumption Expenditures Price Index
                },
                "employment": {
                    "unemployment_rate": "UNRATE",  # Unemployment Rate
                    "nonfarm_payrolls": "PAYEMS",  # Total Nonfarm Payrolls
                    "initial_claims": "ICSA",  # Initial Jobless Claims
                    "labor_force_participation": "CIVPART"  # Labor Force Participation Rate
                },
                "gdp": {
                    "gdp": "GDP",  # Gross Domestic Product
                    "real_gdp": "GDPC1",  # Real Gross Domestic Product
                    "gdp_growth": "A191RL1Q225SBEA"  # Real GDP Growth Rate
                },
                "interest_rates": {
                    "fed_funds_rate": "FEDFUNDS",  # Federal Funds Rate
                    "treasury_10y": "GS10",  # 10-Year Treasury Rate
                    "treasury_2y": "GS2",  # 2-Year Treasury Rate
                    "treasury_3m": "GS3M"  # 3-Month Treasury Rate
                },
                "housing": {
                    "housing_starts": "HOUST",  # Housing Starts
                    "home_price_index": "CSUSHPISA",  # Case-Shiller Home Price Index
                    "mortgage_rate_30y": "MORTGAGE30US"  # 30-Year Fixed Rate Mortgage Average
                },
                "consumer": {
                    "retail_sales": "RSAFS",  # Retail Sales
                    "consumer_sentiment": "UMCSENT",  # Consumer Sentiment Index
                    "personal_income": "PI",  # Personal Income
                    "personal_spending": "PCE"  # Personal Consumption Expenditures
                },
                "industrial": {
                    "industrial_production": "INDPRO",  # Industrial Production Index
                    "capacity_utilization": "TCU",  # Capacity Utilization
                    "ism_manufacturing": "NAPM",  # ISM Manufacturing Index
                    "durable_goods_orders": "DGORDER"  # Durable Goods Orders
                }
            }
            
            # Get data for each series
            data = {}
            
            for category, indicators in series_ids.items():
                data[category] = {}
                
                for name, series_id in indicators.items():
                    try:
                        # Get the series data
                        series = self.fred.get_series(series_id)
                        
                        # Get the most recent value
                        if not series.empty:
                            latest_date = series.index[-1]
                            latest_value = series.iloc[-1]
                            
                            # Calculate year-over-year change for certain indicators
                            yoy_change = None
                            if len(series) > 12:  # Need at least a year of data
                                year_ago_value = series.iloc[-13] if len(series) >= 13 else None
                                if year_ago_value is not None:
                                    yoy_change = (latest_value - year_ago_value) / year_ago_value
                            
                            # Store the data
                            data[category][name] = {
                                "value": float(latest_value),
                                "date": latest_date.strftime("%Y-%m-%d"),
                                "yoy_change": float(yoy_change) if yoy_change is not None else None,
                                "series_id": series_id
                            }
                    except Exception as series_error:
                        logger.error(f"Error getting FRED series {series_id}: {str(series_error)}")
            
            # Determine market regime
            market_regime = self._determine_market_regime_from_fred(data)
            
            # Create overall result
            result = {
                "source": "fred",
                **data,
                "market_regime": market_regime,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting FRED indicators: {str(e)}")
            raise
    
    async def _get_alpha_vantage_indicators(self) -> Dict[str, Any]:
        """Get macroeconomic indicators from Alpha Vantage."""
        try:
            indicators = {
                "inflation": {},
                "employment": {},
                "gdp": {},
                "interest_rates": {}
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get real GDP
                gdp_response = await client.get(
                    "https://www.alphavantage.co/query",
                    params={
                        "function": "REAL_GDP",
                        "interval": "quarterly",
                        "apikey": self.alpha_vantage_key
                    }
                )
                
                if gdp_response.status_code == 200:
                    gdp_data = gdp_response.json()
                    gdp_values = gdp_data.get("data", [])
                    
                    if gdp_values:
                        latest_gdp = gdp_values[0]
                        previous_gdp = gdp_values[1] if len(gdp_values) > 1 else None
                        
                        gdp_value = float(latest_gdp.get("value", 0))
                        gdp_date = latest_gdp.get("date", "")
                        
                        # Calculate growth rate
                        gdp_growth = None
                        if previous_gdp:
                            previous_value = float(previous_gdp.get("value", 0))
                            if previous_value > 0:
                                gdp_growth = (gdp_value - previous_value) / previous_value
                        
                        indicators["gdp"]["real_gdp"] = {
                            "value": gdp_value,
                            "date": gdp_date,
                            "growth": gdp_growth
                        }
                
                # Get CPI
                cpi_response = await client.get(
                    "https://www.alphavantage.co/query",
                    params={
                        "function": "CPI",
                        "interval": "monthly",
                        "apikey": self.alpha_vantage_key
                    }
                )
                
                if cpi_response.status_code == 200:
                    cpi_data = cpi_response.json()
                    cpi_values = cpi_data.get("data", [])
                    
                    if cpi_values:
                        latest_cpi = cpi_values[0]
                        year_ago_cpi = next((cpi for cpi in cpi_values if self._is_year_ago(cpi.get("date", ""), latest_cpi.get("date", ""))), None)
                        
                        cpi_value = float(latest_cpi.get("value", 0))
                        cpi_date = latest_cpi.get("date", "")
                        
                        # Calculate YoY inflation
                        yoy_inflation = None
                        if year_ago_cpi:
                            year_ago_value = float(year_ago_cpi.get("value", 0))
                            if year_ago_value > 0:
                                yoy_inflation = (cpi_value - year_ago_value) / year_ago_value
                        
                        indicators["inflation"]["cpi"] = {
                            "value": cpi_value,
                            "date": cpi_date,
                            "yoy_change": yoy_inflation
                        }
                
                # Get unemployment rate
                unemployment_response = await client.get(
                    "https://www.alphavantage.co/query",
                    params={
                        "function": "UNEMPLOYMENT",
                        "apikey": self.alpha_vantage_key
                    }
                )
                
                if unemployment_response.status_code == 200:
                    unemployment_data = unemployment_response.json()
                    unemployment_values = unemployment_data.get("data", [])
                    
                    if unemployment_values:
                        latest_unemployment = unemployment_values[0]
                        
                        unemployment_value = float(latest_unemployment.get("value", 0))
                        unemployment_date = latest_unemployment.get("date", "")
                        
                        indicators["employment"]["unemployment_rate"] = {
                            "value": unemployment_value,
                            "date": unemployment_date
                        }
                
                # Get federal funds rate
                fed_rate_response = await client.get(
                    "https://www.alphavantage.co/query",
                    params={
                        "function": "FEDERAL_FUNDS_RATE",
                        "interval": "monthly",
                        "apikey": self.alpha_vantage_key
                    }
                )
                
                if fed_rate_response.status_code == 200:
                    fed_rate_data = fed_rate_response.json()
                    fed_rate_values = fed_rate_data.get("data", [])
                    
                    if fed_rate_values:
                        latest_fed_rate = fed_rate_values[0]
                        
                        fed_rate_value = float(latest_fed_rate.get("value", 0))
                        fed_rate_date = latest_fed_rate.get("date", "")
                        
                        indicators["interest_rates"]["fed_funds_rate"] = {
                            "value": fed_rate_value,
                            "date": fed_rate_date
                        }
            
            # Determine market regime
            market_regime = self._determine_market_regime_from_alpha_vantage(indicators)
            
            # Create overall result
            result = {
                "source": "alpha_vantage",
                **indicators,
                "market_regime": market_regime,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting Alpha Vantage indicators: {str(e)}")
            raise
    
    async def _get_world_bank_indicators(self) -> Dict[str, Any]:
        """Get macroeconomic indicators from World Bank API."""
        try:
            indicators = {
                "gdp": {},
                "inflation": {},
                "interest_rates": {}
            }
            
            # Define indicator codes
            indicator_codes = {
                "gdp_growth": "NY.GDP.MKTP.KD.ZG",  # GDP growth (annual %)
                "inflation": "FP.CPI.TOTL.ZG",  # Inflation, consumer prices (annual %)
                "real_interest_rate": "FR.INR.RINR"  # Real interest rate (%)
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get data for United States (country code: US)
                for name, code in indicator_codes.items():
                    response = await client.get(
                        f"{self.world_bank_api}/country/US/indicator/{code}",
                        params={
                            "format": "json",
                            "per_page": 5,  # Get only the most recent values
                            "date": "2018:2023"  # Last 5 years
                        }
                    )
                    
                    if response.status_code != 200:
                        continue
                    
                    data = response.json()
                    
                    # World Bank API returns a list with metadata as first element
                    if len(data) < 2 or not data[1]:
                        continue
                    
                    values = data[1]
                    
                    # Find the most recent non-null value
                    latest_value = next((v for v in values if v.get("value") is not None), None)
                    
                    if latest_value:
                        value = float(latest_value.get("value", 0))
                        year = latest_value.get("date", "")
                        
                        # Store in appropriate category
                        if name == "gdp_growth":
                            indicators["gdp"]["gdp_growth_wb"] = {
                                "value": value,
                                "year": year
                            }
                        elif name == "inflation":
                            indicators["inflation"]["cpi_yoy_wb"] = {
                                "value": value,
                                "year": year
                            }
                        elif name == "real_interest_rate":
                            indicators["interest_rates"]["real_interest_rate_wb"] = {
                                "value": value,
                                "year": year
                            }
            
            # Create overall result
            result = {
                "source": "world_bank",
                **indicators,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting World Bank indicators: {str(e)}")
            raise
    
    async def _get_fallback_indicators(self) -> Dict[str, Any]:
        """Fallback method to get basic macroeconomic indicators."""
        try:
            # Use hardcoded recent values as fallback
            indicators = {
                "inflation": {
                    "cpi_yoy": {
                        "value": 3.7,  # Approximate recent CPI YoY
                        "date": "2023-09-01",
                        "source": "fallback"
                    }
                },
                "employment": {
                    "unemployment_rate": {
                        "value": 3.8,  # Approximate recent unemployment rate
                        "date": "2023-09-01",
                        "source": "fallback"
                    }
                },
                "gdp": {
                    "gdp_growth": {
                        "value": 2.1,  # Approximate recent GDP growth rate
                        "date": "2023-06-30",
                        "source": "fallback"
                    }
                },
                "interest_rates": {
                    "fed_funds_rate": {
                        "value": 5.25,  # Approximate recent Fed funds rate
                        "date": "2023-09-01",
                        "source": "fallback"
                    },
                    "treasury_10y": {
                        "value": 4.6,  # Approximate recent 10-year Treasury yield
                        "date": "2023-09-01",
                        "source": "fallback"
                    }
                },
                "market_regime": {
                    "current_regime": MarketRegime.DISINFLATION,
                    "confidence": 0.6,
                    "source": "fallback"
                }
            }
            
            # Create overall result
            result = {
                "source": "fallback",
                **indicators,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting fallback indicators: {str(e)}")
            raise
    
    def _determine_market_regime(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the current market regime based on available indicators."""
        try:
            # Extract key indicators
            inflation = None
            gdp_growth = None
            unemployment = None
            fed_rate = None
            
            # Get inflation
            if "inflation" in data and "cpi_yoy" in data["inflation"]:
                inflation_data = data["inflation"]["cpi_yoy"]
                if isinstance(inflation_data, dict) and "value" in inflation_data:
                    inflation = inflation_data["value"]
                else:
                    inflation = inflation_data
            
            # Get GDP growth
            if "gdp" in data and "gdp_growth" in data["gdp"]:
                gdp_data = data["gdp"]["gdp_growth"]
                if isinstance(gdp_data, dict) and "value" in gdp_data:
                    gdp_growth = gdp_data["value"]
                else:
                    gdp_growth = gdp_data
            
            # Get unemployment
            if "employment" in data and "unemployment_rate" in data["employment"]:
                unemployment_data = data["employment"]["unemployment_rate"]
                if isinstance(unemployment_data, dict) and "value" in unemployment_data:
                    unemployment = unemployment_data["value"]
                else:
                    unemployment = unemployment_data
            
            # Get Fed funds rate
            if "interest_rates" in data and "fed_funds_rate" in data["interest_rates"]:
                fed_rate_data = data["interest_rates"]["fed_funds_rate"]
                if isinstance(fed_rate_data, dict) and "value" in fed_rate_data:
                    fed_rate = fed_rate_data["value"]
                else:
                    fed_rate = fed_rate_data
            
            # Determine regime based on available indicators
            regime = MarketRegime.EXPANSION  # Default
            confidence = 0.5  # Default confidence
            
            # Check if we have enough data
            if inflation is not None and gdp_growth is not None:
                # High inflation, positive growth -> Expansion or Peak
                if inflation > 3.0 and gdp_growth > 2.0:
                    if fed_rate is not None and fed_rate > 4.0:
                        regime = MarketRegime.PEAK
                        confidence = 0.7
                    else:
                        regime = MarketRegime.EXPANSION
                        confidence = 0.8
                
                # High inflation, negative/low growth -> Stagflation
                elif inflation > 3.0 and gdp_growth < 1.0:
                    regime = MarketRegime.STAGFLATION
                    confidence = 0.75
                
                # Low inflation, negative growth -> Contraction
                elif inflation < 2.0 and gdp_growth < 0:
                    regime = MarketRegime.CONTRACTION
                    confidence = 0.8
                
                # Low inflation, low positive growth -> Recovery or Trough
                elif inflation < 2.0 and gdp_growth > 0 and gdp_growth < 2.0:
                    if unemployment is not None and unemployment > 5.0:
                        regime = MarketRegime.TROUGH
                        confidence = 0.6
                    else:
                        regime = MarketRegime.RECOVERY
                        confidence = 0.7
                
                # Falling inflation, positive growth -> Disinflation
                elif 2.0 <= inflation <= 3.0 and gdp_growth > 1.5:
                    regime = MarketRegime.DISINFLATION
                    confidence = 0.65
                
                # Rising inflation, positive growth -> Reflation
                elif 2.0 <= inflation <= 3.0 and gdp_growth > 0 and gdp_growth < 1.5:
                    regime = MarketRegime.REFLATION
                    confidence = 0.6
            
            return {
                "current_regime": regime,
                "confidence": confidence,
                "indicators_used": {
                    "inflation": inflation,
                    "gdp_growth": gdp_growth,
                    "unemployment": unemployment,
                    "fed_rate": fed_rate
                }
            }
            
        except Exception as e:
            logger.error(f"Error determining market regime: {str(e)}")
            return {
                "current_regime": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _determine_market_regime_from_fred(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Determine market regime specifically from FRED data."""
        try:
            # Extract key indicators
            inflation = None
            gdp_growth = None
            unemployment = None
            fed_rate = None
            
            # Get inflation
            if "inflation" in data and "cpi_yoy" in data["inflation"]:
                inflation = data["inflation"]["cpi_yoy"].get("value")
            
            # Get GDP growth
            if "gdp" in data and "gdp_growth" in data["gdp"]:
                gdp_growth = data["gdp"]["gdp_growth"].get("value")
            
            # Get unemployment
            if "employment" in data and "unemployment_rate" in data["employment"]:
                unemployment = data["employment"]["unemployment_rate"].get("value")
            
            # Get Fed funds rate
            if "interest_rates" in data and "fed_funds_rate" in data["interest_rates"]:
                fed_rate = data["interest_rates"]["fed_funds_rate"].get("value")
            
            # Use the common method to determine regime
            return self._determine_market_regime({
                "inflation": {"cpi_yoy": inflation},
                "gdp": {"gdp_growth": gdp_growth},
                "employment": {"unemployment_rate": unemployment},
                "interest_rates": {"fed_funds_rate": fed_rate}
            })
            
        except Exception as e:
            logger.error(f"Error determining market regime from FRED: {str(e)}")
            return {
                "current_regime": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _determine_market_regime_from_alpha_vantage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Determine market regime specifically from Alpha Vantage data."""
        try:
            # Extract key indicators
            inflation = None
            gdp_growth = None
            unemployment = None
            fed_rate = None
            
            # Get inflation
            if "inflation" in data and "cpi" in data["inflation"]:
                inflation = data["inflation"]["cpi"].get("yoy_change")
                if inflation is not None:
                    inflation *= 100  # Convert to percentage
            
            # Get GDP growth
            if "gdp" in data and "real_gdp" in data["gdp"]:
                gdp_growth = data["gdp"]["real_gdp"].get("growth")
                if gdp_growth is not None:
                    gdp_growth *= 100  # Convert to percentage
            
            # Get unemployment
            if "employment" in data and "unemployment_rate" in data["employment"]:
                unemployment = data["employment"]["unemployment_rate"].get("value")
            
            # Get Fed funds rate
            if "interest_rates" in data and "fed_funds_rate" in data["interest_rates"]:
                fed_rate = data["interest_rates"]["fed_funds_rate"].get("value")
            
            # Use the common method to determine regime
            return self._determine_market_regime({
                "inflation": {"cpi_yoy": inflation},
                "gdp": {"gdp_growth": gdp_growth},
                "employment": {"unemployment_rate": unemployment},
                "interest_rates": {"fed_funds_rate": fed_rate}
            })
            
        except Exception as e:
            logger.error(f"Error determining market regime from Alpha Vantage: {str(e)}")
            return {
                "current_regime": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _is_year_ago(self, date1: str, date2: str) -> bool:
        """Check if date1 is approximately one year before date2."""
        try:
            d1 = datetime.strptime(date1, "%Y-%m-%d")
            d2 = datetime.strptime(date2, "%Y-%m-%d")
            
            # Check if the dates are 11-13 months apart (approximately a year)
            diff = (d2.year - d1.year) * 12 + d2.month - d1.month
            return 11 <= diff <= 13
            
        except Exception:
            return False
