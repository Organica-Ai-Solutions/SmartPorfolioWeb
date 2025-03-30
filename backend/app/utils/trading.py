import os
import logging
from typing import Dict, Any
from fastapi import HTTPException
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def get_trading_credentials() -> Dict[str, Any]:
    """
    Get trading credentials based on the configured trading mode.
    Returns a dictionary with api_key, secret_key, is_paper, and url_override.
    """
    # Force load from .env file
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(dotenv_path=env_path, override=True)
    
    trading_mode = os.getenv("TRADING_MODE", "paper").lower()
    logger.info(f"Getting trading credentials for mode: {trading_mode}")
    
    # Get credentials based on trading mode
    if trading_mode == "paper":
        api_key = os.getenv("ALPACA_PAPER_API_KEY", "").strip()
        secret_key = os.getenv("ALPACA_PAPER_SECRET_KEY", "").strip()
        url_override = "https://paper-api.alpaca.markets"
    else:
        api_key = os.getenv("ALPACA_API_KEY", "").strip()
        secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()
        url_override = "https://api.alpaca.markets"
    
    # Log the API key format for debugging (masking sensitive parts)
    if api_key:
        logger.debug(f"API key format: {api_key[:4]}...{api_key[-4:]}")
    
    # Validate credentials
    if not api_key or not secret_key:
        logger.error(f"Missing API credentials for {trading_mode} trading")
        raise HTTPException(
            status_code=500,
            detail=f"Missing API credentials for {trading_mode} trading"
        )
    
    return {
        "api_key": api_key,
        "secret_key": secret_key,
        "is_paper": trading_mode == "paper",
        "url_override": url_override
    }

def get_account_data(trading_client) -> Dict[str, Any]:
    """
    Get account data from Alpaca with retries.
    Returns a dictionary with account information.
    """
    try:
        account = trading_client.get_account()
        
        # Convert account data to dictionary and filter sensitive information
        account_data = {
            "id": account.id,
            "account_number": account.account_number,
            "status": account.status,
            "currency": account.currency,
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
            "regt_buying_power": float(account.regt_buying_power),
            "daytrading_buying_power": float(account.daytrading_buying_power),
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
            "transfers_blocked": account.transfers_blocked,
            "account_blocked": account.account_blocked,
            "created_at": account.created_at.isoformat() if account.created_at else None,
            "trade_suspended_by_user": account.trade_suspended_by_user,
            "multiplier": account.multiplier,
            "shorting_enabled": account.shorting_enabled,
            "equity": float(account.equity),
            "last_equity": float(account.last_equity),
            "long_market_value": float(account.long_market_value),
            "short_market_value": float(account.short_market_value),
            "initial_margin": float(account.initial_margin),
            "maintenance_margin": float(account.maintenance_margin),
            "last_maintenance_margin": float(account.last_maintenance_margin),
            "sma": float(account.sma),
            "daytrade_count": account.daytrade_count
        }
        
        return account_data
        
    except Exception as e:
        logger.error(f"Error getting account data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get account data: {str(e)}"
        ) 