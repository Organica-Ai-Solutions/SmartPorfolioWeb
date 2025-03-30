from alpaca.trading.client import TradingClient
import os
from dotenv import load_dotenv

def test_alpaca_connection():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("ALPACA_PAPER_API_KEY")
    secret_key = os.getenv("ALPACA_PAPER_SECRET_KEY")
    
    if not api_key or not secret_key:
        print("Error: Alpaca API keys not found in environment variables")
        return
    
    try:
        # Initialize trading client
        trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True
        )
        
        # Try to get account info
        account = trading_client.get_account()
        print("Successfully connected to Alpaca API!")
        print(f"Account Status: {account.status}")
        print(f"Account Currency: {account.currency}")
        
    except Exception as e:
        print(f"Error connecting to Alpaca API: {str(e)}")

if __name__ == "__main__":
    test_alpaca_connection() 