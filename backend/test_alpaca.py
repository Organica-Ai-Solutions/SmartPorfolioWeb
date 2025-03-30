from alpaca.trading.client import TradingClient
import os

def test_alpaca_connection():
    # Use API keys directly
    api_key = "PKSLACYMUE8GBGV2SKS8"
    secret_key = "F53szaag0zYA5g2G5D92InJAbt1LqwJshOFvkFcJ"
    
    print(f"Using API key: {api_key[:4]}...{api_key[-4:]}")
    
    try:
        # Initialize trading client
        trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True,
            url_override="https://paper-api.alpaca.markets"
        )
        
        # Try to get account info
        account = trading_client.get_account()
        print("Successfully connected to Alpaca API!")
        print(f"Account ID: {account.id}")
        print(f"Account Status: {account.status}")
        print(f"Cash: ${float(account.cash)}")
        
    except Exception as e:
        print(f"Error connecting to Alpaca API: {str(e)}")

if __name__ == "__main__":
    test_alpaca_connection() 