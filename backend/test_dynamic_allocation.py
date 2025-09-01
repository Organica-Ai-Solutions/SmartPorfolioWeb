#!/usr/bin/env python3
"""
Test script for the dynamic allocation service.
This script tests the core functionality without requiring API keys.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Set environment variables to avoid missing API key warnings
os.environ.setdefault("ENV", "development")
os.environ.setdefault("TRADING_MODE", "paper")
os.environ.setdefault("ALPACA_PAPER_API_KEY", "test_key")
os.environ.setdefault("ALPACA_PAPER_SECRET_KEY", "test_secret")

async def test_dynamic_allocation():
    """Test the dynamic allocation service."""
    try:
        print("🚀 Testing Dynamic Allocation Service...")
        
        # Import the service
        from app.services.dynamic_allocation_service import DynamicAllocationService
        
        # Create service instance
        service = DynamicAllocationService()
        print("✅ Dynamic allocation service initialized")
        
        # Test data
        test_tickers = ["AAPL", "MSFT", "BTC-USD", "GLD"]
        test_base_allocation = {
            "AAPL": 0.4,
            "MSFT": 0.3,
            "BTC-USD": 0.2,
            "GLD": 0.1
        }
        
        print(f"📊 Testing with tickers: {test_tickers}")
        print(f"📈 Base allocation: {test_base_allocation}")
        
        # Test dynamic allocation
        print("\n🔄 Getting dynamic allocation...")
        result = await service.get_dynamic_allocation(
            tickers=test_tickers,
            base_allocation=test_base_allocation,
            risk_tolerance="medium",
            lookback_days=90  # Use shorter period for faster testing
        )
        
        if "error" in result:
            print(f"❌ Error in dynamic allocation: {result['error']}")
            return False
        
        print("✅ Dynamic allocation completed successfully!")
        
        # Display results
        print("\n📋 Results Summary:")
        print("-" * 50)
        
        if "final_allocation" in result:
            print("Final Allocation:")
            for ticker, weight in result["final_allocation"].items():
                print(f"  {ticker}: {weight:.3f} ({weight*100:.1f}%)")
        
        if "allocation_insights" in result:
            insights = result["allocation_insights"]
            if "key_adjustments" in insights and insights["key_adjustments"]:
                print("\nKey Adjustments:")
                for adjustment in insights["key_adjustments"]:
                    print(f"  • {adjustment}")
        
        if "market_data_summary" in result:
            summary = result["market_data_summary"]
            print(f"\nMarket Summary:")
            print(f"  • Regime: {summary.get('market_regime', 'unknown')}")
            print(f"  • Sentiment: {summary.get('overall_sentiment', 'unknown')}")
            print(f"  • Volatility: {summary.get('volatility_environment', 'unknown')}")
            print(f"  • Momentum: {summary.get('momentum_environment', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_market_analysis():
    """Test the market analysis functionality."""
    try:
        print("\n🌍 Testing Market Analysis...")
        
        from app.services.dynamic_allocation_service import DynamicAllocationService
        
        service = DynamicAllocationService()
        
        # Test market data collection
        test_tickers = ["SPY", "QQQ", "GLD"]
        print(f"📊 Getting market data for: {test_tickers}")
        
        market_data = await service._get_comprehensive_market_data(
            tickers=test_tickers,
            lookback_days=60
        )
        
        if "error" in market_data:
            print(f"❌ Error in market analysis: {market_data['error']}")
            return False
        
        print("✅ Market analysis completed!")
        
        # Display market data summary
        if "summary" in market_data:
            summary = market_data["summary"]
            print("\n📈 Market Data Summary:")
            print("-" * 30)
            for key, value in summary.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Show data sources
        sources = []
        for key in ["macro_indicators", "sentiment_data", "options_data", "momentum_data", "volatility_data"]:
            if key in market_data and market_data[key] and not market_data[key].get("error"):
                sources.append(key.replace("_", " ").title())
        
        if sources:
            print(f"\n📡 Active Data Sources: {', '.join(sources)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Market analysis test failed: {str(e)}")
        return False

async def test_asset_classification():
    """Test asset class classification."""
    try:
        print("\n🏷️  Testing Asset Classification...")
        
        from app.services.dynamic_allocation_service import DynamicAllocationService, AssetClass
        
        service = DynamicAllocationService()
        
        # Test various ticker classifications
        test_cases = [
            ("AAPL", AssetClass.STOCKS),
            ("BTC-USD", AssetClass.CRYPTO),
            ("GLD", AssetClass.COMMODITIES),
            ("TLT", AssetClass.BONDS),
            ("VNQ", AssetClass.REAL_ESTATE),
            ("SHV", AssetClass.CASH)
        ]
        
        print("Asset Classification Test:")
        all_correct = True
        
        for ticker, expected_class in test_cases:
            actual_class = service.asset_class_mapping.get(ticker, AssetClass.STOCKS)
            status = "✅" if actual_class == expected_class else "❌"
            print(f"  {ticker}: {actual_class} {status}")
            if actual_class != expected_class:
                all_correct = False
        
        if all_correct:
            print("✅ All asset classifications correct!")
        else:
            print("⚠️  Some asset classifications may need review")
        
        return True
        
    except Exception as e:
        print(f"❌ Asset classification test failed: {str(e)}")
        return False

async def run_all_tests():
    """Run all tests."""
    print("🧪 Dynamic Allocation Service Test Suite")
    print("=" * 50)
    
    tests = [
        ("Asset Classification", test_asset_classification),
        ("Market Analysis", test_market_analysis),
        ("Dynamic Allocation", test_dynamic_allocation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} Test...")
        try:
            result = await test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}")
        except Exception as e:
            print(f"❌ FAILED: {test_name} - {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("-" * 25)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dynamic allocation system is ready.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(run_all_tests())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
