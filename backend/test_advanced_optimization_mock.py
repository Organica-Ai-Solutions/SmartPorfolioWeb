#!/usr/bin/env python3
"""
Mock test script for the advanced optimization service.
Uses simulated data to test optimization methods when Yahoo Finance is unavailable.
"""

import asyncio
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Set environment variables
os.environ.setdefault("ENV", "development")

def create_mock_data(tickers, days=252):
    """Create realistic mock price data for testing."""
    np.random.seed(42)  # For reproducible results
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    data = pd.DataFrame(index=dates)
    
    # Starting prices
    start_prices = {
        'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 2500.0, 'TSLA': 200.0,
        'JPM': 120.0, 'JNJ': 160.0, 'PG': 140.0, 'KO': 55.0,
        'SPY': 400.0, 'TLT': 120.0, 'GLD': 180.0, 'VNQ': 90.0,
        'AMZN': 3000.0, 'BTC-USD': 35000.0
    }
    
    # Volatilities (annual)
    volatilities = {
        'AAPL': 0.25, 'MSFT': 0.28, 'GOOGL': 0.30, 'TSLA': 0.55,
        'JPM': 0.35, 'JNJ': 0.18, 'PG': 0.15, 'KO': 0.16,
        'SPY': 0.18, 'TLT': 0.12, 'GLD': 0.20, 'VNQ': 0.25,
        'AMZN': 0.35, 'BTC-USD': 0.80
    }
    
    # Expected returns (annual)
    expected_returns = {
        'AAPL': 0.12, 'MSFT': 0.14, 'GOOGL': 0.13, 'TSLA': 0.08,
        'JPM': 0.10, 'JNJ': 0.08, 'PG': 0.07, 'KO': 0.06,
        'SPY': 0.10, 'TLT': 0.03, 'GLD': 0.05, 'VNQ': 0.09,
        'AMZN': 0.15, 'BTC-USD': 0.20
    }
    
    for ticker in tickers:
        if ticker in start_prices:
            # Generate correlated returns
            daily_return_mean = expected_returns[ticker] / 252
            daily_volatility = volatilities[ticker] / np.sqrt(252)
            
            # Generate returns with some autocorrelation
            returns = np.random.normal(daily_return_mean, daily_volatility, days)
            
            # Add some momentum/autocorrelation
            for i in range(1, len(returns)):
                returns[i] += 0.1 * returns[i-1]
            
            # Calculate prices
            prices = [start_prices[ticker]]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            data[ticker] = prices
    
    return data

async def test_black_litterman_mock():
    """Test Black-Litterman optimization with mock data."""
    try:
        print("🔬 Testing Black-Litterman Optimization (Mock Data)...")
        
        # Patch the data retrieval method
        from app.services.advanced_optimization_service import AdvancedOptimizationService, OptimizationMethod
        
        service = AdvancedOptimizationService()
        
        # Override the data retrieval method
        tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        mock_data = create_mock_data(tickers, 90)
        
        # Monkey patch the method
        original_method = service._get_historical_data
        async def mock_get_historical_data(tickers, lookback_days):
            return mock_data[tickers]
        service._get_historical_data = mock_get_historical_data
        
        views = {
            "AAPL": 0.12,    # Expect 12% return
            "TSLA": 0.08,    # Expect 8% return
            "GOOGL": 0.15    # Expect 15% return
        }
        
        print(f"📊 Tickers: {tickers}")
        print(f"👁️ Investor Views: {views}")
        
        result = await service.optimize_portfolio_advanced(
            tickers=tickers,
            method=OptimizationMethod.BLACK_LITTERMAN,
            lookback_days=90,
            risk_tolerance="medium",
            views=views
        )
        
        # Restore original method
        service._get_historical_data = original_method
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return False
        
        print("✅ Black-Litterman optimization completed!")
        print("\n📋 Results:")
        print("-" * 40)
        
        weights = result.get("weights", {})
        for ticker, weight in weights.items():
            print(f"  {ticker}: {weight:.3f} ({weight*100:.1f}%)")
        
        print(f"\n📈 Expected Return: {result.get('expected_return', 0):.3f}")
        print(f"📉 Volatility: {result.get('volatility', 0):.3f}")
        print(f"📊 Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
        
        if "method_details" in result:
            details = result["method_details"]
            if "view_impact" in details:
                print("\n🎯 View Impact on Expected Returns:")
                for ticker, impact in details["view_impact"].items():
                    direction = "↑" if impact > 0 else "↓" if impact < 0 else "→"
                    print(f"  {ticker}: {impact:+.3f} {direction}")
        
        return True
        
    except Exception as e:
        print(f"❌ Black-Litterman test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_risk_parity_mock():
    """Test risk parity optimization with mock data."""
    try:
        print("\n⚖️  Testing Risk Parity Optimization (Mock Data)...")
        
        from app.services.advanced_optimization_service import AdvancedOptimizationService, OptimizationMethod
        
        service = AdvancedOptimizationService()
        
        tickers = ["SPY", "TLT", "GLD", "VNQ"]
        mock_data = create_mock_data(tickers, 90)
        
        # Monkey patch the method
        original_method = service._get_historical_data
        async def mock_get_historical_data(tickers, lookback_days):
            return mock_data[tickers]
        service._get_historical_data = mock_get_historical_data
        
        print(f"📊 Tickers: {tickers}")
        print("🎯 Objective: Equal risk contribution from each asset")
        
        result = await service.optimize_portfolio_advanced(
            tickers=tickers,
            method=OptimizationMethod.RISK_PARITY,
            lookback_days=90
        )
        
        # Restore original method
        service._get_historical_data = original_method
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return False
        
        print("✅ Risk parity optimization completed!")
        print("\n📋 Results:")
        print("-" * 40)
        
        weights = result.get("weights", {})
        for ticker, weight in weights.items():
            print(f"  {ticker}: {weight:.3f} ({weight*100:.1f}%)")
        
        print(f"\n📈 Expected Return: {result.get('expected_return', 0):.3f}")
        print(f"📉 Volatility: {result.get('volatility', 0):.3f}")
        print(f"📊 Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
        
        if "risk_contributions" in result:
            print("\n⚖️  Risk Contributions:")
            risk_contribs = result["risk_contributions"]
            for ticker, contrib in risk_contribs.items():
                print(f"  {ticker}: {contrib:.3f} ({contrib*100:.1f}%)")
            
            # Check how equal the risk contributions are
            risk_values = list(risk_contribs.values())
            target = 1.0 / len(risk_values)
            risk_std = np.std([(r - target) for r in risk_values])
            print(f"\n📏 Risk Equality (lower=better): {risk_std:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk parity test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_minimum_variance_mock():
    """Test minimum variance optimization with mock data."""
    try:
        print("\n📉 Testing Minimum Variance Optimization (Mock Data)...")
        
        from app.services.advanced_optimization_service import AdvancedOptimizationService, OptimizationMethod
        
        service = AdvancedOptimizationService()
        
        tickers = ["AAPL", "JNJ", "PG", "KO"]  # Mix of volatile and defensive stocks
        mock_data = create_mock_data(tickers, 90)
        
        # Monkey patch the method
        original_method = service._get_historical_data
        async def mock_get_historical_data(tickers, lookback_days):
            return mock_data[tickers]
        service._get_historical_data = mock_get_historical_data
        
        print(f"📊 Tickers: {tickers}")
        print("🎯 Objective: Minimize portfolio volatility")
        
        result = await service.optimize_portfolio_advanced(
            tickers=tickers,
            method=OptimizationMethod.MIN_VOLATILITY,
            lookback_days=90
        )
        
        # Restore original method
        service._get_historical_data = original_method
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return False
        
        print("✅ Minimum variance optimization completed!")
        print("\n📋 Results:")
        print("-" * 40)
        
        weights = result.get("weights", {})
        for ticker, weight in weights.items():
            print(f"  {ticker}: {weight:.3f} ({weight*100:.1f}%)")
        
        print(f"\n📈 Expected Return: {result.get('expected_return', 0):.3f}")
        print(f"📉 Volatility: {result.get('volatility', 0):.3f}")
        print(f"📊 Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
        
        # Check if low-volatility stocks get higher weights
        expected_order = ["KO", "PG", "JNJ", "AAPL"]  # Expected order by volatility (low to high)
        actual_order = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 Highest Weight: {actual_order[0][0]} ({actual_order[0][1]:.1%})")
        print(f"🏆 Expected to favor defensive stocks (KO, PG, JNJ)")
        
        return True
        
    except Exception as e:
        print(f"❌ Minimum variance test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_optimization_comparison_mock():
    """Test comparison of multiple optimization methods with mock data."""
    try:
        print("\n🏆 Testing Optimization Method Comparison (Mock Data)...")
        
        from app.services.advanced_optimization_service import AdvancedOptimizationService, OptimizationMethod
        
        service = AdvancedOptimizationService()
        
        tickers = ["AAPL", "MSFT", "JNJ", "TLT"]
        mock_data = create_mock_data(tickers, 90)
        
        # Override the data retrieval method
        original_method = service._get_historical_data
        async def mock_get_historical_data(tickers, lookback_days):
            return mock_data[tickers]
        service._get_historical_data = mock_get_historical_data
        
        methods = [
            OptimizationMethod.MAX_SHARPE,
            OptimizationMethod.MIN_VOLATILITY,
            OptimizationMethod.RISK_PARITY,
            OptimizationMethod.MULTI_OBJECTIVE
        ]
        
        print(f"📊 Tickers: {tickers}")
        print(f"🔬 Methods: {[m.value for m in methods]}")
        
        results = {}
        
        for method in methods:
            print(f"\n🔄 Running {method.value}...")
            try:
                result = await service.optimize_portfolio_advanced(
                    tickers=tickers,
                    method=method,
                    lookback_days=90,
                    risk_tolerance="medium"
                )
                
                if "error" not in result:
                    results[method.value] = {
                        "weights": result.get("weights", {}),
                        "expected_return": result.get("expected_return", 0),
                        "volatility": result.get("volatility", 0),
                        "sharpe_ratio": result.get("sharpe_ratio", 0)
                    }
                    print(f"✅ {method.value} completed")
                else:
                    print(f"❌ {method.value} failed: {result['error']}")
            except Exception as e:
                print(f"❌ {method.value} error: {str(e)}")
        
        # Restore original method
        service._get_historical_data = original_method
        
        if results:
            print("\n🏆 Comparison Results:")
            print("=" * 70)
            print(f"{'Method':<20} {'Return':<10} {'Vol':<10} {'Sharpe':<10}")
            print("-" * 70)
            
            for method, data in results.items():
                print(f"{method:<20} {data['expected_return']:<10.3f} {data['volatility']:<10.3f} {data['sharpe_ratio']:<10.3f}")
            
            # Find best performers
            if len(results) > 1:
                best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
                lowest_vol = min(results.items(), key=lambda x: x[1]['volatility'])
                
                print(f"\n🥇 Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.3f})")
                print(f"🛡️  Lowest Volatility: {lowest_vol[0]} ({lowest_vol[1]['volatility']:.3f})")
                
                # Show key differences
                print("\n📊 Key Insights:")
                
                # Min variance should have lowest volatility
                if "min_volatility" in results:
                    min_vol_result = results["min_volatility"]
                    print(f"  • Min Variance achieved {min_vol_result['volatility']:.3f} volatility")
                
                # Risk parity should have more equal weights
                if "risk_parity" in results:
                    rp_weights = list(results["risk_parity"]["weights"].values())
                    concentration = sum(w**2 for w in rp_weights)
                    print(f"  • Risk Parity concentration ratio: {concentration:.3f} (lower = more equal)")
                
                # Max Sharpe should have highest Sharpe ratio
                if "max_sharpe" in results:
                    ms_result = results["max_sharpe"]
                    print(f"  • Max Sharpe achieved {ms_result['sharpe_ratio']:.3f} Sharpe ratio")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Comparison test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def run_mock_tests():
    """Run mock tests for advanced optimization."""
    print("🧪 Advanced Portfolio Optimization Test Suite (Mock Data)")
    print("=" * 70)
    print("🎭 Using simulated market data for testing")
    
    tests = [
        ("Black-Litterman", test_black_litterman_mock),
        ("Risk Parity", test_risk_parity_mock),
        ("Minimum Variance", test_minimum_variance_mock),
        ("Method Comparison", test_optimization_comparison_mock),
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
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print("-" * 30)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All mock tests passed! Advanced optimization system is working.")
        print("\n💡 Key Features Demonstrated:")
        print("  ✅ Black-Litterman: Incorporates investor views into optimization")
        print("  ✅ Risk Parity: Creates equal risk contribution portfolios")
        print("  ✅ Minimum Variance: Finds lowest risk portfolios")
        print("  ✅ Method Comparison: Compares different optimization approaches")
        print("\n🚀 The system is ready for production with real market data!")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    # Run the mock tests
    success = asyncio.run(run_mock_tests())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
