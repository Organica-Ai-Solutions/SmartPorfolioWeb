#!/usr/bin/env python3
"""
Test script for the advanced optimization service.
Tests Black-Litterman, factor-based, minimum variance, and risk parity optimization.
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

async def test_black_litterman():
    """Test Black-Litterman optimization."""
    try:
        print("🔬 Testing Black-Litterman Optimization...")
        
        from app.services.advanced_optimization_service import AdvancedOptimizationService, OptimizationMethod
        
        service = AdvancedOptimizationService()
        
        # Test data
        tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
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
        return False

async def test_factor_based():
    """Test factor-based optimization."""
    try:
        print("\n🏭 Testing Factor-Based Optimization...")
        
        from app.services.advanced_optimization_service import AdvancedOptimizationService, OptimizationMethod, FactorType
        
        service = AdvancedOptimizationService()
        
        tickers = ["AAPL", "MSFT", "JPM", "JNJ", "PG"]
        
        # Define factor constraints
        factor_constraints = {
            FactorType.VALUE: (10.0, 25.0),      # P/E ratio between 10-25
            FactorType.QUALITY: (0.10, 1.0),     # ROE at least 10%
            FactorType.SIZE: (0.0, 0.3)          # Max 30% in small caps
        }
        
        print(f"📊 Tickers: {tickers}")
        print(f"🏭 Factor Constraints: {factor_constraints}")
        
        result = await service.optimize_portfolio_advanced(
            tickers=tickers,
            method=OptimizationMethod.FACTOR_BASED,
            lookback_days=90,
            factor_constraints=factor_constraints
        )
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return False
        
        print("✅ Factor-based optimization completed!")
        print("\n📋 Results:")
        print("-" * 40)
        
        weights = result.get("weights", {})
        for ticker, weight in weights.items():
            print(f"  {ticker}: {weight:.3f} ({weight*100:.1f}%)")
        
        print(f"\n📈 Expected Return: {result.get('expected_return', 0):.3f}")
        print(f"📉 Volatility: {result.get('volatility', 0):.3f}")
        print(f"📊 Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
        
        if "factor_exposures" in result:
            print("\n🏭 Portfolio Factor Exposures:")
            for factor, exposure in result["factor_exposures"].items():
                print(f"  {factor}: {exposure:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Factor-based test failed: {str(e)}")
        return False

async def test_risk_parity():
    """Test risk parity optimization."""
    try:
        print("\n⚖️  Testing Risk Parity Optimization...")
        
        from app.services.advanced_optimization_service import AdvancedOptimizationService, OptimizationMethod
        
        service = AdvancedOptimizationService()
        
        tickers = ["SPY", "TLT", "GLD", "VNQ"]  # Stocks, Bonds, Gold, REITs
        
        print(f"📊 Tickers: {tickers}")
        print("🎯 Objective: Equal risk contribution from each asset")
        
        result = await service.optimize_portfolio_advanced(
            tickers=tickers,
            method=OptimizationMethod.RISK_PARITY,
            lookback_days=90
        )
        
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
            risk_std = (sum((r - 0.25)**2 for r in risk_values) / len(risk_values))**0.5
            print(f"\n📏 Risk Equality (lower=better): {risk_std:.4f}")
        
        if "method_details" in result:
            details = result["method_details"]
            if "effective_number_of_bets" in details:
                print(f"🎲 Effective Number of Bets: {details['effective_number_of_bets']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk parity test failed: {str(e)}")
        return False

async def test_minimum_variance():
    """Test minimum variance optimization."""
    try:
        print("\n📉 Testing Minimum Variance Optimization...")
        
        from app.services.advanced_optimization_service import AdvancedOptimizationService, OptimizationMethod
        
        service = AdvancedOptimizationService()
        
        tickers = ["AAPL", "MSFT", "JNJ", "PG", "KO"]  # Mix of tech and defensive stocks
        
        print(f"📊 Tickers: {tickers}")
        print("🎯 Objective: Minimize portfolio volatility")
        
        result = await service.optimize_portfolio_advanced(
            tickers=tickers,
            method=OptimizationMethod.MIN_VOLATILITY,
            lookback_days=90
        )
        
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
        
        if "portfolio_variance" in result:
            print(f"📊 Portfolio Variance: {result['portfolio_variance']:.6f}")
        
        if "method_details" in result:
            details = result["method_details"]
            if "diversification_ratio" in details:
                print(f"🌟 Diversification Ratio: {details['diversification_ratio']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Minimum variance test failed: {str(e)}")
        return False

async def test_multi_objective():
    """Test multi-objective optimization."""
    try:
        print("\n🎯 Testing Multi-Objective Optimization...")
        
        from app.services.advanced_optimization_service import AdvancedOptimizationService, OptimizationMethod
        
        service = AdvancedOptimizationService()
        
        tickers = ["AAPL", "MSFT", "BTC-USD", "GLD", "TLT"]
        
        print(f"📊 Tickers: {tickers}")
        print("🎯 Objective: Balance return, risk, and diversification")
        
        result = await service.optimize_portfolio_advanced(
            tickers=tickers,
            method=OptimizationMethod.MULTI_OBJECTIVE,
            lookback_days=90,
            risk_tolerance="medium"
        )
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return False
        
        print("✅ Multi-objective optimization completed!")
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
            if "diversification_index" in details:
                print(f"🌟 Diversification Index: {details['diversification_index']:.3f}")
            if "concentration_ratio" in details:
                print(f"📊 Concentration Ratio: {details['concentration_ratio']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-objective test failed: {str(e)}")
        return False

async def test_optimization_comparison():
    """Test comparison of multiple optimization methods."""
    try:
        print("\n🏆 Testing Optimization Method Comparison...")
        
        from app.services.advanced_optimization_service import AdvancedOptimizationService, OptimizationMethod
        
        service = AdvancedOptimizationService()
        
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
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
            result = await service.optimize_portfolio_advanced(
                tickers=tickers,
                method=method,
                lookback_days=60,
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
        
        if results:
            print("\n🏆 Comparison Results:")
            print("=" * 60)
            print(f"{'Method':<20} {'Return':<10} {'Vol':<10} {'Sharpe':<10}")
            print("-" * 60)
            
            for method, data in results.items():
                print(f"{method:<20} {data['expected_return']:<10.3f} {data['volatility']:<10.3f} {data['sharpe_ratio']:<10.3f}")
            
            # Find best performers
            best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
            lowest_vol = min(results.items(), key=lambda x: x[1]['volatility'])
            
            print(f"\n🥇 Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.3f})")
            print(f"🛡️  Lowest Volatility: {lowest_vol[0]} ({lowest_vol[1]['volatility']:.3f})")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Comparison test failed: {str(e)}")
        return False

async def run_all_tests():
    """Run all advanced optimization tests."""
    print("🧪 Advanced Portfolio Optimization Test Suite")
    print("=" * 60)
    
    tests = [
        ("Black-Litterman", test_black_litterman),
        ("Factor-Based", test_factor_based),
        ("Risk Parity", test_risk_parity),
        ("Minimum Variance", test_minimum_variance),
        ("Multi-Objective", test_multi_objective),
        ("Method Comparison", test_optimization_comparison),
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
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("-" * 30)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Advanced optimization system is ready.")
        print("\n💡 Available Methods:")
        print("  • Black-Litterman: Incorporate investor views")
        print("  • Factor-Based: Control factor exposures")
        print("  • Risk Parity: Equal risk contribution")
        print("  • Minimum Variance: Lowest volatility")
        print("  • Multi-Objective: Balance multiple goals")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(run_all_tests())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
