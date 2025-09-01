// ============================================================================
// RISK MANAGEMENT HOOKS - SmartPortfolio AI Frontend
// ============================================================================
// Custom React hooks for advanced risk management features

import { useCallback, useMemo } from 'react';
import { useAppContext } from '../contexts/AppContext';
import { apiClient } from '../services/api';
import {
  RiskManagementRequest,
  RiskManagementResult,
  VolatilityPositionSizingRequest,
  TailRiskHedgingRequest,
  DrawdownControlsRequest,
  PortfolioWeights,
  RiskRegime,
  DrawdownSeverity
} from '../types/api';

// ============================================================================
// COMPREHENSIVE RISK MANAGEMENT HOOK
// ============================================================================

export const useRiskManagement = () => {
  const { state, dispatch } = useAppContext();
  const { risk, portfolio, preferences } = state;

  const analyzeRisk = useCallback(async (
    portfolioWeights?: PortfolioWeights,
    portfolioValue?: number,
    options: {
      targetVol?: number;
      hedgeBudget?: number;
      peakValue?: number;
    } = {}
  ): Promise<RiskManagementResult | null> => {
    const weights = portfolioWeights || portfolio.weights;
    const value = portfolioValue || portfolio.value;
    
    if (Object.keys(weights).length === 0) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'risk', error: 'No portfolio weights available for risk analysis' }
      });
      return null;
    }

    dispatch({ type: 'SET_RISK_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: { module: 'risk', error: null } });

    try {
      const request: RiskManagementRequest = {
        portfolio_weights: weights,
        tickers: portfolio.tickers,
        portfolio_value: value,
        peak_value: options.peakValue,
        target_vol: options.targetVol || 0.15,
        hedge_budget: options.hedgeBudget || 0.05
      };

      const result = await apiClient.comprehensiveRiskManagement(request);
      dispatch({ type: 'SET_RISK_ANALYSIS', payload: result });

      // Determine risk regime based on metrics
      const riskRegime = determineRiskRegime(result.risk_metrics);
      dispatch({ type: 'SET_RISK_REGIME', payload: riskRegime });

      // Update hedging status
      const hasHedging = result.applied_strategies.some(strategy => 
        strategy.includes('hedge') || strategy.includes('protection')
      );
      dispatch({ type: 'SET_HEDGING_ACTIVE', payload: hasHedging });

      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `risk-analysis-complete-${Date.now()}`,
          type: 'info',
          title: 'Risk Analysis Complete',
          message: `Portfolio volatility: ${(result.risk_metrics.portfolio_volatility * 100).toFixed(2)}%. Applied ${result.applied_strategies.length} risk strategies.`,
          timestamp: new Date().toISOString(),
          duration: 6000
        }
      });

      return result;

    } catch (error: any) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'risk', error: error.message }
      });
      return null;
    } finally {
      dispatch({ type: 'SET_RISK_LOADING', payload: false });
    }
  }, [portfolio.weights, portfolio.tickers, portfolio.value, dispatch]);

  const determineRiskRegime = useCallback((metrics: any): RiskRegime => {
    const volatility = metrics.portfolio_volatility;
    const var95 = Math.abs(metrics.var_95);
    const maxDrawdown = Math.abs(metrics.max_drawdown);

    // Risk scoring based on multiple metrics
    let riskScore = 0;
    
    if (volatility > 0.25) riskScore += 3;
    else if (volatility > 0.20) riskScore += 2;
    else if (volatility > 0.15) riskScore += 1;

    if (var95 > 0.05) riskScore += 3;
    else if (var95 > 0.03) riskScore += 2;
    else if (var95 > 0.02) riskScore += 1;

    if (maxDrawdown > 0.20) riskScore += 3;
    else if (maxDrawdown > 0.15) riskScore += 2;
    else if (maxDrawdown > 0.10) riskScore += 1;

    if (riskScore >= 7) return 'extreme';
    if (riskScore >= 5) return 'high';
    if (riskScore >= 3) return 'moderate';
    return 'low';
  }, []);

  return {
    analyzeRisk,
    analysis: risk.analysis,
    regime: risk.regime,
    hedgingActive: risk.hedgingActive,
    isAnalyzing: risk.isAnalyzing
  };
};

// ============================================================================
// VOLATILITY POSITION SIZING HOOK
// ============================================================================

export const useVolatilityPositionSizing = () => {
  const { state, dispatch } = useAppContext();
  const { portfolio } = state;

  const optimizePositionSizing = useCallback(async (
    targetVol: number = 0.15,
    lookbackDays: number = 63,
    weights?: PortfolioWeights
  ) => {
    const portfolioWeights = weights || portfolio.weights;
    
    if (Object.keys(portfolioWeights).length === 0) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'risk', error: 'No portfolio weights for position sizing' }
      });
      return null;
    }

    try {
      const request: VolatilityPositionSizingRequest = {
        portfolio_weights: portfolioWeights,
        target_portfolio_vol: targetVol,
        lookback_days
      };

      const result = await apiClient.volatilityPositionSizing(request);

      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `position-sizing-complete-${Date.now()}`,
          type: 'success',
          title: 'Position Sizing Optimized',
          message: `Adjusted positions to target ${(targetVol * 100).toFixed(1)}% volatility`,
          timestamp: new Date().toISOString(),
          duration: 5000,
          actions: [{
            label: 'Apply Sizing',
            action: () => {
              dispatch({ type: 'SET_PORTFOLIO_WEIGHTS', payload: result.adjusted_weights });
            }
          }]
        }
      });

      return result;

    } catch (error: any) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'risk', error: error.message }
      });
      return null;
    }
  }, [portfolio.weights, dispatch]);

  const calculatePositionSizes = useCallback((
    assetVols: Record<string, number>,
    targetVol: number,
    correlationMatrix?: number[][]
  ) => {
    // Inverse volatility weighting
    const invVolWeights: Record<string, number> = {};
    const symbols = Object.keys(assetVols);
    
    const invVolSum = symbols.reduce((sum, symbol) => sum + (1 / assetVols[symbol]), 0);
    
    symbols.forEach(symbol => {
      invVolWeights[symbol] = (1 / assetVols[symbol]) / invVolSum;
    });

    // Adjust for target portfolio volatility
    if (correlationMatrix) {
      // More sophisticated calculation considering correlations
      // This would require matrix calculations
      // For now, we'll use the simple inverse volatility approach
    }

    return invVolWeights;
  }, []);

  return {
    optimizePositionSizing,
    calculatePositionSizes
  };
};

// ============================================================================
// TAIL RISK HEDGING HOOK
// ============================================================================

export const useTailRiskHedging = () => {
  const { state, dispatch } = useAppContext();
  const { risk, portfolio, market } = state;

  const implementTailHedging = useCallback(async (
    riskRegime?: RiskRegime,
    hedgeBudget: number = 0.05,
    strategies: string[] = ['vix_protection', 'safe_haven']
  ) => {
    const currentRegime = riskRegime || risk.regime;
    
    if (Object.keys(portfolio.weights).length === 0) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'risk', error: 'No portfolio weights for tail hedging' }
      });
      return null;
    }

    try {
      const request: TailRiskHedgingRequest = {
        portfolio_weights: portfolio.weights,
        risk_regime: currentRegime,
        hedge_budget: hedgeBudget,
        hedge_strategies: strategies
      };

      const result = await apiClient.tailRiskHedging(request);

      const hedgeMessage = getHedgeStrategyMessage(currentRegime, strategies);
      
      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `tail-hedge-complete-${Date.now()}`,
          type: 'info',
          title: 'Tail Risk Hedging Applied',
          message: hedgeMessage,
          timestamp: new Date().toISOString(),
          duration: 7000,
          actions: [{
            label: 'Apply Hedge',
            action: () => {
              dispatch({ type: 'SET_PORTFOLIO_WEIGHTS', payload: result.hedged_portfolio });
              dispatch({ type: 'SET_HEDGING_ACTIVE', payload: true });
            }
          }]
        }
      });

      return result;

    } catch (error: any) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'risk', error: error.message }
      });
      return null;
    }
  }, [risk.regime, portfolio.weights, dispatch]);

  const getHedgeStrategyMessage = useCallback((regime: RiskRegime, strategies: string[]) => {
    const regimeMessages: Record<RiskRegime, string> = {
      'low': 'Minimal hedging for low-risk environment',
      'moderate': 'Balanced hedging with moderate protection',
      'high': 'Enhanced hedging for elevated risk conditions',
      'extreme': 'Maximum hedging for crisis protection'
    };

    const strategyCount = strategies.length;
    return `${regimeMessages[regime]}. Applied ${strategyCount} hedge strategies.`;
  }, []);

  const getRecommendedHedgeStrategies = useCallback((
    regime: RiskRegime,
    marketCondition: string
  ): string[] => {
    const strategyMap: Record<RiskRegime, string[]> = {
      'low': ['safe_haven'],
      'moderate': ['vix_protection', 'safe_haven'],
      'high': ['vix_protection', 'safe_haven', 'currency_hedge'],
      'extreme': ['vix_protection', 'safe_haven', 'currency_hedge', 'tail_protection_etfs']
    };

    let strategies = strategyMap[regime] || [];

    // Adjust based on market condition
    if (marketCondition === 'crisis') {
      strategies = [...strategies, 'tail_protection_etfs'];
    }

    return [...new Set(strategies)]; // Remove duplicates
  }, []);

  return {
    implementTailHedging,
    getRecommendedHedgeStrategies,
    hedgingActive: risk.hedgingActive
  };
};

// ============================================================================
// DRAWDOWN CONTROLS HOOK
// ============================================================================

export const useDrawdownControls = () => {
  const { state, dispatch } = useAppContext();
  const { portfolio } = state;

  const applyDrawdownControls = useCallback(async (
    currentDrawdown: number,
    peakValue: number,
    currentValue?: number
  ) => {
    const portfolioValue = currentValue || portfolio.value;
    
    if (Object.keys(portfolio.weights).length === 0) {
      return null;
    }

    try {
      const request: DrawdownControlsRequest = {
        portfolio_weights: portfolio.weights,
        current_drawdown: currentDrawdown,
        peak_value: peakValue,
        current_value: portfolioValue
      };

      const result = await apiClient.drawdownControls(request);

      const severity = getDrawdownSeverity(currentDrawdown);
      const severityMessages: Record<DrawdownSeverity, string> = {
        'minor': 'Minor drawdown detected. Minimal adjustments applied.',
        'moderate': 'Moderate drawdown. Risk reduction measures activated.',
        'major': 'Major drawdown. Significant defensive positioning implemented.',
        'severe': 'Severe drawdown. Maximum capital preservation mode activated.'
      };

      dispatch({
        type: 'SET_DRAWDOWN_CONTROLS',
        payload: true
      });

      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `drawdown-controls-${Date.now()}`,
          type: currentDrawdown > 0.15 ? 'warning' : 'info',
          title: 'Drawdown Controls Activated',
          message: severityMessages[severity],
          timestamp: new Date().toISOString(),
          duration: 8000,
          actions: [{
            label: 'Apply Controls',
            action: () => {
              dispatch({ type: 'SET_PORTFOLIO_WEIGHTS', payload: result.controlled_weights });
            }
          }]
        }
      });

      return result;

    } catch (error: any) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'risk', error: error.message }
      });
      return null;
    }
  }, [portfolio.weights, portfolio.value, dispatch]);

  const getDrawdownSeverity = useCallback((drawdown: number): DrawdownSeverity => {
    const absDrawdown = Math.abs(drawdown);
    
    if (absDrawdown >= 0.20) return 'severe';
    if (absDrawdown >= 0.15) return 'major';
    if (absDrawdown >= 0.10) return 'moderate';
    return 'minor';
  }, []);

  const calculateDrawdown = useCallback((
    currentValue: number,
    peakValue: number
  ): { drawdown: number; severity: DrawdownSeverity } => {
    const drawdown = (currentValue - peakValue) / peakValue;
    const severity = getDrawdownSeverity(drawdown);
    
    return { drawdown, severity };
  }, [getDrawdownSeverity]);

  const getDrawdownRecommendations = useCallback((severity: DrawdownSeverity): string[] => {
    const recommendations: Record<DrawdownSeverity, string[]> = {
      'minor': [
        'Monitor position sizes',
        'Review stop-loss levels',
        'Maintain current allocation'
      ],
      'moderate': [
        'Reduce equity exposure by 10-20%',
        'Increase cash buffer',
        'Consider defensive sectors'
      ],
      'major': [
        'Reduce equity exposure by 20-30%',
        'Increase bond allocation',
        'Implement hedging strategies'
      ],
      'severe': [
        'Move to capital preservation mode',
        'Increase cash to 40%+',
        'Consider market timing strategies'
      ]
    };

    return recommendations[severity] || [];
  }, []);

  return {
    applyDrawdownControls,
    calculateDrawdown,
    getDrawdownSeverity,
    getDrawdownRecommendations
  };
};

// ============================================================================
// STRESS TESTING HOOK
// ============================================================================

export const useStressTesting = () => {
  const { state } = useAppContext();
  const { portfolio } = state;

  const runStressTest = useCallback(async (
    scenarios: Array<{
      name: string;
      shocks: Record<string, number>; // Symbol -> shock percentage
    }>
  ) => {
    const results: Array<{
      scenario: string;
      portfolioImpact: number;
      worstAsset: { symbol: string; impact: number };
      bestAsset: { symbol: string; impact: number };
    }> = [];

    scenarios.forEach(scenario => {
      let portfolioImpact = 0;
      let worstImpact = 0;
      let bestImpact = 0;
      let worstAsset = '';
      let bestAsset = '';

      Object.entries(portfolio.weights).forEach(([symbol, weight]) => {
        const shock = scenario.shocks[symbol] || 0;
        const assetImpact = weight * shock;
        portfolioImpact += assetImpact;

        if (shock < worstImpact) {
          worstImpact = shock;
          worstAsset = symbol;
        }
        if (shock > bestImpact) {
          bestImpact = shock;
          bestAsset = symbol;
        }
      });

      results.push({
        scenario: scenario.name,
        portfolioImpact,
        worstAsset: { symbol: worstAsset, impact: worstImpact },
        bestAsset: { symbol: bestAsset, impact: bestImpact }
      });
    });

    return results;
  }, [portfolio.weights]);

  const getStandardScenarios = useCallback(() => [
    {
      name: 'Market Crash (-20%)',
      shocks: {
        'SPY': -0.20,
        'QQQ': -0.25,
        'AAPL': -0.22,
        'MSFT': -0.18,
        'GOOGL': -0.23,
        'TLT': 0.05,
        'GLD': 0.02
      }
    },
    {
      name: 'Interest Rate Spike',
      shocks: {
        'TLT': -0.15,
        'REITs': -0.18,
        'Banks': 0.10,
        'Growth': -0.12,
        'Value': -0.05
      }
    },
    {
      name: 'Inflation Surge',
      shocks: {
        'GLD': 0.08,
        'TIPS': 0.05,
        'Energy': 0.15,
        'TLT': -0.10,
        'Growth': -0.15
      }
    },
    {
      name: 'Geopolitical Crisis',
      shocks: {
        'VIX': 0.50,
        'USD': 0.05,
        'Oil': 0.20,
        'Defense': 0.08,
        'Travel': -0.25
      }
    }
  ], []);

  return {
    runStressTest,
    getStandardScenarios
  };
};

// ============================================================================
// COMPREHENSIVE RISK METRICS HOOK
// ============================================================================

export const useRiskMetrics = () => {
  const { state } = useAppContext();
  const { portfolio, market } = state;

  const calculateRiskMetrics = useCallback((
    weights: PortfolioWeights,
    returns: Record<string, number[]>,
    correlations?: number[][]
  ) => {
    const symbols = Object.keys(weights);
    
    // Portfolio return
    const portfolioReturns = symbols.reduce((portRet, symbol, index) => {
      const assetReturns = returns[symbol] || [];
      return portRet.map((ret, i) => ret + weights[symbol] * (assetReturns[i] || 0));
    }, new Array(Math.max(...Object.values(returns).map(r => r.length))).fill(0));

    // Basic metrics
    const meanReturn = portfolioReturns.reduce((sum, ret) => sum + ret, 0) / portfolioReturns.length;
    const variance = portfolioReturns.reduce((sum, ret) => sum + Math.pow(ret - meanReturn, 2), 0) / portfolioReturns.length;
    const volatility = Math.sqrt(variance * 252); // Annualized

    // VaR calculations
    const sortedReturns = [...portfolioReturns].sort((a, b) => a - b);
    const var95 = sortedReturns[Math.floor(sortedReturns.length * 0.05)];
    const var99 = sortedReturns[Math.floor(sortedReturns.length * 0.01)];

    // CVaR (Expected Shortfall)
    const cvar95Index = Math.floor(sortedReturns.length * 0.05);
    const cvar95 = sortedReturns.slice(0, cvar95Index).reduce((sum, ret) => sum + ret, 0) / cvar95Index;

    // Maximum Drawdown
    let peak = portfolioReturns[0];
    let maxDrawdown = 0;
    let cumulativeReturn = 1;
    
    portfolioReturns.forEach(ret => {
      cumulativeReturn *= (1 + ret);
      if (cumulativeReturn > peak) {
        peak = cumulativeReturn;
      }
      const drawdown = (peak - cumulativeReturn) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    });

    // Sharpe Ratio (assuming 2% risk-free rate)
    const riskFreeRate = 0.02;
    const excessReturn = meanReturn * 252 - riskFreeRate;
    const sharpeRatio = excessReturn / volatility;

    // Sortino Ratio (downside deviation)
    const downside = portfolioReturns.filter(ret => ret < meanReturn);
    const downsideVariance = downside.reduce((sum, ret) => sum + Math.pow(ret - meanReturn, 2), 0) / downside.length;
    const downsideVolatility = Math.sqrt(downsideVariance * 252);
    const sortinoRatio = excessReturn / downsideVolatility;

    return {
      expectedReturn: meanReturn * 252,
      volatility,
      sharpeRatio,
      sortinoRatio,
      var95: Math.abs(var95),
      var99: Math.abs(var99),
      cvar95: Math.abs(cvar95),
      maxDrawdown,
      skewness: calculateSkewness(portfolioReturns),
      kurtosis: calculateKurtosis(portfolioReturns)
    };
  }, []);

  const calculateSkewness = useCallback((returns: number[]) => {
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
    const skewness = returns.reduce((sum, ret) => sum + Math.pow((ret - mean) / Math.sqrt(variance), 3), 0) / returns.length;
    return skewness;
  }, []);

  const calculateKurtosis = useCallback((returns: number[]) => {
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
    const kurtosis = returns.reduce((sum, ret) => sum + Math.pow((ret - mean) / Math.sqrt(variance), 4), 0) / returns.length;
    return kurtosis - 3; // Excess kurtosis
  }, []);

  const getRiskGrade = useCallback((metrics: any) => {
    let score = 0;
    
    // Volatility score (0-3)
    if (metrics.volatility < 0.10) score += 0;
    else if (metrics.volatility < 0.15) score += 1;
    else if (metrics.volatility < 0.20) score += 2;
    else score += 3;

    // VaR score (0-3)
    if (metrics.var95 < 0.02) score += 0;
    else if (metrics.var95 < 0.03) score += 1;
    else if (metrics.var95 < 0.05) score += 2;
    else score += 3;

    // Max Drawdown score (0-3)
    if (metrics.maxDrawdown < 0.10) score += 0;
    else if (metrics.maxDrawdown < 0.15) score += 1;
    else if (metrics.maxDrawdown < 0.20) score += 2;
    else score += 3;

    // Convert to letter grade
    if (score <= 2) return 'A';
    if (score <= 4) return 'B';
    if (score <= 6) return 'C';
    if (score <= 8) return 'D';
    return 'F';
  }, []);

  return {
    calculateRiskMetrics,
    getRiskGrade,
    calculateSkewness,
    calculateKurtosis
  };
};

export default useRiskManagement;
