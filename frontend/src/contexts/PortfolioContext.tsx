// ============================================================================
// PORTFOLIO CONTEXT - SmartPortfolio AI Frontend
// ============================================================================
// Specialized context for portfolio management operations

import React, { createContext, useContext, ReactNode, useCallback } from 'react';
import { useAppContext } from './AppContext';
import { apiClient } from '../services/api';
import {
  Portfolio,
  PortfolioWeights,
  PortfolioMetrics,
  OptimizationRequest,
  OptimizationResult,
  OptimizationMethod,
  AIPortfolioManagementRequest,
  AIPortfolioManagementResult
} from '../types/api';

// ============================================================================
// CONTEXT TYPE
// ============================================================================

interface PortfolioContextType {
  // State getters
  portfolio: Portfolio | null;
  weights: PortfolioWeights;
  metrics: PortfolioMetrics | null;
  tickers: string[];
  value: number;
  isLoading: boolean;
  
  // Portfolio operations
  updateWeights: (weights: PortfolioWeights) => void;
  addTicker: (ticker: string, weight?: number) => void;
  removeTicker: (ticker: string) => void;
  setPortfolioValue: (value: number) => void;
  
  // Analysis operations
  analyzePortfolio: (enhanced?: boolean) => Promise<void>;
  rebalancePortfolio: (targetWeights: PortfolioWeights) => Promise<void>;
  
  // Optimization operations
  optimizePortfolio: (request: OptimizationRequest) => Promise<OptimizationResult>;
  blackLittermanOptimization: (views: Record<string, number>) => Promise<OptimizationResult>;
  compareOptimizationMethods: (methods: OptimizationMethod[]) => Promise<any>;
  
  // AI-powered operations
  aiPortfolioManagement: (request: AIPortfolioManagementRequest) => Promise<AIPortfolioManagementResult>;
  
  // Utility operations
  resetPortfolio: () => void;
  loadSamplePortfolio: () => void;
  exportPortfolio: () => string;
  importPortfolio: (data: string) => void;
}

// ============================================================================
// CONTEXT CREATION
// ============================================================================

const PortfolioContext = createContext<PortfolioContextType | undefined>(undefined);

// ============================================================================
// PROVIDER COMPONENT
// ============================================================================

interface PortfolioProviderProps {
  children: ReactNode;
}

export const PortfolioProvider: React.FC<PortfolioProviderProps> = ({ children }) => {
  const { state, dispatch } = useAppContext();
  const { portfolio, preferences } = state;

  // ============================================================================
  // STATE GETTERS
  // ============================================================================

  const getCurrentPortfolio = useCallback(() => portfolio.current, [portfolio.current]);
  const getCurrentWeights = useCallback(() => portfolio.weights, [portfolio.weights]);
  const getCurrentMetrics = useCallback(() => portfolio.metrics, [portfolio.metrics]);
  const getCurrentTickers = useCallback(() => portfolio.tickers, [portfolio.tickers]);
  const getCurrentValue = useCallback(() => portfolio.value, [portfolio.value]);
  const getIsLoading = useCallback(() => portfolio.isLoading, [portfolio.isLoading]);

  // ============================================================================
  // PORTFOLIO OPERATIONS
  // ============================================================================

  const updateWeights = useCallback((weights: PortfolioWeights) => {
    dispatch({ type: 'SET_PORTFOLIO_WEIGHTS', payload: weights });
    
    // Update tickers list
    const tickers = Object.keys(weights).filter(ticker => weights[ticker] > 0);
    dispatch({ type: 'SET_PORTFOLIO_TICKERS', payload: tickers });
    
    // Add notification
    dispatch({
      type: 'ADD_NOTIFICATION',
      payload: {
        id: `weights-updated-${Date.now()}`,
        type: 'success',
        title: 'Portfolio Updated',
        message: `Portfolio weights updated for ${tickers.length} assets`,
        timestamp: new Date().toISOString(),
        duration: 3000
      }
    });
  }, [dispatch]);

  const addTicker = useCallback((ticker: string, weight: number = 0.1) => {
    const currentWeights = portfolio.weights;
    const existingWeight = currentWeights[ticker] || 0;
    
    if (existingWeight > 0) {
      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `ticker-exists-${Date.now()}`,
          type: 'warning',
          title: 'Ticker Already Exists',
          message: `${ticker} is already in the portfolio with ${(existingWeight * 100).toFixed(1)}% allocation`,
          timestamp: new Date().toISOString(),
          duration: 3000
        }
      });
      return;
    }

    // Calculate scaling factor to accommodate new weight
    const totalWeight = Object.values(currentWeights).reduce((sum, w) => sum + w, 0);
    const scalingFactor = (1 - weight) / totalWeight;
    
    const newWeights: PortfolioWeights = {
      ...Object.fromEntries(
        Object.entries(currentWeights).map(([symbol, w]) => [symbol, w * scalingFactor])
      ),
      [ticker]: weight
    };

    updateWeights(newWeights);
  }, [portfolio.weights, updateWeights, dispatch]);

  const removeTicker = useCallback((ticker: string) => {
    const currentWeights = portfolio.weights;
    const { [ticker]: removedWeight, ...remainingWeights } = currentWeights;
    
    if (!removedWeight) return;

    // Redistribute the weight proportionally
    const totalRemainingWeight = Object.values(remainingWeights).reduce((sum, w) => sum + w, 0);
    
    const newWeights: PortfolioWeights = totalRemainingWeight > 0
      ? Object.fromEntries(
          Object.entries(remainingWeights).map(([symbol, w]) => [
            symbol, 
            w + (w / totalRemainingWeight) * removedWeight
          ])
        )
      : {};

    updateWeights(newWeights);
    
    dispatch({
      type: 'ADD_NOTIFICATION',
      payload: {
        id: `ticker-removed-${Date.now()}`,
        type: 'info',
        title: 'Ticker Removed',
        message: `${ticker} removed from portfolio. Weights redistributed.`,
        timestamp: new Date().toISOString(),
        duration: 3000
      }
    });
  }, [portfolio.weights, updateWeights, dispatch]);

  const setPortfolioValue = useCallback((value: number) => {
    dispatch({ type: 'SET_PORTFOLIO_VALUE', payload: value });
  }, [dispatch]);

  // ============================================================================
  // ANALYSIS OPERATIONS
  // ============================================================================

  const analyzePortfolio = useCallback(async (enhanced: boolean = false) => {
    if (portfolio.tickers.length === 0) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'portfolio', error: 'No tickers in portfolio' }
      });
      return;
    }

    dispatch({ type: 'SET_PORTFOLIO_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: { module: 'portfolio', error: null } });

    try {
      const endpoint = enhanced ? 'analyzePortfolioEnhanced' : 'analyzePortfolio';
      const result = await apiClient[endpoint]({
        tickers: portfolio.tickers,
        weights: portfolio.weights,
        benchmark: 'SPY',
        lookback_days: 252,
        ...(enhanced && {
          enable_ai_insights: preferences.enableAI,
          enable_regime_analysis: preferences.enableAI
        })
      });

      if (result.portfolio_metrics || result.basic_metrics) {
        dispatch({
          type: 'SET_PORTFOLIO_METRICS',
          payload: result.portfolio_metrics || result.basic_metrics
        });
      }

      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `analysis-complete-${Date.now()}`,
          type: 'success',
          title: 'Analysis Complete',
          message: enhanced ? 'Enhanced analysis with AI insights completed' : 'Portfolio analysis completed',
          timestamp: new Date().toISOString(),
          duration: 5000
        }
      });

    } catch (error: any) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'portfolio', error: error.message }
      });
      
      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `analysis-error-${Date.now()}`,
          type: 'error',
          title: 'Analysis Failed',
          message: error.message || 'Failed to analyze portfolio',
          timestamp: new Date().toISOString(),
          duration: 5000
        }
      });
    } finally {
      dispatch({ type: 'SET_PORTFOLIO_LOADING', payload: false });
    }
  }, [portfolio.tickers, portfolio.weights, preferences.enableAI, dispatch]);

  const rebalancePortfolio = useCallback(async (targetWeights: PortfolioWeights) => {
    dispatch({ type: 'SET_PORTFOLIO_LOADING', payload: true });

    try {
      const result = await apiClient.rebalancePortfolio({
        current_weights: portfolio.weights,
        target_weights: targetWeights,
        portfolio_value: portfolio.value,
        execution_method: 'smart_routing'
      });

      updateWeights(targetWeights);

      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `rebalance-complete-${Date.now()}`,
          type: 'success',
          title: 'Rebalancing Complete',
          message: `Portfolio rebalanced with ${result.trades?.length || 0} trades. Estimated cost: $${result.estimated_costs?.toFixed(2) || '0.00'}`,
          timestamp: new Date().toISOString(),
          duration: 5000
        }
      });

    } catch (error: any) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'portfolio', error: error.message }
      });
    } finally {
      dispatch({ type: 'SET_PORTFOLIO_LOADING', payload: false });
    }
  }, [portfolio.weights, portfolio.value, updateWeights, dispatch]);

  // ============================================================================
  // OPTIMIZATION OPERATIONS
  // ============================================================================

  const optimizePortfolio = useCallback(async (request: OptimizationRequest): Promise<OptimizationResult> => {
    dispatch({ type: 'SET_OPTIMIZATION_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: { module: 'optimization', error: null } });

    try {
      const result = await apiClient.optimizePortfolio({
        ...request,
        tickers: request.tickers || portfolio.tickers,
        risk_tolerance: request.risk_tolerance || preferences.riskTolerance
      });

      dispatch({ type: 'SET_OPTIMIZATION_RESULT', payload: result });
      dispatch({ type: 'ADD_OPTIMIZATION_HISTORY', payload: result });

      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `optimization-complete-${Date.now()}`,
          type: 'success',
          title: 'Optimization Complete',
          message: `${request.method} optimization completed. Expected return: ${(result.metrics.expected_return * 100).toFixed(2)}%`,
          timestamp: new Date().toISOString(),
          duration: 5000,
          actions: [{
            label: 'Apply Weights',
            action: () => updateWeights(result.weights)
          }]
        }
      });

      return result;

    } catch (error: any) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'optimization', error: error.message }
      });
      throw error;
    } finally {
      dispatch({ type: 'SET_OPTIMIZATION_LOADING', payload: false });
    }
  }, [portfolio.tickers, preferences.riskTolerance, updateWeights, dispatch]);

  const blackLittermanOptimization = useCallback(async (views: Record<string, number>): Promise<OptimizationResult> => {
    return optimizePortfolio({
      tickers: portfolio.tickers,
      method: 'black_litterman',
      views,
      view_confidences: Object.fromEntries(
        Object.keys(views).map(ticker => [ticker, 0.5]) // Default 50% confidence
      )
    } as any);
  }, [portfolio.tickers, optimizePortfolio]);

  const compareOptimizationMethods = useCallback(async (methods: OptimizationMethod[]) => {
    dispatch({ type: 'SET_OPTIMIZATION_LOADING', payload: true });

    try {
      const result = await apiClient.compareOptimizationMethods({
        tickers: portfolio.tickers,
        methods,
        lookback_days: 252
      });

      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `comparison-complete-${Date.now()}`,
          type: 'success',
          title: 'Method Comparison Complete',
          message: `Compared ${methods.length} optimization methods. Best Sharpe: ${result.recommendations.best_sharpe}`,
          timestamp: new Date().toISOString(),
          duration: 8000
        }
      });

      return result;

    } catch (error: any) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'optimization', error: error.message }
      });
      throw error;
    } finally {
      dispatch({ type: 'SET_OPTIMIZATION_LOADING', payload: false });
    }
  }, [portfolio.tickers, dispatch]);

  // ============================================================================
  // AI-POWERED OPERATIONS
  // ============================================================================

  const aiPortfolioManagement = useCallback(async (request: AIPortfolioManagementRequest): Promise<AIPortfolioManagementResult> => {
    dispatch({ type: 'SET_PORTFOLIO_LOADING', payload: true });
    dispatch({ type: 'SET_ML_PREDICTING', payload: true });

    try {
      const result = await apiClient.aiPortfolioManagement({
        ...request,
        tickers: request.tickers || portfolio.tickers,
        base_allocation: request.base_allocation || portfolio.weights,
        portfolio_value: request.portfolio_value || portfolio.value,
        enable_ml_predictions: request.enable_ml_predictions ?? preferences.enableAI,
        enable_risk_management: request.enable_risk_management ?? preferences.enableRiskManagement,
        enable_tax_optimization: request.enable_tax_optimization ?? preferences.enableTaxOptimization,
        risk_tolerance: preferences.riskTolerance
      });

      // Update portfolio with AI recommendations
      updateWeights(result.final_allocation);

      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `ai-management-complete-${Date.now()}`,
          type: 'success',
          title: 'AI Portfolio Management Complete',
          message: `AI analysis complete. ML confidence: ${(result.ai_insights.ml_model_confidence * 100).toFixed(1)}%. Expected return: ${(result.performance_projection.expected_annual_return * 100).toFixed(2)}%`,
          timestamp: new Date().toISOString(),
          duration: 10000,
          actions: result.recommendations.length > 0 ? [{
            label: 'View Recommendations',
            action: () => {
              // This would open a recommendations modal
              dispatch({ type: 'OPEN_MODAL', payload: 'optimization' });
            }
          }] : undefined
        }
      });

      return result;

    } catch (error: any) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'portfolio', error: error.message }
      });
      throw error;
    } finally {
      dispatch({ type: 'SET_PORTFOLIO_LOADING', payload: false });
      dispatch({ type: 'SET_ML_PREDICTING', payload: false });
    }
  }, [
    portfolio.tickers, 
    portfolio.weights, 
    portfolio.value, 
    preferences.enableAI, 
    preferences.enableRiskManagement, 
    preferences.enableTaxOptimization,
    preferences.riskTolerance,
    updateWeights, 
    dispatch
  ]);

  // ============================================================================
  // UTILITY OPERATIONS
  // ============================================================================

  const resetPortfolio = useCallback(() => {
    dispatch({ type: 'SET_PORTFOLIO_WEIGHTS', payload: {} });
    dispatch({ type: 'SET_PORTFOLIO_TICKERS', payload: [] });
    dispatch({ type: 'SET_PORTFOLIO_METRICS', payload: null });
    dispatch({ type: 'SET_PORTFOLIO_VALUE', payload: 0 });
    
    dispatch({
      type: 'ADD_NOTIFICATION',
      payload: {
        id: `portfolio-reset-${Date.now()}`,
        type: 'info',
        title: 'Portfolio Reset',
        message: 'Portfolio has been reset to empty state',
        timestamp: new Date().toISOString(),
        duration: 3000
      }
    });
  }, [dispatch]);

  const loadSamplePortfolio = useCallback(() => {
    const sampleWeights: PortfolioWeights = {
      'AAPL': 0.25,
      'MSFT': 0.20,
      'GOOGL': 0.15,
      'SPY': 0.30,
      'TLT': 0.10
    };
    
    updateWeights(sampleWeights);
    setPortfolioValue(100000);
    
    dispatch({
      type: 'ADD_NOTIFICATION',
      payload: {
        id: `sample-loaded-${Date.now()}`,
        type: 'success',
        title: 'Sample Portfolio Loaded',
        message: 'Loaded a diversified sample portfolio worth $100,000',
        timestamp: new Date().toISOString(),
        duration: 4000
      }
    });
  }, [updateWeights, setPortfolioValue, dispatch]);

  const exportPortfolio = useCallback((): string => {
    const portfolioData = {
      weights: portfolio.weights,
      value: portfolio.value,
      tickers: portfolio.tickers,
      metrics: portfolio.metrics,
      exportDate: new Date().toISOString()
    };
    
    return JSON.stringify(portfolioData, null, 2);
  }, [portfolio]);

  const importPortfolio = useCallback((data: string) => {
    try {
      const portfolioData = JSON.parse(data);
      
      if (portfolioData.weights) {
        updateWeights(portfolioData.weights);
      }
      if (portfolioData.value) {
        setPortfolioValue(portfolioData.value);
      }
      if (portfolioData.metrics) {
        dispatch({ type: 'SET_PORTFOLIO_METRICS', payload: portfolioData.metrics });
      }
      
      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `portfolio-imported-${Date.now()}`,
          type: 'success',
          title: 'Portfolio Imported',
          message: 'Portfolio data successfully imported',
          timestamp: new Date().toISOString(),
          duration: 4000
        }
      });
      
    } catch (error) {
      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `import-error-${Date.now()}`,
          type: 'error',
          title: 'Import Failed',
          message: 'Failed to import portfolio data. Please check the format.',
          timestamp: new Date().toISOString(),
          duration: 5000
        }
      });
    }
  }, [updateWeights, setPortfolioValue, dispatch]);

  // ============================================================================
  // CONTEXT VALUE
  // ============================================================================

  const value: PortfolioContextType = {
    // State getters
    portfolio: getCurrentPortfolio(),
    weights: getCurrentWeights(),
    metrics: getCurrentMetrics(),
    tickers: getCurrentTickers(),
    value: getCurrentValue(),
    isLoading: getIsLoading(),
    
    // Portfolio operations
    updateWeights,
    addTicker,
    removeTicker,
    setPortfolioValue,
    
    // Analysis operations
    analyzePortfolio,
    rebalancePortfolio,
    
    // Optimization operations
    optimizePortfolio,
    blackLittermanOptimization,
    compareOptimizationMethods,
    
    // AI-powered operations
    aiPortfolioManagement,
    
    // Utility operations
    resetPortfolio,
    loadSamplePortfolio,
    exportPortfolio,
    importPortfolio,
  };

  return (
    <PortfolioContext.Provider value={value}>
      {children}
    </PortfolioContext.Provider>
  );
};

// ============================================================================
// HOOK
// ============================================================================

export const usePortfolio = (): PortfolioContextType => {
  const context = useContext(PortfolioContext);
  if (context === undefined) {
    throw new Error('usePortfolio must be used within a PortfolioProvider');
  }
  return context;
};

export default PortfolioProvider;
