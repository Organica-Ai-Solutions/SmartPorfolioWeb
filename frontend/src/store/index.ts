// ============================================================================
// MAIN STORE - SmartPortfolio AI Frontend
// ============================================================================
// Central state management combining all contexts and hooks

import React from 'react';
import { AppProvider } from '../contexts/AppContext';
import { PortfolioProvider } from '../contexts/PortfolioContext';

// ============================================================================
// COMBINED PROVIDER COMPONENT
// ============================================================================

interface ProvidersProps {
  children: React.ReactNode;
}

export const Providers: React.FC<ProvidersProps> = ({ children }) => {
  return (
    <AppProvider>
      <PortfolioProvider>
        {children}
      </PortfolioProvider>
    </AppProvider>
  );
};

// ============================================================================
// STATE MANAGEMENT EXPORTS
// ============================================================================

// Context exports
export { 
  AppProvider, 
  useAppContext, 
  useAppState, 
  useAppDispatch,
  useSystemState,
  usePortfolioState,
  useMLState,
  useOptimizationState,
  useRiskState,
  useMarketState,
  useUIState,
  usePreferences,
  useErrors
} from '../contexts/AppContext';

export { 
  PortfolioProvider, 
  usePortfolio 
} from '../contexts/PortfolioContext';

// Hook exports
export {
  useAI,
  useMLTraining,
  usePricePredictions,
  useMarketRegime,
  useReinforcementLearning,
  useEnsemblePredictions,
  useAIStatus
} from '../hooks/useAI';

export {
  useRiskManagement,
  useVolatilityPositionSizing,
  useTailRiskHedging,
  useDrawdownControls,
  useStressTesting,
  useRiskMetrics
} from '../hooks/useRiskManagement';

// API exports
export { apiClient, APIError } from '../services/api';
export { API_ENDPOINTS, ENDPOINT_CATEGORIES, ENDPOINT_STATS } from '../services/endpoints';

// Type exports
export type {
  AppState,
  AppAction,
  Notification
} from '../contexts/AppContext';

export type * from '../types/api';

// ============================================================================
// CENTRALIZED STATE SELECTORS
// ============================================================================

export const createSelectors = () => {
  return {
    // Portfolio selectors
    getPortfolioValue: (state: any) => state.portfolio.value,
    getPortfolioWeights: (state: any) => state.portfolio.weights,
    getPortfolioTickers: (state: any) => state.portfolio.tickers,
    getPortfolioMetrics: (state: any) => state.portfolio.metrics,
    isPortfolioLoading: (state: any) => state.portfolio.isLoading,

    // ML selectors
    getMLPredictions: (state: any) => state.ml.predictions,
    getMLModels: (state: any) => state.ml.models,
    getMLConfidence: (state: any) => state.ml.confidence,
    getRegimeAnalysis: (state: any) => state.ml.regimeAnalysis,
    isMLTraining: (state: any) => state.ml.isTraining,
    isMLPredicting: (state: any) => state.ml.isPredicting,

    // Risk selectors
    getRiskAnalysis: (state: any) => state.risk.analysis,
    getRiskRegime: (state: any) => state.risk.regime,
    isHedgingActive: (state: any) => state.risk.hedgingActive,
    areDrawdownControlsActive: (state: any) => state.risk.drawdownControls,
    isRiskAnalyzing: (state: any) => state.risk.isAnalyzing,

    // Optimization selectors
    getCurrentOptimization: (state: any) => state.optimization.current,
    getOptimizationMethod: (state: any) => state.optimization.method,
    getOptimizationHistory: (state: any) => state.optimization.history,
    isOptimizing: (state: any) => state.optimization.isOptimizing,

    // Market selectors
    getMarketData: (state: any) => state.market.data,
    getMarketCondition: (state: any) => state.market.condition,
    getMarketSentiment: (state: any) => state.market.sentiment,
    getMarketVolatility: (state: any) => state.market.volatility,
    getTechnicalIndicators: (state: any) => state.market.indicators,
    isMarketUpdating: (state: any) => state.market.isUpdating,

    // System selectors
    getSystemHealth: (state: any) => state.system.health,
    getSystemCapabilities: (state: any) => state.system.capabilities,
    isSystemLoading: (state: any) => state.system.isLoading,

    // UI selectors
    getTheme: (state: any) => state.ui.theme,
    isSidebarOpen: (state: any) => state.ui.sidebar.isOpen,
    getActiveTab: (state: any) => state.ui.sidebar.activeTab,
    getModals: (state: any) => state.ui.modals,
    getNotifications: (state: any) => state.ui.notifications,
    isOnline: (state: any) => state.ui.isOnline,

    // Preferences selectors
    getRiskTolerance: (state: any) => state.preferences.riskTolerance,
    getInvestmentHorizon: (state: any) => state.preferences.investmentHorizon,
    isAIEnabled: (state: any) => state.preferences.enableAI,
    isRiskManagementEnabled: (state: any) => state.preferences.enableRiskManagement,
    isTaxOptimizationEnabled: (state: any) => state.preferences.enableTaxOptimization,
    isAutoRebalanceEnabled: (state: any) => state.preferences.autoRebalance,
    getRebalanceThreshold: (state: any) => state.preferences.rebalanceThreshold,

    // Error selectors
    getSystemError: (state: any) => state.errors.system,
    getPortfolioError: (state: any) => state.errors.portfolio,
    getMLError: (state: any) => state.errors.ml,
    getOptimizationError: (state: any) => state.errors.optimization,
    getRiskError: (state: any) => state.errors.risk,
    getMarketError: (state: any) => state.errors.market,
    hasAnyError: (state: any) => Object.values(state.errors).some(error => error !== null)
  };
};

// ============================================================================
// STATE MANAGEMENT UTILITIES
// ============================================================================

export const stateUtils = {
  // Local storage utilities
  saveToLocalStorage: (key: string, data: any) => {
    try {
      localStorage.setItem(`smartportfolio-${key}`, JSON.stringify(data));
    } catch (error) {
      console.warn(`Failed to save ${key} to localStorage:`, error);
    }
  },

  loadFromLocalStorage: (key: string, defaultValue: any = null) => {
    try {
      const item = localStorage.getItem(`smartportfolio-${key}`);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.warn(`Failed to load ${key} from localStorage:`, error);
      return defaultValue;
    }
  },

  clearLocalStorage: () => {
    const keys = Object.keys(localStorage).filter(key => key.startsWith('smartportfolio-'));
    keys.forEach(key => localStorage.removeItem(key));
  },

  // State validation utilities
  validatePortfolioWeights: (weights: Record<string, number>) => {
    const total = Object.values(weights).reduce((sum, weight) => sum + weight, 0);
    const tolerance = 0.01; // 1% tolerance
    
    return {
      isValid: Math.abs(total - 1) <= tolerance,
      total,
      deviation: total - 1,
      weights: Object.keys(weights).length
    };
  },

  // State transformation utilities
  normalizeWeights: (weights: Record<string, number>) => {
    const total = Object.values(weights).reduce((sum, weight) => sum + weight, 0);
    if (total === 0) return weights;
    
    return Object.fromEntries(
      Object.entries(weights).map(([symbol, weight]) => [symbol, weight / total])
    );
  },

  // Performance utilities
  calculatePortfolioReturn: (weights: Record<string, number>, returns: Record<string, number>) => {
    return Object.entries(weights).reduce((portfolioReturn, [symbol, weight]) => {
      return portfolioReturn + weight * (returns[symbol] || 0);
    }, 0);
  },

  calculatePortfolioVolatility: (
    weights: Record<string, number>, 
    volatilities: Record<string, number>,
    correlations?: Record<string, Record<string, number>>
  ) => {
    const symbols = Object.keys(weights);
    
    if (!correlations) {
      // Simple weighted volatility without correlations
      return Math.sqrt(
        symbols.reduce((variance, symbol) => {
          const weight = weights[symbol];
          const vol = volatilities[symbol] || 0;
          return variance + Math.pow(weight * vol, 2);
        }, 0)
      );
    }

    // Full covariance calculation
    let portfolioVariance = 0;
    
    symbols.forEach(symbol1 => {
      symbols.forEach(symbol2 => {
        const weight1 = weights[symbol1];
        const weight2 = weights[symbol2];
        const vol1 = volatilities[symbol1] || 0;
        const vol2 = volatilities[symbol2] || 0;
        const correlation = symbol1 === symbol2 ? 1 : (correlations[symbol1]?.[symbol2] || 0);
        
        portfolioVariance += weight1 * weight2 * vol1 * vol2 * correlation;
      });
    });
    
    return Math.sqrt(portfolioVariance);
  }
};

// ============================================================================
// ASYNC STATE MANAGEMENT
// ============================================================================

export const asyncStateUtils = {
  // Batch operations
  createBatchUpdater: (dispatch: any) => {
    const updates: any[] = [];
    let timeoutId: NodeJS.Timeout | null = null;

    return {
      add: (action: any) => {
        updates.push(action);
        
        if (timeoutId) {
          clearTimeout(timeoutId);
        }
        
        timeoutId = setTimeout(() => {
          updates.forEach(action => dispatch(action));
          updates.length = 0; // Clear array
          timeoutId = null;
        }, 10); // Batch within 10ms
      },
      flush: () => {
        if (timeoutId) {
          clearTimeout(timeoutId);
          timeoutId = null;
        }
        updates.forEach(action => dispatch(action));
        updates.length = 0;
      }
    };
  },

  // Async operation wrapper
  createAsyncWrapper: (dispatch: any) => {
    return async <T>(
      operation: () => Promise<T>,
      loadingAction: any,
      successAction?: (result: T) => any,
      errorAction?: (error: string) => any
    ): Promise<T | null> => {
      dispatch(loadingAction);
      
      try {
        const result = await operation();
        if (successAction) {
          dispatch(successAction(result));
        }
        return result;
      } catch (error: any) {
        if (errorAction) {
          dispatch(errorAction(error.message || 'Unknown error'));
        }
        return null;
      }
    };
  },

  // Retry mechanism
  createRetryWrapper: <T>(
    operation: () => Promise<T>,
    maxRetries: number = 3,
    delay: number = 1000
  ) => {
    return async (): Promise<T> => {
      let lastError: Error;
      
      for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
          return await operation();
        } catch (error) {
          lastError = error as Error;
          
          if (attempt < maxRetries) {
            await new Promise(resolve => setTimeout(resolve, delay * attempt));
          }
        }
      }
      
      throw lastError!;
    };
  }
};

// ============================================================================
// DEVELOPMENT UTILITIES
// ============================================================================

export const devUtils = {
  // State debugger
  logState: (state: any, label: string = 'State') => {
    if (process.env.NODE_ENV === 'development') {
      console.group(`🔍 ${label}`);
      console.log('Portfolio:', state.portfolio);
      console.log('ML:', state.ml);
      console.log('Risk:', state.risk);
      console.log('Market:', state.market);
      console.log('UI:', state.ui);
      console.log('Preferences:', state.preferences);
      console.log('Errors:', state.errors);
      console.groupEnd();
    }
  },

  // Performance monitor
  createPerformanceMonitor: () => {
    const times: Record<string, number> = {};
    
    return {
      start: (label: string) => {
        times[label] = performance.now();
      },
      end: (label: string) => {
        const startTime = times[label];
        if (startTime) {
          const duration = performance.now() - startTime;
          if (process.env.NODE_ENV === 'development') {
            console.log(`⏱️ ${label}: ${duration.toFixed(2)}ms`);
          }
          delete times[label];
          return duration;
        }
        return 0;
      }
    };
  },

  // State validator
  validateState: (state: any) => {
    const issues: string[] = [];
    
    // Portfolio validation
    if (state.portfolio.weights) {
      const validation = stateUtils.validatePortfolioWeights(state.portfolio.weights);
      if (!validation.isValid) {
        issues.push(`Portfolio weights sum to ${(validation.total * 100).toFixed(2)}% instead of 100%`);
      }
    }
    
    // ML validation
    if (state.ml.confidence < 0 || state.ml.confidence > 1) {
      issues.push(`ML confidence ${state.ml.confidence} is outside valid range [0, 1]`);
    }
    
    // Risk validation
    if (state.market.volatility < 0) {
      issues.push(`Market volatility ${state.market.volatility} cannot be negative`);
    }
    
    return {
      isValid: issues.length === 0,
      issues
    };
  }
};

// ============================================================================
// MAIN EXPORT
// ============================================================================

export default {
  Providers,
  selectors: createSelectors(),
  utils: stateUtils,
  asyncUtils: asyncStateUtils,
  devUtils
};
