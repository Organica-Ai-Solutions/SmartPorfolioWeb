// ============================================================================
// APP CONTEXT - SmartPortfolio AI Frontend
// ============================================================================
// Main application context providing global state management

import React, { createContext, useContext, useReducer, useEffect, ReactNode } from 'react';
import {
  Portfolio,
  PortfolioWeights,
  PortfolioMetrics,
  MLPredictionResult,
  RegimeAnalysisResult,
  OptimizationResult,
  RiskManagementResult,
  HealthCheck,
  SystemCapabilities,
  MarketData,
  TechnicalIndicators,
  MLModelStatus,
  OptimizationMethod,
  PredictionHorizon,
  RiskRegime,
  MarketCondition
} from '../types/api';

// ============================================================================
// STATE TYPES
// ============================================================================

export interface AppState {
  // System State
  system: {
    health: HealthCheck | null;
    capabilities: SystemCapabilities | null;
    isLoading: boolean;
    lastUpdated: string | null;
  };

  // Portfolio State
  portfolio: {
    current: Portfolio | null;
    weights: PortfolioWeights;
    metrics: PortfolioMetrics | null;
    value: number;
    tickers: string[];
    isLoading: boolean;
    lastUpdated: string | null;
  };

  // ML Intelligence State
  ml: {
    models: MLModelStatus | null;
    predictions: MLPredictionResult | null;
    regimeAnalysis: RegimeAnalysisResult | null;
    isTraining: boolean;
    isPredicting: boolean;
    lastTraining: string | null;
    confidence: number;
  };

  // Optimization State
  optimization: {
    current: OptimizationResult | null;
    method: OptimizationMethod;
    isOptimizing: boolean;
    history: OptimizationResult[];
    lastOptimized: string | null;
  };

  // Risk Management State
  risk: {
    analysis: RiskManagementResult | null;
    regime: RiskRegime;
    hedgingActive: boolean;
    drawdownControls: boolean;
    isAnalyzing: boolean;
    lastAnalyzed: string | null;
  };

  // Market Data State
  market: {
    data: Record<string, MarketData>;
    indicators: TechnicalIndicators | null;
    condition: MarketCondition;
    sentiment: number; // -1 to 1
    volatility: number;
    isUpdating: boolean;
    lastUpdated: string | null;
  };

  // UI State
  ui: {
    theme: 'light' | 'dark';
    sidebar: {
      isOpen: boolean;
      activeTab: string;
    };
    modals: {
      optimization: boolean;
      riskManagement: boolean;
      mlTraining: boolean;
      settings: boolean;
    };
    notifications: Notification[];
    isOnline: boolean;
  };

  // User Preferences
  preferences: {
    riskTolerance: 'conservative' | 'moderate' | 'aggressive';
    investmentHorizon: 'short' | 'medium' | 'long';
    enableAI: boolean;
    enableRiskManagement: boolean;
    enableTaxOptimization: boolean;
    autoRebalance: boolean;
    rebalanceThreshold: number;
  };

  // Errors & Loading
  errors: {
    system: string | null;
    portfolio: string | null;
    ml: string | null;
    optimization: string | null;
    risk: string | null;
    market: string | null;
  };
}

export interface Notification {
  id: string;
  type: 'success' | 'warning' | 'error' | 'info';
  title: string;
  message: string;
  timestamp: string;
  duration?: number;
  actions?: Array<{
    label: string;
    action: () => void;
  }>;
}

// ============================================================================
// ACTION TYPES
// ============================================================================

export type AppAction =
  // System Actions
  | { type: 'SET_SYSTEM_HEALTH'; payload: HealthCheck }
  | { type: 'SET_SYSTEM_CAPABILITIES'; payload: SystemCapabilities }
  | { type: 'SET_SYSTEM_LOADING'; payload: boolean }
  
  // Portfolio Actions
  | { type: 'SET_PORTFOLIO'; payload: Portfolio }
  | { type: 'SET_PORTFOLIO_WEIGHTS'; payload: PortfolioWeights }
  | { type: 'SET_PORTFOLIO_METRICS'; payload: PortfolioMetrics }
  | { type: 'SET_PORTFOLIO_TICKERS'; payload: string[] }
  | { type: 'SET_PORTFOLIO_VALUE'; payload: number }
  | { type: 'SET_PORTFOLIO_LOADING'; payload: boolean }
  
  // ML Actions
  | { type: 'SET_ML_MODELS'; payload: MLModelStatus }
  | { type: 'SET_ML_PREDICTIONS'; payload: MLPredictionResult }
  | { type: 'SET_REGIME_ANALYSIS'; payload: RegimeAnalysisResult }
  | { type: 'SET_ML_TRAINING'; payload: boolean }
  | { type: 'SET_ML_PREDICTING'; payload: boolean }
  | { type: 'SET_ML_CONFIDENCE'; payload: number }
  
  // Optimization Actions
  | { type: 'SET_OPTIMIZATION_RESULT'; payload: OptimizationResult }
  | { type: 'SET_OPTIMIZATION_METHOD'; payload: OptimizationMethod }
  | { type: 'SET_OPTIMIZATION_LOADING'; payload: boolean }
  | { type: 'ADD_OPTIMIZATION_HISTORY'; payload: OptimizationResult }
  
  // Risk Management Actions
  | { type: 'SET_RISK_ANALYSIS'; payload: RiskManagementResult }
  | { type: 'SET_RISK_REGIME'; payload: RiskRegime }
  | { type: 'SET_HEDGING_ACTIVE'; payload: boolean }
  | { type: 'SET_DRAWDOWN_CONTROLS'; payload: boolean }
  | { type: 'SET_RISK_LOADING'; payload: boolean }
  
  // Market Data Actions
  | { type: 'SET_MARKET_DATA'; payload: Record<string, MarketData> }
  | { type: 'SET_TECHNICAL_INDICATORS'; payload: TechnicalIndicators }
  | { type: 'SET_MARKET_CONDITION'; payload: MarketCondition }
  | { type: 'SET_MARKET_SENTIMENT'; payload: number }
  | { type: 'SET_MARKET_VOLATILITY'; payload: number }
  | { type: 'SET_MARKET_UPDATING'; payload: boolean }
  
  // UI Actions
  | { type: 'SET_THEME'; payload: 'light' | 'dark' }
  | { type: 'TOGGLE_SIDEBAR' }
  | { type: 'SET_ACTIVE_TAB'; payload: string }
  | { type: 'OPEN_MODAL'; payload: keyof AppState['ui']['modals'] }
  | { type: 'CLOSE_MODAL'; payload: keyof AppState['ui']['modals'] }
  | { type: 'ADD_NOTIFICATION'; payload: Notification }
  | { type: 'REMOVE_NOTIFICATION'; payload: string }
  | { type: 'SET_ONLINE_STATUS'; payload: boolean }
  
  // Preferences Actions
  | { type: 'SET_RISK_TOLERANCE'; payload: 'conservative' | 'moderate' | 'aggressive' }
  | { type: 'SET_INVESTMENT_HORIZON'; payload: 'short' | 'medium' | 'long' }
  | { type: 'TOGGLE_AI_ENABLED' }
  | { type: 'TOGGLE_RISK_MANAGEMENT' }
  | { type: 'TOGGLE_TAX_OPTIMIZATION' }
  | { type: 'TOGGLE_AUTO_REBALANCE' }
  | { type: 'SET_REBALANCE_THRESHOLD'; payload: number }
  
  // Error Actions
  | { type: 'SET_ERROR'; payload: { module: keyof AppState['errors']; error: string | null } }
  | { type: 'CLEAR_ERRORS' }
  
  // Bulk Actions
  | { type: 'RESET_STATE' }
  | { type: 'UPDATE_TIMESTAMPS' };

// ============================================================================
// INITIAL STATE
// ============================================================================

export const initialState: AppState = {
  system: {
    health: null,
    capabilities: null,
    isLoading: false,
    lastUpdated: null,
  },

  portfolio: {
    current: null,
    weights: {},
    metrics: null,
    value: 0,
    tickers: [],
    isLoading: false,
    lastUpdated: null,
  },

  ml: {
    models: null,
    predictions: null,
    regimeAnalysis: null,
    isTraining: false,
    isPredicting: false,
    lastTraining: null,
    confidence: 0,
  },

  optimization: {
    current: null,
    method: 'max_sharpe',
    isOptimizing: false,
    history: [],
    lastOptimized: null,
  },

  risk: {
    analysis: null,
    regime: 'moderate',
    hedgingActive: false,
    drawdownControls: false,
    isAnalyzing: false,
    lastAnalyzed: null,
  },

  market: {
    data: {},
    indicators: null,
    condition: 'calm',
    sentiment: 0,
    volatility: 0.15,
    isUpdating: false,
    lastUpdated: null,
  },

  ui: {
    theme: 'light',
    sidebar: {
      isOpen: true,
      activeTab: 'portfolio',
    },
    modals: {
      optimization: false,
      riskManagement: false,
      mlTraining: false,
      settings: false,
    },
    notifications: [],
    isOnline: true,
  },

  preferences: {
    riskTolerance: 'moderate',
    investmentHorizon: 'medium',
    enableAI: true,
    enableRiskManagement: true,
    enableTaxOptimization: false,
    autoRebalance: false,
    rebalanceThreshold: 0.05, // 5%
  },

  errors: {
    system: null,
    portfolio: null,
    ml: null,
    optimization: null,
    risk: null,
    market: null,
  },
};

// ============================================================================
// REDUCER
// ============================================================================

export function appReducer(state: AppState, action: AppAction): AppState {
  const timestamp = new Date().toISOString();

  switch (action.type) {
    // System Actions
    case 'SET_SYSTEM_HEALTH':
      return {
        ...state,
        system: {
          ...state.system,
          health: action.payload,
          lastUpdated: timestamp,
        },
      };

    case 'SET_SYSTEM_CAPABILITIES':
      return {
        ...state,
        system: {
          ...state.system,
          capabilities: action.payload,
          lastUpdated: timestamp,
        },
      };

    case 'SET_SYSTEM_LOADING':
      return {
        ...state,
        system: {
          ...state.system,
          isLoading: action.payload,
        },
      };

    // Portfolio Actions
    case 'SET_PORTFOLIO':
      return {
        ...state,
        portfolio: {
          ...state.portfolio,
          current: action.payload,
          lastUpdated: timestamp,
        },
      };

    case 'SET_PORTFOLIO_WEIGHTS':
      return {
        ...state,
        portfolio: {
          ...state.portfolio,
          weights: action.payload,
          lastUpdated: timestamp,
        },
      };

    case 'SET_PORTFOLIO_METRICS':
      return {
        ...state,
        portfolio: {
          ...state.portfolio,
          metrics: action.payload,
          lastUpdated: timestamp,
        },
      };

    case 'SET_PORTFOLIO_TICKERS':
      return {
        ...state,
        portfolio: {
          ...state.portfolio,
          tickers: action.payload,
          lastUpdated: timestamp,
        },
      };

    case 'SET_PORTFOLIO_VALUE':
      return {
        ...state,
        portfolio: {
          ...state.portfolio,
          value: action.payload,
          lastUpdated: timestamp,
        },
      };

    case 'SET_PORTFOLIO_LOADING':
      return {
        ...state,
        portfolio: {
          ...state.portfolio,
          isLoading: action.payload,
        },
      };

    // ML Actions
    case 'SET_ML_MODELS':
      return {
        ...state,
        ml: {
          ...state.ml,
          models: action.payload,
        },
      };

    case 'SET_ML_PREDICTIONS':
      return {
        ...state,
        ml: {
          ...state.ml,
          predictions: action.payload,
          confidence: action.payload.portfolio_insights.avg_confidence,
        },
      };

    case 'SET_REGIME_ANALYSIS':
      return {
        ...state,
        ml: {
          ...state.ml,
          regimeAnalysis: action.payload,
        },
      };

    case 'SET_ML_TRAINING':
      return {
        ...state,
        ml: {
          ...state.ml,
          isTraining: action.payload,
          lastTraining: action.payload ? timestamp : state.ml.lastTraining,
        },
      };

    case 'SET_ML_PREDICTING':
      return {
        ...state,
        ml: {
          ...state.ml,
          isPredicting: action.payload,
        },
      };

    case 'SET_ML_CONFIDENCE':
      return {
        ...state,
        ml: {
          ...state.ml,
          confidence: action.payload,
        },
      };

    // Optimization Actions
    case 'SET_OPTIMIZATION_RESULT':
      return {
        ...state,
        optimization: {
          ...state.optimization,
          current: action.payload,
          lastOptimized: timestamp,
        },
      };

    case 'SET_OPTIMIZATION_METHOD':
      return {
        ...state,
        optimization: {
          ...state.optimization,
          method: action.payload,
        },
      };

    case 'SET_OPTIMIZATION_LOADING':
      return {
        ...state,
        optimization: {
          ...state.optimization,
          isOptimizing: action.payload,
        },
      };

    case 'ADD_OPTIMIZATION_HISTORY':
      return {
        ...state,
        optimization: {
          ...state.optimization,
          history: [action.payload, ...state.optimization.history].slice(0, 10), // Keep last 10
        },
      };

    // Risk Management Actions
    case 'SET_RISK_ANALYSIS':
      return {
        ...state,
        risk: {
          ...state.risk,
          analysis: action.payload,
          lastAnalyzed: timestamp,
        },
      };

    case 'SET_RISK_REGIME':
      return {
        ...state,
        risk: {
          ...state.risk,
          regime: action.payload,
        },
      };

    case 'SET_HEDGING_ACTIVE':
      return {
        ...state,
        risk: {
          ...state.risk,
          hedgingActive: action.payload,
        },
      };

    case 'SET_DRAWDOWN_CONTROLS':
      return {
        ...state,
        risk: {
          ...state.risk,
          drawdownControls: action.payload,
        },
      };

    case 'SET_RISK_LOADING':
      return {
        ...state,
        risk: {
          ...state.risk,
          isAnalyzing: action.payload,
        },
      };

    // Market Data Actions
    case 'SET_MARKET_DATA':
      return {
        ...state,
        market: {
          ...state.market,
          data: action.payload,
          lastUpdated: timestamp,
        },
      };

    case 'SET_TECHNICAL_INDICATORS':
      return {
        ...state,
        market: {
          ...state.market,
          indicators: action.payload,
          lastUpdated: timestamp,
        },
      };

    case 'SET_MARKET_CONDITION':
      return {
        ...state,
        market: {
          ...state.market,
          condition: action.payload,
        },
      };

    case 'SET_MARKET_SENTIMENT':
      return {
        ...state,
        market: {
          ...state.market,
          sentiment: action.payload,
        },
      };

    case 'SET_MARKET_VOLATILITY':
      return {
        ...state,
        market: {
          ...state.market,
          volatility: action.payload,
        },
      };

    case 'SET_MARKET_UPDATING':
      return {
        ...state,
        market: {
          ...state.market,
          isUpdating: action.payload,
        },
      };

    // UI Actions
    case 'SET_THEME':
      return {
        ...state,
        ui: {
          ...state.ui,
          theme: action.payload,
        },
      };

    case 'TOGGLE_SIDEBAR':
      return {
        ...state,
        ui: {
          ...state.ui,
          sidebar: {
            ...state.ui.sidebar,
            isOpen: !state.ui.sidebar.isOpen,
          },
        },
      };

    case 'SET_ACTIVE_TAB':
      return {
        ...state,
        ui: {
          ...state.ui,
          sidebar: {
            ...state.ui.sidebar,
            activeTab: action.payload,
          },
        },
      };

    case 'OPEN_MODAL':
      return {
        ...state,
        ui: {
          ...state.ui,
          modals: {
            ...state.ui.modals,
            [action.payload]: true,
          },
        },
      };

    case 'CLOSE_MODAL':
      return {
        ...state,
        ui: {
          ...state.ui,
          modals: {
            ...state.ui.modals,
            [action.payload]: false,
          },
        },
      };

    case 'ADD_NOTIFICATION':
      return {
        ...state,
        ui: {
          ...state.ui,
          notifications: [action.payload, ...state.ui.notifications],
        },
      };

    case 'REMOVE_NOTIFICATION':
      return {
        ...state,
        ui: {
          ...state.ui,
          notifications: state.ui.notifications.filter(n => n.id !== action.payload),
        },
      };

    case 'SET_ONLINE_STATUS':
      return {
        ...state,
        ui: {
          ...state.ui,
          isOnline: action.payload,
        },
      };

    // Preferences Actions
    case 'SET_RISK_TOLERANCE':
      return {
        ...state,
        preferences: {
          ...state.preferences,
          riskTolerance: action.payload,
        },
      };

    case 'SET_INVESTMENT_HORIZON':
      return {
        ...state,
        preferences: {
          ...state.preferences,
          investmentHorizon: action.payload,
        },
      };

    case 'TOGGLE_AI_ENABLED':
      return {
        ...state,
        preferences: {
          ...state.preferences,
          enableAI: !state.preferences.enableAI,
        },
      };

    case 'TOGGLE_RISK_MANAGEMENT':
      return {
        ...state,
        preferences: {
          ...state.preferences,
          enableRiskManagement: !state.preferences.enableRiskManagement,
        },
      };

    case 'TOGGLE_TAX_OPTIMIZATION':
      return {
        ...state,
        preferences: {
          ...state.preferences,
          enableTaxOptimization: !state.preferences.enableTaxOptimization,
        },
      };

    case 'TOGGLE_AUTO_REBALANCE':
      return {
        ...state,
        preferences: {
          ...state.preferences,
          autoRebalance: !state.preferences.autoRebalance,
        },
      };

    case 'SET_REBALANCE_THRESHOLD':
      return {
        ...state,
        preferences: {
          ...state.preferences,
          rebalanceThreshold: action.payload,
        },
      };

    // Error Actions
    case 'SET_ERROR':
      return {
        ...state,
        errors: {
          ...state.errors,
          [action.payload.module]: action.payload.error,
        },
      };

    case 'CLEAR_ERRORS':
      return {
        ...state,
        errors: {
          system: null,
          portfolio: null,
          ml: null,
          optimization: null,
          risk: null,
          market: null,
        },
      };

    // Bulk Actions
    case 'RESET_STATE':
      return {
        ...initialState,
        ui: {
          ...initialState.ui,
          theme: state.ui.theme, // Preserve theme
        },
        preferences: state.preferences, // Preserve preferences
      };

    case 'UPDATE_TIMESTAMPS':
      return {
        ...state,
        system: { ...state.system, lastUpdated: timestamp },
        portfolio: { ...state.portfolio, lastUpdated: timestamp },
        market: { ...state.market, lastUpdated: timestamp },
      };

    default:
      return state;
  }
}

// ============================================================================
// CONTEXT CREATION
// ============================================================================

interface AppContextType {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
}

export const AppContext = createContext<AppContextType | undefined>(undefined);

// ============================================================================
// PROVIDER COMPONENT
// ============================================================================

interface AppProviderProps {
  children: ReactNode;
  initialState?: Partial<AppState>;
}

export const AppProvider: React.FC<AppProviderProps> = ({ 
  children, 
  initialState: customInitialState 
}) => {
  const [state, dispatch] = useReducer(appReducer, {
    ...initialState,
    ...customInitialState,
  });

  // Auto-save preferences to localStorage
  useEffect(() => {
    localStorage.setItem('smartportfolio-preferences', JSON.stringify(state.preferences));
  }, [state.preferences]);

  // Auto-save UI theme to localStorage
  useEffect(() => {
    localStorage.setItem('smartportfolio-theme', state.ui.theme);
    document.documentElement.setAttribute('data-theme', state.ui.theme);
  }, [state.ui.theme]);

  // Load saved preferences on mount
  useEffect(() => {
    const savedPreferences = localStorage.getItem('smartportfolio-preferences');
    if (savedPreferences) {
      try {
        const preferences = JSON.parse(savedPreferences);
        Object.entries(preferences).forEach(([key, value]) => {
          if (key === 'riskTolerance') {
            dispatch({ type: 'SET_RISK_TOLERANCE', payload: value as any });
          } else if (key === 'investmentHorizon') {
            dispatch({ type: 'SET_INVESTMENT_HORIZON', payload: value as any });
          } else if (key === 'rebalanceThreshold') {
            dispatch({ type: 'SET_REBALANCE_THRESHOLD', payload: value as number });
          }
          // Add more preference mappings as needed
        });
      } catch (error) {
        console.warn('Failed to load saved preferences:', error);
      }
    }

    const savedTheme = localStorage.getItem('smartportfolio-theme') as 'light' | 'dark';
    if (savedTheme) {
      dispatch({ type: 'SET_THEME', payload: savedTheme });
    }
  }, []);

  // Online/offline detection
  useEffect(() => {
    const handleOnline = () => dispatch({ type: 'SET_ONLINE_STATUS', payload: true });
    const handleOffline = () => dispatch({ type: 'SET_ONLINE_STATUS', payload: false });

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Auto-remove notifications after duration
  useEffect(() => {
    const timeouts: NodeJS.Timeout[] = [];

    state.ui.notifications.forEach(notification => {
      if (notification.duration && notification.duration > 0) {
        const timeout = setTimeout(() => {
          dispatch({ type: 'REMOVE_NOTIFICATION', payload: notification.id });
        }, notification.duration);
        timeouts.push(timeout);
      }
    });

    return () => {
      timeouts.forEach(clearTimeout);
    };
  }, [state.ui.notifications]);

  const value: AppContextType = {
    state,
    dispatch,
  };

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
};

// ============================================================================
// HOOK
// ============================================================================

export const useAppContext = (): AppContextType => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
};

// ============================================================================
// CONVENIENCE HOOKS
// ============================================================================

export const useAppState = () => {
  const { state } = useAppContext();
  return state;
};

export const useAppDispatch = () => {
  const { dispatch } = useAppContext();
  return dispatch;
};

// Specific state hooks
export const useSystemState = () => useAppState().system;
export const usePortfolioState = () => useAppState().portfolio;
export const useMLState = () => useAppState().ml;
export const useOptimizationState = () => useAppState().optimization;
export const useRiskState = () => useAppState().risk;
export const useMarketState = () => useAppState().market;
export const useUIState = () => useAppState().ui;
export const usePreferences = () => useAppState().preferences;
export const useErrors = () => useAppState().errors;

export default AppProvider;
