// ============================================================================
// API ENDPOINTS - SmartPortfolio AI Frontend
// ============================================================================
// Complete documentation and configuration for all 39 API endpoints

export const API_ENDPOINTS = {
  // ============================================================================
  // SYSTEM & HEALTH ENDPOINTS (5)
  // ============================================================================
  SYSTEM: {
    HEALTH: {
      method: 'GET' as const,
      path: '/health',
      description: 'System health check with service status',
      response: 'HealthCheck'
    },
    READINESS: {
      method: 'GET' as const,
      path: '/readiness',
      description: 'Service readiness probe',
      response: '{ status: string }'
    },
    CAPABILITIES: {
      method: 'GET' as const,
      path: '/portfolio-management-capabilities',
      description: 'Complete system capabilities overview',
      response: 'SystemCapabilities'
    },
    ACCOUNT: {
      method: 'GET' as const,
      path: '/account',
      description: 'User account information and trading status',
      response: 'AccountInfo'
    },
    ENV_CHECK: {
      method: 'GET' as const,
      path: '/env-check',
      description: 'Environment configuration and API key status',
      response: '{ api_keys: Record<string, boolean> }'
    }
  },

  // ============================================================================
  // MACHINE LEARNING ENDPOINTS (6)
  // ============================================================================
  ML: {
    TRAIN_MODELS: {
      method: 'POST' as const,
      path: '/ml/train-models',
      description: 'Train ML models for price prediction using multiple algorithms',
      request: 'MLTrainingRequest',
      response: 'MLTrainingResult',
      features: ['Random Forest', 'Gradient Boosting', 'Neural Networks', 'XGBoost', 'LightGBM']
    },
    PREDICT_MOVEMENTS: {
      method: 'POST' as const,
      path: '/ml/predict-movements',
      description: 'Generate AI-powered price movement predictions with confidence scores',
      request: 'MLPredictionRequest',
      response: 'MLPredictionResult',
      features: ['Multi-horizon predictions', 'Confidence scoring', 'Factor analysis']
    },
    IDENTIFY_REGIMES: {
      method: 'POST' as const,
      path: '/ml/identify-regimes',
      description: 'Market regime detection using clustering algorithms',
      request: 'RegimeAnalysisRequest',
      response: 'RegimeAnalysisResult',
      features: ['8 distinct regimes', 'Transition probabilities', 'Historical analysis']
    },
    RL_OPTIMIZATION: {
      method: 'POST' as const,
      path: '/ml/rl-optimization',
      description: 'Reinforcement learning-based portfolio optimization',
      request: 'RLOptimizationRequest',
      response: 'RLOptimizationResult',
      features: ['Q-learning', 'Dynamic adaptation', 'Market state analysis']
    },
    ENSEMBLE_PREDICTION: {
      method: 'POST' as const,
      path: '/ml/ensemble-prediction',
      description: 'Combined ML predictions using ensemble methods',
      request: 'EnsemblePredictionRequest',
      response: 'EnsemblePredictionResult',
      features: ['Multiple model fusion', 'Weighted predictions', 'Consensus scoring']
    },
    MODEL_STATUS: {
      method: 'GET' as const,
      path: '/ml/model-status',
      description: 'ML model training status and performance metrics',
      response: 'MLModelStatus'
    }
  },

  // ============================================================================
  // PORTFOLIO OPTIMIZATION ENDPOINTS (7)
  // ============================================================================
  OPTIMIZATION: {
    ADVANCED: {
      method: 'POST' as const,
      path: '/optimize-portfolio-advanced',
      description: 'Advanced portfolio optimization with multiple methods',
      request: 'OptimizationRequest',
      response: 'OptimizationResult',
      features: ['6 optimization methods', 'Risk constraints', 'Factor exposure']
    },
    BLACK_LITTERMAN: {
      method: 'POST' as const,
      path: '/black-litterman-optimization',
      description: 'Black-Litterman model combining market equilibrium with investor views',
      request: 'BlackLittermanRequest',
      response: 'OptimizationResult',
      features: ['Market equilibrium', 'Investor views', 'Confidence weighting']
    },
    FACTOR_BASED: {
      method: 'POST' as const,
      path: '/factor-based-optimization',
      description: 'Factor-based optimization with style constraints',
      request: 'FactorBasedRequest',
      response: 'OptimizationResult',
      features: ['Value/Growth/Quality factors', 'Style constraints', 'Factor exposure control']
    },
    RISK_PARITY: {
      method: 'POST' as const,
      path: '/risk-parity-optimization',
      description: 'Risk parity optimization for equal risk contribution',
      request: 'OptimizationRequest',
      response: 'OptimizationResult',
      features: ['Equal risk contribution', 'Volatility weighting', 'Risk budgeting']
    },
    MIN_VARIANCE: {
      method: 'POST' as const,
      path: '/minimum-variance-optimization',
      description: 'Minimum variance optimization for risk minimization',
      request: 'OptimizationRequest',
      response: 'OptimizationResult',
      features: ['Volatility minimization', 'Conservative approach', 'Low-risk allocation']
    },
    COMPARE_METHODS: {
      method: 'POST' as const,
      path: '/compare-optimization-methods',
      description: 'Compare multiple optimization methods side by side',
      request: '{ tickers: string[], methods: string[], lookback_days?: number }',
      response: 'OptimizationComparison',
      features: ['Method comparison', 'Performance metrics', 'Best method recommendation']
    },
    WITH_RISK_CONTROLS: {
      method: 'POST' as const,
      path: '/optimize-with-risk-controls',
      description: 'Portfolio optimization with integrated risk management',
      request: 'RiskManagementRequest',
      response: 'OptimizationResult',
      features: ['Risk-aware optimization', 'Integrated controls', 'Hedging integration']
    }
  },

  // ============================================================================
  // RISK MANAGEMENT ENDPOINTS (6)
  // ============================================================================
  RISK: {
    COMPREHENSIVE: {
      method: 'POST' as const,
      path: '/comprehensive-risk-management',
      description: 'Complete risk management with multiple strategies',
      request: 'RiskManagementRequest',
      response: 'RiskManagementResult',
      features: ['Multi-strategy approach', 'Dynamic hedging', 'Risk regime detection']
    },
    VOLATILITY_SIZING: {
      method: 'POST' as const,
      path: '/volatility-position-sizing',
      description: 'Dynamic position sizing based on asset volatility',
      request: 'VolatilityPositionSizingRequest',
      response: '{ adjusted_weights: PortfolioWeights, volatility_metrics: any }',
      features: ['Inverse volatility weighting', 'Risk target matching', 'Dynamic adjustment']
    },
    TAIL_HEDGING: {
      method: 'POST' as const,
      path: '/tail-risk-hedging',
      description: 'Tail risk hedging strategies for downside protection',
      request: 'TailRiskHedgingRequest',
      response: '{ hedged_portfolio: PortfolioWeights, hedge_details: any }',
      features: ['VIX protection', 'Safe haven allocation', 'Crisis hedging']
    },
    DRAWDOWN_CONTROLS: {
      method: 'POST' as const,
      path: '/drawdown-controls',
      description: 'Drawdown control mechanisms for loss limitation',
      request: 'DrawdownControlsRequest',
      response: '{ controlled_weights: PortfolioWeights, drawdown_metrics: any }',
      features: ['Automatic derisking', 'Loss thresholds', 'Capital preservation']
    },
    INFO: {
      method: 'GET' as const,
      path: '/risk-management-info',
      description: 'Risk management configuration and available strategies',
      response: '{ risk_regimes: string[], hedging_strategies: string[], drawdown_thresholds: Record<string, number> }'
    }
  },

  // ============================================================================
  // LIQUIDITY MANAGEMENT ENDPOINTS (4)
  // ============================================================================
  LIQUIDITY: {
    CASH_BUFFER: {
      method: 'POST' as const,
      path: '/calculate-cash-buffer',
      description: 'Dynamic cash buffer optimization based on market conditions',
      request: 'CashBufferRequest',
      response: 'CashBufferResult',
      features: ['Volatility-based sizing', 'Stress scenario planning', 'Dynamic adjustment']
    },
    REBALANCE_FREQUENCY: {
      method: 'POST' as const,
      path: '/determine-rebalancing-frequency',
      description: 'Optimal rebalancing frequency based on costs and volatility',
      request: '{ portfolio_weights: PortfolioWeights, volatility_forecast: number, transaction_costs: number }',
      response: '{ recommended_frequency: string, cost_benefit_analysis: any }',
      features: ['Cost optimization', 'Frequency recommendation', 'Transaction analysis']
    },
    ASSET_SCORING: {
      method: 'POST' as const,
      path: '/score-asset-liquidity',
      description: 'Comprehensive asset liquidity scoring and stress testing',
      request: 'LiquidityScoreRequest',
      response: 'LiquidityScoreResult',
      features: ['5-tier scoring', 'Stress scenarios', 'Liquidation timelines']
    },
    AWARE_ALLOCATION: {
      method: 'POST' as const,
      path: '/liquidity-aware-allocation',
      description: 'Portfolio allocation considering liquidity constraints',
      request: '{ target_allocation: PortfolioWeights, market_condition: string, stress_test: boolean }',
      response: '{ optimized_allocation: PortfolioWeights, liquidity_metrics: any }',
      features: ['Liquidity constraints', 'Stress testing', 'Allocation optimization']
    }
  },

  // ============================================================================
  // TAX OPTIMIZATION ENDPOINTS (4)
  // ============================================================================
  TAX: {
    LOSS_HARVESTING: {
      method: 'POST' as const,
      path: '/tax-loss-harvesting',
      description: 'Tax-loss harvesting opportunities identification',
      request: 'TaxLossHarvestingRequest',
      response: 'TaxLossHarvestingResult',
      features: ['Loss identification', 'Tax savings calculation', 'Wash sale avoidance']
    },
    ASSET_LOCATION: {
      method: 'POST' as const,
      path: '/optimize-asset-location',
      description: 'Multi-account asset location optimization for tax efficiency',
      request: 'AssetLocationRequest',
      response: 'AssetLocationResult',
      features: ['Multi-account optimization', 'Tax efficiency maximization', 'Account type consideration']
    },
    AWARE_REBALANCING: {
      method: 'POST' as const,
      path: '/tax-aware-rebalancing',
      description: 'Tax-aware portfolio rebalancing with impact minimization',
      request: '{ current_allocation: PortfolioWeights, target_allocation: PortfolioWeights, cost_basis: Record<string, number>, account_type: string }',
      response: '{ optimized_trades: any[], tax_impact: number, after_tax_improvement: number }',
      features: ['Tax impact minimization', 'Optimal trade sequencing', 'After-tax optimization']
    },
    ALPHA_CALCULATION: {
      method: 'POST' as const,
      path: '/calculate-tax-alpha',
      description: 'Tax alpha calculation and efficiency measurement',
      request: '{ portfolio_weights: PortfolioWeights, account_types: Record<string, string>, tax_rates: Record<string, number> }',
      response: '{ annual_tax_alpha: number, tax_efficiency_score: number, improvement_opportunities: any[] }',
      features: ['Alpha quantification', 'Efficiency scoring', 'Improvement identification']
    }
  },

  // ============================================================================
  // PORTFOLIO ANALYSIS ENDPOINTS (4)
  // ============================================================================
  ANALYSIS: {
    COMPREHENSIVE: {
      method: 'POST' as const,
      path: '/analyze-portfolio',
      description: 'Comprehensive portfolio analysis with risk metrics',
      request: '{ tickers: string[], weights: PortfolioWeights, benchmark?: string, lookback_days?: number }',
      response: '{ portfolio_metrics: PortfolioMetrics, asset_metrics: any, risk_analysis: any, performance_attribution: any }',
      features: ['Complete metrics', 'Risk analysis', 'Performance attribution']
    },
    SIMPLE: {
      method: 'POST' as const,
      path: '/analyze-portfolio-simple',
      description: 'Simple portfolio analysis with basic metrics',
      request: '{ tickers: string[], weights: PortfolioWeights }',
      response: '{ expected_return: number, volatility: number, sharpe_ratio: number }',
      features: ['Basic metrics only', 'Fast analysis', 'Core statistics']
    },
    ENHANCED: {
      method: 'POST' as const,
      path: '/analyze-portfolio-enhanced',
      description: 'Enhanced analysis with AI insights and regime analysis',
      request: '{ tickers: string[], weights: PortfolioWeights, enable_ai_insights?: boolean, enable_regime_analysis?: boolean }',
      response: '{ basic_metrics: PortfolioMetrics, ai_insights?: any, regime_analysis?: any, dynamic_allocation?: any }',
      features: ['AI insights', 'Regime analysis', 'Dynamic recommendations']
    },
    REBALANCE: {
      method: 'POST' as const,
      path: '/rebalance-portfolio',
      description: 'Portfolio rebalancing with execution planning',
      request: '{ current_weights: PortfolioWeights, target_weights: PortfolioWeights, portfolio_value: number, execution_method?: string }',
      response: '{ trades: any[], estimated_costs: number, expected_slippage: number, execution_plan: any }',
      features: ['Trade planning', 'Cost estimation', 'Execution optimization']
    }
  },

  // ============================================================================
  // DYNAMIC ALLOCATION ENDPOINTS (2)
  // ============================================================================
  DYNAMIC: {
    ALLOCATION: {
      method: 'POST' as const,
      path: '/dynamic-allocation',
      description: 'Dynamic asset allocation based on market conditions',
      request: '{ tickers: string[], base_allocation: PortfolioWeights, enable_regime_detection?: boolean, enable_momentum_signals?: boolean, lookback_days?: number }',
      response: '{ final_allocation: PortfolioWeights, regime_analysis: any, momentum_signals: any, allocation_changes: any }',
      features: ['Regime-based allocation', 'Momentum signals', 'Dynamic adjustment']
    },
    MARKET_ANALYSIS: {
      method: 'POST' as const,
      path: '/market-analysis',
      description: 'Comprehensive market analysis with technical indicators',
      request: '{ symbols?: string[], lookback_days?: number }',
      response: '{ market_metrics: any, technical_indicators: TechnicalIndicators, sentiment_analysis: any, regime_detection: any }',
      features: ['Technical analysis', 'Sentiment scoring', 'Market regime detection']
    }
  },

  // ============================================================================
  // COMPREHENSIVE AI MANAGEMENT ENDPOINTS (2)
  // ============================================================================
  AI_MANAGEMENT: {
    AI_PORTFOLIO: {
      method: 'POST' as const,
      path: '/ai-portfolio-management',
      description: 'Complete AI-powered portfolio management with all features',
      request: 'AIPortfolioManagementRequest',
      response: 'AIPortfolioManagementResult',
      features: ['Complete AI integration', 'All-in-one optimization', 'Comprehensive recommendations']
    },
    COMPREHENSIVE: {
      method: 'POST' as const,
      path: '/comprehensive-portfolio-management',
      description: 'Comprehensive portfolio management with all available features',
      request: '{ tickers: string[], base_allocation: PortfolioWeights, portfolio_value: number, risk_tolerance: string, investment_horizon: string, enable_all_features?: boolean }',
      response: '{ optimized_allocation: PortfolioWeights, ai_insights: any, risk_analysis: any, tax_optimization: any, liquidity_analysis: any, execution_plan: any, recommendations: string[] }',
      features: ['All features enabled', 'Holistic approach', 'Complete optimization']
    }
  }
} as const;

// ============================================================================
// ENDPOINT CATEGORIES
// ============================================================================

export const ENDPOINT_CATEGORIES = {
  SYSTEM: 'System & Health',
  ML: 'Machine Learning',
  OPTIMIZATION: 'Portfolio Optimization',
  RISK: 'Risk Management',
  LIQUIDITY: 'Liquidity Management',
  TAX: 'Tax Optimization',
  ANALYSIS: 'Portfolio Analysis',
  DYNAMIC: 'Dynamic Allocation',
  AI_MANAGEMENT: 'AI Management'
} as const;

// ============================================================================
// ENDPOINT STATISTICS
// ============================================================================

export const ENDPOINT_STATS = {
  TOTAL_ENDPOINTS: 39,
  BY_CATEGORY: {
    [ENDPOINT_CATEGORIES.SYSTEM]: 5,
    [ENDPOINT_CATEGORIES.ML]: 6,
    [ENDPOINT_CATEGORIES.OPTIMIZATION]: 7,
    [ENDPOINT_CATEGORIES.RISK]: 5,
    [ENDPOINT_CATEGORIES.LIQUIDITY]: 4,
    [ENDPOINT_CATEGORIES.TAX]: 4,
    [ENDPOINT_CATEGORIES.ANALYSIS]: 4,
    [ENDPOINT_CATEGORIES.DYNAMIC]: 2,
    [ENDPOINT_CATEGORIES.AI_MANAGEMENT]: 2
  },
  BY_METHOD: {
    GET: 5,
    POST: 34
  }
} as const;

// ============================================================================
// ENDPOINT UTILITIES
// ============================================================================

export const getEndpointsByCategory = (category: keyof typeof ENDPOINT_CATEGORIES) => {
  return API_ENDPOINTS[category] || {};
};

export const getAllEndpoints = () => {
  const allEndpoints: Array<{
    category: string;
    name: string;
    method: string;
    path: string;
    description: string;
    features?: string[];
  }> = [];

  Object.entries(API_ENDPOINTS).forEach(([categoryKey, categoryEndpoints]) => {
    const categoryName = ENDPOINT_CATEGORIES[categoryKey as keyof typeof ENDPOINT_CATEGORIES];
    
    Object.entries(categoryEndpoints).forEach(([endpointKey, endpoint]) => {
      allEndpoints.push({
        category: categoryName,
        name: endpointKey,
        method: endpoint.method,
        path: endpoint.path,
        description: endpoint.description,
        features: endpoint.features
      });
    });
  });

  return allEndpoints;
};

export const getEndpointByPath = (path: string) => {
  const allEndpoints = getAllEndpoints();
  return allEndpoints.find(endpoint => endpoint.path === path);
};

export const getEndpointsByFeature = (feature: string) => {
  const allEndpoints = getAllEndpoints();
  return allEndpoints.filter(endpoint => 
    endpoint.features?.some(f => f.toLowerCase().includes(feature.toLowerCase()))
  );
};

// ============================================================================
// ENDPOINT PERFORMANCE BENCHMARKS
// ============================================================================

export const ENDPOINT_PERFORMANCE = {
  FAST: ['< 200ms', [
    '/health',
    '/readiness',
    '/portfolio-management-capabilities',
    '/analyze-portfolio-simple',
    '/ml/model-status'
  ]],
  MODERATE: ['200ms - 1s', [
    '/optimize-portfolio-advanced',
    '/analyze-portfolio',
    '/volatility-position-sizing',
    '/calculate-cash-buffer',
    '/tax-loss-harvesting'
  ]],
  SLOW: ['1s - 5s', [
    '/ml/predict-movements',
    '/ml/identify-regimes',
    '/comprehensive-risk-management',
    '/ai-portfolio-management'
  ]],
  VERY_SLOW: ['> 5s', [
    '/ml/train-models',
    '/ml/ensemble-prediction',
    '/comprehensive-portfolio-management'
  ]]
} as const;

// ============================================================================
// ENDPOINT DEPENDENCIES
// ============================================================================

export const ENDPOINT_DEPENDENCIES = {
  '/ml/predict-movements': ['ML models must be trained first'],
  '/ml/rl-optimization': ['Portfolio weights required', 'Market data needed'],
  '/ml/ensemble-prediction': ['Multiple ML models required'],
  '/ai-portfolio-management': ['All AI features enabled', 'Portfolio data required'],
  '/comprehensive-portfolio-management': ['Complete portfolio setup', 'All services enabled']
} as const;

// ============================================================================
// ENDPOINT USAGE RECOMMENDATIONS
// ============================================================================

export const ENDPOINT_USAGE = {
  BEGINNER: [
    '/health',
    '/analyze-portfolio-simple',
    '/optimize-portfolio-advanced',
    '/portfolio-management-capabilities'
  ],
  INTERMEDIATE: [
    '/analyze-portfolio',
    '/black-litterman-optimization',
    '/comprehensive-risk-management',
    '/calculate-cash-buffer',
    '/ml/predict-movements'
  ],
  ADVANCED: [
    '/ml/train-models',
    '/ml/ensemble-prediction',
    '/ai-portfolio-management',
    '/comprehensive-portfolio-management',
    '/factor-based-optimization'
  ]
} as const;

export default API_ENDPOINTS;
