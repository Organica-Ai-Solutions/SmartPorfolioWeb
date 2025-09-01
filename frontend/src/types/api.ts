// ============================================================================
// API TYPES - SmartPortfolio AI Frontend
// ============================================================================
// Comprehensive TypeScript types for all AI-powered endpoints

// ============================================================================
// CORE TYPES
// ============================================================================

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string;
}

export interface ApiError {
  detail: string;
  error_code?: string;
  timestamp: string;
}

// ============================================================================
// PORTFOLIO TYPES
// ============================================================================

export interface Portfolio {
  id: string;
  name: string;
  positions: Position[];
  cash: number;
  total_value: number;
  created_at: string;
  updated_at: string;
}

export interface Position {
  symbol: string;
  quantity: number;
  market_value: number;
  cost_basis: number;
  unrealized_pnl: number;
  weight: number;
}

export interface PortfolioWeights {
  [symbol: string]: number;
}

export interface PortfolioMetrics {
  total_return: number;
  annualized_return: number;
  volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
  alpha: number;
  beta: number;
  var_95: number;
  cvar_95: number;
  skewness: number;
  kurtosis: number;
}

export interface AssetMetrics {
  [symbol: string]: {
    return: number;
    volatility: number;
    sharpe_ratio: number;
    beta: number;
    correlation_to_portfolio: number;
  };
}

// ============================================================================
// MACHINE LEARNING TYPES
// ============================================================================

export type PredictionHorizon = "1_day" | "5_days" | "21_days";
export type ModelType = "random_forest" | "gradient_boosting" | "neural_network" | "xgboost" | "lightgbm";
export type MLMarketRegime = "bull_trending" | "bear_trending" | "bull_volatile" | "bear_volatile" | 
                            "sideways_low_vol" | "sideways_high_vol" | "crisis" | "recovery";

export interface MLTrainingRequest {
  symbols: string[];
  horizons: PredictionHorizon[];
  model_types: ModelType[];
  lookback_days?: number;
  retrain?: boolean;
}

export interface MLTrainingResult {
  models_trained: number;
  symbols_processed: number;
  feature_count: number;
  training_results: {
    [symbol: string]: {
      [horizon: string]: {
        [model: string]: {
          r2_score: number;
          mse: number;
          mae: number;
          training_time: number;
        };
      };
    };
  };
}

export interface MLPredictionRequest {
  symbols: string[];
  horizons: PredictionHorizon[];
  include_confidence?: boolean;
}

export interface MLPrediction {
  predicted_return: number;
  confidence: number;
  probability_up: number;
  probability_down: number;
  key_factors: string[];
  model_agreement: number;
}

export interface MLPredictionResult {
  predictions: {
    [symbol: string]: {
      [horizon: string]: MLPrediction;
    };
  };
  portfolio_insights: {
    market_sentiment: "bullish" | "bearish" | "neutral";
    avg_confidence: number;
    prediction_quality: "high" | "medium" | "low";
    recommendation: string;
  };
}

export interface RegimeAnalysisRequest {
  lookback_days?: number;
  retrain_model?: boolean;
}

export interface RegimeAnalysisResult {
  current_regime: MLMarketRegime;
  regime_probability: number;
  regime_duration_days: number;
  next_regime_probabilities: {
    [regime: string]: number;
  };
  key_indicators: {
    market_volatility: number;
    vix_level: number;
    credit_spreads: number;
    yield_curve_slope: number;
  };
  regime_history: {
    date: string;
    regime: MLMarketRegime;
    probability: number;
  }[];
}

export interface RLOptimizationRequest {
  portfolio_weights: PortfolioWeights;
  market_state: {
    volatility: number;
    correlation: number;
    momentum: number;
    sentiment?: number;
  };
  learning_mode?: boolean;
}

export interface RLOptimizationResult {
  recommendation: {
    action: "hold" | "rebalance" | "reduce_risk" | "increase_risk";
    confidence: number;
    expected_return: number;
    allocation_weights: PortfolioWeights;
    reasoning: string[];
  };
  q_values: {
    [action: string]: number;
  };
  exploration_vs_exploitation: {
    exploration_rate: number;
    exploitation_confidence: number;
  };
}

export interface EnsemblePredictionRequest {
  symbols: string[];
  prediction_horizon: PredictionHorizon;
  include_regime_analysis?: boolean;
  include_rl_insights?: boolean;
}

export interface EnsemblePredictionResult {
  ensemble_signals: {
    [symbol: string]: {
      composite_score: number;
      confidence: number;
      recommendation: "strong_buy" | "buy" | "hold" | "sell" | "strong_sell";
      contributing_models: {
        [model: string]: {
          prediction: number;
          weight: number;
          confidence: number;
        };
      };
    };
  };
  final_recommendations: {
    portfolio_recommendation: {
      overall_sentiment: "bullish" | "bearish" | "neutral";
      recommended_action: "increase_risk" | "reduce_risk" | "maintain" | "defensive";
      confidence: number;
    };
    individual_actions: {
      [symbol: string]: {
        action: "buy" | "sell" | "hold" | "reduce" | "increase";
        urgency: "high" | "medium" | "low";
        reasoning: string;
      };
    };
  };
  portfolio_metrics: {
    average_confidence: number;
    prediction_quality_score: number;
    model_consensus: number;
    regime_alignment: number;
  };
}

export interface MLModelStatus {
  price_prediction_models: {
    [symbol: string]: PredictionHorizon[];
  };
  regime_model_trained: boolean;
  rl_agent_initialized: boolean;
  total_models: number;
  last_training_date: string;
  model_performance: {
    [symbol: string]: {
      [horizon: string]: {
        accuracy: number;
        last_updated: string;
      };
    };
  };
}

// ============================================================================
// OPTIMIZATION TYPES
// ============================================================================

export type OptimizationMethod = "max_sharpe" | "min_variance" | "risk_parity" | 
                                "black_litterman" | "factor_based" | "multi_objective";

export type FactorType = "value" | "momentum" | "quality" | "size" | "low_volatility";

export interface OptimizationRequest {
  tickers: string[];
  method: OptimizationMethod;
  lookback_days?: number;
  risk_tolerance?: "conservative" | "moderate" | "aggressive";
  target_return?: number;
  max_weight?: number;
  min_weight?: number;
}

export interface BlackLittermanRequest extends OptimizationRequest {
  views: { [symbol: string]: number };
  view_confidences?: { [symbol: string]: number };
  risk_aversion?: number;
}

export interface FactorBasedRequest extends OptimizationRequest {
  factor_constraints: {
    [factor in FactorType]?: [number, number]; // [min, max]
  };
}

export interface OptimizationResult {
  weights: PortfolioWeights;
  metrics: {
    expected_return: number;
    volatility: number;
    sharpe_ratio: number;
    sortino_ratio?: number;
    max_drawdown?: number;
    var_95?: number;
  };
  optimization_details: {
    method_used: OptimizationMethod;
    convergence: boolean;
    iterations: number;
    objective_value: number;
    constraints_satisfied: boolean;
  };
  factor_exposures?: {
    [factor: string]: number;
  };
  risk_contributions?: {
    [symbol: string]: number;
  };
}

export interface OptimizationComparison {
  methods: OptimizationMethod[];
  results: {
    [method: string]: OptimizationResult;
  };
  recommendations: {
    best_sharpe: OptimizationMethod;
    best_return: OptimizationMethod;
    lowest_risk: OptimizationMethod;
    most_diversified: OptimizationMethod;
  };
  summary_table: {
    [method: string]: {
      return: number;
      risk: number;
      sharpe: number;
      max_weight: number;
      diversification: number;
    };
  };
}

// ============================================================================
// RISK MANAGEMENT TYPES
// ============================================================================

export type RiskRegime = "low" | "moderate" | "high" | "extreme";
export type DrawdownSeverity = "minor" | "moderate" | "major" | "severe";

export interface RiskManagementRequest {
  portfolio_weights: PortfolioWeights;
  tickers: string[];
  portfolio_value: number;
  peak_value?: number;
  target_vol?: number;
  hedge_budget?: number;
}

export interface VolatilityPositionSizingRequest {
  portfolio_weights: PortfolioWeights;
  target_portfolio_vol: number;
  lookback_days?: number;
}

export interface TailRiskHedgingRequest {
  portfolio_weights: PortfolioWeights;
  risk_regime: RiskRegime;
  hedge_budget: number;
  hedge_strategies?: string[];
}

export interface DrawdownControlsRequest {
  portfolio_weights: PortfolioWeights;
  current_drawdown: number;
  peak_value: number;
  current_value: number;
}

export interface RiskManagementResult {
  final_weights: PortfolioWeights;
  risk_metrics: {
    portfolio_volatility: number;
    max_drawdown: number;
    var_95: number;
    cvar_95: number;
    beta: number;
    tracking_error: number;
  };
  applied_strategies: string[];
  hedge_positions?: {
    [symbol: string]: number;
  };
  risk_reduction: {
    volatility_reduction: number;
    drawdown_improvement: number;
    tail_risk_protection: number;
  };
}

// ============================================================================
// LIQUIDITY & TAX TYPES
// ============================================================================

export type MarketCondition = "calm" | "volatile" | "stressed" | "crisis";
export type RebalanceFrequency = "daily" | "weekly" | "monthly" | "quarterly";
export type AccountType = "taxable" | "traditional_ira" | "roth_ira" | "401k" | "hsa";
export type TaxEfficiencyTier = "tax_efficient" | "tax_neutral" | "tax_inefficient";

export interface CashBufferRequest {
  portfolio_value: number;
  current_positions: PortfolioWeights;
  volatility_forecast: number;
  stress_indicators?: {
    credit_spreads: number;
    correlation_increase: number;
    liquidity_stress: number;
  };
}

export interface CashBufferResult {
  target_cash_percentage: number;
  target_cash_amount: number;
  cash_adjustment_needed: number;
  market_condition: MarketCondition;
  funding_plan: {
    action: "raise_cash" | "deploy_cash" | "maintain";
    funding_sources: {
      symbol: string;
      reduction: number;
    }[];
  };
}

export interface LiquidityScoreRequest {
  symbols: string[];
  position_sizes: PortfolioWeights;
}

export interface LiquidityScoreResult {
  asset_liquidity_metrics: {
    [symbol: string]: {
      liquidity_tier: "tier_1" | "tier_2" | "tier_3" | "tier_4" | "tier_5";
      liquidity_score: number;
      days_to_liquidate: number;
      daily_volume: number;
      bid_ask_spread: number;
      market_cap?: number;
    };
  };
  portfolio_liquidity: {
    weighted_liquidity_score: number;
    overall_tier: string;
    time_to_liquidate_50pct: number;
    time_to_liquidate_100pct: number;
  };
  stress_scenarios: {
    [scenario: string]: {
      liquidity_score: number;
      max_liquidation_days: number;
      expected_slippage: number;
    };
  };
}

export interface TaxLossHarvestingRequest {
  portfolio_positions: {
    [symbol: string]: {
      quantity: number;
      current_price: number;
      cost_basis: number;
      purchase_date?: string;
    };
  };
  account_type: AccountType;
  min_loss_threshold?: number;
  wash_sale_avoidance?: boolean;
}

export interface TaxLossHarvestingResult {
  opportunities: {
    symbol: string;
    unrealized_loss: number;
    tax_savings: number;
    replacement_symbol?: string;
    wash_sale_risk: boolean;
    recommendation: "HARVEST" | "HOLD" | "REVIEW";
    reasoning: string;
  }[];
  total_harvestable_losses: number;
  estimated_tax_savings: number;
  net_tax_alpha: number;
}

export interface AssetLocationRequest {
  target_allocation: PortfolioWeights;
  available_accounts: {
    [account: string]: number; // account balance
  };
  asset_tax_efficiency?: {
    [symbol: string]: TaxEfficiencyTier;
  };
}

export interface AssetLocationResult {
  optimized_allocation: {
    [account: string]: PortfolioWeights;
  };
  tax_efficiency_metrics: {
    annual_tax_savings: number;
    efficiency_improvement: number;
    tax_drag_reduction: number;
  };
  placement_reasoning: {
    [symbol: string]: {
      recommended_account: AccountType;
      reasoning: string;
      tax_efficiency_score: number;
    };
  };
}

// ============================================================================
// COMPREHENSIVE AI MANAGEMENT TYPES
// ============================================================================

export interface AIPortfolioManagementRequest {
  tickers: string[];
  base_allocation: PortfolioWeights;
  portfolio_value: number;
  enable_ml_predictions?: boolean;
  enable_regime_analysis?: boolean;
  enable_risk_management?: boolean;
  enable_liquidity_management?: boolean;
  enable_tax_optimization?: boolean;
  prediction_horizon?: PredictionHorizon;
  optimization_method?: OptimizationMethod;
  risk_tolerance?: "conservative" | "moderate" | "aggressive";
  train_models?: boolean;
}

export interface AIPortfolioManagementResult {
  ai_insights: {
    ml_model_confidence: number;
    regime_analysis: {
      current_regime: MLMarketRegime;
      regime_probability: number;
      regime_stability: number;
    };
    predictions_summary: {
      avg_predicted_return: number;
      avg_confidence: number;
      bullish_signals: number;
      bearish_signals: number;
    };
    optimization_improvement: number;
  };
  final_allocation: PortfolioWeights;
  allocation_changes: {
    [symbol: string]: {
      original: number;
      final: number;
      change: number;
      reasoning: string;
    };
  };
  performance_projection: {
    expected_annual_return: number;
    projected_volatility: number;
    projected_sharpe_ratio: number;
    ml_confidence_score: number;
  };
  risk_adjustments: {
    volatility_reduction: number;
    drawdown_protection: number;
    tail_risk_hedging: boolean;
  };
  liquidity_assessment: {
    portfolio_liquidity_score: number;
    cash_buffer_recommendation: number;
    rebalancing_frequency: RebalanceFrequency;
  };
  tax_implications?: {
    estimated_tax_savings: number;
    tax_loss_opportunities: number;
    location_optimization_benefit: number;
  };
  recommendations: string[];
  execution_plan: {
    trades: {
      symbol: string;
      action: "buy" | "sell";
      quantity: number;
      priority: "high" | "medium" | "low";
    }[];
    estimated_costs: number;
    expected_slippage: number;
  };
}

// ============================================================================
// SYSTEM STATUS TYPES
// ============================================================================

export interface HealthCheck {
  status: "healthy" | "degraded" | "unhealthy";
  version: string;
  timestamp: string;
  ml_enabled: boolean;
  services: {
    [service: string]: "operational" | "degraded" | "down";
  };
  uptime: number;
  memory_usage: number;
  cpu_usage: number;
}

export interface SystemCapabilities {
  optimization_methods: {
    [method: string]: string;
  };
  ml_intelligence: {
    price_prediction: string;
    regime_identification: string;
    reinforcement_learning: string;
    prediction_horizons: PredictionHorizon[];
    model_types: ModelType[];
  };
  risk_management: {
    [feature: string]: string;
  };
  liquidity_management: {
    [feature: string]: string;
  };
  tax_optimization: {
    [feature: string]: string;
  };
  data_sources: string[];
  api_version: string;
  last_updated: string;
}

// ============================================================================
// MARKET DATA TYPES
// ============================================================================

export interface MarketData {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  market_cap?: number;
  pe_ratio?: number;
  timestamp: string;
}

export interface HistoricalData {
  symbol: string;
  data: {
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }[];
}

export interface TechnicalIndicators {
  [symbol: string]: {
    rsi: number;
    macd: {
      macd: number;
      signal: number;
      histogram: number;
    };
    moving_averages: {
      ma_20: number;
      ma_50: number;
      ma_200: number;
    };
    bollinger_bands: {
      upper: number;
      middle: number;
      lower: number;
    };
    momentum: {
      "1m": number;
      "3m": number;
      "12m": number;
    };
  };
}

// ============================================================================
// ERROR TYPES
// ============================================================================

export interface ValidationError {
  field: string;
  message: string;
  code: string;
}

export interface APIErrorResponse {
  detail: string | ValidationError[];
  error_code?: string;
  timestamp: string;
  request_id?: string;
}

// ============================================================================
// REQUEST/RESPONSE WRAPPERS
// ============================================================================

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface BulkOperationResponse {
  successful: number;
  failed: number;
  errors: {
    index: number;
    error: string;
  }[];
}

// ============================================================================
// EXPORT ALL TYPES
// ============================================================================

export type {
  // Core
  ApiResponse,
  ApiError,
  
  // Portfolio
  Portfolio,
  Position,
  PortfolioWeights,
  PortfolioMetrics,
  AssetMetrics,
  
  // ML
  MLTrainingRequest,
  MLTrainingResult,
  MLPredictionRequest,
  MLPredictionResult,
  RegimeAnalysisRequest,
  RegimeAnalysisResult,
  RLOptimizationRequest,
  RLOptimizationResult,
  EnsemblePredictionRequest,
  EnsemblePredictionResult,
  MLModelStatus,
  
  // Optimization
  OptimizationRequest,
  BlackLittermanRequest,
  FactorBasedRequest,
  OptimizationResult,
  OptimizationComparison,
  
  // Risk Management
  RiskManagementRequest,
  VolatilityPositionSizingRequest,
  TailRiskHedgingRequest,
  DrawdownControlsRequest,
  RiskManagementResult,
  
  // Liquidity & Tax
  CashBufferRequest,
  CashBufferResult,
  LiquidityScoreRequest,
  LiquidityScoreResult,
  TaxLossHarvestingRequest,
  TaxLossHarvestingResult,
  AssetLocationRequest,
  AssetLocationResult,
  
  // AI Management
  AIPortfolioManagementRequest,
  AIPortfolioManagementResult,
  
  // System
  HealthCheck,
  SystemCapabilities,
  
  // Market Data
  MarketData,
  HistoricalData,
  TechnicalIndicators,
  
  // Errors
  ValidationError,
  APIErrorResponse,
  PaginatedResponse,
  BulkOperationResponse
};
