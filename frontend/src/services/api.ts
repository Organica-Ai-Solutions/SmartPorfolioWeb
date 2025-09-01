// ============================================================================
// API CLIENT - SmartPortfolio AI Frontend
// ============================================================================
// Comprehensive API client for all AI-powered endpoints

import {
  // Core Types
  ApiResponse,
  ApiError,
  HealthCheck,
  SystemCapabilities,
  
  // Portfolio Types
  Portfolio,
  PortfolioWeights,
  PortfolioMetrics,
  
  // ML Types
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
  
  // Optimization Types
  OptimizationRequest,
  BlackLittermanRequest,
  FactorBasedRequest,
  OptimizationResult,
  OptimizationComparison,
  
  // Risk Management Types
  RiskManagementRequest,
  VolatilityPositionSizingRequest,
  TailRiskHedgingRequest,
  DrawdownControlsRequest,
  RiskManagementResult,
  
  // Liquidity & Tax Types
  CashBufferRequest,
  CashBufferResult,
  LiquidityScoreRequest,
  LiquidityScoreResult,
  TaxLossHarvestingRequest,
  TaxLossHarvestingResult,
  AssetLocationRequest,
  AssetLocationResult,
  
  // AI Management Types
  AIPortfolioManagementRequest,
  AIPortfolioManagementResult,
  
  // Market Data Types
  MarketData,
  HistoricalData,
  TechnicalIndicators
} from '../types/api';

// ============================================================================
// API CONFIGURATION
// ============================================================================

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';
const API_TIMEOUT = 30000; // 30 seconds for ML operations

// ============================================================================
// API CLIENT CLASS
// ============================================================================

class SmartPortfolioAPIClient {
  private baseURL: string;
  private timeout: number;
  private defaultHeaders: Record<string, string>;

  constructor() {
    this.baseURL = API_BASE_URL;
    this.timeout = API_TIMEOUT;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    };
  }

  // ============================================================================
  // CORE HTTP METHODS
  // ============================================================================

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const config: RequestInit = {
      ...options,
      headers: {
        ...this.defaultHeaders,
        ...options.headers,
      },
      signal: AbortSignal.timeout(this.timeout),
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({
          detail: `HTTP ${response.status}: ${response.statusText}`,
          error_code: `HTTP_${response.status}`,
          timestamp: new Date().toISOString()
        }));
        
        throw new APIError(errorData.detail || 'Unknown error', errorData.error_code, response.status);
      }

      return await response.json();
    } catch (error) {
      if (error instanceof APIError) {
        throw error;
      }
      
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new APIError('Request timeout', 'TIMEOUT', 408);
        }
        throw new APIError(error.message, 'NETWORK_ERROR', 0);
      }
      
      throw new APIError('Unknown error occurred', 'UNKNOWN_ERROR', 0);
    }
  }

  private async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET' });
  }

  private async post<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  private async put<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  private async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' });
  }

  // ============================================================================
  // SYSTEM & HEALTH ENDPOINTS
  // ============================================================================

  async getHealth(): Promise<HealthCheck> {
    return this.get<HealthCheck>('/health');
  }

  async getReadiness(): Promise<{ status: string }> {
    return this.get<{ status: string }>('/readiness');
  }

  async getCapabilities(): Promise<SystemCapabilities> {
    return this.get<SystemCapabilities>('/portfolio-management-capabilities');
  }

  async getAccount(): Promise<any> {
    return this.get('/account');
  }

  async checkEnvironment(): Promise<{ api_keys: Record<string, boolean> }> {
    return this.get('/env-check');
  }

  // ============================================================================
  // MACHINE LEARNING ENDPOINTS
  // ============================================================================

  async trainMLModels(request: MLTrainingRequest): Promise<MLTrainingResult> {
    return this.post<MLTrainingResult>('/ml/train-models', request);
  }

  async predictMovements(request: MLPredictionRequest): Promise<MLPredictionResult> {
    return this.post<MLPredictionResult>('/ml/predict-movements', request);
  }

  async identifyRegimes(request: RegimeAnalysisRequest): Promise<RegimeAnalysisResult> {
    return this.post<RegimeAnalysisResult>('/ml/identify-regimes', request);
  }

  async rlOptimization(request: RLOptimizationRequest): Promise<RLOptimizationResult> {
    return this.post<RLOptimizationResult>('/ml/rl-optimization', request);
  }

  async ensemblePrediction(request: EnsemblePredictionRequest): Promise<EnsemblePredictionResult> {
    return this.post<EnsemblePredictionResult>('/ml/ensemble-prediction', request);
  }

  async getMLModelStatus(): Promise<MLModelStatus> {
    return this.get<MLModelStatus>('/ml/model-status');
  }

  // ============================================================================
  // PORTFOLIO OPTIMIZATION ENDPOINTS
  // ============================================================================

  async optimizePortfolio(request: OptimizationRequest): Promise<OptimizationResult> {
    return this.post<OptimizationResult>('/optimize-portfolio-advanced', request);
  }

  async blackLittermanOptimization(request: BlackLittermanRequest): Promise<OptimizationResult> {
    return this.post<OptimizationResult>('/black-litterman-optimization', request);
  }

  async factorBasedOptimization(request: FactorBasedRequest): Promise<OptimizationResult> {
    return this.post<OptimizationResult>('/factor-based-optimization', request);
  }

  async riskParityOptimization(request: OptimizationRequest): Promise<OptimizationResult> {
    return this.post<OptimizationResult>('/risk-parity-optimization', request);
  }

  async minimumVarianceOptimization(request: OptimizationRequest): Promise<OptimizationResult> {
    return this.post<OptimizationResult>('/minimum-variance-optimization', request);
  }

  async compareOptimizationMethods(request: {
    tickers: string[];
    methods: string[];
    lookback_days?: number;
  }): Promise<OptimizationComparison> {
    return this.post<OptimizationComparison>('/compare-optimization-methods', request);
  }

  // ============================================================================
  // RISK MANAGEMENT ENDPOINTS
  // ============================================================================

  async comprehensiveRiskManagement(request: RiskManagementRequest): Promise<RiskManagementResult> {
    return this.post<RiskManagementResult>('/comprehensive-risk-management', request);
  }

  async volatilityPositionSizing(request: VolatilityPositionSizingRequest): Promise<{
    adjusted_weights: PortfolioWeights;
    volatility_metrics: any;
  }> {
    return this.post('/volatility-position-sizing', request);
  }

  async tailRiskHedging(request: TailRiskHedgingRequest): Promise<{
    hedged_portfolio: PortfolioWeights;
    hedge_details: any;
  }> {
    return this.post('/tail-risk-hedging', request);
  }

  async drawdownControls(request: DrawdownControlsRequest): Promise<{
    controlled_weights: PortfolioWeights;
    drawdown_metrics: any;
  }> {
    return this.post('/drawdown-controls', request);
  }

  async optimizeWithRiskControls(request: RiskManagementRequest): Promise<OptimizationResult> {
    return this.post<OptimizationResult>('/optimize-with-risk-controls', request);
  }

  async getRiskManagementInfo(): Promise<{
    risk_regimes: string[];
    hedging_strategies: string[];
    drawdown_thresholds: Record<string, number>;
  }> {
    return this.get('/risk-management-info');
  }

  // ============================================================================
  // LIQUIDITY MANAGEMENT ENDPOINTS
  // ============================================================================

  async calculateCashBuffer(request: CashBufferRequest): Promise<CashBufferResult> {
    return this.post<CashBufferResult>('/calculate-cash-buffer', request);
  }

  async determineRebalancingFrequency(request: {
    portfolio_weights: PortfolioWeights;
    volatility_forecast: number;
    transaction_costs: number;
  }): Promise<{
    recommended_frequency: string;
    cost_benefit_analysis: any;
  }> {
    return this.post('/determine-rebalancing-frequency', request);
  }

  async scoreAssetLiquidity(request: LiquidityScoreRequest): Promise<LiquidityScoreResult> {
    return this.post<LiquidityScoreResult>('/score-asset-liquidity', request);
  }

  async liquidityAwareAllocation(request: {
    target_allocation: PortfolioWeights;
    market_condition: string;
    stress_test: boolean;
  }): Promise<{
    optimized_allocation: PortfolioWeights;
    liquidity_metrics: any;
  }> {
    return this.post('/liquidity-aware-allocation', request);
  }

  // ============================================================================
  // TAX OPTIMIZATION ENDPOINTS
  // ============================================================================

  async taxLossHarvesting(request: TaxLossHarvestingRequest): Promise<TaxLossHarvestingResult> {
    return this.post<TaxLossHarvestingResult>('/tax-loss-harvesting', request);
  }

  async optimizeAssetLocation(request: AssetLocationRequest): Promise<AssetLocationResult> {
    return this.post<AssetLocationResult>('/optimize-asset-location', request);
  }

  async taxAwareRebalancing(request: {
    current_allocation: PortfolioWeights;
    target_allocation: PortfolioWeights;
    cost_basis: Record<string, number>;
    account_type: string;
  }): Promise<{
    optimized_trades: any[];
    tax_impact: number;
    after_tax_improvement: number;
  }> {
    return this.post('/tax-aware-rebalancing', request);
  }

  async calculateTaxAlpha(request: {
    portfolio_weights: PortfolioWeights;
    account_types: Record<string, string>;
    tax_rates: Record<string, number>;
  }): Promise<{
    annual_tax_alpha: number;
    tax_efficiency_score: number;
    improvement_opportunities: any[];
  }> {
    return this.post('/calculate-tax-alpha', request);
  }

  // ============================================================================
  // PORTFOLIO ANALYSIS ENDPOINTS
  // ============================================================================

  async analyzePortfolio(request: {
    tickers: string[];
    weights: PortfolioWeights;
    benchmark?: string;
    lookback_days?: number;
  }): Promise<{
    portfolio_metrics: PortfolioMetrics;
    asset_metrics: any;
    risk_analysis: any;
    performance_attribution: any;
  }> {
    return this.post('/analyze-portfolio', request);
  }

  async analyzePortfolioSimple(request: {
    tickers: string[];
    weights: PortfolioWeights;
  }): Promise<{
    expected_return: number;
    volatility: number;
    sharpe_ratio: number;
  }> {
    return this.post('/analyze-portfolio-simple', request);
  }

  async analyzePortfolioEnhanced(request: {
    tickers: string[];
    weights: PortfolioWeights;
    enable_ai_insights?: boolean;
    enable_regime_analysis?: boolean;
  }): Promise<{
    basic_metrics: PortfolioMetrics;
    ai_insights?: any;
    regime_analysis?: any;
    dynamic_allocation?: any;
  }> {
    return this.post('/analyze-portfolio-enhanced', request);
  }

  async rebalancePortfolio(request: {
    current_weights: PortfolioWeights;
    target_weights: PortfolioWeights;
    portfolio_value: number;
    execution_method?: string;
  }): Promise<{
    trades: any[];
    estimated_costs: number;
    expected_slippage: number;
    execution_plan: any;
  }> {
    return this.post('/rebalance-portfolio', request);
  }

  // ============================================================================
  // DYNAMIC ALLOCATION ENDPOINTS
  // ============================================================================

  async dynamicAllocation(request: {
    tickers: string[];
    base_allocation: PortfolioWeights;
    enable_regime_detection?: boolean;
    enable_momentum_signals?: boolean;
    lookback_days?: number;
  }): Promise<{
    final_allocation: PortfolioWeights;
    regime_analysis: any;
    momentum_signals: any;
    allocation_changes: any;
  }> {
    return this.post('/dynamic-allocation', request);
  }

  async marketAnalysis(request: {
    symbols?: string[];
    lookback_days?: number;
  }): Promise<{
    market_metrics: any;
    technical_indicators: TechnicalIndicators;
    sentiment_analysis: any;
    regime_detection: any;
  }> {
    return this.post('/market-analysis', request);
  }

  // ============================================================================
  // COMPREHENSIVE AI MANAGEMENT
  // ============================================================================

  async aiPortfolioManagement(request: AIPortfolioManagementRequest): Promise<AIPortfolioManagementResult> {
    return this.post<AIPortfolioManagementResult>('/ai-portfolio-management', request);
  }

  async comprehensivePortfolioManagement(request: {
    tickers: string[];
    base_allocation: PortfolioWeights;
    portfolio_value: number;
    risk_tolerance: string;
    investment_horizon: string;
    enable_all_features?: boolean;
  }): Promise<{
    optimized_allocation: PortfolioWeights;
    ai_insights: any;
    risk_analysis: any;
    tax_optimization: any;
    liquidity_analysis: any;
    execution_plan: any;
    recommendations: string[];
  }> {
    return this.post('/comprehensive-portfolio-management', request);
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  async getMarketData(symbols: string[]): Promise<MarketData[]> {
    // This would typically be implemented with real-time market data
    // For now, we'll return a placeholder
    return symbols.map(symbol => ({
      symbol,
      price: 100 + Math.random() * 100,
      change: (Math.random() - 0.5) * 10,
      change_percent: (Math.random() - 0.5) * 0.1,
      volume: Math.floor(Math.random() * 1000000),
      timestamp: new Date().toISOString()
    }));
  }

  async getHistoricalData(symbol: string, days: number = 252): Promise<HistoricalData> {
    // Placeholder implementation
    const data = [];
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    for (let i = 0; i < days; i++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);
      
      const basePrice = 100 + Math.random() * 50;
      data.push({
        date: date.toISOString().split('T')[0],
        open: basePrice + Math.random() * 2 - 1,
        high: basePrice + Math.random() * 3,
        low: basePrice - Math.random() * 3,
        close: basePrice + Math.random() * 2 - 1,
        volume: Math.floor(Math.random() * 1000000)
      });
    }

    return { symbol, data };
  }

  // ============================================================================
  // BATCH OPERATIONS
  // ============================================================================

  async batchMLPredictions(requests: MLPredictionRequest[]): Promise<MLPredictionResult[]> {
    return Promise.all(requests.map(request => this.predictMovements(request)));
  }

  async batchOptimizations(requests: OptimizationRequest[]): Promise<OptimizationResult[]> {
    return Promise.all(requests.map(request => this.optimizePortfolio(request)));
  }

  // ============================================================================
  // WEBSOCKET SUPPORT (FUTURE)
  // ============================================================================

  private wsConnections: Map<string, WebSocket> = new Map();

  connectWebSocket(channel: string, onMessage: (data: any) => void): () => void {
    const wsUrl = `${this.baseURL.replace('http', 'ws')}/ws/${channel}`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('WebSocket message parsing error:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.wsConnections.set(channel, ws);

    // Return disconnect function
    return () => {
      ws.close();
      this.wsConnections.delete(channel);
    };
  }

  disconnectWebSocket(channel: string): void {
    const ws = this.wsConnections.get(channel);
    if (ws) {
      ws.close();
      this.wsConnections.delete(channel);
    }
  }

  disconnectAllWebSockets(): void {
    this.wsConnections.forEach(ws => ws.close());
    this.wsConnections.clear();
  }
}

// ============================================================================
// API ERROR CLASS
// ============================================================================

export class APIError extends Error {
  public readonly code: string;
  public readonly status: number;
  public readonly timestamp: string;

  constructor(message: string, code: string = 'UNKNOWN_ERROR', status: number = 0) {
    super(message);
    this.name = 'APIError';
    this.code = code;
    this.status = status;
    this.timestamp = new Date().toISOString();
  }

  toJSON() {
    return {
      name: this.name,
      message: this.message,
      code: this.code,
      status: this.status,
      timestamp: this.timestamp
    };
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

export const apiClient = new SmartPortfolioAPIClient();
export default apiClient;

// ============================================================================
// NAMED EXPORTS FOR CONVENIENCE
// ============================================================================

export const {
  // System
  getHealth,
  getReadiness,
  getCapabilities,
  getAccount,
  checkEnvironment,
  
  // ML
  trainMLModels,
  predictMovements,
  identifyRegimes,
  rlOptimization,
  ensemblePrediction,
  getMLModelStatus,
  
  // Optimization
  optimizePortfolio,
  blackLittermanOptimization,
  factorBasedOptimization,
  riskParityOptimization,
  minimumVarianceOptimization,
  compareOptimizationMethods,
  
  // Risk Management
  comprehensiveRiskManagement,
  volatilityPositionSizing,
  tailRiskHedging,
  drawdownControls,
  optimizeWithRiskControls,
  getRiskManagementInfo,
  
  // Liquidity
  calculateCashBuffer,
  determineRebalancingFrequency,
  scoreAssetLiquidity,
  liquidityAwareAllocation,
  
  // Tax
  taxLossHarvesting,
  optimizeAssetLocation,
  taxAwareRebalancing,
  calculateTaxAlpha,
  
  // Portfolio Analysis
  analyzePortfolio,
  analyzePortfolioSimple,
  analyzePortfolioEnhanced,
  rebalancePortfolio,
  
  // Dynamic Allocation
  dynamicAllocation,
  marketAnalysis,
  
  // AI Management
  aiPortfolioManagement,
  comprehensivePortfolioManagement,
  
  // Utilities
  getMarketData,
  getHistoricalData,
  
  // WebSocket
  connectWebSocket,
  disconnectWebSocket,
  disconnectAllWebSockets
} = apiClient;
