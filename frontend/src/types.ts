export interface AIInsights {
  portfolio_analysis: {
    risk_metrics: {
      sortino_ratio: number;
      treynor_ratio: number;
      information_ratio: number;
      max_drawdown: number;
      var_95: number;
      cvar_95: number;
    };
    diversification_metrics: {
      concentration_ratio: number;
      effective_num_assets: number;
      sector_exposure: Record<string, number>;
      geographic_exposure: Record<string, number>;
    };
  };
  market_analysis: {
    market_regime: string;
    volatility_regime: string;
    correlation_regime: string;
    trend_strength: string;
  };
  risk_analysis: {
    risk_decomposition: {
      systematic_risk: number;
      specific_risk: number;
    };
    stress_test_results: {
      market_crash: number;
      interest_rate_shock: number;
      volatility_spike: number;
    };
    risk_concentration: {
      asset_concentration: string;
      sector_concentration: string;
      factor_concentration: string;
    };
  };
  portfolio_characteristics: {
    style_exposure: Record<string, number>;
    factor_exposure: Record<string, number>;
  };
  explanations: {
    summary: { en: string; es: string };
    risk_analysis: { en: string; es: string };
    diversification_analysis: { en: string; es: string };
    market_context: { en: string; es: string };
    stress_test_interpretation: { en: string; es: string };
  };
  recommendations: string[];
  optimization_suggestions: string[];
  market_outlook: {
    short_term: string;
    medium_term: string;
    long_term: string;
    key_drivers: string[];
  };
} 