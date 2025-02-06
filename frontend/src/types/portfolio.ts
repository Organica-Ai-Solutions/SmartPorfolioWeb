export interface Portfolio {
    id?: number;
    user_id?: number;
    name?: string;
    tickers: string[];
    start_date: string;
    risk_tolerance: 'low' | 'medium' | 'high';
    created_at?: string;
    updated_at?: string;
}

export interface PortfolioMetrics {
    expected_return: number;
    volatility: number;
    sharpe_ratio: number;
    sortino_ratio: number;
    beta: number;
    max_drawdown: number;
    var_95: number;
    cvar_95: number;
}

export interface AssetMetrics {
    annual_return: number;
    annual_volatility: number;
    beta: number;
    weight: number;
    alpha: number;
    volatility: number;
    var_95: number;
    max_drawdown: number;
    correlation: number;
}

export interface TechnicalIndicators {
    bollinger_bands: {
        middle_band: number;
        upper_band: number;
        lower_band: number;
    };
    momentum: number;
    volume_analysis: {
        average_volume: number;
        volume_trend: 'increasing' | 'decreasing';
    };
}

export interface MarketTrends {
    current_price: number;
    ma20: number;
    ma50: number;
    rsi: number;
    macd: number;
    macd_signal: number;
    trend: 'bullish' | 'bearish';
}

export interface MarketAnalysis {
    market_trends: { [key: string]: MarketTrends };
    volatility_analysis: {
        [key: string]: {
            daily_volatility: number;
            annualized_volatility: number;
            high_low_range: number;
            average_true_range: number;
        };
    };
    correlation_analysis: {
        correlation_matrix: { [key: string]: { [key: string]: number } };
        explained_variance_ratio: number[];
        principal_components: number[][];
    };
    technical_indicators: { [key: string]: TechnicalIndicators };
    market_regime: {
        regime: 'normal' | 'high_volatility' | 'risk_off' | 'bear_market' | 'bull_market';
        metrics: {
            average_volatility: number;
            average_correlation: number;
            market_trend: number;
        };
    };
}

export interface PortfolioAnalysis {
    allocations: { [key: string]: number };
    metrics: PortfolioMetrics;
    asset_metrics: { [key: string]: AssetMetrics };
    discrete_allocation: {
        shares: { [key: string]: number };
        leftover: number;
    };
    historical_performance: {
        dates: string[];
        portfolio_values: number[];
        drawdowns: number[];
        rolling_volatility: number[];
        rolling_sharpe: number[];
    };
    market_comparison: {
        dates: string[];
        market_values: number[];
        relative_performance: number[];
    };
    ai_insights?: {
        portfolio_analysis: {
            risk_metrics: any;
            diversification_metrics: any;
        };
        market_analysis: any;
        risk_analysis: any;
        explanations: {
            summary: {
                en: string;
                es: string;
            };
            risk_analysis: {
                en: string;
                es: string;
            };
            diversification_analysis: {
                en: string;
                es: string;
            };
            market_context: {
                en: string;
                es: string;
            };
            stress_test_interpretation: {
                en: string;
                es: string;
            };
        };
        recommendations: string[];
        market_outlook: {
            short_term: string;
            medium_term: string;
            long_term: string;
            key_drivers: string[];
        };
    };
    market_analysis?: MarketAnalysis;
}

export interface Order {
    symbol: string;
    qty: number;
    side: 'buy' | 'sell';
    status?: 'executed' | 'failed';
    order_id?: string;
    error?: string;
}

export interface RebalanceResult {
    message: string;
    orders: Order[];
    account_balance: {
        equity: number;
        cash: number;
        buying_power: number;
    };
}

export interface RiskAlert {
    id: number;
    portfolio_id: number;
    timestamp: string;
    alert_type: string;
    severity: 'low' | 'medium' | 'high';
    message: string;
    metrics: { [key: string]: number };
}

export interface Notification {
    id: number;
    user_id: number;
    timestamp: string;
    type: string;
    title: string;
    message: string;
    read: boolean;
} 