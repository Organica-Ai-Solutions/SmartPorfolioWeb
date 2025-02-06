import { motion } from 'framer-motion';

interface MarketRegimeProps {
    regime: 'normal' | 'high_volatility' | 'risk_off' | 'bear_market' | 'bull_market';
    metrics: {
        average_volatility: number;
        average_correlation: number;
        market_trend: number;
    };
}

export function MarketRegime({ regime, metrics }: MarketRegimeProps) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white/5 border border-white/10 rounded-xl p-6"
        >
            <h3 className="font-semibold mb-4 text-purple-400">Market Regime</h3>
            
            {/* Current Regime */}
            <div className="mb-6">
                <div className={`inline-flex items-center px-4 py-2 rounded-full ${getRegimeColor(regime)}`}>
                    <span className="text-sm font-medium">
                        {formatRegimeName(regime)}
                    </span>
                </div>
                <p className="mt-2 text-sm text-gray-400">
                    {getRegimeDescription(regime)}
                </p>
            </div>

            {/* Regime Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white/5 rounded-lg p-3">
                    <p className="text-xs text-gray-400">Average Volatility</p>
                    <p className={`text-lg ${metrics.average_volatility > 0.25 ? 'text-red-400' : 'text-green-400'}`}>
                        {(metrics.average_volatility * 100).toFixed(2)}%
                    </p>
                </div>

                <div className="bg-white/5 rounded-lg p-3">
                    <p className="text-xs text-gray-400">Average Correlation</p>
                    <p className={`text-lg ${metrics.average_correlation > 0.7 ? 'text-red-400' : 'text-green-400'}`}>
                        {metrics.average_correlation.toFixed(2)}
                    </p>
                </div>

                <div className="bg-white/5 rounded-lg p-3">
                    <p className="text-xs text-gray-400">Market Trend</p>
                    <p className={`text-lg ${metrics.market_trend >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {(metrics.market_trend * 100).toFixed(2)}%
                    </p>
                </div>
            </div>

            {/* Regime Implications */}
            <div className="mt-6">
                <h4 className="text-sm font-medium text-gray-300 mb-3">Investment Implications</h4>
                <div className="bg-white/5 rounded-lg p-4">
                    <ul className="space-y-2 text-sm text-gray-300">
                        {getRegimeImplications(regime).map((implication, index) => (
                            <li key={index} className="flex items-start">
                                <span className="mr-2">•</span>
                                {implication}
                            </li>
                        ))}
                    </ul>
                </div>
            </div>
        </motion.div>
    );
}

function getRegimeColor(regime: string): string {
    switch (regime) {
        case 'normal':
            return 'bg-blue-500/20 text-blue-400';
        case 'high_volatility':
            return 'bg-red-500/20 text-red-400';
        case 'risk_off':
            return 'bg-yellow-500/20 text-yellow-400';
        case 'bear_market':
            return 'bg-red-500/20 text-red-400';
        case 'bull_market':
            return 'bg-green-500/20 text-green-400';
        default:
            return 'bg-gray-500/20 text-gray-400';
    }
}

function formatRegimeName(regime: string): string {
    return regime
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function getRegimeDescription(regime: string): string {
    switch (regime) {
        case 'normal':
            return 'Market conditions are stable with typical levels of volatility and correlation.';
        case 'high_volatility':
            return 'Market is experiencing elevated levels of price fluctuations and uncertainty.';
        case 'risk_off':
            return 'Investors are showing preference for safer assets, with high correlation across markets.';
        case 'bear_market':
            return 'Market is in a sustained downtrend with negative sentiment.';
        case 'bull_market':
            return 'Market is in a sustained uptrend with positive sentiment.';
        default:
            return 'Market conditions are being analyzed.';
    }
}

function getRegimeImplications(regime: string): string[] {
    switch (regime) {
        case 'normal':
            return [
                'Maintain standard asset allocation',
                'Focus on stock selection',
                'Regular rebalancing schedule'
            ];
        case 'high_volatility':
            return [
                'Consider reducing position sizes',
                'Increase cash holdings',
                'Focus on quality stocks with low beta',
                'Consider hedging strategies'
            ];
        case 'risk_off':
            return [
                'Reduce exposure to high-beta assets',
                'Increase allocation to defensive sectors',
                'Consider safe-haven assets',
                'Monitor correlation changes'
            ];
        case 'bear_market':
            return [
                'Defensive positioning',
                'Focus on capital preservation',
                'Consider inverse ETFs or puts',
                'Look for counter-trend opportunities'
            ];
        case 'bull_market':
            return [
                'Consider increasing equity exposure',
                'Focus on growth sectors',
                'Monitor for signs of market excess',
                'Stay invested but vigilant'
            ];
        default:
            return [
                'Maintain balanced portfolio',
                'Monitor market conditions',
                'Stay prepared for changes'
            ];
    }
} 