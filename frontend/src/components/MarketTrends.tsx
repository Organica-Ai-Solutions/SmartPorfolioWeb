import { MarketTrends as MarketTrendsType } from '../types/portfolio';
import { motion } from 'framer-motion';

interface Props {
    trends: MarketTrendsType;
    ticker: string;
}

export function MarketTrends({ trends, ticker }: Props) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white/5 border border-white/10 rounded-xl p-6"
        >
            <h3 className="font-semibold mb-4 text-purple-400">
                Market Trends - {ticker}
            </h3>
            
            <div className="space-y-4">
                {/* Price and Moving Averages */}
                <div className="grid grid-cols-3 gap-4">
                    <div className="bg-white/5 rounded-lg p-3">
                        <p className="text-xs text-gray-400">Current Price</p>
                        <p className="text-lg">${trends.current_price.toFixed(2)}</p>
                    </div>
                    <div className="bg-white/5 rounded-lg p-3">
                        <p className="text-xs text-gray-400">MA20</p>
                        <p className="text-lg">${trends.ma20.toFixed(2)}</p>
                    </div>
                    <div className="bg-white/5 rounded-lg p-3">
                        <p className="text-xs text-gray-400">MA50</p>
                        <p className="text-lg">${trends.ma50.toFixed(2)}</p>
                    </div>
                </div>

                {/* Technical Indicators */}
                <div className="grid grid-cols-2 gap-4">
                    <div className="bg-white/5 rounded-lg p-3">
                        <p className="text-xs text-gray-400">RSI</p>
                        <p className={`text-lg ${
                            trends.rsi > 70 ? 'text-red-400' : 
                            trends.rsi < 30 ? 'text-green-400' : 
                            'text-gray-100'
                        }`}>
                            {trends.rsi.toFixed(2)}
                        </p>
                    </div>
                    <div className="bg-white/5 rounded-lg p-3">
                        <p className="text-xs text-gray-400">MACD</p>
                        <div className="space-y-1">
                            <p className={`text-sm ${trends.macd >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                MACD: {trends.macd.toFixed(3)}
                            </p>
                            <p className="text-sm text-gray-300">
                                Signal: {trends.macd_signal.toFixed(3)}
                            </p>
                        </div>
                    </div>
                </div>

                {/* Overall Trend */}
                <div className="bg-white/5 rounded-lg p-3">
                    <p className="text-xs text-gray-400">Trend</p>
                    <p className={`text-lg font-medium ${
                        trends.trend === 'bullish' ? 'text-green-400' : 'text-red-400'
                    }`}>
                        {trends.trend.charAt(0).toUpperCase() + trends.trend.slice(1)}
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                        Based on MA20 vs MA50 crossover
                    </p>
                </div>
            </div>
        </motion.div>
    );
} 