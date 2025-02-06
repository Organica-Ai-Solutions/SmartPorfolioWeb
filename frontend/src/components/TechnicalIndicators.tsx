import { TechnicalIndicators as TechnicalIndicatorsType } from '../types/portfolio';
import { motion } from 'framer-motion';

interface Props {
    indicators: TechnicalIndicatorsType;
    ticker: string;
}

export function TechnicalIndicators({ indicators, ticker }: Props) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white/5 border border-white/10 rounded-xl p-6"
        >
            <h3 className="font-semibold mb-4 text-purple-400">
                Technical Indicators - {ticker}
            </h3>
            
            <div className="space-y-4">
                {/* Bollinger Bands */}
                <div>
                    <h4 className="text-sm font-medium text-gray-300 mb-2">Bollinger Bands</h4>
                    <div className="grid grid-cols-3 gap-4">
                        <div className="bg-white/5 rounded-lg p-3">
                            <p className="text-xs text-gray-400">Upper</p>
                            <p className="text-lg">${indicators.bollinger_bands.upper_band.toFixed(2)}</p>
                        </div>
                        <div className="bg-white/5 rounded-lg p-3">
                            <p className="text-xs text-gray-400">Middle</p>
                            <p className="text-lg">${indicators.bollinger_bands.middle_band.toFixed(2)}</p>
                        </div>
                        <div className="bg-white/5 rounded-lg p-3">
                            <p className="text-xs text-gray-400">Lower</p>
                            <p className="text-lg">${indicators.bollinger_bands.lower_band.toFixed(2)}</p>
                        </div>
                    </div>
                </div>

                {/* Momentum */}
                <div>
                    <h4 className="text-sm font-medium text-gray-300 mb-2">Momentum (10-day)</h4>
                    <div className="bg-white/5 rounded-lg p-3">
                        <p className={`text-lg ${indicators.momentum >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                            {(indicators.momentum * 100).toFixed(2)}%
                        </p>
                    </div>
                </div>

                {/* Volume Analysis */}
                <div>
                    <h4 className="text-sm font-medium text-gray-300 mb-2">Volume Analysis</h4>
                    <div className="grid grid-cols-2 gap-4">
                        <div className="bg-white/5 rounded-lg p-3">
                            <p className="text-xs text-gray-400">Average Volume</p>
                            <p className="text-lg">
                                {(indicators.volume_analysis.average_volume / 1000000).toFixed(2)}M
                            </p>
                        </div>
                        <div className="bg-white/5 rounded-lg p-3">
                            <p className="text-xs text-gray-400">Volume Trend</p>
                            <p className={`text-lg ${
                                indicators.volume_analysis.volume_trend === 'increasing' 
                                ? 'text-green-400' 
                                : 'text-red-400'
                            }`}>
                                {indicators.volume_analysis.volume_trend.charAt(0).toUpperCase() + 
                                 indicators.volume_analysis.volume_trend.slice(1)}
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </motion.div>
    );
} 