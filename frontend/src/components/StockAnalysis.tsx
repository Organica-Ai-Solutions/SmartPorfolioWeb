import { motion } from 'framer-motion';
import { AssetMetrics } from '../types/portfolio';

interface StockAnalysisProps {
  ticker: string;
  metrics: AssetMetrics;
}

export function StockAnalysis({ ticker, metrics }: StockAnalysisProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white/5 border border-white/10 rounded-xl p-6"
    >
      <h3 className="font-semibold mb-4 text-purple-400">
        {ticker} Analysis
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Returns and Risk */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-gray-300">Returns & Risk</h4>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white/5 rounded-lg p-3">
              <p className="text-xs text-gray-400">Annual Return</p>
              <p className={`text-lg ${metrics.annual_return >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {(metrics.annual_return * 100).toFixed(2)}%
              </p>
            </div>
            <div className="bg-white/5 rounded-lg p-3">
              <p className="text-xs text-gray-400">Volatility</p>
              <p className="text-lg">
                {(metrics.volatility * 100).toFixed(2)}%
              </p>
            </div>
          </div>
        </div>

        {/* Market Metrics */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-gray-300">Market Metrics</h4>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white/5 rounded-lg p-3">
              <p className="text-xs text-gray-400">Beta</p>
              <p className={`text-lg ${
                metrics.beta > 1.2 ? 'text-red-400' : 
                metrics.beta < 0.8 ? 'text-green-400' : 
                'text-gray-100'
              }`}>
                {metrics.beta.toFixed(2)}
              </p>
            </div>
            <div className="bg-white/5 rounded-lg p-3">
              <p className="text-xs text-gray-400">Alpha</p>
              <p className={`text-lg ${metrics.alpha >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {(metrics.alpha * 100).toFixed(2)}%
              </p>
            </div>
          </div>
        </div>

        {/* Risk Metrics */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-gray-300">Risk Analysis</h4>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white/5 rounded-lg p-3">
              <p className="text-xs text-gray-400">VaR (95%)</p>
              <p className="text-lg text-red-400">
                {(metrics.var_95 * 100).toFixed(2)}%
              </p>
            </div>
            <div className="bg-white/5 rounded-lg p-3">
              <p className="text-xs text-gray-400">Max Drawdown</p>
              <p className="text-lg text-red-400">
                {(metrics.max_drawdown * 100).toFixed(2)}%
              </p>
            </div>
          </div>
        </div>

        {/* Portfolio Context */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-gray-300">Portfolio Context</h4>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white/5 rounded-lg p-3">
              <p className="text-xs text-gray-400">Weight</p>
              <p className="text-lg">
                {(metrics.weight * 100).toFixed(2)}%
              </p>
            </div>
            <div className="bg-white/5 rounded-lg p-3">
              <p className="text-xs text-gray-400">Correlation</p>
              <p className={`text-lg ${
                metrics.correlation > 0.7 ? 'text-red-400' : 
                metrics.correlation < 0.3 ? 'text-green-400' : 
                'text-gray-100'
              }`}>
                {metrics.correlation.toFixed(2)}
              </p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
} 