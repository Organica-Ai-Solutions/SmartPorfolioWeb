import { motion } from 'framer-motion';
import { AssetMetrics } from '../types/portfolio';

interface StockAnalysisProps {
  ticker: string;
  metrics: AssetMetrics;
}

const formatMetric = (value: number): string => {
  if (typeof value !== 'number' || isNaN(value)) return '0.00%';
  return (value * 100).toFixed(2) + '%';
};

export function StockAnalysis({ ticker, metrics }: StockAnalysisProps) {
  if (!metrics) {
    console.warn(`No metrics provided for ticker ${ticker}`);
    return null;
  }

  // Ensure all required metrics exist with fallbacks
  const safeMetrics = {
    annual_return: Number(metrics.annual_return) || 0,
    volatility: Number(metrics.volatility) || 0,
    beta: Number(metrics.beta) || 0,
    weight: Number(metrics.weight) || 0,
    alpha: Number(metrics.alpha) || 0,
    var_95: Number(metrics.var_95) || 0,
    max_drawdown: Number(metrics.max_drawdown) || 0,
    correlation: Number(metrics.correlation) || 0
  };

  console.log('Rendering StockAnalysis with safe metrics:', ticker, safeMetrics);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="bg-[#2a2a2a]/95 backdrop-blur-md border border-white/20 rounded-xl p-6"
    >
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-lg font-bold text-purple-400">{ticker}</h3>
        <span className={safeMetrics.annual_return > 0 ? 'text-green-400' : 'text-red-400'}>
          {formatMetric(safeMetrics.annual_return)}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Returns and Risk */}
        <div className="space-y-2">
          <p className="text-sm text-gray-400">Annual Return</p>
          <p className={`text-lg font-semibold ${safeMetrics.annual_return > 0 ? 'text-green-400' : 'text-red-400'}`}>
            {formatMetric(safeMetrics.annual_return)}
          </p>
        </div>
        <div className="space-y-2">
          <p className="text-sm text-gray-400">Volatility</p>
          <p className={`text-lg font-semibold ${safeMetrics.volatility < 0.2 ? 'text-green-400' : 'text-yellow-400'}`}>
            {formatMetric(safeMetrics.volatility)}
          </p>
        </div>

        {/* Market Metrics */}
        <div className="space-y-2">
          <p className="text-sm text-gray-400">Beta</p>
          <p className={`text-lg font-semibold ${
            safeMetrics.beta > 1.2 ? 'text-red-400' : 
            safeMetrics.beta < 0.8 ? 'text-green-400' : 
            'text-white'
          }`}>
            {safeMetrics.beta.toFixed(2)}
          </p>
        </div>
        <div className="space-y-2">
          <p className="text-sm text-gray-400">Weight</p>
          <p className="text-lg font-semibold text-white">
            {formatMetric(safeMetrics.weight)}
          </p>
        </div>

        {/* Additional Metrics */}
        <div className="space-y-2">
          <p className="text-sm text-gray-400">Alpha</p>
          <p className={`text-lg font-semibold ${safeMetrics.alpha > 0 ? 'text-green-400' : 'text-red-400'}`}>
            {formatMetric(safeMetrics.alpha)}
          </p>
        </div>
        <div className="space-y-2">
          <p className="text-sm text-gray-400">Max Drawdown</p>
          <p className="text-lg font-semibold text-red-400">
            {formatMetric(safeMetrics.max_drawdown)}
          </p>
        </div>
      </div>
    </motion.div>
  );
} 