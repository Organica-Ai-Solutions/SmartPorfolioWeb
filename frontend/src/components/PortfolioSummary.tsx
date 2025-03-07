import { motion } from 'framer-motion';
import { PortfolioMetrics } from '../types/portfolio';
import { InformationCircleIcon } from '@heroicons/react/24/outline';
import { Tooltip } from './ui/tooltip';

interface PortfolioSummaryProps {
  metrics: PortfolioMetrics;
}

const formatMetric = (value: number, isPercentage = true): string => {
  if (typeof value !== 'number' || isNaN(value)) return isPercentage ? '0.00%' : '0.00';
  return isPercentage ? (value * 100).toFixed(2) + '%' : value.toFixed(2);
};

export function PortfolioSummary({ metrics }: PortfolioSummaryProps) {
  if (!metrics) {
    console.warn(`No portfolio metrics provided`);
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="bg-[#0a0a0a]/95 backdrop-blur-md border border-white/10 rounded-xl p-6 mb-6"
    >
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-lg font-bold text-blue-400">Portfolio Summary</h3>
        <Tooltip
          content="Overall metrics for your entire portfolio including risk-adjusted returns and downside risk measures."
          side="right"
        >
          <button className="p-1 rounded-full hover:bg-gray-700 transition-colors">
            <InformationCircleIcon className="h-5 w-5 text-gray-400" />
          </button>
        </Tooltip>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
        <div className="space-y-1">
          <p className="text-gray-400 text-sm">Expected Return</p>
          <p className={`text-lg font-bold ${metrics.expected_return > 0 ? 'text-green-400' : 'text-red-400'}`}>
            {formatMetric(metrics.expected_return)}
          </p>
        </div>

        <div className="space-y-1">
          <p className="text-gray-400 text-sm">Volatility</p>
          <p className="text-lg font-bold text-yellow-400">
            {formatMetric(metrics.volatility)}
          </p>
        </div>

        <div className="space-y-1">
          <p className="text-gray-400 text-sm">Sharpe Ratio</p>
          <p className={`text-lg font-bold ${metrics.sharpe_ratio > 1 ? 'text-green-400' : 'text-yellow-400'}`}>
            {formatMetric(metrics.sharpe_ratio, false)}
          </p>
        </div>

        <div className="space-y-1">
          <p className="text-gray-400 text-sm">Sortino Ratio</p>
          <p className={`text-lg font-bold ${metrics.sortino_ratio > 1 ? 'text-green-400' : 'text-yellow-400'}`}>
            {formatMetric(metrics.sortino_ratio, false)}
          </p>
        </div>

        <div className="space-y-1">
          <p className="text-gray-400 text-sm">Beta</p>
          <p className={`text-lg font-bold ${metrics.beta < 1 ? 'text-green-400' : 'text-yellow-400'}`}>
            {formatMetric(metrics.beta, false)}
          </p>
        </div>

        <div className="space-y-1">
          <p className="text-gray-400 text-sm">Max Drawdown</p>
          <p className="text-lg font-bold text-red-400">
            {formatMetric(metrics.max_drawdown)}
          </p>
        </div>

        <div className="space-y-1">
          <p className="text-gray-400 text-sm">VaR (95%)</p>
          <p className="text-lg font-bold text-red-400">
            {formatMetric(metrics.var_95)}
          </p>
        </div>

        <div className="space-y-1">
          <p className="text-gray-400 text-sm">CVaR (95%)</p>
          <p className="text-lg font-bold text-red-400">
            {formatMetric(metrics.cvar_95 || metrics.var_95 * 1.2)}
          </p>
        </div>
      </div>
    </motion.div>
  );
} 