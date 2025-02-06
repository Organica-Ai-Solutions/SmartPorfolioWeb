import { Button } from "./ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog";
import { PortfolioAnalysis } from "../types/portfolio";

interface PortfolioExplanationProps {
  analysis: PortfolioAnalysis;
}

export function PortfolioExplanation({ analysis }: PortfolioExplanationProps) {
  const explanations = analysis.ai_insights?.explanations;
  const metrics = analysis.metrics;

  // Return null if no AI insights are available
  if (!analysis.ai_insights || !explanations) return null;

  // Format metrics for display
  const formatMetric = (value: number, isPercentage = true) => {
    return isPercentage ? `${(value * 100).toFixed(2)}%` : value.toFixed(2);
  };

  // Generate summary based on actual metrics
  const generateSummary = () => {
    const riskLevel = metrics.volatility > 0.3 ? "high" : metrics.volatility > 0.15 ? "moderate" : "low";
    const performanceLevel = metrics.sharpe_ratio > 1.5 ? "strong" : metrics.sharpe_ratio > 1 ? "good" : "poor";
    const diversificationLevel = analysis.allocations && Object.keys(analysis.allocations).length > 5 ? "well" : "poorly";
    
    return {
      en: `Your portfolio shows ${riskLevel} risk with ${performanceLevel} risk-adjusted returns (Sharpe ratio: ${metrics.sharpe_ratio.toFixed(2)}). It is ${diversificationLevel} diversified across ${Object.keys(analysis.allocations || {}).length} assets. The expected annual return is ${formatMetric(metrics.expected_return)} with ${formatMetric(metrics.volatility)} volatility.`,
      es: `Su portafolio muestra un riesgo ${riskLevel === 'high' ? 'alto' : riskLevel === 'moderate' ? 'moderado' : 'bajo'} con rendimientos ajustados por riesgo ${performanceLevel === 'strong' ? 'fuertes' : performanceLevel === 'good' ? 'buenos' : 'pobres'} (Ratio de Sharpe: ${metrics.sharpe_ratio.toFixed(2)}). Está ${diversificationLevel === 'well' ? 'bien' : 'pobremente'} diversificado entre ${Object.keys(analysis.allocations || {}).length} activos. El retorno anual esperado es ${formatMetric(metrics.expected_return)} con una volatilidad del ${formatMetric(metrics.volatility)}.`
    };
  };

  // Generate risk analysis based on actual metrics
  const generateRiskAnalysis = () => {
    return {
      en: `The portfolio has a Sharpe ratio of ${metrics.sharpe_ratio.toFixed(2)} and a Sortino ratio of ${metrics.sortino_ratio.toFixed(2)}, indicating ${metrics.sharpe_ratio > 1 ? 'good' : 'suboptimal'} risk-adjusted returns. Maximum drawdown is ${formatMetric(metrics.max_drawdown)}, with a market beta of ${metrics.beta.toFixed(2)}. Value at Risk (95%) suggests a maximum daily loss of ${formatMetric(metrics.var_95)} under normal market conditions.`,
      es: `El portafolio tiene un ratio de Sharpe de ${metrics.sharpe_ratio.toFixed(2)} y un ratio de Sortino de ${metrics.sortino_ratio.toFixed(2)}, indicando rendimientos ajustados por riesgo ${metrics.sharpe_ratio > 1 ? 'buenos' : 'subóptimos'}. La máxima caída es del ${formatMetric(metrics.max_drawdown)}, con un beta de mercado de ${metrics.beta.toFixed(2)}. El Valor en Riesgo (95%) sugiere una pérdida diaria máxima del ${formatMetric(metrics.var_95)} en condiciones normales de mercado.`
    };
  };

  // Ensure all required sections exist with actual data
  const sections = {
    summary: generateSummary(),
    risk_analysis: generateRiskAnalysis(),
    diversification_analysis: explanations.diversification_analysis || {
      en: `Portfolio consists of ${Object.keys(analysis.allocations || {}).length} assets with varying weights.`,
      es: `El portafolio consiste en ${Object.keys(analysis.allocations || {}).length} activos con pesos variables.`
    },
    market_context: explanations.market_context || {
      en: 'Current market conditions and implications for the portfolio.',
      es: 'Condiciones actuales del mercado e implicaciones para el portafolio.'
    },
    stress_test_interpretation: {
      en: `Under stress scenarios, the portfolio's Value at Risk (95%) is ${formatMetric(metrics.var_95)}, indicating potential losses in extreme market conditions.`,
      es: `Bajo escenarios de estrés, el Valor en Riesgo del portafolio (95%) es ${formatMetric(metrics.var_95)}, indicando pérdidas potenciales en condiciones extremas de mercado.`
    }
  };

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" className="w-full md:w-auto">
          <span className="mr-2">✨</span> AI Explanation
        </Button>
      </DialogTrigger>
      <DialogContent className="flex flex-col gap-0 p-0 sm:max-h-[min(640px,80vh)] sm:max-w-[800px]">
        <div className="overflow-y-auto">
          <DialogHeader className="px-6 pt-6">
            <DialogTitle className="text-xl bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
              Portfolio Analysis Explanation
            </DialogTitle>
            <DialogDescription className="text-base text-gray-300">
              Detailed insights and recommendations for your portfolio
            </DialogDescription>
          </DialogHeader>

          <div className="p-6 space-y-6">
            {/* Summary Section */}
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-purple-400">Summary</h3>
              <div className="bg-white/5 rounded-lg p-4 space-y-2">
                <p className="text-white">{sections.summary.en}</p>
                <p className="text-gray-400 italic">{sections.summary.es}</p>
              </div>
            </div>

            {/* Risk Analysis Section */}
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-purple-400">Risk Analysis</h3>
              <div className="bg-white/5 rounded-lg p-4 space-y-2">
                <p className="text-white">{sections.risk_analysis.en}</p>
                <p className="text-gray-400 italic">{sections.risk_analysis.es}</p>
              </div>
            </div>

            {/* Diversification Analysis Section */}
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-purple-400">Diversification Analysis</h3>
              <div className="bg-white/5 rounded-lg p-4 space-y-2">
                <p className="text-white">{sections.diversification_analysis.en}</p>
                <p className="text-gray-400 italic">{sections.diversification_analysis.es}</p>
              </div>
            </div>

            {/* Market Context Section */}
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-purple-400">Market Context</h3>
              <div className="bg-white/5 rounded-lg p-4 space-y-2">
                <p className="text-white">{sections.market_context.en}</p>
                <p className="text-gray-400 italic">{sections.market_context.es}</p>
              </div>
            </div>

            {/* Stress Test Interpretation Section */}
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-purple-400">Stress Test Analysis</h3>
              <div className="bg-white/5 rounded-lg p-4 space-y-2">
                <p className="text-white">{sections.stress_test_interpretation.en}</p>
                <p className="text-gray-400 italic">{sections.stress_test_interpretation.es}</p>
              </div>
            </div>
          </div>
        </div>

        <DialogFooter className="border-t border-white/10 px-6 py-4">
          <DialogClose asChild>
            <Button type="button" variant="outline">Close</Button>
          </DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
} 