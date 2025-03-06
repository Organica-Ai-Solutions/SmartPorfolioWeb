import React from 'react'
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

type PortfolioExplanationProps = {
  analysis: PortfolioAnalysis
  language?: 'en' | 'es'
}

export function PortfolioExplanation({ analysis, language = 'en' }: PortfolioExplanationProps) {
  const formatMetric = (value: number, isPercentage = false) => {
    return isPercentage ? `${(value * 100).toFixed(2)}%` : value.toFixed(2);
  }

  // Basic explanation without AI insights
  const generateBasicExplanation = () => {
    const metrics = analysis.metrics;
    const diversificationLevel = Object.keys(analysis.allocations || {}).length > 3 ? 'well' : 'poorly';
    const riskLevel = metrics.volatility < 0.15 ? 'low' : metrics.volatility < 0.25 ? 'moderate' : 'high';
    const performanceLevel = metrics.sharpe_ratio > 1 ? 'good' : 'suboptimal';
    
    const explanations = {
      en: `Your portfolio shows ${riskLevel} risk with ${performanceLevel} risk-adjusted returns (Sharpe ratio: ${metrics.sharpe_ratio?.toFixed(2) || 'N/A'}). It is ${diversificationLevel} diversified across ${Object.keys(analysis.allocations || {}).length} assets. The expected annual return is ${formatMetric(metrics.expected_return, true)} with ${formatMetric(metrics.volatility, true)} volatility.`,
      es: `Su cartera muestra un riesgo ${riskLevel === 'low' ? 'bajo' : riskLevel === 'moderate' ? 'moderado' : 'alto'} con rendimientos ajustados por riesgo ${performanceLevel === 'good' ? 'buenos' : 'subóptimos'} (Ratio de Sharpe: ${metrics.sharpe_ratio?.toFixed(2) || 'N/A'}). Está ${diversificationLevel === 'well' ? 'bien' : 'pobremente'} diversificado entre ${Object.keys(analysis.allocations || {}).length} activos. El retorno anual esperado es ${formatMetric(metrics.expected_return, true)} con una volatilidad de ${formatMetric(metrics.volatility, true)}.`
    };
    
    return explanations[language];
  }
  
  // Risk explanation without AI insights
  const generateRiskExplanation = () => {
    const metrics = analysis.metrics;
    
    const explanations = {
      en: `The portfolio has a Sharpe ratio of ${metrics.sharpe_ratio?.toFixed(2) || 'N/A'} and a Sortino ratio of ${metrics.sortino_ratio?.toFixed(2) || 'N/A'}, indicating ${metrics.sharpe_ratio && metrics.sharpe_ratio > 1 ? 'good' : 'suboptimal'} risk-adjusted returns. Maximum drawdown is ${formatMetric(metrics.max_drawdown, true)}, with a market beta of ${metrics.beta?.toFixed(2) || 'N/A'}. Value at Risk (95%) suggests a maximum daily loss of ${formatMetric(metrics.var_95, true)} under normal market conditions.`,
      es: `El portafolio tiene un ratio de Sharpe de ${metrics.sharpe_ratio?.toFixed(2) || 'N/A'} y un ratio de Sortino de ${metrics.sortino_ratio?.toFixed(2) || 'N/A'}, indicando rendimientos ajustados por riesgo ${metrics.sharpe_ratio && metrics.sharpe_ratio > 1 ? 'buenos' : 'subóptimos'}. La máxima caída es del ${formatMetric(metrics.max_drawdown, true)}, con un beta de mercado de ${metrics.beta?.toFixed(2) || 'N/A'}. El Valor en Riesgo (95%) sugiere una pérdida diaria máxima del ${formatMetric(metrics.var_95, true)} en condiciones normales de mercado.`
    };
    
    return explanations[language];
  }
  
  // Check if AI insights are available
  const hasAiInsights = analysis.ai_insights && 
                       !analysis.ai_insights.error;
  
  // Get AI explanations if available
  const aiExplanations = hasAiInsights ? 
    analysis.ai_insights?.explanations?.[language === 'en' ? 'english' : 'spanish'] : 
    null;
  
  // Get AI recommendations if available
  const aiRecommendations = hasAiInsights ?
    analysis.ai_insights?.recommendations || [] :
    [];
  
  // Get market outlook if available
  const marketOutlook = hasAiInsights ?
    analysis.ai_insights?.market_outlook || {} :
    null;

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
            <div>
              <h3 className="text-xl font-semibold mb-2">Portfolio Summary</h3>
              <p className="text-gray-600 dark:text-gray-300">
                {hasAiInsights && aiExplanations?.summary ? 
                  aiExplanations.summary : 
                  generateBasicExplanation()}
              </p>
            </div>
            
            <div>
              <h3 className="text-xl font-semibold mb-2">Risk Analysis</h3>
              <p className="text-gray-600 dark:text-gray-300">
                {hasAiInsights && aiExplanations?.risk_analysis ? 
                  aiExplanations.risk_analysis : 
                  generateRiskExplanation()}
              </p>
            </div>
            
            {hasAiInsights && aiExplanations?.diversification_analysis && (
              <div>
                <h3 className="text-xl font-semibold mb-2">Diversification Analysis</h3>
                <p className="text-gray-600 dark:text-gray-300">
                  {aiExplanations.diversification_analysis}
                </p>
              </div>
            )}
            
            {hasAiInsights && aiExplanations?.market_analysis && (
              <div>
                <h3 className="text-xl font-semibold mb-2">Market Analysis</h3>
                <p className="text-gray-600 dark:text-gray-300">
                  {aiExplanations.market_analysis}
                </p>
              </div>
            )}
            
            {hasAiInsights && aiRecommendations.length > 0 && (
              <div>
                <h3 className="text-xl font-semibold mb-2">Recommendations</h3>
                <ul className="list-disc pl-5 space-y-1 text-gray-600 dark:text-gray-300">
                  {aiRecommendations.map((rec, index) => (
                    <li key={index}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {hasAiInsights && marketOutlook && (
              <div>
                <h3 className="text-xl font-semibold mb-2">Market Outlook</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
                    <h4 className="font-medium mb-1">Short Term</h4>
                    <p className="text-sm">{marketOutlook.short_term || 'No data available'}</p>
                  </div>
                  <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
                    <h4 className="font-medium mb-1">Medium Term</h4>
                    <p className="text-sm">{marketOutlook.medium_term || 'No data available'}</p>
                  </div>
                  <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
                    <h4 className="font-medium mb-1">Long Term</h4>
                    <p className="text-sm">{marketOutlook.long_term || 'No data available'}</p>
                  </div>
                </div>
              </div>
            )}
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