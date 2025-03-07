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
import { motion } from 'framer-motion';
import { 
  ChartBarIcon, 
  ShieldExclamationIcon,
  ArrowPathIcon,
  GlobeAmericasIcon,
  LightBulbIcon,
  ChartBarSquareIcon
} from '@heroicons/react/24/outline';

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

  // Generate diversification explanation without AI insights
  const generateDiversificationExplanation = () => {
    const assetCount = Object.keys(analysis.allocations || {}).length;
    const isWellDiversified = assetCount > 3;
    const largestHolding = Math.max(...Object.values(analysis.allocations || {}).map(v => Number(v) || 0));
    const hasConcenrationRisk = largestHolding > 0.3; // 30% in a single asset
    
    const explanations = {
      en: `Your portfolio includes ${assetCount} different assets, which provides ${isWellDiversified ? 'good' : 'limited'} diversification benefits. ${hasConcenrationRisk ? 'There is concentration risk with a significant portion allocated to a single asset.' : 'The allocation is well balanced across multiple assets, reducing specific risk.'} Optimal diversification depends on correlation between assets and exposure to different sectors and geographies.`,
      es: `Su cartera incluye ${assetCount} activos diferentes, lo que proporciona beneficios de diversificación ${isWellDiversified ? 'buenos' : 'limitados'}. ${hasConcenrationRisk ? 'Existe riesgo de concentración con una porción significativa asignada a un solo activo.' : 'La asignación está bien equilibrada entre múltiples activos, reduciendo el riesgo específico.'} La diversificación óptima depende de la correlación entre activos y la exposición a diferentes sectores y geografías.`
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

  // Generate action items based on portfolio analysis
  const generateActionItems = () => {
    const metrics = analysis.metrics;
    const actionItems = [];
    
    if (metrics.sharpe_ratio < 0.5) {
      actionItems.push({
        en: "Consider revising your asset allocation to improve risk-adjusted returns",
        es: "Considere revisar su asignación de activos para mejorar los rendimientos ajustados por riesgo"
      });
    }
    
    if (metrics.volatility > 0.25) {
      actionItems.push({
        en: "Your portfolio shows high volatility, consider adding more stable assets",
        es: "Su cartera muestra alta volatilidad, considere agregar activos más estables"
      });
    }
    
    if (Object.keys(analysis.allocations || {}).length < 4) {
      actionItems.push({
        en: "Increase diversification by adding more uncorrelated assets",
        es: "Aumente la diversificación agregando más activos no correlacionados"
      });
    }
    
    if (metrics.beta > 1.2) {
      actionItems.push({
        en: "Your portfolio is highly sensitive to market movements, consider reducing beta",
        es: "Su cartera es muy sensible a los movimientos del mercado, considere reducir el beta"
      });
    }
    
    return actionItems.map(item => item[language as 'en' | 'es']);
  };

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" className="w-full md:w-auto">
          <LightBulbIcon className="h-5 w-5 mr-2" /> AI Analysis
        </Button>
      </DialogTrigger>
      <DialogContent className="flex flex-col gap-0 p-0 bg-[#121212] sm:max-h-[min(720px,80vh)] sm:max-w-[800px]">
        <div className="overflow-y-auto">
          <DialogHeader className="px-6 pt-6 pb-2">
            <DialogTitle className="text-xl md:text-2xl bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
              AI-Powered Portfolio Analysis
            </DialogTitle>
            <DialogDescription className="text-base text-gray-300">
              Comprehensive insights and actionable recommendations for your portfolio
            </DialogDescription>
          </DialogHeader>

          <div className="p-6 space-y-8">
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="bg-[#1a1a1a] rounded-xl p-5 border border-purple-500/20"
            >
              <div className="flex items-start mb-3">
                <ChartBarIcon className="h-6 w-6 text-purple-400 mr-2 mt-1" />
                <h3 className="text-xl font-semibold text-white">Portfolio Summary</h3>
              </div>
              <p className="text-gray-300 leading-relaxed">
                {hasAiInsights && aiExplanations?.summary ? 
                  aiExplanations.summary : 
                  generateBasicExplanation()}
              </p>
            </motion.div>
            
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="bg-[#1a1a1a] rounded-xl p-5 border border-yellow-500/20"
            >
              <div className="flex items-start mb-3">
                <ShieldExclamationIcon className="h-6 w-6 text-yellow-400 mr-2 mt-1" />
                <h3 className="text-xl font-semibold text-white">Risk Analysis</h3>
              </div>
              <p className="text-gray-300 leading-relaxed">
                {hasAiInsights && aiExplanations?.risk_analysis ? 
                  aiExplanations.risk_analysis : 
                  generateRiskExplanation()}
              </p>
            </motion.div>
            
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="bg-[#1a1a1a] rounded-xl p-5 border border-blue-500/20"
            >
              <div className="flex items-start mb-3">
                <ArrowPathIcon className="h-6 w-6 text-blue-400 mr-2 mt-1" />
                <h3 className="text-xl font-semibold text-white">Diversification Analysis</h3>
              </div>
              <p className="text-gray-300 leading-relaxed">
                {hasAiInsights && aiExplanations?.diversification_analysis ? 
                  aiExplanations.diversification_analysis : 
                  generateDiversificationExplanation()}
              </p>
            </motion.div>
            
            {hasAiInsights && aiExplanations?.market_analysis && (
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.3 }}
                className="bg-[#1a1a1a] rounded-xl p-5 border border-green-500/20"
              >
                <div className="flex items-start mb-3">
                  <GlobeAmericasIcon className="h-6 w-6 text-green-400 mr-2 mt-1" />
                  <h3 className="text-xl font-semibold text-white">Market Analysis</h3>
                </div>
                <p className="text-gray-300 leading-relaxed">
                  {aiExplanations.market_analysis}
                </p>
              </motion.div>
            )}
            
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="bg-[#1a1a1a] rounded-xl p-5 border border-red-500/20"
            >
              <div className="flex items-start mb-3">
                <LightBulbIcon className="h-6 w-6 text-red-400 mr-2 mt-1" />
                <h3 className="text-xl font-semibold text-white">Recommendations</h3>
              </div>
              <ul className="list-disc pl-5 space-y-2 text-gray-300">
                {hasAiInsights && aiRecommendations.length > 0 ? 
                  aiRecommendations.map((rec, index) => (
                    <li key={index}>{rec}</li>
                  )) : 
                  generateActionItems().map((item, index) => (
                    <li key={index}>{item}</li>
                  ))
                }
              </ul>
            </motion.div>
            
            {hasAiInsights && marketOutlook && (
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.5 }}
                className="bg-[#1a1a1a] rounded-xl p-5 border border-indigo-500/20"
              >
                <div className="flex items-start mb-3">
                  <ChartBarSquareIcon className="h-6 w-6 text-indigo-400 mr-2 mt-1" />
                  <h3 className="text-xl font-semibold text-white">Market Outlook</h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-[#252525] p-4 rounded-lg">
                    <h4 className="font-medium mb-1 text-blue-300">Short Term</h4>
                    <p className="text-sm text-gray-300">{marketOutlook.short_term || 'No data available'}</p>
                  </div>
                  <div className="bg-[#252525] p-4 rounded-lg">
                    <h4 className="font-medium mb-1 text-purple-300">Medium Term</h4>
                    <p className="text-sm text-gray-300">{marketOutlook.medium_term || 'No data available'}</p>
                  </div>
                  <div className="bg-[#252525] p-4 rounded-lg">
                    <h4 className="font-medium mb-1 text-indigo-300">Long Term</h4>
                    <p className="text-sm text-gray-300">{marketOutlook.long_term || 'No data available'}</p>
                  </div>
                </div>
              </motion.div>
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