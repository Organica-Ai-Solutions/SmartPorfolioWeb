import React from 'react';
import { RebalanceResult } from '../types/portfolio';

type RebalanceExplanationProps = {
  result: RebalanceResult;
  language?: 'en' | 'es';
}

export function RebalanceExplanation({ result, language = 'en' }: RebalanceExplanationProps) {
  // Check if AI insights are available
  const hasAiInsights = result.ai_insights && 
                       typeof result.ai_insights === 'object' &&
                       !result.ai_insights.error;
  
  // Get AI explanations if available
  const aiExplanation = hasAiInsights && result.ai_insights?.explanation ? 
    language === 'en' ? result.ai_insights.explanation.english : result.ai_insights.explanation.spanish : 
    null;
  
  // Get AI recommendations if available
  const aiRecommendations = hasAiInsights && result.ai_insights?.recommendations ?
    result.ai_insights.recommendations :
    [];
  
  // Get market impact if available
  const marketImpact = hasAiInsights && result.ai_insights?.market_impact ?
    result.ai_insights.market_impact :
    null;
  
  // Basic explanation without AI insights
  const generateBasicExplanation = () => {
    const orderCount = result.orders?.length || 0;
    const totalValue = result.account_balance?.equity || 0;
    
    const explanations = {
      en: `Rebalancing completed with ${orderCount} orders executed. Your account equity is now $${totalValue.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}.`,
      es: `Rebalanceo completado con ${orderCount} órdenes ejecutadas. El valor de su cuenta ahora es de $${totalValue.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}.`
    };
    
    return explanations[language];
  }

  return (
    <div className="space-y-6 text-gray-800 dark:text-gray-200">
      <div>
        <h3 className="text-xl font-semibold mb-2">Rebalance Summary</h3>
        <p className="text-gray-600 dark:text-gray-300">
          {aiExplanation || generateBasicExplanation()}
        </p>
      </div>
      
      {/* Orders Summary */}
      <div>
        <h3 className="text-xl font-semibold mb-2">Orders Executed</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Symbol</th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Side</th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Quantity</th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">Status</th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
              {result.orders?.map((order, index) => (
                <tr key={index} className={index % 2 === 0 ? 'bg-white dark:bg-gray-900' : 'bg-gray-50 dark:bg-gray-800'}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">{order.symbol}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300 capitalize">{order.side}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">{order.qty}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                      order.status === 'executed' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' : 
                      order.status === 'pending' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' : 
                      'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                    }`}>
                      {order.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* Account Balance */}
      <div>
        <h3 className="text-xl font-semibold mb-2">Account Balance</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg">
            <h4 className="font-medium mb-1">Equity</h4>
            <p className="text-2xl font-bold">${result.account_balance?.equity?.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) || 'N/A'}</p>
          </div>
          <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg">
            <h4 className="font-medium mb-1">Cash</h4>
            <p className="text-2xl font-bold">${result.account_balance?.cash?.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) || 'N/A'}</p>
          </div>
          <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg">
            <h4 className="font-medium mb-1">Buying Power</h4>
            <p className="text-2xl font-bold">${result.account_balance?.buying_power?.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) || 'N/A'}</p>
          </div>
        </div>
      </div>
      
      {/* AI Recommendations */}
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
      
      {/* Market Impact */}
      {hasAiInsights && marketImpact && (
        <div>
          <h3 className="text-xl font-semibold mb-2">Market Impact Analysis</h3>
          <p className="text-gray-600 dark:text-gray-300 mb-4">
            {marketImpact.summary || 'No market impact analysis available.'}
          </p>
          
          {marketImpact.factors && marketImpact.factors.length > 0 && (
            <div>
              <h4 className="text-lg font-medium mb-2">Key Factors</h4>
              <ul className="list-disc pl-5 space-y-1 text-gray-600 dark:text-gray-300">
                {marketImpact.factors.map((factor, index) => (
                  <li key={index}>{factor}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
} 