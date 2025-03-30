import React from 'react';

interface AIInsight {
  title: string;
  description: string;
  type: 'positive' | 'negative' | 'neutral';
}

interface AIInsightsProps {
  insights: AIInsight[];
}

export const AIInsights: React.FC<AIInsightsProps> = ({ insights }) => {
  const getTypeColor = (type: AIInsight['type']) => {
    switch (type) {
      case 'positive':
        return 'text-green-400';
      case 'negative':
        return 'text-red-400';
      default:
        return 'text-blue-400';
    }
  };

  return (
    <div className="space-y-4">
      <h3 className="text-xl font-semibold mb-4">AI Portfolio Insights</h3>
      {insights.map((insight, index) => (
        <div
          key={index}
          className="bg-[#121a2a] p-4 rounded-lg border border-blue-900/30"
        >
          <h4 className={`font-medium ${getTypeColor(insight.type)}`}>
            {insight.title}
          </h4>
          <p className="text-gray-300 mt-2">{insight.description}</p>
        </div>
      ))}
    </div>
  );
}; 