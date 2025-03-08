import React from 'react';
import { motion } from 'framer-motion';
import { ArrowTrendingUpIcon, ArrowTrendingDownIcon, Bars3BottomLeftIcon } from '@heroicons/react/24/outline';

interface SentimentScore {
  sentiment: string;
  score: number;
  sources?: string[];
  key_factors?: string[];
}

interface SentimentData {
  overall_sentiment: string;
  score: number;
  ticker_sentiments: Record<string, SentimentScore>;
  analysis_date: string;
  timeframe: string;
}

interface SentimentAnalysisProps {
  sentimentData: SentimentData;
  onTickerSelect?: (ticker: string) => void;
}

export function SentimentAnalysis({ sentimentData, onTickerSelect }: SentimentAnalysisProps) {
  if (!sentimentData) {
    return (
      <div className="bg-slate-800 rounded-lg p-4 text-center text-slate-400">
        Sentiment data is not available
      </div>
    );
  }

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case 'positive': return 'text-green-400';
      case 'negative': return 'text-red-400';
      case 'neutral': return 'text-blue-400';
      default: return 'text-slate-400';
    }
  };

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case 'positive': return <ArrowTrendingUpIcon className="h-5 w-5 text-green-400" />;
      case 'negative': return <ArrowTrendingDownIcon className="h-5 w-5 text-red-400" />;
      case 'neutral': return <Bars3BottomLeftIcon className="h-5 w-5 text-blue-400" />;
      default: return null;
    }
  };

  const getScorePercentage = (score: number) => {
    return Math.round(score * 100);
  };

  return (
    <div className="space-y-5">
      <motion.div 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-slate-800 rounded-lg p-5 border border-slate-700"
      >
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xl font-semibold text-white">Market Sentiment</h3>
          <div className="text-sm text-slate-400">{sentimentData.timeframe}</div>
        </div>
        
        <div className="flex items-center mb-4">
          <div className={`text-2xl font-bold mr-2 ${getSentimentColor(sentimentData.overall_sentiment)}`}>
            {sentimentData.overall_sentiment.toUpperCase()}
          </div>
          <div className="text-lg text-slate-300">
            {getScorePercentage(sentimentData.score)}%
          </div>
          <div className="ml-2">
            {getSentimentIcon(sentimentData.overall_sentiment)}
          </div>
        </div>
        
        <div className="text-slate-300 text-sm mb-4">
          Analysis date: {new Date(sentimentData.analysis_date).toLocaleDateString()}
        </div>
        
        <div className="space-y-1">
          {Object.entries(sentimentData.ticker_sentiments).map(([ticker, data]) => (
            <div 
              key={ticker}
              className="bg-slate-700 rounded p-3 cursor-pointer hover:bg-slate-600 transition-colors"
              onClick={() => onTickerSelect && onTickerSelect(ticker)}
            >
              <div className="flex justify-between items-center mb-1">
                <div className="font-medium text-white">{ticker}</div>
                <div className="flex items-center">
                  <div className={`mr-1 ${getSentimentColor(data.sentiment)}`}>{data.sentiment}</div>
                  <div className="text-sm text-slate-300">{getScorePercentage(data.score)}%</div>
                </div>
              </div>
              
              {data.key_factors && (
                <div className="text-xs text-slate-400 mt-1">
                  Key factors: {data.key_factors.join(', ')}
                </div>
              )}
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
} 