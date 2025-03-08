import React, { useState } from 'react';
import { Portfolio, PortfolioAnalysis } from '../types/portfolio';

interface FailsafeAnalysisProps {
  portfolio: Portfolio;
  onSuccess: (data: PortfolioAnalysis) => void;
  onError: (message: string) => void;
}

export function FailsafeAnalysis({ portfolio, onSuccess, onError }: FailsafeAnalysisProps) {
  const [isLoading, setIsLoading] = useState(false);
  
  const analyzeDirectly = async () => {
    setIsLoading(true);
    
    try {
      // Create a failsafe analysis result with dummy data
      const dates = ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"];
      
      // Portfolio values and drawdowns need to be Record<string, number>
      const portfolioValues: Record<string, number> = {};
      const drawdowns: Record<string, number> = {};
      
      // Fill in dummy data for each date
      dates.forEach((date, index) => {
        portfolioValues[date] = 10000 + index * 200 - (index === 3 ? 100 : 0);
        drawdowns[date] = index === 3 ? -0.01 : 0;
      });
      
      // Create rolling volatility as number[]
      const rollingVolatility = [0.1, 0.12, 0.11, 0.13, 0.12];
      
      // Market values and relative performance need to be number[]
      const marketValues = [10000, 10150, 10300, 10250, 10380];
      const relativePerformance = [0, 0.5, 0.97, 0.49, 1.16];
      
      const dummyData: PortfolioAnalysis = {
        allocations: {},
        metrics: {
          expected_return: 0.12,
          volatility: 0.18,
          sharpe_ratio: 0.67,
          sortino_ratio: 0.85,
          beta: 1.05,
          max_drawdown: -0.25,
          var_95: -0.02,
          cvar_95: -0.03
        },
        asset_metrics: {},
        discrete_allocation: {},
        historical_performance: {
          dates: dates,
          portfolio_values: portfolioValues,
          drawdowns: drawdowns,
          rolling_volatility: rollingVolatility
        },
        market_comparison: {
          dates: dates,
          market_values: marketValues,
          relative_performance: relativePerformance
        }
      };
      
      // Add allocations for each ticker
      const tickerCount = portfolio.tickers.length;
      const equalWeight = 1.0 / tickerCount;
      
      portfolio.tickers.forEach(ticker => {
        dummyData.allocations[ticker] = equalWeight;
        dummyData.discrete_allocation[ticker] = 10; // Dummy shares
        
        // Add asset metrics
        dummyData.asset_metrics[ticker] = {
          annual_return: 0.15 + Math.random() * 0.1,
          annual_volatility: 0.15 + Math.random() * 0.1,
          beta: 0.9 + Math.random() * 0.2,
          correlation: 0.6 + Math.random() * 0.2,
          weight: dummyData.allocations[ticker],
          alpha: 0.02 + Math.random() * 0.03,
          volatility: 0.15 + Math.random() * 0.1,
          var_95: -0.015 - Math.random() * 0.01,
          max_drawdown: -0.15 - Math.random() * 0.1,
          sharpe_ratio: 0.8 + Math.random() * 0.5,
          sortino_ratio: 1.0 + Math.random() * 0.5
        };
      });
      
      // Add AI insights with correct structure
      dummyData.ai_insights = {
        explanations: {
          english: {
            summary: "Your portfolio is well-balanced with a mix of large-cap technology stocks. These companies have strong fundamentals and growth potential, making them suitable for a medium risk tolerance investor.",
            risk_analysis: "The portfolio has a moderate risk profile with a beta of 1.05 relative to the S&P 500. The largest risk factors are technology sector concentration and exposure to regulatory changes.",
            diversification_analysis: "Your portfolio is concentrated in the technology sector, which increases sector-specific risk. Consider adding assets from healthcare or financial sectors for better diversification.",
            market_analysis: "Technology stocks have shown resilience in recent market conditions. The sector outlook remains positive with expected growth in AI, cloud computing, and digital transformation."
          }
        },
        recommendations: [
          "Consider adding some healthcare stocks for further diversification",
          "The portfolio may benefit from some exposure to value stocks to balance growth",
          "Monitor tech regulatory developments which could impact GOOGL and MSFT"
        ],
        market_outlook: {
          short_term: "Neutral with bullish bias",
          medium_term: "Bullish",
          long_term: "Strongly bullish"
        }
      };
      
      // Return the dummy data
      onSuccess(dummyData);
    } catch (error) {
      onError("Failed to create fallback analysis. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };
  
  // Auto-trigger analysis when component mounts
  React.useEffect(() => {
    analyzeDirectly();
  }, []);
  
  return (
    <div className="p-4 bg-blue-100/10 rounded-lg">
      {isLoading ? (
        <div className="text-center">
          <p>Generating fallback analysis...</p>
        </div>
      ) : (
        <div className="text-center">
          <p>Using fallback analysis mode.</p>
        </div>
      )}
    </div>
  );
} 