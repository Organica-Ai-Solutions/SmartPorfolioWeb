import React, { useState, useEffect } from 'react'
import { Portfolio, PortfolioAnalysis, PortfolioMetrics } from './types/portfolio'
import axios from 'axios'
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, ArcElement, Filler } from 'chart.js'
import { Line } from 'react-chartjs-2'
import { SparklesCore } from './components/ui/sparkles'
import { motion } from 'framer-motion'
import { TickerSuggestions } from './components/TickerSuggestions'
import { TooltipProvider } from "./components/ui/tooltip"
import Settings, { AlpacaSettings } from './components/Settings'
import { RebalanceResult } from './types/portfolio'
import { AllocationChart } from './components/AllocationChart'
import { PerformanceChart } from './components/PerformanceChart'
import { HeaderComponent } from './components/HeaderComponent'
import { HeroComponent } from './components/HeroComponent'
import { SentimentAnalysis } from './components/SentimentAnalysis'
import { ChartBarIcon, ChatBubbleLeftRightIcon, XMarkIcon } from '@heroicons/react/24/outline'
import { API_URL as BASE_API_URL } from './config'

ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler
)

// Update the API_URL to use a relative path or environment variable
const API_URL = BASE_API_URL

type TimePeriod = '3m' | '6m' | '1y' | '5y' | 'max'

// Define TabButton component - this is missing and causing the error
interface TabButtonProps {
  name: string;
  activeTab: string;
  setActiveTab: (tab: string) => void;
  children: React.ReactNode;
}

const TabButton = ({ name, activeTab, setActiveTab, children }: TabButtonProps) => {
  return (
    <button
      className={`px-4 py-2 font-medium text-sm border-b-2 ${
        activeTab === name
          ? 'border-blue-500 text-blue-500'
          : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-300'
      } transition-colors`}
      onClick={() => setActiveTab(name)}
    >
      {children}
    </button>
  );
};

// Define a simple StockDataType to fix the error
type StockDataType = {
  symbol: string;
  name?: string;
  price?: number;
  change?: number;
};

// Function to calculate start date based on period
const getStartDate = (period: TimePeriod): string => {
  const today = new Date()
  switch (period) {
    case '3m':
      today.setMonth(today.getMonth() - 3)
      break
    case '6m':
      today.setMonth(today.getMonth() - 6)
      break
    case '1y':
      today.setFullYear(today.getFullYear() - 1)
      break
    case '5y':
      today.setFullYear(today.getFullYear() - 5)
      break
    case 'max':
      today.setFullYear(today.getFullYear() - 10) // Using 10 years as max
      break
  }
  return today.toISOString().split('T')[0]
}

function App() {
  const [loading, setLoading] = useState(false);
  const [stockData, setStockData] = useState<StockDataType[]>([]);
  
  // Ensure start_date is initialized properly with today's date as fallback
  const today = new Date().toISOString().split('T')[0];
  const defaultStartDate = getStartDate('1y');
  
  const [portfolio, setPortfolio] = useState<Portfolio>({
    tickers: [],
    start_date: defaultStartDate, // Default to 1 year
    risk_tolerance: 'medium'
  });
  const [activeTab, setActiveTab] = useState('portfolio');
  const [newTicker, setNewTicker] = useState('')
  const [analysis, setAnalysis] = useState<PortfolioAnalysis | null>(null)
  const [error, setError] = useState('')
  const [chartData, setChartData] = useState<any>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [rebalanceResult, setRebalanceResult] = useState<RebalanceResult | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [sentimentData, setSentimentData] = useState<any>(null);

  console.log("Portfolio state initialized:", portfolio);

  const addTicker = (ticker: string) => {
    if (ticker && !portfolio.tickers.includes(ticker.toUpperCase())) {
      setPortfolio({
        ...portfolio,
        tickers: [...portfolio.tickers, ticker.toUpperCase()]
      })
      setNewTicker('')
    }
  }

  const removeTicker = (ticker: string) => {
    setPortfolio({
      ...portfolio,
      tickers: portfolio.tickers.filter(t => t !== ticker)
    })
  }

  const getSentimentAnalysis = async () => {
    if (!portfolio.tickers.length) return;
    
    try {
      const response = await fetch(`${API_URL}/ai-sentiment-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
        tickers: portfolio.tickers,
        start_date: portfolio.start_date,
          risk_tolerance: portfolio.risk_tolerance,
        }),
      });
      
      if (!response.ok) {
        console.warn("Sentiment analysis not available");
        return;
      }
      
      const data = await response.json();
      console.log("Sentiment analysis response:", data);
      setSentimentData(data);
    } catch (err) {
      console.warn("Error getting sentiment analysis:", err);
    }
  };

  const analyzePortfolio = async () => {
    if (!portfolio.tickers || portfolio.tickers.length === 0) {
      setError("Please add at least one ticker to your portfolio.")
      return
    }

    setIsAnalyzing(true)
    setError('')
    console.log("Analyzing portfolio with tickers:", portfolio.tickers)
    
    try {
      console.log('Making API request to:', `${API_URL}/analyze-portfolio`)
      console.log('Request payload:', portfolio)
      
      // Always try to use the real data endpoint
      const response = await axios.post(`${API_URL}/analyze-portfolio`, portfolio, {
        timeout: 30000, // Set timeout to 30 seconds for slower data retrieval
        headers: {
          'Content-Type': 'application/json',
        }
      })
      
      console.log('Raw API response:', response.data)
      
      if (!response.data) {
        throw new Error('Invalid response data')
      }

      // Check if we're using real or mock data
      const isRealData = response.data.metadata?.is_real_data === true
      if (isRealData) {
        console.log('Using REAL market data from:', response.data.metadata.data_source)
        console.log('Data points:', response.data.metadata.data_points)
        console.log('Date range:', response.data.metadata.date_range)
      } else {
        console.warn('Using MOCK data - real market data could not be retrieved')
      }

      // Update the analysis state with the response data
      setAnalysis(response.data)
      
      // Try to get sentiment data
      try {
        const sentimentResponse = await axios.post(`${API_URL}/ai-sentiment-analysis`, portfolio)
        if (sentimentResponse.data) {
          setSentimentData(sentimentResponse.data)
        }
      } catch (sentimentErr) {
        console.warn("Error getting sentiment analysis:", sentimentErr)
      }

      setError('')
      // Add an informational message if using mock data
      if (!isRealData) {
        setError('Using simulated data for analysis. Real market data could not be retrieved.')
      }
      console.log('Portfolio analysis completed successfully')
      
    } catch (error: any) {
      console.error('API Error:', error)
      console.error('Error response:', error.response?.data)
      console.error('Error status:', error.response?.status)
      console.error('Error headers:', error.response?.headers)
      console.error('Error message:', error.message)
      setError(error.response?.data?.detail || error.message || 'Failed to analyze portfolio. Please try again.')
    } finally {
      setIsAnalyzing(false)
    }
  }

  // Helper function to fetch AI insights
  const fetchAIInsights = async (baseAnalysis: PortfolioAnalysis) => {
    try {
      const aiResponse = await fetch(`${API_URL}/ai-portfolio-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tickers: portfolio.tickers,
          start_date: portfolio.start_date,
          risk_tolerance: portfolio.risk_tolerance,
        }),
      });
      
      if (aiResponse.ok) {
        const aiData = await aiResponse.json();
        console.log("AI portfolio analysis response:", aiData);
        
        // If we have AI insights, add them to the analysis
        if (aiData && aiData.ai_insights) {
          const updatedAnalysis = { ...baseAnalysis, ai_insights: aiData.ai_insights };
          setAnalysis(updatedAnalysis);
        }
      } else {
        console.warn("AI insights not available, continuing with basic analysis");
      }
      
      // Also try to get sentiment data
      fetchSentimentData();
      
    } catch (aiErr) {
      console.warn("Error getting AI insights:", aiErr);
    }
  };
  
  // Helper function to fetch sentiment data
  const fetchSentimentData = async () => {
    try {
      const sentimentResponse = await fetch(`${API_URL}/ai-sentiment-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tickers: portfolio.tickers,
          start_date: portfolio.start_date,
          risk_tolerance: portfolio.risk_tolerance,
        }),
      });
      
      if (sentimentResponse.ok) {
        const sentimentData = await sentimentResponse.json();
        console.log("Sentiment analysis response:", sentimentData);
        setSentimentData(sentimentData);
      }
    } catch (sentimentErr) {
      console.warn("Error getting sentiment analysis:", sentimentErr);
    }
  };

  const rebalancePortfolio = async () => {
    if (!analysis) return

    try {
      setLoading(true)
      setError('')
      setRebalanceResult(null)
      
      // Get Alpaca settings from localStorage
      const savedSettings = localStorage.getItem('alpacaSettings');
      let alpacaSettings: AlpacaSettings | null = null;
      
      if (savedSettings) {
        try {
        alpacaSettings = JSON.parse(savedSettings) as AlpacaSettings;
        console.log('Using Alpaca settings:', {
          apiKey: alpacaSettings.apiKey ? '****' + alpacaSettings.apiKey.slice(-4) : 'not set',
          secretKey: alpacaSettings.secretKey ? '****' + alpacaSettings.secretKey.slice(-4) : 'not set',
          isPaper: alpacaSettings.isPaper
        });
        } catch (parseError) {
          console.error('Error parsing Alpaca settings:', parseError);
          setError('Invalid Alpaca settings. Please reconfigure in Settings.');
          setSettingsOpen(true);
          setLoading(false);
          return;
        }
      } else {
        console.log('No Alpaca settings found in localStorage');
      }
      
      // Validate Alpaca settings
      if (!alpacaSettings?.apiKey || !alpacaSettings?.secretKey) {
        setError('Alpaca API keys are required for paper trading. Please configure them in Settings.');
        setSettingsOpen(true);
        setLoading(false);
        return;
      }

      // Validate portfolio data
      if (!analysis.allocations || Object.keys(analysis.allocations).length === 0) {
        setError('No valid portfolio allocations found. Please analyze your portfolio first.');
        setLoading(false);
        return;
      }

      // Validate discrete allocation
      if (!analysis.discrete_allocation || Object.keys(analysis.discrete_allocation).length === 0) {
        setError('No valid discrete allocation found. Please analyze your portfolio first.');
        setLoading(false);
        return;
      }

      // Use the AI rebalance explanation endpoint
      const response = await axios.post(`${API_URL}/ai-rebalance-explanation`, {
        allocations: analysis.allocations,
        discrete_allocation: analysis.discrete_allocation,
        alpaca_api_key: alpacaSettings.apiKey,
        alpaca_secret_key: alpacaSettings.secretKey,
        use_paper_trading: alpacaSettings.isPaper,
        portfolio_value: 10000 // Default paper trading value
      });
      
      console.log('Rebalancing response:', response.data);
      
      // Validate rebalance result
      if (!response.data || !response.data.orders) {
        setError('Invalid rebalance response from server. Please try again.');
        setLoading(false);
        return;
      }

      setRebalanceResult(response.data);
      
      // Show success message
      setError('Portfolio rebalanced successfully! Check the orders tab for details.');
      
    } catch (err: any) {
      console.error('Rebalance error:', err);
      if (err.response?.data?.detail) {
        setError(`Error: ${err.response.data.detail}`);
      } else if (err.message) {
        setError(`Error: ${err.message}`);
      } else {
        setError('Error rebalancing portfolio. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  }

  // Initialize chart data when analysis changes
  useEffect(() => {
    if (analysis?.allocations && Object.keys(analysis.allocations).length > 0) {
      const labels = Object.keys(analysis.allocations)
      const data = Object.values(analysis.allocations).map(value => {
        const numValue = Number(value) * 100
        return isNaN(numValue) || !isFinite(numValue) ? 0 : parseFloat(numValue.toFixed(2))
      })

      setChartData({
        labels,
        datasets: [{
          data,
          backgroundColor: [
          'rgba(255, 99, 132, 0.8)',
          'rgba(54, 162, 235, 0.8)',
          'rgba(255, 206, 86, 0.8)',
          'rgba(75, 192, 192, 0.8)',
          'rgba(153, 102, 255, 0.8)',
          ],
          borderColor: [
            'rgba(255, 99, 132, 1)',
            'rgba(54, 162, 235, 1)',
            'rgba(255, 206, 86, 1)',
            'rgba(75, 192, 192, 1)',
            'rgba(153, 102, 255, 1)',
          ],
              borderWidth: 1
            }]
      })
    }
  }, [analysis?.allocations])

  // Handler for period change
  const handlePeriodChange = (period: TimePeriod) => {
    const newStartDate = getStartDate(period)
    console.log(`Setting new start date: ${newStartDate} for period: ${period}`)
    
    setPortfolio({
      ...portfolio,
      start_date: newStartDate
    })
  }

  // Update the prepareChartData function to handle the updated data structure
  const prepareChartData = (analysis: PortfolioAnalysis) => {
    if (!analysis || !analysis.historical_performance) return null;
    
    // Get dates from the portfolio_values object
    const dates = Object.keys(analysis.historical_performance.portfolio_values || {});
    if (dates.length === 0) return null;
    
    // Get values from the objects
    const portfolioValues = dates.map(date => {
      const value = analysis.historical_performance.portfolio_values[date];
      return typeof value === 'number' ? value : 0;
    });
    
    const drawdowns = dates.map(date => {
      const value = analysis.historical_performance.drawdowns[date];
      return typeof value === 'number' ? value : 0;
    });
    
    return {
      labels: dates,
      datasets: [
        {
          label: 'Portfolio Value',
          data: Array.isArray(analysis.historical_performance.portfolio_values) 
            ? analysis.historical_performance.portfolio_values.map((value: any) => {
                const numValue = Number(value);
                return isNaN(numValue) || !isFinite(numValue) ? null : numValue;
              }) 
            : Object.values(analysis.historical_performance.portfolio_values || {}).map((value: any) => {
                const numValue = Number(value);
                return isNaN(numValue) || !isFinite(numValue) ? null : numValue;
              }),
          borderColor: 'rgb(147, 51, 234)',
          backgroundColor: 'rgba(147, 51, 234, 0.1)',
          tension: 0.4,
          fill: true,
          yAxisID: 'y'
        },
        {
          label: 'Market Benchmark',
          data: Array.isArray(analysis.market_comparison?.market_values) 
            ? analysis.market_comparison.market_values.map((value: any) => {
                const numValue = Number(value);
                return isNaN(numValue) || !isFinite(numValue) ? null : numValue;
              }) 
            : Object.values(analysis.market_comparison?.market_values || {}).map((value: any) => {
                const numValue = Number(value);
                return isNaN(numValue) || !isFinite(numValue) ? null : numValue;
              }),
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderDash: [5, 5],
          tension: 0.4,
          fill: false,
          yAxisID: 'y'
        },
        {
          label: 'Volatility',
          data: Array.isArray(analysis.historical_performance.rolling_volatility) 
            ? analysis.historical_performance.rolling_volatility.map((value: any) => {
                const numValue = Number(value);
                return isNaN(numValue) || !isFinite(numValue) ? null : numValue;
              }) 
            : Object.values(analysis.historical_performance.rolling_volatility || {}).map((value: any) => {
                const numValue = Number(value);
                return isNaN(numValue) || !isFinite(numValue) ? null : numValue;
              }),
          borderColor: 'rgb(239, 68, 68)',
          backgroundColor: 'rgba(239, 68, 68, 0)',
          borderDash: [3, 3],
          tension: 0.4,
          pointRadius: 0,
          yAxisID: 'y1'
        }
      ]
    };
  };

  // Update the mergeWithDefaults function to match the interface
  function mergeWithDefaults(apiData: any, defaultData: PortfolioAnalysis): PortfolioAnalysis {
    const result = { ...defaultData };
    
    // Handle allocations if available
    if (apiData.allocations && typeof apiData.allocations === 'object') {
      result.allocations = apiData.allocations;
    }
    
    // Handle metrics if available
    if (apiData.metrics && typeof apiData.metrics === 'object') {
      result.metrics = {
        ...defaultData.metrics,
        ...apiData.metrics
      };
    }
    
    // Handle asset_metrics if available
    if (apiData.asset_metrics && typeof apiData.asset_metrics === 'object') {
      result.asset_metrics = apiData.asset_metrics;
    }
    
    // Handle discrete_allocation which might be nested
    if (apiData.discrete_allocation) {
      if (apiData.discrete_allocation.shares) {
        // If it's in the format {shares: {...}, leftover: ...}
        result.discrete_allocation = apiData.discrete_allocation.shares;
      } else {
        // If it's already in the expected format
        result.discrete_allocation = apiData.discrete_allocation;
      }
    }
    
    // Handle historical_performance if available
    if (apiData.historical_performance) {
      const hp = apiData.historical_performance;
      
      // Keep the dates if available
      if (hp.dates && Array.isArray(hp.dates)) {
        result.historical_performance.dates = hp.dates;
      }
      
      // Handle portfolio_values which might be an array or object
      if (hp.portfolio_values) {
        if (Array.isArray(hp.portfolio_values)) {
          // Convert array to object using dates as keys
          const portfolioValues: Record<string, number> = {};
          // Ensure we have dates, use the defaultData dates if not provided
          const dates = result.historical_performance.dates || defaultData.historical_performance.dates || [];
          dates.forEach((date, i) => {
            if (i < hp.portfolio_values.length) {
              portfolioValues[date] = hp.portfolio_values[i];
            }
          });
          result.historical_performance.portfolio_values = portfolioValues;
        } else {
          // It's already an object
          result.historical_performance.portfolio_values = hp.portfolio_values;
        }
      }
      
      // Handle drawdowns which might be an array or object
      if (hp.drawdowns) {
        if (Array.isArray(hp.drawdowns)) {
          // Convert array to object using dates as keys
          const drawdowns: Record<string, number> = {};
          // Ensure we have dates, use the defaultData dates if not provided
          const dates = result.historical_performance.dates || defaultData.historical_performance.dates || [];
          dates.forEach((date, i) => {
            if (i < hp.drawdowns.length) {
              drawdowns[date] = hp.drawdowns[i];
            }
          });
          result.historical_performance.drawdowns = drawdowns;
        } else {
          // It's already an object
          result.historical_performance.drawdowns = hp.drawdowns;
        }
      }
      
      // Handle rolling_volatility if available
      if (hp.rolling_volatility) {
        result.historical_performance.rolling_volatility = hp.rolling_volatility;
      }
    }
    
    // Handle market_comparison if available
    if (apiData.market_comparison) {
      const mc = apiData.market_comparison;
      
      // Ensure market_comparison is properly initialized
      if (!result.market_comparison) {
        result.market_comparison = {
          dates: defaultData.historical_performance.dates || [],
          market_values: [],
          relative_performance: []
        };
      }
      
      // Keep the dates if available
      if (mc.dates && Array.isArray(mc.dates)) {
        result.market_comparison.dates = mc.dates;
      }
      
      // Handle market_values if available
      if (mc.market_values && Array.isArray(mc.market_values)) {
        result.market_comparison.market_values = mc.market_values;
      }
      
      // Handle relative_performance if available
      if (mc.relative_performance && Array.isArray(mc.relative_performance)) {
        result.market_comparison.relative_performance = mc.relative_performance;
      }
    }
    
    // Handle AI insights if available
    if (apiData.ai_insights) {
      result.ai_insights = {
        ...defaultData.ai_insights,
        ...apiData.ai_insights
      };
    }
    
    return result;
  }

  // Update the validatePortfolioData function to be more specific
  const validatePortfolioData = (data: PortfolioAnalysis): boolean => {
    if (!data) {
      console.error('Data is null or undefined');
      return false;
    }
    
    // Check allocations
    if (!data.allocations || typeof data.allocations !== 'object') {
      console.error('Invalid allocations:', data.allocations);
      return false;
    }
    
    // Check metrics
    if (!data.metrics || typeof data.metrics !== 'object') {
      console.error('Invalid metrics:', data.metrics);
      return false;
    }
    
    // Check required metrics with type safety
    const metrics = data.metrics as PortfolioMetrics;
    if (typeof metrics.expected_return !== 'number' ||
        typeof metrics.volatility !== 'number' ||
        typeof metrics.sharpe_ratio !== 'number' ||
        typeof metrics.sortino_ratio !== 'number' ||
        typeof metrics.beta !== 'number' ||
        typeof metrics.max_drawdown !== 'number' ||
        typeof metrics.var_95 !== 'number' ||
        typeof metrics.cvar_95 !== 'number') {
      console.error('Missing or invalid metrics:', metrics);
      return false;
    }
    
    // Check historical performance
    if (!data.historical_performance || typeof data.historical_performance !== 'object') {
      console.error('Invalid historical performance:', data.historical_performance);
      return false;
    }
    
    if (!Array.isArray(data.historical_performance.dates)) {
      console.error('Invalid historical performance dates:', data.historical_performance.dates);
      return false;
    }
    
    if (!data.historical_performance.portfolio_values || typeof data.historical_performance.portfolio_values !== 'object') {
      console.error('Invalid portfolio values:', data.historical_performance.portfolio_values);
      return false;
    }
    
    // Check market comparison
    if (!data.market_comparison || typeof data.market_comparison !== 'object') {
      console.error('Invalid market comparison:', data.market_comparison);
      return false;
    }
    
    if (!Array.isArray(data.market_comparison.dates)) {
      console.error('Invalid market comparison dates:', data.market_comparison.dates);
      return false;
    }
    
    if (!Array.isArray(data.market_comparison.market_values)) {
      console.error('Invalid market values:', data.market_comparison.market_values);
      return false;
    }
    
    return true;
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 text-gray-800 dark:text-gray-200">
    <TooltipProvider>
        <div className="container mx-auto px-4 py-8">
          <HeaderComponent />
          
          {!analysis && (
            <HeroComponent />
          )}
          
          <div className="mt-8 max-w-4xl mx-auto">
            {error && (
              <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-4 rounded">
                <p>{error}</p>
                </div>
            )}
            
            {/* Data source indicator if we have analysis */}
            {analysis && analysis.metadata && (
              <div className={`mb-4 p-3 rounded-md text-sm ${analysis.metadata.is_real_data ? 
                'bg-green-100 text-green-800 border-l-4 border-green-500 dark:bg-green-900 dark:text-green-100' : 
                'bg-yellow-100 text-yellow-800 border-l-4 border-yellow-500 dark:bg-yellow-900 dark:text-yellow-100'}`}>
                <p className="font-medium">
                  {analysis.metadata.is_real_data ? 
                    `Using real market data from ${analysis.metadata.data_source}` : 
                    'Using simulated market data'
                  }
                </p>
                {analysis.metadata.is_real_data && (
                  <p className="text-xs mt-1 opacity-80">
                    {analysis.metadata.data_points} data points from {analysis.metadata.date_range?.start} to {analysis.metadata.date_range?.end}
                  </p>
                )}
              </div>
            )}
            
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4">Selected Tickers</h2>
              <div className="flex flex-wrap gap-2 mb-6">
                {portfolio.tickers.map((ticker, i) => (
                  <div key={i} className="flex items-center bg-blue-100 dark:bg-blue-900 px-3 py-1 rounded-full text-blue-800 dark:text-blue-100">
                    {ticker}
                    <button onClick={() => removeTicker(ticker)} className="ml-2 text-blue-600 dark:text-blue-300 hover:text-blue-800 dark:hover:text-blue-100">
                      <XMarkIcon className="h-4 w-4" />
                    </button>
                  </div>
                ))}
                {portfolio.tickers.length === 0 && (
                  <p className="text-gray-500 dark:text-gray-400">Add tickers to analyze</p>
                )}
                    </div>
                    
              <div className="flex items-center mb-6">
                <input
                  type="text"
                  value={newTicker}
                  onChange={(e) => setNewTicker(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      addTicker(newTicker);
                    }
                  }}
                  placeholder="Add ticker (e.g., AAPL, MSFT, BTC-USD)"
                  className="border rounded p-2 flex-1 dark:bg-gray-700 dark:border-gray-600"
                />
                        <button
                  onClick={() => addTicker(newTicker)}
                  className="ml-2 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
                        >
                  Add
                        </button>
                          </div>
              
              {suggestions.length > 0 && (
                <div className="mb-6">
                  <TickerSuggestions
                    suggestions={suggestions}
                    onSelect={handleSuggestionSelect}
                  />
                  </div>
              )}

              <h2 className="text-xl font-semibold mb-4">Risk Tolerance</h2>
                      <select
                        value={portfolio.risk_tolerance}
                        onChange={(e) => setPortfolio({...portfolio, risk_tolerance: e.target.value})}
                className="border rounded p-2 w-full mb-6 dark:bg-gray-700 dark:border-gray-600"
                      >
                        <option value="low">Low</option>
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                      </select>

              <h2 className="text-xl font-semibold mb-4">Analysis Period</h2>
                  <select
                value={today}
                onChange={(e) => {
                  handlePeriodChange(e.target.value as TimePeriod);
                }}
                className="border rounded p-2 w-full mb-6 dark:bg-gray-700 dark:border-gray-600"
                      >
                        <option value="3m">3 Months</option>
                        <option value="6m">6 Months</option>
                <option value="1y">1 Year</option>
                        <option value="5y">5 Years</option>
                        <option value="max">Max</option>
                  </select>
              
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                Debug: Tickers: {portfolio.tickers.join(', ')}<br />
                Start date: {portfolio.start_date}<br />
                Loading: {loading ? 'Yes' : 'No'}<br />
                Button should be enabled: {portfolio.tickers.length > 0 ? 'Yes' : 'No'}
                      </div>
                      
                  <button
                    onClick={analyzePortfolio}
                disabled={isAnalyzing || portfolio.tickers.length === 0}
                className={`w-full py-3 rounded-md font-medium transition-colors ${
                  isAnalyzing || portfolio.tickers.length === 0
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed dark:bg-gray-700 dark:text-gray-500'
                    : 'bg-blue-500 text-white hover:bg-blue-600'
                }`}
                      >
                        {isAnalyzing ? 'Analyzing...' : 'Analyze Portfolio'}
                  </button>
                  
              {analysis && (
                  <button
                    onClick={rebalancePortfolio}
                  disabled={loading}
                  className={`w-full mt-4 py-3 rounded-md font-medium transition-colors ${
                    loading
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed dark:bg-gray-700 dark:text-gray-500'
                      : 'bg-green-500 text-white hover:bg-green-600'
                  }`}
                >
                  {loading ? 'Processing...' : 'Rebalance Portfolio'}
                  </button>
                )}
                  </div>
                  
                  {analysis && (
              <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
                  <h2 className="text-xl font-semibold mb-4">Portfolio Allocation</h2>
                          <AllocationChart allocations={analysis.allocations} />
                          </div>
                      
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
                  <h2 className="text-xl font-semibold mb-4">Portfolio Metrics</h2>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">Expected Return</p>
                      <p className="text-xl font-semibold">{(analysis.metrics.expected_return * 100).toFixed(2)}%</p>
                            </div>
                                      <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">Volatility</p>
                      <p className="text-xl font-semibold">{(analysis.metrics.volatility * 100).toFixed(2)}%</p>
                                      </div>
                                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">Sharpe Ratio</p>
                      <p className="text-xl font-semibold">{analysis.metrics.sharpe_ratio.toFixed(2)}</p>
                                    </div>
                                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">Beta</p>
                      <p className="text-xl font-semibold">{analysis.metrics.beta.toFixed(2)}</p>
                                    </div>
                            </div>
                              </div>
                              
                <div className="md:col-span-2 bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
                  <h2 className="text-xl font-semibold mb-4">Performance</h2>
                  <PerformanceChart data={analysis.historical_performance} />
                                  </div>
                                  
                {sentimentData && (
                  <div className="md:col-span-2 bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
                    <h2 className="flex items-center text-xl font-semibold mb-4">
                      <ChatBubbleLeftRightIcon className="h-5 w-5 mr-2" />
                      Market Sentiment
                    </h2>
                    <SentimentAnalysis data={sentimentData} />
                                    </div>
                )}
                
                {/* Crypto-specific information section */}
                {analysis && analysis.asset_metrics && Object.keys(analysis.asset_metrics).some(ticker => 
                  ticker.includes('-USD') || ticker.includes('-USDT')
                ) && (
                  <div className="md:col-span-2 bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
                    <h2 className="text-xl font-semibold mb-4">Cryptocurrency Assets</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {Object.entries(analysis.asset_metrics)
                        .filter(([ticker]) => ticker.includes('-USD') || ticker.includes('-USDT'))
                        .map(([ticker, metrics]) => (
                          <div key={ticker} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                            <h3 className="text-lg font-medium text-blue-600 dark:text-blue-400">{ticker}</h3>
                            <div className="mt-2 space-y-1">
                              <p className="text-sm">
                                <span className="text-gray-500 dark:text-gray-400">Weight: </span>
                                <span className="font-medium">{(metrics.weight * 100).toFixed(2)}%</span>
                              </p>
                              <p className="text-sm">
                                <span className="text-gray-500 dark:text-gray-400">Expected Return: </span>
                                <span className="font-medium">{(metrics.annual_return * 100).toFixed(2)}%</span>
                              </p>
                              <p className="text-sm">
                                <span className="text-gray-500 dark:text-gray-400">Volatility: </span>
                                <span className="font-medium">{(metrics.annual_volatility * 100).toFixed(2)}%</span>
                              </p>
                              <p className="text-sm">
                                <span className="text-gray-500 dark:text-gray-400">Sharpe Ratio: </span>
                                <span className="font-medium">{metrics.sharpe_ratio.toFixed(2)}</span>
                              </p>
                                          </div>
                            </div>
                        ))}
                          </div>
                    <div className="mt-4 text-sm text-gray-500 dark:text-gray-400 bg-blue-50 dark:bg-blue-900/30 p-3 rounded">
                      <p>Cryptocurrency assets tend to have higher volatility than traditional assets. Consider your risk tolerance when allocating to these assets.</p>
                        </div>
                    </div>
                  )}
                </div>
              )}
          </div>
      </div>
    </TooltipProvider>
    </div>
  );
}

// Define a WatchlistTab component
const WatchlistTab = ({ stockData }: { stockData: StockDataType[] }) => {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Watchlist</h2>
      {stockData.length === 0 ? (
        <p>No stocks in your watchlist yet.</p>
      ) : (
        <ul className="space-y-2">
          {stockData.map((stock) => (
            <li key={stock.symbol} className="p-3 bg-slate-800 rounded-lg flex justify-between">
              <span>{stock.symbol}</span>
              {stock.price && <span>${stock.price.toFixed(2)}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default App

