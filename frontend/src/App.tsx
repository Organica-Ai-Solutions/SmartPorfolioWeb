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
import { ChartBarIcon, ChatBubbleLeftRightIcon } from '@heroicons/react/24/outline'
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
    <TooltipProvider>
      <div className="relative min-h-screen bg-[#030303] text-white">
        {/* Background sparkles */}
        <div className="fixed inset-0 w-full h-full pointer-events-none">
          <SparklesCore
            id="tsparticlesfullpage"
            background="transparent"
            minSize={0.8}
            maxSize={1.6}
            particleDensity={100}
            className="w-full h-full"
            particleColor="#FFFFFF"
            speed={1}
          />
        </div>

        <div className="relative z-10 min-h-screen py-8 px-4">
          <div className="container mx-auto max-w-6xl">
            {/* Use our custom header and hero components */}
            <HeaderComponent onSettingsClick={() => setSettingsOpen(true)} />
            <HeroComponent />

            {/* Main card */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="relative bg-[#0a0a0a]/95 backdrop-blur-xl rounded-3xl p-6 md:p-8 shadow-2xl border border-white/10 mt-8"
            >
              <div className="border-b border-gray-700 mb-8">
                <nav className="-mb-px flex space-x-8">
                  <TabButton
                    name="portfolio"
                    activeTab={activeTab}
                    setActiveTab={setActiveTab}
                  >
                    Portfolio
                  </TabButton>
                  <TabButton
                    name="watchlist"
                    activeTab={activeTab}
                    setActiveTab={setActiveTab}
                  >
                    Watchlist
                  </TabButton>
                  <TabButton name="insights" activeTab={activeTab} setActiveTab={setActiveTab}>
                    <ChartBarIcon className="h-5 w-5" />
                    <span>Insights</span>
                  </TabButton>
                  {sentimentData && (
                    <TabButton name="sentiment" activeTab={activeTab} setActiveTab={setActiveTab}>
                      <ChatBubbleLeftRightIcon className="h-5 w-5" />
                      <span>Sentiment</span>
                    </TabButton>
                  )}
                </nav>
                </div>

              {activeTab === 'portfolio' && (
      <div>
                  {/* Portfolio Management Section */}
                  <div className="mb-6">
                    <h2 className="text-2xl font-bold mb-4">Portfolio Management</h2>
                    
                    {/* Ticker Input */}
                    <div className="mb-4">
                      <TickerSuggestions onSelectTicker={addTicker} />
                    </div>
                    
                    {/* Selected Tickers */}
                    <div className="mb-4">
                      <h3 className="text-xl font-semibold mb-2">Selected Tickers</h3>
                      <div className="flex flex-wrap gap-2">
                        {portfolio.tickers.map(ticker => (
                          <div 
                        key={ticker}
                            className="bg-slate-700 px-3 py-1 rounded-full flex items-center gap-2"
                          >
                            <span>{ticker}</span>
                        <button
                          onClick={() => removeTicker(ticker)}
                              className="text-slate-400 hover:text-white"
                        >
                              &times;
                        </button>
                          </div>
                    ))}
                      </div>
                  </div>
                </div>

                    {/* Risk Tolerance */}
                    <div className="mb-4">
                      <h3 className="text-xl font-semibold mb-2">Risk Tolerance</h3>
                      <select
                        value={portfolio.risk_tolerance}
                        onChange={(e) => setPortfolio({...portfolio, risk_tolerance: e.target.value})}
                        className="bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 w-full"
                      >
                        <option value="low">Low</option>
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                      </select>
                  </div>

                    {/* Analysis Period */}
                    <div className="mb-4">
                      <h3 className="text-xl font-semibold mb-2">Analysis Period</h3>
                  <select
                        onChange={(e) => handlePeriodChange(e.target.value as TimePeriod)}
                        className="bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 w-full"
                      >
                        <option value="3m">3 Months</option>
                        <option value="6m">6 Months</option>
                        <option value="1y" selected>1 Year</option>
                        <option value="5y">5 Years</option>
                        <option value="max">Max</option>
                  </select>
                </div>

                    {/* Analyze Button */}
                    <div className="mb-4 flex flex-col md:flex-row gap-4">
                      {/* Debug info for button state */}
                      <div className="mb-2 p-2 bg-blue-900/30 rounded-lg text-xs">
                        <p>Debug: Tickers: {portfolio.tickers.length > 0 ? portfolio.tickers.join(', ') : 'None'}</p>
                        <p>Start date: {portfolio.start_date || 'Not set'}</p>
                        <p>Loading: {loading ? 'Yes' : 'No'}</p>
                        <p>Button should be enabled: {!(loading || portfolio.tickers.length === 0) ? 'Yes' : 'No'}</p>
                      </div>
                      
                  <button
                    onClick={analyzePortfolio}
                        disabled={loading || portfolio.tickers.length === 0}
                        className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg font-medium hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {isAnalyzing ? 'Analyzing...' : 'Analyze Portfolio'}
                  </button>
                  
                  <button
                    onClick={rebalancePortfolio}
                        disabled={!analysis || loading}
                        className="px-6 py-3 bg-gradient-to-r from-green-500 to-teal-500 rounded-lg font-medium hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? 'Rebalancing...' : 'Rebalance Portfolio'}
                  </button>
                </div>

                    {/* Error Message */}
                    {error && (
                      <div className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
                        {error}
                  </div>
                )}
                  </div>
              )}
              {activeTab === 'watchlist' && <WatchlistTab stockData={stockData} />}
              {activeTab === 'insights' && (
                <div className="mt-8 border-t border-gray-700 pt-6">
                  <h2 className="text-2xl font-bold mb-6">Portfolio Analysis Results</h2>
                  
                  {isAnalyzing ? (
                    <div className="flex justify-center items-center py-12">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                      <span className="ml-3">Analyzing portfolio...</span>
                    </div>
                  ) : !analysis ? (
                    <div className="text-center py-12 text-gray-400">
                      <p>No analysis data available. Please analyze your portfolio first.</p>
                    </div>
                  ) : (
                    <>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                        {/* Performance Chart */}
                        <div className="bg-[#121a2a] p-6 rounded-xl shadow-lg border border-blue-900/30">
                          <div className="flex justify-between items-center mb-4">
                            <h3 className="text-xl font-semibold text-blue-400">Performance</h3>
                          </div>
                          {chartData && <PerformanceChart data={chartData} />}
                        </div>

                        {/* Allocation Chart */}
                        <div className="bg-[#121a2a] p-6 rounded-xl shadow-lg border border-blue-900/30">
                          <h3 className="text-xl font-semibold mb-3 text-blue-400">Asset Allocation</h3>
                          {analysis.allocations && <AllocationChart allocations={analysis.allocations} />}
                        </div>
                      </div>

                      {/* Metrics */}
                      <div className="bg-[#121a2a] p-6 rounded-xl shadow-lg border border-blue-900/30 mb-6">
                        <h3 className="text-xl font-semibold mb-4 text-blue-400">Portfolio Metrics</h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                          {analysis.metrics && (
                            <>
                              <div className="bg-[#0a0a17] p-4 rounded-lg border border-indigo-900/20">
                                <div className="text-slate-400 text-sm">Expected Return</div>
                                <div className="text-xl font-bold text-green-400">
                                  {(analysis.metrics.expected_return * 100).toFixed(2)}%
                                </div>
                              </div>
                              <div className="bg-[#0a0a17] p-4 rounded-lg border border-indigo-900/20">
                                <div className="text-slate-400 text-sm">Volatility</div>
                                <div className="text-xl font-bold text-yellow-400">
                                  {(analysis.metrics.volatility * 100).toFixed(2)}%
                                </div>
                              </div>
                              <div className="bg-[#0a0a17] p-4 rounded-lg border border-indigo-900/20">
                                <div className="text-slate-400 text-sm">Sharpe Ratio</div>
                                <div className="text-xl font-bold text-blue-400">
                                  {analysis.metrics.sharpe_ratio.toFixed(2)}
                                </div>
                              </div>
                              <div className="bg-[#0a0a17] p-4 rounded-lg border border-indigo-900/20">
                                <div className="text-slate-400 text-sm">Max Drawdown</div>
                                <div className="text-xl font-bold text-red-400">
                                  {(analysis.metrics.max_drawdown * 100).toFixed(2)}%
                                </div>
                              </div>
                            </>
                          )}
                        </div>
                      </div>
                    </>
                  )}
                </div>
              )}
              {activeTab === 'sentiment' && sentimentData && (
                <div className="mt-4">
                  <SentimentAnalysis 
                    sentimentData={sentimentData} 
                    onTickerSelect={(ticker) => {
                      // Optionally handle ticker selection, e.g., to show more details
                      console.log("Selected ticker:", ticker);
                    }}
                  />
                          </div>
              )}
            </motion.div>
          </div>
        </div>

        {/* Settings Dialog */}
        <Settings isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
      </div>
    </TooltipProvider>
  )
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

