import React, { useState, useEffect, useRef } from 'react'
import { Portfolio, PortfolioAnalysis } from './types/portfolio'
import axios from 'axios'
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip as ChartTooltip, Legend, ArcElement, Filler } from 'chart.js'
import { Line, Pie } from 'react-chartjs-2'
import { SparklesCore } from './components/ui/sparkles'
import { motion, AnimatePresence } from 'framer-motion'
import { Button } from './components/ui/button'
import { TickerSuggestions } from './components/TickerSuggestions'
import { PortfolioExplanation } from './components/PortfolioExplanation'
import { Tooltip, TooltipProvider } from "./components/ui/tooltip"
import { StockAnalysis } from './components/StockAnalysis'
import { InformationCircleIcon } from "@heroicons/react/24/outline"
import Settings, { AlpacaSettings } from './components/Settings'
import { Cog6ToothIcon } from '@heroicons/react/24/outline'
import { RebalanceResult } from './types/portfolio'
import { RebalanceExplanation } from './components/RebalanceExplanation'
import { AllocationChart } from './components/AllocationChart'
import { PerformanceChart } from './components/PerformanceChart'
import { PortfolioSummary } from './components/PortfolioSummary'
import { AssetMetricsDetail } from './components/AssetMetricsDetail'
import { HeaderComponent } from './components/HeaderComponent'
import { HeroComponent } from './components/HeroComponent'
import { SentimentAnalysis } from './components/SentimentAnalysis'
import { ChartBarIcon, ChatBubbleLeftRightIcon } from '@heroicons/react/24/outline'

ChartJS.register(
  ArcElement,
  ChartTooltip,
  Legend,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler
)

// Update the API_URL to use a relative path or environment variable
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'

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
      setError("Please add at least one ticker to your portfolio.");
      return;
    }

    setIsAnalyzing(true);
    setError('');
    console.log("Analyzing portfolio with tickers:", portfolio.tickers);
    
    // Create a default data structure with guaranteed values for all required fields
    const initialDefaultData: PortfolioAnalysis = {
      allocations: {},
      metrics: {
        expected_return: 0.05, 
        volatility: 0.1,
        sharpe_ratio: 0.5,
        sortino_ratio: 0.6,
        beta: 1.0,
        max_drawdown: -0.1,
        var_95: -0.02,
        cvar_95: -0.03
      },
      asset_metrics: {},
      discrete_allocation: {},
      historical_performance: {
        dates: ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"],
        portfolio_values: {
          "2023-01-01": 10000,
          "2023-02-01": 10050,
          "2023-03-01": 10100,
          "2023-04-01": 10150,
          "2023-05-01": 10200
        },
        drawdowns: {
          "2023-01-01": 0,
          "2023-02-01": 0,
          "2023-03-01": 0,
          "2023-04-01": -0.01,
          "2023-05-01": -0.02
        }
      },
      market_comparison: {
        dates: ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"],
        market_values: [10000, 10150, 10300, 10250, 10380],
        relative_performance: [0, 0.5, 0.97, 0.49, 1.16]
      },
      ai_insights: {
        explanations: {
          english: {
            summary: "This is a placeholder assessment. The portfolio appears balanced.",
            risk_analysis: "Placeholder risk analysis.",
            diversification_analysis: "Placeholder diversification analysis.",
            market_analysis: "Placeholder market analysis."
          }
        },
        recommendations: ["Consider diversifying across more sectors.", "Review allocation of high-volatility assets."]
      }
    };

    try {
      // Make the API request
      const response = await fetch(`${API_URL}/analyze-portfolio-simple`, {
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
        console.warn(`API returned ${response.status}: ${response.statusText}`);
        // Continue with default data if the API fails
      } else {
        try {
          const text = await response.text();
          
          try {
            // Parse the JSON response
            const data = JSON.parse(text);
            console.log("Portfolio analysis response:", data);
            
            // Variable we can modify (not a constant)
            let analysisData = {...initialDefaultData};
            
            try {
              if (data) {
                // Merge API data with default data to ensure all fields exist
                analysisData = mergeWithDefaults(data, initialDefaultData);
              }
              setAnalysis(analysisData);
              setError('');
            } catch (error) {
              console.error("Error processing response:", error);
              setError("Invalid response data");
              // Still use default data so UI doesn't break
              setAnalysis(analysisData);
            } finally {
              setIsAnalyzing(false);
            }
            
            // ... rest of the function, replace defaultData with analysisData
            console.log("Final processed data:", analysisData);
            
            // Try to get AI insights if we don't already have them
            if (!analysisData.ai_insights || !analysisData.ai_insights.explanations) {
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
                  
                  if (aiData && aiData.ai_insights) {
                    const updatedData = { ...analysisData, ai_insights: aiData.ai_insights };
                    setAnalysis(updatedData);
                  }
                }
              } catch (aiError) {
                console.warn("Error getting AI insights:", aiError);
                // Continue without AI insights if there's an error
              }
            }
            
            // Try to get sentiment data
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
            } catch (sentimentError) {
              console.warn("Error getting sentiment analysis:", sentimentError);
              // Continue without sentiment data if there's an error
            }
            
            // Set active tab to insights and prepare chart data
            setActiveTab('insights');
            const chartData = prepareChartData(analysisData);
            setChartData(chartData);
            
          } catch (parseError) {
            console.error("Error parsing API response:", parseError);
            // Continue with default data if parsing fails
          }
        } catch (parseError) {
          console.error("Error parsing API response:", parseError);
          // Continue with default data if parsing fails
        }
      }
    } catch (error) {
      console.error("Error in portfolio analysis:", error);
      setError("Failed to analyze portfolio. Please try again.");
    }
  };

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
        alpacaSettings = JSON.parse(savedSettings) as AlpacaSettings;
        console.log('Using Alpaca settings:', {
          apiKey: alpacaSettings.apiKey ? '****' + alpacaSettings.apiKey.slice(-4) : 'not set',
          secretKey: alpacaSettings.secretKey ? '****' + alpacaSettings.secretKey.slice(-4) : 'not set',
          isPaper: alpacaSettings.isPaper
        });
      } else {
        console.log('No Alpaca settings found in localStorage');
      }
      
      // Check if API keys are available
      if (!alpacaSettings?.apiKey || !alpacaSettings?.secretKey) {
        setError('Alpaca API keys are required. Please configure them in Settings.');
        setSettingsOpen(true);
        setLoading(false);
        return;
      }

      // Use the AI rebalance explanation endpoint
      const response = await axios.post(`${API_URL}/ai-rebalance-explanation`, {
        allocations: analysis.allocations,
        alpaca_api_key: alpacaSettings.apiKey,
        alpaca_secret_key: alpacaSettings.secretKey,
        use_paper_trading: alpacaSettings.isPaper
      });
      
      console.log('Rebalancing response:', response.data);
      setRebalanceResult(response.data);
    } catch (err: any) {
      if (err.response?.data?.detail) {
        setError(`Error: ${err.response.data.detail}`);
      } else {
        setError('Error rebalancing portfolio. Please try again.');
      }
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  // Initialize chart data
  useEffect(() => {
    if (analysis?.allocations && Object.keys(analysis.allocations).length > 0) {
      try {
        const labels = Object.keys(analysis.allocations);
        const data = Object.values(analysis.allocations)
          .map(value => {
            const numValue = Number(value) * 100;
            if (isNaN(numValue) || !isFinite(numValue)) {
              console.warn('Invalid allocation value:', value);
              return 0;
            }
            return parseFloat(numValue.toFixed(2)); // Round to 2 decimal places
          });

        // Validate data before setting chart data
        if (labels.length !== data.length) {
          console.error('Mismatch between labels and data length');
          return; // Don't update chart data if validation fails
        }

        if (data.some(value => isNaN(value) || !isFinite(value))) {
          console.error('Invalid numeric values in data');
          return; // Don't update chart data if any value is invalid
        }

        const backgroundColors = [
          'rgba(255, 99, 132, 0.8)',
          'rgba(54, 162, 235, 0.8)',
          'rgba(255, 206, 86, 0.8)',
          'rgba(75, 192, 192, 0.8)',
          'rgba(153, 102, 255, 0.8)',
          'rgba(255, 159, 64, 0.8)',
          'rgba(199, 199, 199, 0.8)',
          'rgba(83, 102, 255, 0.8)',
          'rgba(40, 159, 64, 0.8)',
          'rgba(210, 199, 199, 0.8)',
        ];
        const borderColors = backgroundColors.map(color => color.replace('0.8', '1'));

        // Only update chart data if we have valid data
        if (labels.length > 0 && data.length > 0) {
          setChartData({
            labels,
            datasets: [{
              data,
              backgroundColor: backgroundColors.slice(0, labels.length),
              borderColor: borderColors.slice(0, labels.length),
              borderWidth: 1
            }]
          });
        }
      } catch (err) {
        console.error('Error setting chart data:', err);
        // Don't reset chart data on error, keep previous state
      }
    }
  }, [analysis?.allocations]);

  // Handler for period change
  const handlePeriodChange = (period: TimePeriod) => {
    const newStartDate = getStartDate(period);
    console.log(`Setting new start date: ${newStartDate} for period: ${period}`);
    
    setPortfolio({
      ...portfolio,
      start_date: newStartDate
    });
    
    // Debug after state update
    setTimeout(() => {
      console.log("Portfolio after period change:", portfolio);
    }, 100);
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
              {activeTab === 'insights' && analysis && (
                <div className="mt-8 border-t border-gray-700 pt-6">
                  <h2 className="text-2xl font-bold mb-6">Portfolio Analysis Results</h2>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    {/* Performance Chart - Enhanced with S&P 500 comparison */}
                    <div className="bg-[#121a2a] p-6 rounded-xl shadow-lg border border-blue-900/30">
                      <div className="flex justify-between items-center mb-4">
                        <h3 className="text-xl font-semibold text-blue-400">Performance Comparison</h3>
                        <div className="flex items-center space-x-4 text-xs">
                          <div className="flex items-center">
                            <div className="w-3 h-3 rounded-full bg-purple-500 mr-1"></div>
                            <span>Your Portfolio</span>
                          </div>
                          <div className="flex items-center">
                            <div className="w-3 h-3 rounded-full bg-blue-500 mr-1"></div>
                            <span>S&P 500 Index</span>
                          </div>
                        </div>
                      </div>
                      {chartData && <PerformanceChart data={chartData} />}
                      <div className="mt-3 text-sm text-slate-400">
                        <p>This chart compares the performance of your optimized portfolio against the S&P 500 index benchmark.</p>
                      </div>
                    </div>

                    {/* Allocation Chart - Add explanation about optimization */}
                    <div className="bg-[#121a2a] p-6 rounded-xl shadow-lg border border-blue-900/30">
                      <h3 className="text-xl font-semibold mb-3 text-blue-400">Optimized Asset Allocation</h3>
                      <AllocationChart allocations={analysis.allocations} />
                      <div className="mt-3 text-sm text-slate-400">
                        <p>Our algorithm optimizes your allocation based on risk/reward metrics. The weights are adjusted to favor assets with better risk-adjusted returns while maintaining diversification appropriate for your selected risk profile ({portfolio.risk_tolerance}).</p>
                      </div>
                    </div>
                  </div>
                  
                  {/* Portfolio Metrics - Updated with colored status indicators */}
                  <div className="bg-[#121a2a] p-6 rounded-xl shadow-lg border border-blue-900/30 mb-6">
                    <h3 className="text-xl font-semibold mb-4 text-blue-400">Portfolio Summary</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                      <div className="bg-[#0a0a17] p-4 rounded-lg border border-indigo-900/20">
                        <div className="flex justify-between mb-1">
                          <div className="text-slate-400 text-sm">Expected Return</div>
                          <div className="h-2 w-2 rounded-full bg-green-500"></div>
                        </div>
                        <div className="text-xl font-bold text-green-400">{(analysis.metrics.expected_return * 100).toFixed(2)}%</div>
                      </div>
                      <div className="bg-[#0a0a17] p-4 rounded-lg border border-indigo-900/20">
                        <div className="flex justify-between mb-1">
                          <div className="text-slate-400 text-sm">Volatility</div>
                          <div className="h-2 w-2 rounded-full bg-yellow-500"></div>
                        </div>
                        <div className="text-xl font-bold text-yellow-400">{(analysis.metrics.volatility * 100).toFixed(2)}%</div>
                      </div>
                      <div className="bg-[#0a0a17] p-4 rounded-lg border border-indigo-900/20">
                        <div className="flex justify-between mb-1">
                          <div className="text-slate-400 text-sm">Sharpe Ratio</div>
                          <div className="h-2 w-2 rounded-full bg-blue-500"></div>
                        </div>
                        <div className="text-xl font-bold text-blue-400">{analysis.metrics.sharpe_ratio.toFixed(2)}</div>
                      </div>
                      <div className="bg-[#0a0a17] p-4 rounded-lg border border-indigo-900/20">
                        <div className="flex justify-between mb-1">
                          <div className="text-slate-400 text-sm">Max Drawdown</div>
                          <div className="h-2 w-2 rounded-full bg-red-500"></div>
                        </div>
                        <div className="text-xl font-bold text-red-400">{(analysis.metrics.max_drawdown * 100).toFixed(2)}%</div>
                      </div>
                    </div>
                  </div>
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

