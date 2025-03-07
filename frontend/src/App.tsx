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
  const [portfolio, setPortfolio] = useState<Portfolio>({
    tickers: [],
    start_date: getStartDate('1y'), // Default to 1 year
    risk_tolerance: 'medium'
  })
  const [newTicker, setNewTicker] = useState('')
  const [analysis, setAnalysis] = useState<PortfolioAnalysis | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [chartData, setChartData] = useState<any>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [rebalanceResult, setRebalanceResult] = useState<RebalanceResult | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

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

  const analyzePortfolio = async () => {
    try {
      setIsAnalyzing(true);
      setError('');
      
      console.log("Analyzing portfolio with tickers:", portfolio.tickers);
      
      const response = await fetch(`${API_URL}/ai-portfolio-analysis`, {
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
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze portfolio');
      }
      
      const data = await response.json();
      console.log("Portfolio analysis response:", data);
      
      // Validate that we have the required data
      if (!data.allocations || !data.historical_performance) {
        throw new Error('Invalid response data');
      }
      
      setAnalysis(data);
      
      // Prepare chart data
      const chartData = prepareChartData(data);
      setChartData(chartData);
      
    } catch (err: any) {
      console.error('Error analyzing portfolio:', err);
      setError(err.message || 'Failed to analyze portfolio');
    } finally {
      setIsAnalyzing(false);
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
    setPortfolio({
      ...portfolio,
      start_date: getStartDate(period)
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
          data: portfolioValues,
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.5)',
          tension: 0.1
        },
        {
          label: 'Drawdowns',
          data: drawdowns,
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.5)',
          tension: 0.1
        }
      ]
    };
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
            {/* Header */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-center mb-8 flex justify-between items-center"
            >
              <div className="flex-1">
                <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-500 mb-4">
                  Smart Portfolio Manager
                </h1>
                <p className="text-lg md:text-xl text-gray-300">
                  Optimize your investments with AI-powered portfolio analysis
                </p>
              </div>
              
              {/* Settings Button */}
              <button
                onClick={() => setSettingsOpen(true)}
                className="p-2 rounded-full hover:bg-gray-700 transition-colors self-start"
                aria-label="Settings"
              >
                <Cog6ToothIcon className="h-6 w-6 text-gray-300" />
              </button>
            </motion.div>

            {/* Main card */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="relative bg-[#0a0a0a]/95 backdrop-blur-xl rounded-3xl p-6 md:p-8 shadow-2xl border border-white/10"
            >
              <div className="space-y-10">
                {/* AI Ticker Suggestions */}
                <div>
                  <h2 className="text-2xl font-bold text-center mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                    AI Ticker Suggestions
                  </h2>
                  
                  {/* Selected Tickers Display */}
                  <div className="mb-6 flex flex-wrap gap-2">
                    {portfolio.tickers.map((ticker) => (
                      <motion.div
                        key={ticker}
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        exit={{ scale: 0 }}
                        className="px-3 py-1.5 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-lg
                                 border border-purple-500/30 text-white flex items-center gap-2"
                      >
                        <span className="font-medium">{ticker}</span>
                        <button
                          onClick={() => removeTicker(ticker)}
                          className="hover:bg-white/10 rounded-full w-5 h-5 flex items-center justify-center
                                   transition-colors text-purple-400 hover:text-purple-300"
                        >
                          ×
                        </button>
                      </motion.div>
                    ))}
                  </div>
                  
                  <TickerSuggestions onSelectTicker={addTicker} />
                </div>

                {/* Time Period and Risk Selection */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Time Period Selector */}
                  <div className="flex gap-2 justify-start items-center flex-wrap">
                    {(['3m', '6m', '1y', '5y', 'max'] as TimePeriod[]).map((period) => (
                      <button
                        key={period}
                        onClick={() => handlePeriodChange(period)}
                        className={`px-4 py-2 rounded-full transition-all ${
                          portfolio.start_date === getStartDate(period)
                            ? 'bg-gradient-to-r from-purple-500 to-blue-500 text-white'
                            : 'bg-white/5 hover:bg-white/10 text-white/80'
                        }`}
                      >
                        {period.toUpperCase()}
                      </button>
                    ))}
                  </div>

                  {/* Risk Tolerance Selector */}
                  <select
                    value={portfolio.risk_tolerance}
                    onChange={(e) => setPortfolio({ ...portfolio, risk_tolerance: e.target.value as 'low' | 'medium' | 'high' })}
                    className="px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="low">Low Risk</option>
                    <option value="medium">Medium Risk</option>
                    <option value="high">High Risk</option>
                  </select>
                </div>

                {/* Action Buttons */}
                <div className="flex flex-col md:flex-row gap-4 justify-center">
                  <button
                    onClick={analyzePortfolio}
                    disabled={loading || portfolio.tickers.length === 0 || !portfolio.start_date}
                    className="px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg font-medium 
                             hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed
                             w-full md:w-auto text-center"
                  >
                    {isAnalyzing ? 'Analyzing...' : 'Analyze Portfolio'}
                  </button>
                  
                  <button
                    onClick={rebalancePortfolio}
                    disabled={loading || !analysis}
                    className="px-8 py-4 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg font-medium 
                             hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed
                             w-full md:w-auto text-center"
                  >
                    {loading ? 'Rebalancing...' : 'Rebalance Portfolio'}
                  </button>
                </div>

                {/* Loading State - Modified to be less intrusive */}
                {loading && (
                  <div className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50">
                    <div className="bg-[#2a2a2a]/95 backdrop-blur-xl rounded-xl p-8 border border-white/20 shadow-2xl">
                      <div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mb-4 mx-auto"></div>
                      <p className="text-lg font-medium text-white text-center">Analyzing Portfolio...</p>
                      <p className="text-sm text-gray-400 text-center mt-2">This may take a few moments</p>
                    </div>
                  </div>
                )}

                {/* Analysis Results */}
                {analysis && !loading && (
                  <motion.div
                    key="analysis-results"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.5 }}
                  >
                    <motion.div 
                      initial={{ y: 20, opacity: 0 }}
                      animate={{ y: 0, opacity: 1 }}
                      transition={{ delay: 0.3, duration: 0.5 }}
                      className="text-center mb-10"
                    >
                      <h2 className="text-3xl font-bold mb-3 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                        Portfolio Analysis Results
                      </h2>
                      <p className="text-lg text-gray-300">
                        Optimized asset allocation based on your preferences
                      </p>
                    </motion.div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
                      <div>
                        <AllocationChart allocations={analysis.allocations} />
                      </div>
                      <div>
                        <PerformanceChart data={chartData} />
                      </div>
                    </div>

                    <div className="space-y-10">
                      {/* Portfolio Summary Card */}
                      {analysis && analysis.metrics && (
                        <div>
                          <PortfolioSummary metrics={analysis.metrics} />
                          <div className="flex justify-center mb-6">
                            <div className="bg-[#0a0a0a]/80 backdrop-blur-sm border border-white/10 rounded-xl p-6 w-full">
                              <div className="flex justify-between items-center mb-4">
                                <h3 className="text-lg font-bold text-blue-400">AI Analysis</h3>
                                <PortfolioExplanation analysis={analysis} language="en" />
                              </div>
                              <p className="text-gray-300">
                                Get detailed AI-powered insights about your portfolio including risk assessment, 
                                diversification analysis, and personalized recommendations.
                              </p>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Individual Stock Analysis Section */}
                      {analysis && analysis.asset_metrics && Object.keys(analysis.asset_metrics).length > 0 && (
                        <div className="space-y-6">
                          <div className="flex items-center justify-between">
                            <h2 className="text-xl font-semibold text-purple-400">Individual Stock Analysis</h2>
                            <Tooltip content={{
                              title: "Individual Stock Metrics",
                              description: "Detailed analysis of each stock in your portfolio, including returns, risk metrics, and market performance indicators."
                            }}>
                              <Button variant="ghost" className="p-2">
                                <InformationCircleIcon className="w-5 h-5" />
                              </Button>
                            </Tooltip>
                          </div>
                          
                          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            {Object.entries(analysis.asset_metrics).map(([ticker, assetMetric]) => (
                              <StockAnalysis 
                                key={ticker} 
                                ticker={ticker} 
                                metrics={assetMetric}
                              />
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Historical Performance Chart */}
                      <div className="bg-[#2a2a2a]/95 backdrop-blur-md border border-white/20 rounded-xl p-6">
                        <div className="flex justify-between items-center mb-4">
                          <h3 className="font-semibold text-purple-400">Historical Performance</h3>
                          <Tooltip
                            content="This chart compares your portfolio's performance against the S&P 500 benchmark. The purple line shows your portfolio value, the blue line shows the S&P 500, and the red dotted line shows rolling volatility (risk level over time)."
                            side="left"
                            sideOffset={5}
                          >
                            <button 
                              type="button"
                              className="p-2 hover:bg-white/10 rounded-full transition-colors"
                            >
                              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-purple-400">
                                <circle cx="12" cy="12" r="10"/>
                                <path d="M12 16v-4"/>
                                <path d="M12 8h.01"/>
                              </svg>
                            </button>
                          </Tooltip>
                        </div>
                        <div className="w-full h-[300px]">
                          {analysis?.historical_performance?.dates?.length > 0 ? (
                            <Line
                              data={{
                                labels: analysis.historical_performance.dates,
                                datasets: [
                                  {
                                    label: 'Portfolio Value',
                                    data: analysis.historical_performance.portfolio_values?.map(value => {
                                      const numValue = Number(value);
                                      return isNaN(numValue) || !isFinite(numValue) ? null : numValue;
                                    }) || [],
                                    borderColor: 'rgb(147, 51, 234)',
                                    backgroundColor: 'rgba(147, 51, 234, 0.1)',
                                    tension: 0.4,
                                    fill: true,
                                    yAxisID: 'y'
                                  },
                                  {
                                    label: 'S&P 500',
                                    data: analysis.market_comparison?.market_values?.map(value => {
                                      const numValue = Number(value);
                                      return isNaN(numValue) || !isFinite(numValue) ? null : numValue;
                                    }) || [],
                                    borderColor: 'rgb(59, 130, 246)',
                                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                    tension: 0.4,
                                    fill: true,
                                    yAxisID: 'y'
                                  },
                                  {
                                    label: 'Rolling Volatility',
                                    data: analysis.historical_performance.rolling_volatility?.map(value => {
                                      const numValue = Number(value);
                                      return isNaN(numValue) || !isFinite(numValue) ? null : numValue;
                                    }) || [],
                                    borderColor: 'rgb(239, 68, 68)',
                                    borderDash: [5, 5],
                                    tension: 0.4,
                                    fill: false,
                                    yAxisID: 'y1'
                                  }
                                ]
                              }}
                              options={{
                                responsive: true,
                                maintainAspectRatio: false,
                                interaction: {
                                  mode: 'index',
                                  intersect: false,
                                },
                                plugins: {
                                  legend: {
                                    position: 'top' as const,
                                    labels: {
                                      color: 'white'
                                    }
                                  },
                                  tooltip: {
                                    enabled: true,
                                    mode: 'index',
                                    intersect: false,
                                    callbacks: {
                                      label: function(context) {
                                        const value = context.raw as number;
                                        if (context.dataset.label === 'Rolling Volatility') {
                                          return `${context.dataset.label}: ${(value * 100).toFixed(2)}%`;
                                        }
                                        return `${context.dataset.label}: ${value.toFixed(2)}`;
                                      }
                                    }
                                  }
                                },
                                scales: {
                                  x: {
                                    grid: {
                                      color: 'rgba(255, 255, 255, 0.1)'
                                    },
                                    ticks: {
                                      color: 'white'
                                    }
                                  },
                                  y: {
                                    type: 'linear',
                                    display: true,
                                    position: 'left',
                                    grid: {
                                      color: 'rgba(255, 255, 255, 0.1)'
                                    },
                                    ticks: {
                                      color: 'white'
                                    }
                                  },
                                  y1: {
                                    type: 'linear',
                                    display: true,
                                    position: 'right',
                                    grid: {
                                      drawOnChartArea: false
                                    },
                                    ticks: {
                                      color: 'white',
                                      callback: function(value) {
                                        return (Number(value) * 100).toFixed(1) + '%';
                                      }
                                    }
                                  }
                                }
                              }}
                            />
                          ) : (
                            <div className="h-full flex items-center justify-center text-gray-400">
                              No historical performance data available
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Drawdown Chart */}
                      <div className="bg-[#2a2a2a]/95 backdrop-blur-md border border-white/20 rounded-xl p-6">
                        <div className="flex justify-between items-center mb-4">
                          <h3 className="font-semibold text-purple-400">Drawdown Analysis</h3>
                          <Tooltip
                            content="The drawdown chart shows how much your portfolio has declined from its peak value at any given time. This helps visualize the maximum losses you might experience and how long it takes to recover from them."
                            side="left"
                            sideOffset={5}
                          >
                            <button 
                              type="button"
                              className="p-2 hover:bg-white/10 rounded-full transition-colors"
                            >
                              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-purple-400">
                                <circle cx="12" cy="12" r="10"/>
                                <path d="M12 16v-4"/>
                                <path d="M12 8h.01"/>
                              </svg>
                            </button>
                          </Tooltip>
                        </div>
                        <div className="w-full h-[200px]">
                          <Line
                            data={{
                              labels: analysis.historical_performance.dates,
                              datasets: [
                                {
                                  label: 'Drawdown',
                                  data: analysis.historical_performance.drawdowns,
                                  borderColor: 'rgb(239, 68, 68)',
                                  backgroundColor: 'rgba(239, 68, 68, 0.1)',
                                  tension: 0.4,
                                  fill: true
                                }
                              ]
                            }}
                            options={{
                              responsive: true,
                              maintainAspectRatio: false,
                              plugins: {
                                legend: {
                                  display: true,
                                  position: 'top' as const,
                                  labels: {
                                    color: 'white'
                                  }
                                }
                              },
                              scales: {
                                x: {
                                  grid: {
                                    color: 'rgba(255, 255, 255, 0.1)'
                                  },
                                  ticks: {
                                    color: 'white'
                                  }
                                },
                                y: {
                                  grid: {
                                    color: 'rgba(255, 255, 255, 0.1)'
                                  },
                                  ticks: {
                                    color: 'white',
                                    callback: function(value) {
                                      return (Number(value) * 100).toFixed(1) + '%'
                                    }
                                  }
                                }
                              }
                            }}
                          />
                        </div>
                      </div>

                      {/* Asset Metrics */}
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mt-6">
                        {Object.entries(analysis.asset_metrics).map(([ticker, assetMetric]) => (
                          <div key={ticker} className="bg-[#2a2a2a]/95 backdrop-blur-md border border-white/20 rounded-xl p-6">
                            <h3 className="font-semibold mb-4 text-purple-400">{ticker} Metrics</h3>
                            <div className="space-y-3 text-white">
                              {analysis && analysis.metrics && (
                                <>
                                  <p className="flex justify-between">
                                    <span>Expected Return:</span>
                                    <span className="font-medium text-green-400">
                                      {(analysis.metrics.expected_return * 100).toFixed(2)}%
                                    </span>
                                  </p>
                                  <p className="flex justify-between">
                                    <span>Volatility:</span>
                                    <span className="font-medium text-yellow-400">
                                      {(analysis.metrics.volatility * 100).toFixed(2)}%
                                    </span>
                                  </p>
                                  <p className="flex justify-between">
                                    <span>Sharpe Ratio:</span>
                                    <span className="font-medium text-blue-400">
                                      {analysis.metrics.sharpe_ratio.toFixed(2)}
                                    </span>
                                  </p>
                                  <p className="flex justify-between">
                                    <span>Sortino Ratio:</span>
                                    <span className="font-medium text-purple-400">
                                      {analysis.metrics.sortino_ratio.toFixed(2)}
                                    </span>
                                  </p>
                                  <p className="flex justify-between">
                                    <span>Market Beta:</span>
                                    <span className={`font-medium ${analysis.metrics.beta > 1 ? 'text-red-400' : 'text-green-400'}`}>
                                      {analysis.metrics.beta.toFixed(2)}
                                    </span>
                                  </p>
                                  <p className="flex justify-between">
                                    <span>Maximum Drawdown:</span>
                                    <span className="font-medium text-red-400">
                                      {(analysis.metrics.max_drawdown * 100).toFixed(2)}%
                                    </span>
                                  </p>
                                  <p className="flex justify-between">
                                    <span>Value at Risk (95%):</span>
                                    <span className="font-medium text-orange-400">
                                      {(analysis.metrics.var_95 * 100).toFixed(2)}%
                                    </span>
                                  </p>
                                </>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* Rebalance Results */}
                {rebalanceResult && (
                  <div className="mt-6">
                    <h2 className="text-2xl font-bold mb-4">Rebalance Results</h2>
                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
                      <RebalanceExplanation result={rebalanceResult} language="en" />
                    </div>
                  </div>
                )}

                {/* Error Message */}
                {error && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-red-400 text-center p-4 bg-red-500/10 rounded-lg border border-red-500/20"
                  >
                    {error}
                  </motion.div>
                )}
              </div>
            </motion.div>
          </div>
        </div>
      </div>

      {/* Settings Dialog */}
      <Settings isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </TooltipProvider>
  )
}

export default App
