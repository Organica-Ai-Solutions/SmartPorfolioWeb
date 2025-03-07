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
                  
                  {/* Analysis Results Section - Only show when analysis data exists */}
                  {analysis && (
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

                      {/* New Risk Analysis Section */}
                      <div className="bg-[#121a2a] p-6 rounded-xl shadow-lg border border-blue-900/30 mb-6">
                        <h3 className="text-xl font-semibold mb-4 text-red-400">Risk Analysis</h3>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                          <div className="bg-[#0a0a17] p-4 rounded-lg border border-indigo-900/20">
                            <h4 className="text-base font-medium text-slate-300 mb-2">Value at Risk (95%)</h4>
                            <div className="text-xl font-bold text-red-400">{(analysis.metrics.var_95 * 100).toFixed(2)}%</div>
                            <p className="text-xs text-slate-500 mt-1">Maximum expected daily loss</p>
                          </div>
                          <div className="bg-[#0a0a17] p-4 rounded-lg border border-indigo-900/20">
                            <h4 className="text-base font-medium text-slate-300 mb-2">Beta</h4>
                            <div className="text-xl font-bold text-yellow-400">{analysis.metrics.beta.toFixed(2)}</div>
                            <p className="text-xs text-slate-500 mt-1">Portfolio sensitivity to market</p>
                          </div>
                          <div className="bg-[#0a0a17] p-4 rounded-lg border border-indigo-900/20">
                            <h4 className="text-base font-medium text-slate-300 mb-2">Sortino Ratio</h4>
                            <div className="text-xl font-bold text-blue-400">{analysis.metrics.sortino_ratio.toFixed(2)}</div>
                            <p className="text-xs text-slate-500 mt-1">Return per unit of downside risk</p>
                          </div>
                        </div>
                        
                        {/* Risk Gauge */}
                        <div className="mt-6 bg-[#0a0a17] p-4 rounded-lg border border-indigo-900/20">
                          <h4 className="text-base font-medium text-slate-300 mb-3">Risk Level</h4>
                          <div className="relative h-6 bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 rounded-full overflow-hidden">
                            {/* Dynamic marker based on volatility/risk ratio */}
                            <div 
                              className="absolute top-0 w-3 h-6 bg-white"
                              style={{ 
                                left: `${Math.min(Math.max((analysis.metrics.volatility / 0.3) * 100, 0), 95)}%`,
                                transform: 'translateX(-50%)'
                              }}
                            ></div>
                          </div>
                          <div className="flex justify-between text-xs text-slate-400 mt-1">
                            <span>Conservative</span>
                            <span>Moderate</span>
                            <span>Aggressive</span>
                          </div>
                        </div>
                      </div>
                      
                      {/* Asset Metrics - Enhanced with trend indicators */}
                      <div className="bg-[#121a2a] p-6 rounded-xl shadow-lg border border-blue-900/30 mb-6">
                        <h3 className="text-xl font-semibold mb-4 text-blue-400">Asset Details</h3>
                        <div className="overflow-x-auto">
                          <table className="min-w-full">
                            <thead>
                              <tr>
                                <th className="py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Asset</th>
                                <th className="py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Weight</th>
                                <th className="py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Return</th>
                                <th className="py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Volatility</th>
                                <th className="py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Beta</th>
                                <th className="py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Alpha</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-800">
                          {Object.entries(analysis.asset_metrics).map(([ticker, metrics]) => (
                                <tr key={ticker} className="hover:bg-slate-800/50 transition-colors">
                                  <td className="py-4 text-sm font-medium text-white">{ticker}</td>
                                  <td className="py-4 text-sm text-blue-400 font-semibold">{(metrics.weight * 100).toFixed(2)}%</td>
                                  <td className="py-4 text-sm">
                                    <span className={metrics.annual_return > 0 ? 'text-green-400' : 'text-red-400'}>
                                      {(metrics.annual_return * 100).toFixed(2)}%
                                      {metrics.annual_return > 0 ? ' ↑' : ' ↓'}
                                    </span>
                                  </td>
                                  <td className="py-4 text-sm text-yellow-400">{(metrics.annual_volatility * 100).toFixed(2)}%</td>
                                  <td className="py-4 text-sm">
                                    <span className={metrics.beta < 1 ? 'text-green-400' : 'text-yellow-400'}>
                                      {metrics.beta.toFixed(2)}
                                    </span>
                                  </td>
                                  <td className="py-4 text-sm">
                                    <span className={(metrics.alpha || 0) > 0 ? 'text-green-400' : 'text-red-400'}>
                                      {((metrics.alpha || 0) * 100).toFixed(2)}%
                                    </span>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                      
                      {/* AI Insights - Completely Revamped */}
                      {analysis.ai_insights && (
                        <div className="bg-[#121a2a] p-6 rounded-xl shadow-lg border border-blue-900/30 mb-6">
                          <div className="flex items-center justify-between mb-6">
                            <h3 className="text-xl font-semibold text-purple-400">AI Portfolio Insights</h3>
                            <div className="bg-gradient-to-r from-blue-500 to-purple-500 text-white text-xs px-3 py-1 rounded-full font-medium">
                              Powered by AI
                            </div>
                          </div>
                          
                          {/* AI Summary with Gradient Card */}
                          <div className="bg-gradient-to-br from-[#1a1a3a] to-[#0a0a20] p-5 rounded-lg border border-indigo-900/40 mb-6 shadow-lg">
                            <div className="flex items-start gap-4">
                              <div className="p-3 bg-purple-900/30 rounded-full mt-1">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                            </svg>
                      </div>
                              <div className="flex-1">
                                <h4 className="text-lg font-semibold text-gradient bg-gradient-to-r from-blue-400 to-purple-400 mb-3">Portfolio Analysis Summary</h4>
                                <p className="text-white/90 leading-relaxed mb-4">
                                  {/* Safe access to nested objects */}
                                  {analysis.ai_insights?.explanations?.english?.summary || 
                                   "Your portfolio shows a balanced approach with moderate risk and potential for growth. The allocation is optimized based on historical performance and volatility metrics to maximize your risk-adjusted returns."}
                                </p>
                                
                                {/* Key metrics visualization */}
                                <div className="grid grid-cols-3 gap-3 my-4">
                                  <div className="bg-[#0a0a20]/70 p-3 rounded-lg">
                                    <div className="text-xs text-blue-300 mb-1">Risk Level</div>
                                    <div className="text-lg font-bold text-white">
                                      {analysis.metrics.volatility < 0.15 ? 'Low' : 
                                       analysis.metrics.volatility < 0.25 ? 'Moderate' : 'High'}
                          </div>
                                  </div>
                                  <div className="bg-[#0a0a20]/70 p-3 rounded-lg">
                                    <div className="text-xs text-green-300 mb-1">Growth Potential</div>
                                    <div className="text-lg font-bold text-white">
                                      {analysis.metrics.expected_return < 0.1 ? 'Conservative' : 
                                       analysis.metrics.expected_return < 0.2 ? 'Moderate' : 'Aggressive'}
                                    </div>
                                  </div>
                                  <div className="bg-[#0a0a20]/70 p-3 rounded-lg">
                                    <div className="text-xs text-yellow-300 mb-1">Diversification</div>
                                    <div className="text-lg font-bold text-white">
                                      {Object.keys(analysis.asset_metrics).length <= 2 ? 'Limited' :
                                       Object.keys(analysis.asset_metrics).length <= 5 ? 'Moderate' : 'Strong'}
                                    </div>
                                  </div>
                                </div>
                              </div>
                      </div>
      </div>

                          {/* Recommendations section */}
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* Left column - Recommendations */}
                            <div className="bg-[#0a0a17] p-5 rounded-lg border border-indigo-900/20">
                              <div className="flex items-center mb-4">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-yellow-400 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                            </svg>
                                <h4 className="text-base font-semibold text-yellow-400">Recommended Actions</h4>
                      </div>
                              
                              {analysis.ai_insights.recommendations && 
                               Array.isArray(analysis.ai_insights.recommendations) && 
                               analysis.ai_insights.recommendations.length > 0 ? (
                                <ul className="space-y-2">
                                  {analysis.ai_insights.recommendations.map((rec, idx) => (
                                    <li key={idx} className="flex items-start">
                                      <div className="bg-yellow-500/20 text-yellow-500 rounded-full h-5 w-5 flex items-center justify-center text-xs mr-3 mt-0.5">{idx + 1}</div>
                                      <p className="text-white/80 text-sm">{rec}</p>
                                    </li>
                                  ))}
                                </ul>
                              ) : (
                                <ul className="space-y-2">
                                  <li className="flex items-start">
                                    <div className="bg-yellow-500/20 text-yellow-500 rounded-full h-5 w-5 flex items-center justify-center text-xs mr-3 mt-0.5">1</div>
                                    <p className="text-white/80 text-sm">Consider increasing diversification by adding assets from different sectors.</p>
                                  </li>
                                  <li className="flex items-start">
                                    <div className="bg-yellow-500/20 text-yellow-500 rounded-full h-5 w-5 flex items-center justify-center text-xs mr-3 mt-0.5">2</div>
                                    <p className="text-white/80 text-sm">Rebalance portfolio regularly to maintain optimized allocation.</p>
                                  </li>
                                  <li className="flex items-start">
                                    <div className="bg-yellow-500/20 text-yellow-500 rounded-full h-5 w-5 flex items-center justify-center text-xs mr-3 mt-0.5">3</div>
                                    <p className="text-white/80 text-sm">Monitor market conditions and adjust risk tolerance as needed.</p>
                                  </li>
                                </ul>
                              )}
                      </div>
                            
                            {/* Right column - Market outlook */}
                            <div className="bg-[#0a0a17] p-5 rounded-lg border border-indigo-900/20">
                              <div className="flex items-center mb-4">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-400 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 13v-1m4 1v-3m4 3V8M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                                </svg>
                                <h4 className="text-base font-semibold text-blue-400">Market Outlook</h4>
                    </div>

                              {analysis.ai_insights.market_outlook && 
                               typeof analysis.ai_insights.market_outlook === 'object' ? (
                                <div className="space-y-4">
                                  <div className="space-y-2">
                                    <div className="flex justify-between items-center">
                                      <span className="text-sm text-slate-400">Short Term</span>
                                      <span className="text-sm font-medium text-white px-2 py-1 rounded bg-blue-500/20">
                                        {analysis.ai_insights.market_outlook.short_term || 'Neutral'}
                              </span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                      <span className="text-sm text-slate-400">Medium Term</span>
                                      <span className="text-sm font-medium text-white px-2 py-1 rounded bg-blue-500/20">
                                        {analysis.ai_insights.market_outlook.medium_term || 'Neutral'}
                              </span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                      <span className="text-sm text-slate-400">Long Term</span>
                                      <span className="text-sm font-medium text-white px-2 py-1 rounded bg-blue-500/20">
                                        {analysis.ai_insights.market_outlook.long_term || 'Neutral'}
                              </span>
                                    </div>
                                  </div>
                                  
                                  <div className="pt-3 border-t border-slate-800">
                                    <h5 className="text-sm font-medium text-slate-300 mb-2">Key Market Drivers</h5>
                                    <div className="flex flex-wrap gap-2">
                                      {(analysis.ai_insights.market_outlook as any)?.key_drivers && 
                                       Array.isArray((analysis.ai_insights.market_outlook as any).key_drivers) ? (
                                        (analysis.ai_insights.market_outlook as any).key_drivers.map((driver: string, idx: number) => (
                                          <span key={idx} className="text-xs bg-blue-900/30 text-blue-300 px-2 py-1 rounded">
                                            {driver}
                              </span>
                                        ))
                                      ) : (
                                        <>
                                          <span className="text-xs bg-blue-900/30 text-blue-300 px-2 py-1 rounded">
                                            Inflation
                              </span>
                                          <span className="text-xs bg-blue-900/30 text-blue-300 px-2 py-1 rounded">
                                            Interest Rates
                                          </span>
                                          <span className="text-xs bg-blue-900/30 text-blue-300 px-2 py-1 rounded">
                                            Economic Growth
                                          </span>
                                        </>
                                      )}
                          </div>
                        </div>
                    </div>
                              ) : (
                                <p className="text-white/70 text-sm">Market outlook data not available.</p>
                              )}
                    </div>
                          </div>
                        </div>
                      )}
                    </div>
                )}
              </div>
              )}
              {activeTab === 'watchlist' && <WatchlistTab stockData={stockData} />}
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
