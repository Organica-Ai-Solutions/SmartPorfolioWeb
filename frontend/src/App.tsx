import { useState, useEffect } from 'react'
import { Portfolio, PortfolioAnalysis } from './types/portfolio'
import axios from 'axios'
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement } from 'chart.js'
import { Pie, Line } from 'react-chartjs-2'
import { SparklesCore } from './components/ui/sparkles'
import { motion } from 'framer-motion'
import { TickerSuggestions } from './components/TickerSuggestions'
import { PortfolioExplanation } from './components/PortfolioExplanation'
import { Tooltip as UiTooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './components/ui/tooltip'

ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement
)

const API_URL = 'http://localhost:8000'

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
      if (portfolio.tickers.length < 2) {
        setError('Please add at least 2 tickers to analyze')
        return
      }
      if (!portfolio.start_date) {
        setError('Please select a start date')
        return
      }

      // Validate tickers
      const invalidTickers = portfolio.tickers.filter(ticker => !/^[A-Z]+$/.test(ticker))
      if (invalidTickers.length > 0) {
        setError(`Invalid ticker(s): ${invalidTickers.join(', ')}`)
        return
      }

      setLoading(true)
      setError('')
      const previousAnalysis = analysis
      
      console.log('Making API request with data:', {
        tickers: portfolio.tickers,
        start_date: portfolio.start_date,
        risk_tolerance: portfolio.risk_tolerance
      })

      const response = await axios.post(`${API_URL}/analyze-portfolio`, {
        tickers: portfolio.tickers,
        start_date: portfolio.start_date,
        risk_tolerance: portfolio.risk_tolerance
      })

      // Log response for debugging
      console.log('Backend response:', response.data)

      // Check for optimization errors first
      if (response.data.error) {
        let errorMessage = response.data.error;
        if (errorMessage.includes('minimum volatility')) {
          errorMessage = 'Portfolio is too volatile for the selected risk level. Try adding more diverse assets or selecting a higher risk tolerance.';
        } else if (errorMessage.includes('solver status: infeasible')) {
          errorMessage = 'Unable to optimize portfolio with current constraints. Try adding more diverse assets.';
        } else if (errorMessage.includes('tuple index out of range')) {
          errorMessage = 'Error in portfolio optimization. Try selecting different assets or a different risk level.';
        }
        setError(errorMessage);
        setAnalysis(previousAnalysis); // Keep previous analysis on error
        return;
      }

      // Validate response data structure
      if (!response.data) {
        setError('No data received from server');
        setAnalysis(previousAnalysis); // Keep previous analysis
        return;
      }

      // Validate required data structures
      const validationErrors = []
      if (!response.data.allocations && !response.data.allocation) {
        validationErrors.push('Invalid allocation data')
      }
      if (!response.data.metrics || typeof response.data.metrics !== 'object') {
        validationErrors.push('Invalid metrics data')
      }
      if (!response.data.historical_performance || !response.data.historical_performance.dates) {
        validationErrors.push('Invalid historical performance data')
      }

      if (validationErrors.length > 0) {
        setError(`Invalid response format: ${validationErrors.join(', ')}`);
        setAnalysis(previousAnalysis); // Keep previous analysis
        return;
      }

      // Transform data with strict validation and safe defaults
      const transformedAnalysis = {
        allocations: response.data.allocations || response.data.allocation.reduce((acc: {[key: string]: number}, curr: any) => {
          if (curr && typeof curr.ticker === 'string' && typeof curr.weight === 'number' && !isNaN(curr.weight) && isFinite(curr.weight)) {
            acc[curr.ticker] = curr.weight
          }
          return acc
        }, {}),
        metrics: {
          expected_return: Number(response.data.metrics.expected_return) || 0,
          volatility: Number(response.data.metrics.volatility) || 0,
          sharpe_ratio: Number(response.data.metrics.sharpe_ratio) || 0,
          sortino_ratio: Number(response.data.metrics.sortino_ratio) || 0,
          beta: Number(response.data.metrics.beta) || 0,
          max_drawdown: Number(response.data.metrics.max_drawdown) || 0,
          var_95: Number(response.data.metrics.var_95) || 0,
          cvar_95: Number(response.data.metrics.cvar_95) || 0
        },
        historical_performance: {
          dates: Array.isArray(response.data.historical_performance.dates) 
            ? response.data.historical_performance.dates 
            : [],
          portfolio_values: Array.isArray(response.data.historical_performance.portfolio_values)
            ? response.data.historical_performance.portfolio_values.map((v: any) => {
                const num = Number(v);
                return isNaN(num) || !isFinite(num) ? null : num;
              }).filter((v: number | null): v is number => v !== null)
            : [],
          drawdowns: Array.isArray(response.data.historical_performance.drawdowns)
            ? response.data.historical_performance.drawdowns.map((v: any) => {
                const num = Number(v);
                return isNaN(num) || !isFinite(num) ? null : num;
              }).filter((v: number | null): v is number => v !== null)
            : [],
          rolling_volatility: Array.isArray(response.data.historical_performance.rolling_volatility)
            ? response.data.historical_performance.rolling_volatility.map((v: any) => {
                const num = Number(v);
                return isNaN(num) || !isFinite(num) ? null : num;
              }).filter((v: number | null): v is number => v !== null)
            : [],
          rolling_sharpe: Array.isArray(response.data.historical_performance.rolling_sharpe)
            ? response.data.historical_performance.rolling_sharpe.map((v: any) => {
                const num = Number(v);
                return isNaN(num) || !isFinite(num) ? null : num;
              }).filter((v: number | null): v is number => v !== null)
            : []
        },
        market_comparison: {
          dates: Array.isArray(response.data.market_comparison?.dates)
            ? response.data.market_comparison.dates
            : [],
          market_values: Array.isArray(response.data.market_comparison?.market_values)
            ? response.data.market_comparison.market_values.map((v: any) => {
                const num = Number(v);
                return isNaN(num) || !isFinite(num) ? null : num;
              }).filter((v: number | null): v is number => v !== null)
            : [],
          relative_performance: Array.isArray(response.data.market_comparison?.relative_performance)
            ? response.data.market_comparison.relative_performance.map((v: any) => {
                const num = Number(v);
                return isNaN(num) || !isFinite(num) ? null : num;
              }).filter((v: number | null): v is number => v !== null)
            : []
        },
        asset_metrics: response.data.asset_metrics || {},
        ai_insights: response.data.ai_insights || {
          explanations: {
            summary: { en: '', es: '' },
            risk_analysis: { en: '', es: '' },
            diversification_analysis: { en: '', es: '' },
            market_context: { en: '', es: '' },
            stress_test_interpretation: { en: '', es: '' }
          }
        },
        discrete_allocation: response.data.discrete_allocation || {
          shares: {},
          leftover: 0
        }
      }

      // Validate transformed data before setting state
      if (!transformedAnalysis.allocations || Object.keys(transformedAnalysis.allocations).length === 0) {
        setError('No valid allocation data after transformation');
        setAnalysis(previousAnalysis); // Keep previous analysis
        return;
      }

      // Validate historical performance data
      const hasValidHistoricalData = 
        transformedAnalysis.historical_performance.dates.length > 0 && 
        transformedAnalysis.historical_performance.portfolio_values.length > 0 &&
        transformedAnalysis.historical_performance.portfolio_values.some((v: number | null) => v !== null && !isNaN(v) && isFinite(v));

      if (!hasValidHistoricalData) {
        setError('Invalid or missing historical performance data');
        setAnalysis(previousAnalysis); // Keep previous analysis
        return;
      }

      // Only update state if we have valid data
      setAnalysis(transformedAnalysis)
      setError('')
    } catch (err: any) {
      console.error('Portfolio analysis error:', err)
      let errorMessage = err.response?.data?.detail || err.message || 'Error analyzing portfolio'
      
      // Make error messages more user-friendly
      if (errorMessage.includes('minimum volatility')) {
        errorMessage = 'Portfolio is too volatile for the selected risk level. Try adding more diverse assets or selecting a higher risk tolerance.';
      } else if (errorMessage.includes('solver status: infeasible')) {
        errorMessage = 'Unable to optimize portfolio with current constraints. Try adding more diverse assets.';
      } else if (errorMessage.includes('tuple index out of range')) {
        errorMessage = 'Error in portfolio optimization. Try selecting different assets or a different risk level.';
      }
      
      setError(errorMessage)
      // Keep previous analysis state on error
    } finally {
      setLoading(false)
    }
  }

  const rebalancePortfolio = async () => {
    if (!analysis) return

    try {
      setLoading(true)
      setError('')
      const response = await axios.post(`${API_URL}/rebalance-portfolio`, {
        allocations: analysis.allocations
      })
      console.log('Rebalancing response:', response.data)
    } catch (err) {
      setError('Error rebalancing portfolio. Please try again.')
      console.error(err)
    } finally {
      setLoading(false)
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

  return (
    <TooltipProvider>
      <div className="relative min-h-screen bg-gradient-to-b from-[#1a1a1a] to-[#0d0d0d] text-white">
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

        {/* Main content */}
        <div className="relative z-10 min-h-screen py-8 px-4">
          <div className="container mx-auto max-w-6xl">
            {/* Header */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-center mb-8"
            >
              <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-500 mb-4">
                Smart Portfolio Manager
              </h1>
              <p className="text-lg md:text-xl text-gray-300">
                Optimize your investments with AI-powered portfolio analysis
              </p>
            </motion.div>

            {/* Main card */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="relative bg-[#2a2a2a]/95 backdrop-blur-xl rounded-3xl p-6 md:p-8 shadow-2xl border border-white/20"
            >
              <div className="space-y-8">
                {/* AI Ticker Suggestions */}
                <div className="mb-8">
                  <h2 className="text-2xl font-bold text-center mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                    AI Ticker Suggestions
                  </h2>
                  <TickerSuggestions onSelectTicker={addTicker} />
                </div>

                {/* Time Period and Risk Selection */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
                    {loading ? 'Analyzing...' : 'Analyze Portfolio'}
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
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.5 }}
                    className="mt-8 space-y-6"
                  >
                    <h2 className="text-2xl font-bold text-center bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                      Portfolio Analysis Results
                    </h2>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                      {/* Metrics Card */}
                      <div className="bg-[#2a2a2a]/95 backdrop-blur-md border border-white/20 rounded-xl p-6">
                        <h3 className="font-semibold mb-4 text-purple-400">Portfolio Metrics</h3>
                        <div className="space-y-3 text-white">
                          <p className="flex justify-between">
                            <span>Expected Annual Return:</span>
                            <span className="font-medium text-green-400">
                              {(analysis.metrics.expected_return * 100).toFixed(2)}%
                            </span>
                          </p>
                          <p className="flex justify-between">
                            <span>Annual Volatility:</span>
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
                        </div>
                      </div>

                      {/* Allocation Chart */}
                      <div className="bg-[#2a2a2a]/95 backdrop-blur-md border border-white/20 rounded-xl p-6">
                        <div className="flex justify-between items-center mb-4">
                          <h3 className="font-semibold text-purple-400">Optimal Allocations</h3>
                          <UiTooltip>
                            <TooltipTrigger asChild>
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
                            </TooltipTrigger>
                            <TooltipContent 
                              side="left" 
                              sideOffset={5}
                              className="z-[100] max-w-[300px]"
                            >
                              <p>This pie chart shows the optimal portfolio allocation based on Modern Portfolio Theory. Each slice represents the percentage that should be invested in each asset to achieve the best risk-adjusted returns.</p>
                            </TooltipContent>
                          </UiTooltip>
                        </div>
                        {chartData && chartData.labels && chartData.labels.length > 0 ? (
                          <div className="w-full h-[400px] flex items-center justify-center">
                            <Pie 
                              data={chartData}
                              options={{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                  legend: {
                                    position: 'right' as const,
                                    labels: {
                                      color: 'white',
                                      padding: 20,
                                      font: {
                                        size: 14
                                      },
                                      boxWidth: 20,
                                      boxHeight: 20
                                    }
                                  }
                                }
                              }}
                            />
                          </div>
                        ) : (
                          <div className="h-[400px] flex items-center justify-center text-gray-400">
                            No allocation data available
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Historical Performance Chart */}
                    <div className="bg-[#2a2a2a]/95 backdrop-blur-md border border-white/20 rounded-xl p-6">
                      <div className="flex justify-between items-center mb-4">
                        <h3 className="font-semibold text-purple-400">Historical Performance</h3>
                        <UiTooltip>
                          <TooltipTrigger asChild>
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
                          </TooltipTrigger>
                          <TooltipContent 
                            side="left" 
                            sideOffset={5}
                            className="z-[100] max-w-[300px]"
                          >
                            <p>This chart compares your portfolio's performance against the S&P 500 benchmark. The purple line shows your portfolio value, the blue line shows the S&P 500, and the red dotted line shows rolling volatility (risk level over time).</p>
                          </TooltipContent>
                        </UiTooltip>
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
                        <UiTooltip>
                          <TooltipTrigger asChild>
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
                          </TooltipTrigger>
                          <TooltipContent 
                            side="left" 
                            sideOffset={5}
                            className="z-[100] max-w-[300px]"
                          >
                            <p>The drawdown chart shows how much your portfolio has declined from its peak value at any given time. This helps visualize the maximum losses you might experience and how long it takes to recover from them.</p>
                          </TooltipContent>
                        </UiTooltip>
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
                      {Object.entries(analysis.asset_metrics).map(([ticker, metrics]) => (
                        <div key={ticker} className="bg-[#2a2a2a]/95 backdrop-blur-md border border-white/20 rounded-xl p-6">
                          <h3 className="font-semibold mb-4 text-purple-400">{ticker} Metrics</h3>
                          <div className="space-y-3 text-white">
                            <p className="flex justify-between">
                              <span>Beta:</span>
                              <span className={`font-medium ${metrics.beta > 1 ? 'text-red-400' : 'text-green-400'}`}>
                                {metrics.beta.toFixed(2)}
                              </span>
                            </p>
                            <p className="flex justify-between">
                              <span>Alpha:</span>
                              <span className={`font-medium ${metrics.alpha > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                {(metrics.alpha * 100).toFixed(2)}%
                              </span>
                            </p>
                            <p className="flex justify-between">
                              <span>Volatility:</span>
                              <span className="font-medium text-yellow-400">
                                {(metrics.volatility * 100).toFixed(2)}%
                              </span>
                            </p>
                            <p className="flex justify-between">
                              <span>Max Drawdown:</span>
                              <span className="font-medium text-red-400">
                                {(metrics.max_drawdown * 100).toFixed(2)}%
                              </span>
                            </p>
                            <p className="flex justify-between">
                              <span>Correlation:</span>
                              <span className="font-medium text-blue-400">
                                {metrics.correlation.toFixed(2)}
                              </span>
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* AI Insights */}
                    <div className="mt-8">
                      <PortfolioExplanation analysis={analysis} />
                    </div>
                  </motion.div>
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
    </TooltipProvider>
  )
}

export default App
