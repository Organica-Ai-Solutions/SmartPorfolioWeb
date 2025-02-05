import { useState } from 'react'
import { Portfolio, PortfolioAnalysis } from './types/portfolio'
import axios from 'axios'
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js'
import { Pie } from 'react-chartjs-2'
import { SparklesCore } from './components/ui/sparkles'
import { motion } from 'framer-motion'
import { TickerSuggestions } from './components/TickerSuggestions'

ChartJS.register(ArcElement, Tooltip, Legend)

const API_URL = 'http://localhost:8001'

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

      setLoading(true)
      setError('')
      setAnalysis(null) // Reset analysis before new request

      const response = await axios.post(`${API_URL}/analyze-portfolio`, {
        tickers: portfolio.tickers,
        start_date: portfolio.start_date,
        risk_tolerance: portfolio.risk_tolerance
      })

      console.log('Analysis response:', response.data) // Debug log

      // Validate response data before setting state
      if (response.data && response.data.allocation && response.data.metrics) {
        const allocations = response.data.allocation.reduce((acc: {[key: string]: number}, curr: any) => {
          acc[curr.ticker] = curr.weight
          return acc
        }, {})

        setAnalysis({
          allocations,
          metrics: response.data.metrics,
          asset_metrics: {},
          discrete_allocation: {
            shares: {},
            leftover: 0
          }
        })
      } else {
        throw new Error('Invalid response format from server')
      }
    } catch (err: any) {
      console.error('Analysis error:', err)
      if (err.response?.data?.detail) {
        setError(err.response.data.detail)
      } else if (err.message) {
        setError(err.message)
      } else {
        setError('Error analyzing portfolio. Please try again.')
      }
      setAnalysis(null)
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

  const chartData = analysis ? {
    labels: Object.keys(analysis.allocations),
    datasets: [
      {
        data: Object.values(analysis.allocations),
        backgroundColor: [
          '#FF6384',
          '#36A2EB',
          '#FFCE56',
          '#4BC0C0',
          '#9966FF',
          '#FF9F40'
        ]
      }
    ]
  } : null

  // Handler for period change
  const handlePeriodChange = (period: TimePeriod) => {
    setPortfolio({
      ...portfolio,
      start_date: getStartDate(period)
    })
  }

  return (
    <div className="relative min-h-screen bg-[#0d0d0d] text-white">
      {/* Background sparkles */}
      <div className="fixed inset-0 w-full h-full pointer-events-none z-0">
        <SparklesCore
          id="tsparticlesfullpage"
          background="transparent"
          minSize={0.6}
          maxSize={1.4}
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
            className="relative bg-black/40 backdrop-blur-xl rounded-3xl p-6 md:p-8 shadow-2xl border border-white/10"
          >
            <div className="space-y-8">
              {/* AI Ticker Suggestions */}
              <div className="mb-8">
                <h2 className="text-2xl font-bold text-center mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                  AI Ticker Suggestions
                </h2>
                <TickerSuggestions onSelectTicker={addTicker} />
              </div>

              {/* Portfolio Input Section */}
              <div className="space-y-6">
                <div className="flex flex-col md:flex-row gap-4">
                  <div className="flex-1">
                    <input
                      type="text"
                      value={newTicker}
                      onChange={(e) => setNewTicker(e.target.value.toUpperCase())}
                      placeholder="Enter stock ticker"
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <button
                    onClick={() => addTicker(newTicker)}
                    className="px-8 py-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg font-medium hover:opacity-90 transition-opacity"
                  >
                    Add
                  </button>
                </div>

                {/* Selected Tickers */}
                <div className="flex flex-wrap gap-2 min-h-[40px]">
                  {portfolio.tickers.map(ticker => (
                    <motion.span
                      key={ticker}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      className="px-4 py-2 bg-gradient-to-r from-purple-500/20 to-blue-500/20 rounded-full 
                               border border-purple-500/30 text-white flex items-center gap-2"
                    >
                      {ticker}
                      <button
                        onClick={() => removeTicker(ticker)}
                        className="text-purple-400 hover:text-purple-300 transition-colors"
                      >
                        ×
                      </button>
                    </motion.span>
                  ))}
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

              {/* Loading State */}
              {loading && (
                <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
                  <div className="text-center">
                    <div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mb-4"></div>
                    <p className="text-lg font-medium text-white">Analyzing Portfolio...</p>
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
                  className="mt-8 space-y-6 relative z-20"
                >
                  <h2 className="text-2xl font-bold text-center bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                    Portfolio Analysis Results
                  </h2>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* Metrics Card */}
                    <div className="bg-black/40 backdrop-blur-md border border-white/10 rounded-xl p-6">
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
                      </div>
                    </div>

                    {/* Allocation Chart */}
                    <div className="bg-black/40 backdrop-blur-md border border-white/10 rounded-xl p-6">
                      <h3 className="font-semibold mb-4 text-purple-400">Optimal Allocations</h3>
                      {chartData && (
                        <div className="w-full aspect-square relative">
                          <Pie 
                            data={chartData} 
                            options={{ 
                              plugins: { 
                                legend: { 
                                  position: 'bottom',
                                  labels: {
                                    color: 'white',
                                    padding: 20,
                                    font: {
                                      size: 12
                                    }
                                  }
                                },
                                tooltip: {
                                  backgroundColor: 'rgba(0, 0, 0, 0.8)',
                                  titleColor: 'white',
                                  bodyColor: 'white',
                                  callbacks: {
                                    label: function(context) {
                                      const value = context.raw as number;
                                      return ` ${(value * 100).toFixed(1)}%`;
                                    }
                                  }
                                }
                              },
                              animation: {
                                animateRotate: true,
                                animateScale: true
                              }
                            }} 
                          />
                        </div>
                      )}
                    </div>
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
  )
}

export default App
