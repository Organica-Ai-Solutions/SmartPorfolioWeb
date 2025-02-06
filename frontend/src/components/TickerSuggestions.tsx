import { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { IconCloud } from './ui/interactive-icon-cloud'

interface TickerSuggestionsProps {
  onSelectTicker: (ticker: string) => void
  maxTickers?: number
}

// Popular tickers by sector - cleaned up to include only actual tradable tickers
const popularTickers = {
  tech: ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'ADBE', 'CRM', 'INTC', 'AMD', 'PYPL', 'ORCL', 'CSCO'],
  finance: ['JPM', 'V', 'GS', 'MA', 'BAC', 'WFC', 'BLK', 'MS', 'C', 'AXP', 'SCHW', 'USB', 'PNC', 'COF', 'BX'],
  consumer: ['WMT', 'DIS', 'SBUX', 'NKE', 'MCD', 'HD', 'TGT', 'COST', 'PG', 'KO', 'PEP', 'ABNB', 'MAR', 'YUM', 'EL'],
  health: ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'DHR', 'BMY', 'LLY', 'AMGN', 'GILD', 'ISRG', 'CVS', 'CI', 'HUM', 'BIIB'],
  energy: ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL', 'KMI', 'WMB', 'DVN', 'BP'],
  industrial: ['BA', 'CAT', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'MMM', 'DE', 'EMR', 'ETN', 'ITW', 'CSX', 'UNP', 'FDX'],
  materials: ['LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'NUE', 'SCCO', 'VMC', 'MLM', 'ALB', 'CF', 'PPG'],
  realestate: ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'DLR', 'O', 'WELL', 'SPG', 'AVB', 'EQR', 'VTR', 'BXP', 'ARE', 'HST'],
  crypto: [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD', 
    'AVAX-USD', 'MATIC-USD', 'LINK-USD', 'XRP-USD', 'DOGE-USD',
    'UNI-USD', 'AAVE-USD', 'ATOM-USD', 'BNB-USD', 'LTC-USD'
  ]
}

// Clean up tickerToSlug mapping to only include actual stock tickers
const tickerToSlug: Record<string, string> = {
  // Technology
  AAPL: "apple",
  GOOGL: "google",
  MSFT: "microsoft",
  AMZN: "amazonaws",
  META: "facebook",
  NFLX: "netflix",
  TSLA: "tesla",
  NVDA: "nvidia",
  ADBE: "adobe",
  ORCL: "oracle",
  CSCO: "cisco",
  IBM: "ibm",
  INTC: "intel",
  AMD: "amd",
  QCOM: "qualcomm",
  CRM: "salesforce",
  AVGO: "broadcom",
  HPQ: "hp",
  DELL: "dell",
  VMW: "vmware",
  SAP: "sap",
  
  // Consumer
  DIS: "disney",
  SBUX: "starbucks",
  ABNB: "airbnb",
  PG: "procter-and-gamble",
  KO: "coca-cola",
  PEP: "pepsi",
  MCD: "mcdonalds",
  WMT: "walmart",
  TGT: "target",
  COST: "costco",
  NKE: "nike",
  
  // Financial
  JPM: "jpmorgan",
  V: "visa",
  MA: "mastercard",
  PYPL: "paypal",
  MS: "morgan-stanley",
  GS: "goldman-sachs",
  BAC: "bank-of-america",
  C: "citigroup",
  WFC: "wells-fargo",
  AXP: "american-express",
  SCHW: "charles-schwab",
  BLK: "blackrock",
  
  // E-commerce & Tech Services
  SHOP: "shopify",
  ETSY: "etsy",
  EBAY: "ebay",
  UBER: "uber",
  LYFT: "lyft",
  DASH: "doordash"
}

export function TickerSuggestions({ onSelectTicker, maxTickers = 10 }: TickerSuggestionsProps) {
  const [error, setError] = useState('')
  const [selectedTickers, setSelectedTickers] = useState<string[]>([])
  const [inputValue, setInputValue] = useState('')
  const [showAutocomplete, setShowAutocomplete] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  // Get all tickers including cryptos
  const allTickers = Object.values(popularTickers).flat()
  // Only show tickers that have corresponding icons
  const availableTickers = allTickers.filter(ticker => tickerToSlug[ticker])
  const filteredTickers = availableTickers.filter(ticker => 
    ticker.toLowerCase().includes(inputValue.toLowerCase()) && 
    !selectedTickers.includes(ticker)
  ).slice(0, 5)

  const handleTickerClick = (symbol: string) => {
    if (selectedTickers.length >= maxTickers) {
      setError(`Maximum ${maxTickers} tickers allowed`)
      return
    }
    if (!selectedTickers.includes(symbol)) {
      setSelectedTickers([...selectedTickers, symbol])
      onSelectTicker(symbol)
      setInputValue('')
      setShowAutocomplete(false)
      inputRef.current?.focus()
    }
  }

  const handleRemoveTicker = (tickerToRemove: string) => {
    setSelectedTickers(selectedTickers.filter(t => t !== tickerToRemove))
    setError('')
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value.toUpperCase()
    setInputValue(value)
    setShowAutocomplete(value.length > 0)
    setError('')
  }

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && inputValue) {
      const validTicker = allTickers.find(t => t === inputValue)
      if (validTicker) {
        handleTickerClick(validTicker)
      } else {
        setError('Invalid ticker symbol')
      }
    }
  }

  return (
    <div className="space-y-6 p-4">
      {/* Interactive Icon Cloud with reduced height for better visibility */}
      <div className="relative h-[400px]">
        <IconCloud 
          tickers={allTickers} 
          onTickerSelect={handleTickerClick} 
        />
      </div>

      {/* Selected Tickers */}
      <div className="flex flex-wrap gap-2 min-h-[40px]">
        <AnimatePresence>
          {selectedTickers.map((ticker) => (
            <motion.div
              key={ticker}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0, opacity: 0 }}
              className="px-3 py-1.5 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-lg
                       border border-purple-500/30 text-white flex items-center gap-2
                       shadow-lg"
            >
              <span className="font-medium">{ticker}</span>
              <button
                onClick={() => handleRemoveTicker(ticker)}
                className="hover:bg-white/10 rounded-full w-5 h-5 flex items-center justify-center
                         transition-colors text-purple-400 hover:text-purple-300"
              >
                ×
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Input Section */}
      <div className="flex gap-3">
        <div className="relative flex-1">
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={handleInputChange}
            onKeyDown={handleInputKeyDown}
            placeholder="Enter stock ticker (e.g., AAPL)"
            className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg 
                     text-white placeholder-gray-400 focus:outline-none focus:ring-2
                     focus:ring-purple-500/50 transition-all"
            maxLength={5}
          />
          
          <div className="absolute right-3 top-3 text-sm text-gray-400">
            {selectedTickers.length}/{maxTickers}
          </div>

          {/* Autocomplete Dropdown */}
          <AnimatePresence>
            {showAutocomplete && filteredTickers.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute z-50 w-full mt-1 bg-black/90 border border-white/10 rounded-lg 
                         shadow-lg overflow-hidden backdrop-blur-sm"
              >
                {filteredTickers.map((ticker) => (
                  <button
                    key={ticker}
                    onClick={() => handleTickerClick(ticker)}
                    className="w-full px-4 py-2.5 text-left hover:bg-white/10 text-white flex items-center
                             justify-between transition-colors"
                  >
                    <span className="font-medium">{ticker}</span>
                    <span className="text-xs text-gray-400">Click to add</span>
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <button
          onClick={() => handleTickerClick(inputValue)}
          className="px-8 py-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg 
                   font-medium hover:opacity-90 transition-all duration-200 text-white
                   shadow-lg hover:shadow-purple-500/20 whitespace-nowrap"
        >
          Add
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-red-400 text-sm text-center"
        >
          {error}
        </motion.div>
      )}
    </div>
  )
} 