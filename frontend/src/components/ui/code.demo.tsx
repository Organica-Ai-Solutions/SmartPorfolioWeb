import { IconCloud } from "./interactive-icon-cloud"

const tickers = [
  // Tech Stocks
  'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'ADBE', 'CRM', 'INTC', 'AMD', 'PYPL', 'ORCL', 'CSCO', 'IBM', 'QCOM', 'TXN', 'NOW', 'AVGO',
  
  // Financial Stocks
  'JPM', 'V', 'GS', 'MA', 'BAC', 'WFC', 'BLK', 'MS', 'C', 'AXP', 'SCHW', 'USB', 'PNC', 'TFC', 'COF', 'BX', 'SPGI', 'CME', 'ICE', 'CB',
  
  // Consumer Stocks
  'WMT', 'DIS', 'NFLX', 'SBUX', 'NKE', 'MCD', 'HD', 'TGT', 'COST', 'PG', 'KO', 'PEP', 'ABNB', 'BKNG', 'MAR', 'YUM', 'MO', 'EL', 'NKE', 'LVS',
  
  // Healthcare Stocks
  'JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'DHR', 'BMY', 'LLY', 'AMGN', 'GILD', 'ISRG', 'CVS', 'CI', 'HUM', 'BIIB', 'VRTX', 'REGN', 'ZTS', 'BSX', 'BDX',
  
  // Energy Stocks
  'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL', 'KMI', 'WMB', 'DVN', 'BP', 'ENB', 'TTE', 'SU', 'CNQ', 'EPD',
  
  // Industrial Stocks
  'BA', 'CAT', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'MMM', 'DE', 'EMR', 'ETN', 'ITW', 'CSX', 'UNP', 'FDX', 'NSC', 'WM', 'RSG', 'DAL', 'LUV',
  
  // Materials Stocks
  'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'NUE', 'SCCO', 'VMC', 'MLM', 'ALB', 'CF', 'MOS', 'IFF', 'PPG', 'IP', 'BLL', 'AVY',
  
  // Real Estate Stocks
  'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'DLR', 'O', 'WELL', 'SPG', 'AVB', 'EQR', 'VTR', 'BXP', 'ARE', 'HST', 'KIM', 'REG', 'UDR', 'FRT', 'VNO'
]

export function IconCloudDemo() {
  return (
    <div className="relative flex size-full max-w-lg items-center justify-center overflow-hidden rounded-lg border bg-background px-20 pb-20 pt-8">
      <IconCloud tickers={tickers} onTickerSelect={console.log} />
    </div>
  )
} 