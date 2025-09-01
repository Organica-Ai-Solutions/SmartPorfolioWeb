# 🎨 SmartPortfolio AI Frontend

**Modern, responsive frontend for institutional-grade portfolio management** built with React 18, TypeScript, and Vite. Interfaces with our powerful AI backend to deliver professional-grade portfolio analytics and management.

## 🌟 **FEATURES**

### **🤖 AI Integration**
- **ML Predictions Dashboard** - Real-time price predictions with confidence scores
- **Market Regime Visualization** - Interactive regime detection and analysis
- **Ensemble Insights** - Combined ML model predictions and recommendations
- **Portfolio AI Chat** - Natural language interaction with AI advisor

### **📊 Advanced Analytics**
- **Multi-Objective Optimization** - Interactive Black-Litterman, factor-based optimization
- **Risk Management Controls** - Real-time risk metrics and hedging strategies  
- **Performance Attribution** - Factor decomposition and return analysis
- **Tax Analytics** - Tax-loss harvesting opportunities and efficiency metrics

### **💎 Professional Features**
- **Multi-Account Management** - Taxable, IRA, 401k account optimization
- **Liquidity Dashboard** - Asset liquidity scoring and stress testing
- **Execution Analytics** - VWAP/TWAP performance and smart routing
- **Regime-Based Allocation** - Dynamic allocation based on market conditions

### **🎯 User Experience**
- **Real-Time Updates** - Live market data and portfolio tracking
- **Interactive Charts** - Advanced financial visualizations with Chart.js
- **Responsive Design** - Optimized for desktop, tablet, and mobile
- **Dark/Light Mode** - Professional theming options

## 🚀 **TECH STACK**

### **⚡ Core Technologies**
- **React 18** - Latest React with concurrent features
- **TypeScript** - Type-safe development
- **Vite** - Lightning-fast development server
- **TailwindCSS** - Utility-first styling framework
- **Chart.js** - Professional financial charting

### **📊 Data Visualization**
```typescript
// Advanced financial charts
import {
  LineChart, CandlestickChart, PortfolioAllocation,
  RiskMetricsChart, PerformanceAttribution,
  CorrelationHeatmap, DrawdownChart
} from '@/components/charts'
```

### **🤖 AI Integration**
```typescript
// AI-powered components
import {
  MLPredictionsPanel, RegimeDetector,
  PortfolioOptimizer, RiskAnalyzer,
  TaxOptimizer, LiquidityManager
} from '@/components/ai'
```

## 🏁 **QUICK START**

### **1. Install Dependencies**
```bash
cd frontend
npm install
```

### **2. Configure Environment**
Create `.env` file:
```bash
# Backend API URL
VITE_API_URL=http://localhost:8001

# Optional: Analytics
VITE_GOOGLE_ANALYTICS_ID=your_ga_id
```

### **3. Start Development Server**
```bash
npm run dev
```

**Visit `http://localhost:5173`** to see the AI-powered frontend!

## 🔧 **DEVELOPMENT**

### **Available Scripts**
```bash
# Development
npm run dev          # Start dev server with HMR
npm run build        # Production build
npm run preview      # Preview production build

# Code Quality
npm run lint         # ESLint checking
npm run type-check   # TypeScript checking
npm run format       # Prettier formatting
```

### **Project Structure**
```
frontend/
├── src/
│   ├── components/
│   │   ├── ai/          # AI-powered components
│   │   ├── charts/      # Financial visualizations
│   │   ├── portfolio/   # Portfolio management
│   │   └── ui/          # Reusable UI components
│   ├── hooks/
│   │   ├── useMLPredictions.ts
│   │   ├── usePortfolioOptimization.ts
│   │   └── useRiskManagement.ts
│   ├── services/
│   │   ├── api.ts       # Backend API client
│   │   ├── ml.ts        # ML service integration
│   │   └── portfolio.ts # Portfolio data service
│   ├── types/
│   │   ├── portfolio.ts # Portfolio type definitions
│   │   ├── ml.ts        # ML prediction types
│   │   └── api.ts       # API response types
│   └── utils/
│       ├── calculations.ts # Financial calculations
│       ├── formatters.ts   # Data formatting
│       └── validators.ts   # Input validation
```

### **API Integration Examples**

#### **ML Predictions Hook**
```typescript
export const useMLPredictions = (symbols: string[]) => {
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading] = useState(false)

  const generatePredictions = async () => {
    setLoading(true)
    try {
      const response = await fetch('/ml/predict-movements', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbols,
          horizons: ['5_days'],
          include_confidence: true
        })
      })
      const data = await response.json()
      setPredictions(data.predictions)
    } catch (error) {
      console.error('ML prediction error:', error)
    } finally {
      setLoading(false)
    }
  }

  return { predictions, loading, generatePredictions }
}
```

#### **Portfolio Optimization Component**
```typescript
const PortfolioOptimizer: React.FC = () => {
  const [optimization, setOptimization] = useState(null)
  
  const runOptimization = async (method: string) => {
    const response = await fetch('/optimize-portfolio-advanced', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tickers: ['AAPL', 'MSFT', 'SPY'],
        method,
        risk_tolerance: 'medium'
      })
    })
    
    const result = await response.json()
    setOptimization(result)
  }

  return (
    <div className="optimization-panel">
      <h3>AI Portfolio Optimization</h3>
      <AllocationChart weights={optimization?.weights} />
      <MetricsPanel metrics={optimization?.metrics} />
    </div>
  )
}
```

## 🎨 **UI COMPONENTS**

### **Financial Charts**
```typescript
// Portfolio allocation pie chart
<AllocationChart 
  allocation={portfolio.weights}
  interactive={true}
  showLegend={true}
/>

// Performance line chart with benchmarks
<PerformanceChart
  portfolioReturns={returns}
  benchmark="SPY"
  timeframe="1Y"
/>

// Risk metrics dashboard
<RiskMetrics
  volatility={metrics.volatility}
  sharpeRatio={metrics.sharpe}
  maxDrawdown={metrics.drawdown}
/>
```

### **AI Components**
```typescript
// ML prediction panel
<MLPredictionsPanel
  symbols={['AAPL', 'MSFT']}
  horizon="5_days"
  onPredictionUpdate={handlePredictions}
/>

// Market regime indicator
<RegimeIndicator
  currentRegime={regime.current}
  probability={regime.confidence}
  nextRegimes={regime.transitions}
/>
```

## 📱 **RESPONSIVE DESIGN**

### **TailwindCSS Configuration**
```typescript
// tailwind.config.js
export default {
  content: ['./src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          500: '#3b82f6',
          900: '#1e3a8a'
        },
        success: '#10b981',
        warning: '#f59e0b',
        danger: '#ef4444'
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif']
      }
    }
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography')
  ]
}
```

### **Mobile-First Approach**
```typescript
// Responsive layout example
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
  <PortfolioSummary className="md:col-span-2" />
  <RiskMetrics className="lg:col-span-1" />
  <MLPredictions className="md:col-span-3" />
</div>
```

## 🔐 **SECURITY & PERFORMANCE**

### **Environment Variables**
```bash
# .env (never commit this file)
VITE_API_URL=https://smartportfolio-ai.onrender.com
VITE_WS_URL=wss://smartportfolio-ai.onrender.com/ws
```

### **Performance Optimization**
```typescript
// Lazy loading for heavy components
const MLPredictionsPanel = lazy(() => import('./components/MLPredictionsPanel'))
const AdvancedCharts = lazy(() => import('./components/AdvancedCharts'))

// Memoization for expensive calculations
const portfolioMetrics = useMemo(() => 
  calculatePortfolioMetrics(positions, prices), 
  [positions, prices]
)

// Debounced API calls
const debouncedOptimization = useDebounce(runOptimization, 500)
```

---

🌟 **Modern frontend for institutional-grade AI portfolio management**  
🏆 **Built to showcase the power of our advanced backend**
