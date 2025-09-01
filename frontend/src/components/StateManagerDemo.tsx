// ============================================================================
// STATE MANAGER DEMO - SmartPortfolio AI Frontend
// ============================================================================
// Comprehensive demonstration of the state management system

import React, { useEffect, useState } from 'react';
import {
  useAppState,
  useAppDispatch,
  usePortfolio,
  useAI,
  useRiskManagement,
  apiClient
} from '../store';

const StateManagerDemo: React.FC = () => {
  const dispatch = useAppDispatch();
  const state = useAppState();
  const portfolio = usePortfolio();
  const ai = useAI();
  const riskManagement = useRiskManagement();
  
  const [isDemo, setIsDemo] = useState(false);

  // ============================================================================
  // DEMO ACTIONS
  // ============================================================================

  const runPortfolioDemo = async () => {
    console.log('🎯 Running Portfolio Demo...');
    
    // 1. Load sample portfolio
    portfolio.loadSamplePortfolio();
    
    // 2. Analyze portfolio
    await portfolio.analyzePortfolio(true);
    
    // 3. Run optimization
    const optimizationResult = await portfolio.optimizePortfolio({
      tickers: ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'TLT'],
      method: 'black_litterman',
      risk_tolerance: 'moderate'
    });
    
    console.log('Portfolio optimization result:', optimizationResult);
  };

  const runAIDemo = async () => {
    console.log('🤖 Running AI Demo...');
    
    // 1. Check AI readiness
    const readiness = ai.getAIReadiness();
    console.log('AI Readiness:', readiness);
    
    if (!readiness.ready) {
      // 2. Train models if needed
      await ai.training.trainModels(['AAPL', 'MSFT', 'SPY']);
    }
    
    // 3. Generate predictions
    const predictions = await ai.predictions.generatePredictions(['AAPL', 'MSFT', 'SPY']);
    console.log('ML Predictions:', predictions);
    
    // 4. Identify market regime
    const regime = await ai.regime.identifyRegime();
    console.log('Market Regime:', regime);
    
    // 5. Run ensemble prediction
    const ensemble = await ai.ensemble.generateEnsemblePrediction(['AAPL', 'MSFT', 'SPY']);
    console.log('Ensemble Prediction:', ensemble);
  };

  const runRiskManagementDemo = async () => {
    console.log('⚡ Running Risk Management Demo...');
    
    // 1. Analyze current risk
    const riskAnalysis = await riskManagement.analyzeRisk();
    console.log('Risk Analysis:', riskAnalysis);
    
    // 2. Apply tail hedging
    // const hedging = await riskManagement.implementTailHedging();
    // console.log('Tail Hedging:', hedging);
  };

  const runComprehensiveDemo = async () => {
    setIsDemo(true);
    
    try {
      console.log('🚀 Running Comprehensive AI Portfolio Management Demo...');
      
      // 1. Portfolio setup
      await runPortfolioDemo();
      
      // 2. AI analysis
      await runAIDemo();
      
      // 3. Risk management
      await runRiskManagementDemo();
      
      // 4. Complete AI management
      const aiResult = await portfolio.aiPortfolioManagement({
        tickers: ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'TLT'],
        base_allocation: portfolio.weights,
        portfolio_value: portfolio.value,
        enable_ml_predictions: true,
        enable_regime_analysis: true,
        enable_risk_management: true,
        prediction_horizon: '5_days'
      });
      
      console.log('🎉 Complete AI Portfolio Management Result:', aiResult);
      
    } catch (error) {
      console.error('Demo error:', error);
    } finally {
      setIsDemo(false);
    }
  };

  // ============================================================================
  // STATE DISPLAY COMPONENTS
  // ============================================================================

  const SystemStatus = () => (
    <div className="bg-white p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-3">🔧 System Status</h3>
      <div className="space-y-2 text-sm">
        <div>Health: {state.system.health?.status || 'Unknown'}</div>
        <div>Online: {state.ui.isOnline ? '✅' : '❌'}</div>
        <div>Theme: {state.ui.theme}</div>
        <div>AI Enabled: {state.preferences.enableAI ? '✅' : '❌'}</div>
        <div>Risk Management: {state.preferences.enableRiskManagement ? '✅' : '❌'}</div>
      </div>
    </div>
  );

  const PortfolioStatus = () => (
    <div className="bg-white p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-3">💼 Portfolio Status</h3>
      <div className="space-y-2 text-sm">
        <div>Value: ${portfolio.value.toLocaleString()}</div>
        <div>Tickers: {portfolio.tickers.length}</div>
        <div>Loading: {portfolio.isLoading ? '⏳' : '✅'}</div>
        <div>Last Updated: {state.portfolio.lastUpdated ? new Date(state.portfolio.lastUpdated).toLocaleString() : 'Never'}</div>
        
        {Object.keys(portfolio.weights).length > 0 && (
          <div className="mt-3">
            <div className="font-medium">Weights:</div>
            {Object.entries(portfolio.weights).map(([symbol, weight]) => (
              <div key={symbol} className="ml-2">
                {symbol}: {(weight * 100).toFixed(1)}%
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  const AIStatus = () => (
    <div className="bg-white p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-3">🤖 AI Status</h3>
      <div className="space-y-2 text-sm">
        <div>Training: {state.ml.isTraining ? '⏳' : '✅'}</div>
        <div>Predicting: {state.ml.isPredicting ? '⏳' : '✅'}</div>
        <div>Confidence: {(state.ml.confidence * 100).toFixed(1)}%</div>
        <div>Models: {state.ml.models?.total_models || 0}</div>
        <div>Last Training: {state.ml.lastTraining ? new Date(state.ml.lastTraining).toLocaleString() : 'Never'}</div>
        
        {state.ml.regimeAnalysis && (
          <div className="mt-3">
            <div className="font-medium">Current Regime:</div>
            <div className="ml-2">{state.ml.regimeAnalysis.current_regime.replace(/_/g, ' ').toUpperCase()}</div>
            <div className="ml-2">Confidence: {(state.ml.regimeAnalysis.regime_probability * 100).toFixed(1)}%</div>
          </div>
        )}
      </div>
    </div>
  );

  const RiskStatus = () => (
    <div className="bg-white p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-3">⚡ Risk Status</h3>
      <div className="space-y-2 text-sm">
        <div>Regime: {state.risk.regime.toUpperCase()}</div>
        <div>Hedging Active: {state.risk.hedgingActive ? '✅' : '❌'}</div>
        <div>Drawdown Controls: {state.risk.drawdownControls ? '✅' : '❌'}</div>
        <div>Analyzing: {state.risk.isAnalyzing ? '⏳' : '✅'}</div>
        <div>Market Volatility: {(state.market.volatility * 100).toFixed(1)}%</div>
        <div>Market Condition: {state.market.condition.toUpperCase()}</div>
      </div>
    </div>
  );

  const NotificationsList = () => (
    <div className="bg-white p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-3">🔔 Notifications ({state.ui.notifications.length})</h3>
      <div className="space-y-2 max-h-40 overflow-y-auto">
        {state.ui.notifications.slice(0, 5).map(notification => (
          <div key={notification.id} className={`p-2 rounded text-sm ${
            notification.type === 'success' ? 'bg-green-100 text-green-800' :
            notification.type === 'warning' ? 'bg-yellow-100 text-yellow-800' :
            notification.type === 'error' ? 'bg-red-100 text-red-800' :
            'bg-blue-100 text-blue-800'
          }`}>
            <div className="font-medium">{notification.title}</div>
            <div>{notification.message}</div>
            <div className="text-xs opacity-70">
              {new Date(notification.timestamp).toLocaleTimeString()}
            </div>
          </div>
        ))}
        
        {state.ui.notifications.length === 0 && (
          <div className="text-gray-500 text-sm">No notifications</div>
        )}
      </div>
    </div>
  );

  const ErrorsDisplay = () => {
    const errors = Object.entries(state.errors).filter(([, error]) => error !== null);
    
    if (errors.length === 0) return null;
    
    return (
      <div className="bg-red-50 p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-3 text-red-800">❌ Errors</h3>
        <div className="space-y-2">
          {errors.map(([module, error]) => (
            <div key={module} className="text-sm text-red-700">
              <span className="font-medium">{module}:</span> {error}
            </div>
          ))}
        </div>
      </div>
    );
  };

  // ============================================================================
  // DEMO CONTROLS
  // ============================================================================

  const DemoControls = () => (
    <div className="bg-blue-50 p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-3">🎮 Demo Controls</h3>
      <div className="space-y-2">
        <button
          onClick={runPortfolioDemo}
          disabled={isDemo}
          className="w-full px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
        >
          Run Portfolio Demo
        </button>
        
        <button
          onClick={runAIDemo}
          disabled={isDemo}
          className="w-full px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
        >
          Run AI Demo
        </button>
        
        <button
          onClick={runRiskManagementDemo}
          disabled={isDemo}
          className="w-full px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
        >
          Run Risk Demo
        </button>
        
        <button
          onClick={runComprehensiveDemo}
          disabled={isDemo}
          className="w-full px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50"
        >
          {isDemo ? 'Running Demo...' : 'Run Complete Demo'}
        </button>
        
        <div className="border-t pt-2 mt-4">
          <h4 className="font-medium mb-2">Quick Actions:</h4>
          
          <button
            onClick={() => dispatch({ type: 'SET_THEME', payload: state.ui.theme === 'light' ? 'dark' : 'light' })}
            className="w-full px-3 py-1 bg-gray-600 text-white rounded text-sm hover:bg-gray-700 mb-1"
          >
            Toggle Theme
          </button>
          
          <button
            onClick={() => dispatch({ type: 'TOGGLE_AI_ENABLED' })}
            className="w-full px-3 py-1 bg-gray-600 text-white rounded text-sm hover:bg-gray-700 mb-1"
          >
            Toggle AI
          </button>
          
          <button
            onClick={() => dispatch({ type: 'CLEAR_ERRORS' })}
            className="w-full px-3 py-1 bg-gray-600 text-white rounded text-sm hover:bg-gray-700 mb-1"
          >
            Clear Errors
          </button>
          
          <button
            onClick={() => portfolio.resetPortfolio()}
            className="w-full px-3 py-1 bg-gray-600 text-white rounded text-sm hover:bg-gray-700"
          >
            Reset Portfolio
          </button>
        </div>
      </div>
    </div>
  );

  // ============================================================================
  // API CLIENT DEMO
  // ============================================================================

  const APIDemo = () => {
    const [apiHealth, setApiHealth] = useState<any>(null);
    const [apiCapabilities, setApiCapabilities] = useState<any>(null);
    
    const checkAPIHealth = async () => {
      try {
        const health = await apiClient.getHealth();
        setApiHealth(health);
        dispatch({ type: 'SET_SYSTEM_HEALTH', payload: health });
      } catch (error) {
        console.error('API Health check failed:', error);
      }
    };
    
    const checkAPICapabilities = async () => {
      try {
        const capabilities = await apiClient.getCapabilities();
        setApiCapabilities(capabilities);
        dispatch({ type: 'SET_SYSTEM_CAPABILITIES', payload: capabilities });
      } catch (error) {
        console.error('API Capabilities check failed:', error);
      }
    };
    
    return (
      <div className="bg-gray-50 p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-3">🌐 API Demo</h3>
        <div className="space-y-2">
          <button
            onClick={checkAPIHealth}
            className="w-full px-3 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-700"
          >
            Check API Health
          </button>
          
          <button
            onClick={checkAPICapabilities}
            className="w-full px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700"
          >
            Get API Capabilities
          </button>
          
          {apiHealth && (
            <div className="text-xs bg-white p-2 rounded">
              <div className="font-medium">Health Status:</div>
              <div>Status: {apiHealth.status}</div>
              <div>Version: {apiHealth.version}</div>
              <div>ML Enabled: {apiHealth.ml_enabled ? '✅' : '❌'}</div>
            </div>
          )}
          
          {apiCapabilities && (
            <div className="text-xs bg-white p-2 rounded">
              <div className="font-medium">Capabilities:</div>
              <div>Optimization Methods: {Object.keys(apiCapabilities.optimization_methods || {}).length}</div>
              <div>ML Features: {Object.keys(apiCapabilities.ml_intelligence || {}).length}</div>
              <div>Risk Features: {Object.keys(apiCapabilities.risk_management || {}).length}</div>
            </div>
          )}
        </div>
      </div>
    );
  };

  // ============================================================================
  // INITIALIZATION
  // ============================================================================

  useEffect(() => {
    // Initialize the demo with some default notifications
    dispatch({
      type: 'ADD_NOTIFICATION',
      payload: {
        id: 'welcome',
        type: 'info',
        title: 'Welcome to SmartPortfolio AI',
        message: 'State management system initialized successfully!',
        timestamp: new Date().toISOString(),
        duration: 5000
      }
    });
  }, [dispatch]);

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6 text-center">
          🚀 SmartPortfolio AI State Management Demo
        </h1>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Status Panels */}
          <SystemStatus />
          <PortfolioStatus />
          <AIStatus />
          <RiskStatus />
          <NotificationsList />
          
          {/* Error Display */}
          <ErrorsDisplay />
          
          {/* Demo Controls */}
          <DemoControls />
          
          {/* API Demo */}
          <APIDemo />
        </div>
        
        <div className="mt-8 bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">📊 State Management Features</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
            <div>
              <h3 className="font-semibold mb-2">✅ Implemented Features:</h3>
              <ul className="space-y-1">
                <li>• 39 API endpoints integrated</li>
                <li>• TypeScript type safety</li>
                <li>• React Context state management</li>
                <li>• Custom hooks for AI features</li>
                <li>• Risk management hooks</li>
                <li>• Portfolio management context</li>
                <li>• Error handling & notifications</li>
                <li>• Local storage persistence</li>
                <li>• Performance monitoring</li>
                <li>• State validation utilities</li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-semibold mb-2">🤖 AI Features:</h3>
              <ul className="space-y-1">
                <li>• ML model training</li>
                <li>• Price predictions</li>
                <li>• Market regime detection</li>
                <li>• Reinforcement learning</li>
                <li>• Ensemble predictions</li>
                <li>• Confidence scoring</li>
                <li>• Model status monitoring</li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-semibold mb-2">🎯 Portfolio Features:</h3>
              <ul className="space-y-1">
                <li>• 6 optimization methods</li>
                <li>• Black-Litterman model</li>
                <li>• Factor-based optimization</li>
                <li>• Risk management</li>
                <li>• Tax optimization</li>
                <li>• Liquidity management</li>
                <li>• Dynamic allocation</li>
                <li>• Performance analysis</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StateManagerDemo;
