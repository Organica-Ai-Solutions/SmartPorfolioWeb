// ============================================================================
// AI HOOKS - SmartPortfolio AI Frontend
// ============================================================================
// Custom React hooks for AI and Machine Learning features

import { useCallback, useEffect, useState } from 'react';
import { useAppContext } from '../contexts/AppContext';
import { apiClient } from '../services/api';
import {
  MLTrainingRequest,
  MLPredictionRequest,
  RegimeAnalysisRequest,
  RLOptimizationRequest,
  EnsemblePredictionRequest,
  MLTrainingResult,
  MLPredictionResult,
  RegimeAnalysisResult,
  RLOptimizationResult,
  EnsemblePredictionResult,
  MLModelStatus,
  PredictionHorizon,
  ModelType,
  MLMarketRegime,
  PortfolioWeights
} from '../types/api';

// ============================================================================
// MACHINE LEARNING TRAINING HOOK
// ============================================================================

export const useMLTraining = () => {
  const { state, dispatch } = useAppContext();
  const { ml, portfolio } = state;

  const trainModels = useCallback(async (
    symbols?: string[],
    horizons: PredictionHorizon[] = ['5_days'],
    modelTypes: ModelType[] = ['random_forest', 'gradient_boosting', 'neural_network'],
    retrain: boolean = false
  ): Promise<MLTrainingResult | null> => {
    const tickers = symbols || portfolio.tickers;
    
    if (tickers.length === 0) {
      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `no-tickers-${Date.now()}`,
          type: 'warning',
          title: 'No Tickers Selected',
          message: 'Please add tickers to your portfolio before training ML models',
          timestamp: new Date().toISOString(),
          duration: 5000
        }
      });
      return null;
    }

    dispatch({ type: 'SET_ML_TRAINING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: { module: 'ml', error: null } });

    try {
      const request: MLTrainingRequest = {
        symbols: tickers,
        horizons,
        model_types: modelTypes,
        lookback_days: 252,
        retrain
      };

      const result = await apiClient.trainMLModels(request);

      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `training-complete-${Date.now()}`,
          type: 'success',
          title: 'ML Training Complete',
          message: `Successfully trained ${result.models_trained} models for ${result.symbols_processed} symbols with ${result.feature_count} features`,
          timestamp: new Date().toISOString(),
          duration: 8000
        }
      });

      // Refresh model status
      const modelStatus = await apiClient.getMLModelStatus();
      dispatch({ type: 'SET_ML_MODELS', payload: modelStatus });

      return result;

    } catch (error: any) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'ml', error: error.message }
      });

      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `training-error-${Date.now()}`,
          type: 'error',
          title: 'ML Training Failed',
          message: error.message || 'Failed to train ML models',
          timestamp: new Date().toISOString(),
          duration: 5000
        }
      });

      return null;
    } finally {
      dispatch({ type: 'SET_ML_TRAINING', payload: false });
    }
  }, [portfolio.tickers, dispatch]);

  const getTrainingStatus = useCallback(() => ({
    isTraining: ml.isTraining,
    lastTraining: ml.lastTraining,
    models: ml.models
  }), [ml.isTraining, ml.lastTraining, ml.models]);

  return {
    trainModels,
    getTrainingStatus,
    isTraining: ml.isTraining,
    models: ml.models
  };
};

// ============================================================================
// PRICE PREDICTION HOOK
// ============================================================================

export const usePricePredictions = () => {
  const { state, dispatch } = useAppContext();
  const { ml, portfolio } = state;

  const generatePredictions = useCallback(async (
    symbols?: string[],
    horizons: PredictionHorizon[] = ['5_days'],
    includeConfidence: boolean = true
  ): Promise<MLPredictionResult | null> => {
    const tickers = symbols || portfolio.tickers;
    
    if (tickers.length === 0) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'ml', error: 'No tickers available for prediction' }
      });
      return null;
    }

    dispatch({ type: 'SET_ML_PREDICTING', payload: true });

    try {
      const request: MLPredictionRequest = {
        symbols: tickers,
        horizons,
        include_confidence: includeConfidence
      };

      const result = await apiClient.predictMovements(request);
      dispatch({ type: 'SET_ML_PREDICTIONS', payload: result });

      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `predictions-complete-${Date.now()}`,
          type: 'success',
          title: 'Predictions Generated',
          message: `Generated predictions for ${tickers.length} symbols. Average confidence: ${(result.portfolio_insights.avg_confidence * 100).toFixed(1)}%`,
          timestamp: new Date().toISOString(),
          duration: 5000
        }
      });

      return result;

    } catch (error: any) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'ml', error: error.message }
      });
      return null;
    } finally {
      dispatch({ type: 'SET_ML_PREDICTING', payload: false });
    }
  }, [portfolio.tickers, dispatch]);

  const getPredictionSummary = useCallback(() => {
    if (!ml.predictions) return null;

    const predictions = ml.predictions.predictions;
    const symbols = Object.keys(predictions);
    
    const bullishCount = symbols.filter(symbol => 
      Object.values(predictions[symbol]).some(pred => pred.predicted_return > 0)
    ).length;
    
    const bearishCount = symbols.length - bullishCount;
    
    const avgConfidence = symbols.reduce((sum, symbol) => {
      const symbolPredictions = Object.values(predictions[symbol]);
      const avgSymbolConfidence = symbolPredictions.reduce((sum, pred) => sum + pred.confidence, 0) / symbolPredictions.length;
      return sum + avgSymbolConfidence;
    }, 0) / symbols.length;

    return {
      totalSymbols: symbols.length,
      bullishSignals: bullishCount,
      bearishSignals: bearishCount,
      averageConfidence: avgConfidence,
      sentiment: bullishCount > bearishCount ? 'bullish' : bearishCount > bullishCount ? 'bearish' : 'neutral'
    };
  }, [ml.predictions]);

  return {
    generatePredictions,
    getPredictionSummary,
    predictions: ml.predictions,
    isPredicting: ml.isPredicting,
    confidence: ml.confidence
  };
};

// ============================================================================
// MARKET REGIME DETECTION HOOK
// ============================================================================

export const useMarketRegime = () => {
  const { state, dispatch } = useAppContext();
  const { ml } = state;

  const identifyRegime = useCallback(async (
    lookbackDays: number = 252,
    retrainModel: boolean = false
  ): Promise<RegimeAnalysisResult | null> => {
    try {
      const request: RegimeAnalysisRequest = {
        lookback_days: lookbackDays,
        retrain_model: retrainModel
      };

      const result = await apiClient.identifyRegimes(request);
      dispatch({ type: 'SET_REGIME_ANALYSIS', payload: result });

      // Update market condition based on regime
      const regimeToCondition: Record<MLMarketRegime, 'calm' | 'volatile' | 'stressed' | 'crisis'> = {
        'bull_trending': 'calm',
        'bear_trending': 'stressed',
        'bull_volatile': 'volatile',
        'bear_volatile': 'crisis',
        'sideways_low_vol': 'calm',
        'sideways_high_vol': 'volatile',
        'crisis': 'crisis',
        'recovery': 'volatile'
      };

      dispatch({ 
        type: 'SET_MARKET_CONDITION', 
        payload: regimeToCondition[result.current_regime] || 'calm'
      });

      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `regime-identified-${Date.now()}`,
          type: 'info',
          title: 'Market Regime Identified',
          message: `Current regime: ${result.current_regime.replace(/_/g, ' ').toUpperCase()} (${(result.regime_probability * 100).toFixed(1)}% confidence)`,
          timestamp: new Date().toISOString(),
          duration: 6000
        }
      });

      return result;

    } catch (error: any) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'ml', error: error.message }
      });
      return null;
    }
  }, [dispatch]);

  const getRegimeInsights = useCallback(() => {
    if (!ml.regimeAnalysis) return null;

    const regime = ml.regimeAnalysis;
    const regimeDescriptions: Record<MLMarketRegime, string> = {
      'bull_trending': 'Strong upward trending market with low volatility',
      'bear_trending': 'Declining market with moderate volatility',
      'bull_volatile': 'Rising market with high volatility and uncertainty',
      'bear_volatile': 'Falling market with extreme volatility',
      'sideways_low_vol': 'Stable, range-bound market with low volatility',
      'sideways_high_vol': 'Choppy, directionless market with high volatility',
      'crisis': 'Extreme stress conditions with panic selling',
      'recovery': 'Early recovery phase with improving sentiment'
    };

    const regimeStrategies: Record<MLMarketRegime, string[]> = {
      'bull_trending': ['Increase equity allocation', 'Reduce cash positions', 'Momentum strategies'],
      'bear_trending': ['Defensive positioning', 'Increase bond allocation', 'Consider short strategies'],
      'bull_volatile': ['Moderate risk-taking', 'Volatility strategies', 'Maintain diversification'],
      'bear_volatile': ['Capital preservation', 'High cash allocation', 'Safe haven assets'],
      'sideways_low_vol': ['Range trading', 'Income strategies', 'Balanced allocation'],
      'sideways_high_vol': ['Short-term trading', 'Volatility arbitrage', 'Hedge positions'],
      'crisis': ['Maximum defense', 'Flight to quality', 'Preserve capital'],
      'recovery': ['Gradual re-risking', 'Value opportunities', 'Contrarian positioning']
    };

    return {
      description: regimeDescriptions[regime.current_regime],
      strategies: regimeStrategies[regime.current_regime],
      stability: regime.regime_duration_days > 30 ? 'stable' : regime.regime_duration_days > 10 ? 'moderate' : 'unstable',
      nextRegimeProb: Object.entries(regime.next_regime_probabilities)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 3)
    };
  }, [ml.regimeAnalysis]);

  return {
    identifyRegime,
    getRegimeInsights,
    regimeAnalysis: ml.regimeAnalysis
  };
};

// ============================================================================
// REINFORCEMENT LEARNING HOOK
// ============================================================================

export const useReinforcementLearning = () => {
  const { state, dispatch } = useAppContext();
  const { ml, portfolio, market } = state;

  const optimizeWithRL = useCallback(async (
    weights?: PortfolioWeights,
    learningMode: boolean = false
  ): Promise<RLOptimizationResult | null> => {
    const portfolioWeights = weights || portfolio.weights;
    
    if (Object.keys(portfolioWeights).length === 0) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'ml', error: 'No portfolio weights available for RL optimization' }
      });
      return null;
    }

    try {
      const request: RLOptimizationRequest = {
        portfolio_weights: portfolioWeights,
        market_state: {
          volatility: market.volatility,
          correlation: 0.6, // This would come from market analysis
          momentum: market.sentiment,
          sentiment: market.sentiment
        },
        learning_mode: learningMode
      };

      const result = await apiClient.rlOptimization(request);

      const actionMessages: Record<string, string> = {
        'hold': 'Maintain current allocation',
        'rebalance': 'Adjust portfolio weights',
        'reduce_risk': 'Decrease risk exposure',
        'increase_risk': 'Increase risk exposure'
      };

      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `rl-complete-${Date.now()}`,
          type: 'info',
          title: 'RL Analysis Complete',
          message: `Recommendation: ${actionMessages[result.recommendation.action]} (${(result.recommendation.confidence * 100).toFixed(1)}% confidence)`,
          timestamp: new Date().toISOString(),
          duration: 8000,
          actions: result.recommendation.action !== 'hold' ? [{
            label: 'Apply Recommendation',
            action: () => {
              dispatch({ type: 'SET_PORTFOLIO_WEIGHTS', payload: result.recommendation.allocation_weights });
            }
          }] : undefined
        }
      });

      return result;

    } catch (error: any) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'ml', error: error.message }
      });
      return null;
    }
  }, [portfolio.weights, market.volatility, market.sentiment, dispatch]);

  return {
    optimizeWithRL
  };
};

// ============================================================================
// ENSEMBLE PREDICTION HOOK
// ============================================================================

export const useEnsemblePredictions = () => {
  const { state, dispatch } = useAppContext();
  const { ml, portfolio } = state;

  const generateEnsemblePrediction = useCallback(async (
    symbols?: string[],
    horizon: PredictionHorizon = '5_days',
    includeRegime: boolean = true,
    includeRL: boolean = true
  ): Promise<EnsemblePredictionResult | null> => {
    const tickers = symbols || portfolio.tickers;
    
    if (tickers.length === 0) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'ml', error: 'No tickers available for ensemble prediction' }
      });
      return null;
    }

    dispatch({ type: 'SET_ML_PREDICTING', payload: true });

    try {
      const request: EnsemblePredictionRequest = {
        symbols: tickers,
        prediction_horizon: horizon,
        include_regime_analysis: includeRegime,
        include_rl_insights: includeRL
      };

      const result = await apiClient.ensemblePrediction(request);

      // Calculate summary statistics
      const signals = Object.values(result.ensemble_signals);
      const avgConfidence = signals.reduce((sum, signal) => sum + signal.confidence, 0) / signals.length;
      const bullishSignals = signals.filter(signal => signal.composite_score > 0).length;
      
      dispatch({
        type: 'ADD_NOTIFICATION',
        payload: {
          id: `ensemble-complete-${Date.now()}`,
          type: 'success',
          title: 'Ensemble Analysis Complete',
          message: `Analyzed ${tickers.length} symbols. Average confidence: ${(avgConfidence * 100).toFixed(1)}%. ${bullishSignals} bullish signals detected.`,
          timestamp: new Date().toISOString(),
          duration: 8000
        }
      });

      return result;

    } catch (error: any) {
      dispatch({
        type: 'SET_ERROR',
        payload: { module: 'ml', error: error.message }
      });
      return null;
    } finally {
      dispatch({ type: 'SET_ML_PREDICTING', payload: false });
    }
  }, [portfolio.tickers, dispatch]);

  return {
    generateEnsemblePrediction
  };
};

// ============================================================================
// COMPREHENSIVE AI HOOK
// ============================================================================

export const useAI = () => {
  const training = useMLTraining();
  const predictions = usePricePredictions();
  const regime = useMarketRegime();
  const rl = useReinforcementLearning();
  const ensemble = useEnsemblePredictions();
  
  const { state } = useAppContext();
  const { ml } = state;

  // Comprehensive AI analysis workflow
  const runCompleteAIAnalysis = useCallback(async (
    symbols?: string[],
    trainFirst: boolean = false
  ) => {
    try {
      // Step 1: Train models if requested
      if (trainFirst) {
        await training.trainModels(symbols);
      }

      // Step 2: Identify market regime
      const regimeResult = await regime.identifyRegime();

      // Step 3: Generate predictions
      const predictionResult = await predictions.generatePredictions(symbols);

      // Step 4: Run ensemble analysis
      const ensembleResult = await ensemble.generateEnsemblePrediction(symbols);

      // Step 5: RL optimization
      const rlResult = await rl.optimizeWithRL();

      return {
        regime: regimeResult,
        predictions: predictionResult,
        ensemble: ensembleResult,
        rl: rlResult
      };

    } catch (error) {
      console.error('Complete AI analysis failed:', error);
      return null;
    }
  }, [training, regime, predictions, ensemble, rl]);

  // AI readiness check
  const getAIReadiness = useCallback(() => {
    const hasModels = ml.models && ml.models.total_models > 0;
    const hasRecentTraining = ml.lastTraining && 
      (Date.now() - new Date(ml.lastTraining).getTime()) < 24 * 60 * 60 * 1000; // 24 hours
    
    return {
      ready: hasModels && hasRecentTraining,
      hasModels,
      hasRecentTraining,
      modelCount: ml.models?.total_models || 0,
      lastTraining: ml.lastTraining
    };
  }, [ml.models, ml.lastTraining]);

  return {
    // Individual features
    training,
    predictions,
    regime,
    rl,
    ensemble,

    // Comprehensive workflows
    runCompleteAIAnalysis,
    getAIReadiness,

    // Overall state
    isTraining: ml.isTraining,
    isPredicting: ml.isPredicting,
    confidence: ml.confidence,
    models: ml.models
  };
};

// ============================================================================
// AI STATUS MONITORING HOOK
// ============================================================================

export const useAIStatus = () => {
  const { state, dispatch } = useAppContext();
  const [refreshInterval, setRefreshInterval] = useState<NodeJS.Timeout | null>(null);

  const refreshModelStatus = useCallback(async () => {
    try {
      const status = await apiClient.getMLModelStatus();
      dispatch({ type: 'SET_ML_MODELS', payload: status });
    } catch (error) {
      console.error('Failed to refresh ML model status:', error);
    }
  }, [dispatch]);

  const startAutoRefresh = useCallback((intervalMs: number = 30000) => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
    
    const interval = setInterval(refreshModelStatus, intervalMs);
    setRefreshInterval(interval);
  }, [refreshModelStatus, refreshInterval]);

  const stopAutoRefresh = useCallback(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
      setRefreshInterval(null);
    }
  }, [refreshInterval]);

  useEffect(() => {
    // Initial load
    refreshModelStatus();
    
    // Cleanup on unmount
    return () => {
      if (refreshInterval) {
        clearInterval(refreshInterval);
      }
    };
  }, []);

  return {
    refreshModelStatus,
    startAutoRefresh,
    stopAutoRefresh,
    isAutoRefreshing: refreshInterval !== null,
    models: state.ml.models
  };
};

export default useAI;
