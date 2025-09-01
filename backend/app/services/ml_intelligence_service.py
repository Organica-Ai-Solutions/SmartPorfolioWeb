import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass
import asyncio
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    import sklearn
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_regression
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Deep Learning (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

# Advanced ML (optional)
try:
    import xgboost as xgb
    import lightgbm as lgb
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False

logger = logging.getLogger(__name__)

class PredictionHorizon(str, Enum):
    SHORT_TERM = "1_day"        # Next day
    MEDIUM_TERM = "5_days"      # Next week
    LONG_TERM = "21_days"       # Next month
    QUARTERLY = "63_days"       # Next quarter

class ModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    SVM = "svm"
    ENSEMBLE = "ensemble"
    LSTM = "lstm"              # If deep learning available
    TRANSFORMER = "transformer" # If deep learning available
    XGBOOST = "xgboost"        # If XGBoost available
    LIGHTGBM = "lightgbm"      # If LightGBM available

class MarketRegime(str, Enum):
    BULL_TRENDING = "bull_trending"        # Strong upward trend
    BEAR_TRENDING = "bear_trending"        # Strong downward trend
    BULL_VOLATILE = "bull_volatile"        # Upward but volatile
    BEAR_VOLATILE = "bear_volatile"        # Downward but volatile
    SIDEWAYS_LOW_VOL = "sideways_low_vol"  # Range-bound, low volatility
    SIDEWAYS_HIGH_VOL = "sideways_high_vol" # Range-bound, high volatility
    CRISIS = "crisis"                      # Extreme volatility/correlation
    RECOVERY = "recovery"                  # Post-crisis recovery

class RLAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REBALANCE = "rebalance"

@dataclass
class PredictionResult:
    symbol: str
    horizon: PredictionHorizon
    predicted_return: float
    confidence: float
    probability_up: float
    probability_down: float
    key_factors: List[str]
    model_used: ModelType
    timestamp: datetime

@dataclass
class MarketRegimeResult:
    current_regime: MarketRegime
    regime_probability: float
    regime_duration: int  # days in current regime
    next_regime_probability: Dict[MarketRegime, float]
    key_indicators: Dict[str, float]
    timestamp: datetime

@dataclass
class RLRecommendation:
    action: RLAction
    confidence: float
    expected_return: float
    risk_adjustment: float
    allocation_weights: Dict[str, float]
    reasoning: List[str]
    timestamp: datetime

class MLIntelligenceService:
    """Advanced ML-powered intelligence service for portfolio management."""
    
    def __init__(self, models_dir: str = "ml_models"):
        if not ML_AVAILABLE:
            raise ImportError("ML libraries not available. Install scikit-learn, pandas, numpy")
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.price_models = {}  # symbol -> {horizon -> model}
        self.regime_model = None
        self.rl_agent = None
        
        # Feature engineering settings
        self.technical_indicators = [
            'sma_5', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi', 'macd', 'bollinger_upper', 'bollinger_lower',
            'volume_sma', 'high_low_pct', 'open_close_pct'
        ]
        
        self.macro_features = [
            'vix', 'dxy', 'treasury_10y', 'oil_price', 'gold_price'
        ]
        
        # Regime detection features
        self.regime_features = [
            'market_return', 'market_volatility', 'correlation_spy_bonds',
            'vix_level', 'yield_curve_slope', 'sector_rotation'
        ]
        
        # Caching
        self._feature_cache = {}
        self._prediction_cache = {}
        self._regime_cache = None
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
        # Model configurations
        self.model_configs = {
            ModelType.RANDOM_FOREST: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42
            },
            ModelType.GRADIENT_BOOSTING: {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            },
            ModelType.NEURAL_NETWORK: {
                'hidden_layer_sizes': (100, 50),
                'max_iter': 500,
                'random_state': 42
            },
            ModelType.SVM: {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale'
            }
        }
    
    async def train_price_prediction_models(
        self,
        symbols: List[str],
        horizons: List[PredictionHorizon] = None,
        model_types: List[ModelType] = None,
        lookback_days: int = 252,
        retrain: bool = False
    ) -> Dict[str, Any]:
        """
        Train ML models to predict short-term price movements.
        """
        try:
            logger.info(f"Training price prediction models for {len(symbols)} symbols")
            
            if horizons is None:
                horizons = [PredictionHorizon.SHORT_TERM, PredictionHorizon.MEDIUM_TERM]
            
            if model_types is None:
                model_types = [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING, ModelType.NEURAL_NETWORK]
            
            training_results = {}
            
            for symbol in symbols:
                logger.info(f"Training models for {symbol}")
                symbol_results = {}
                
                # Get training data
                training_data = await self._prepare_training_data(symbol, lookback_days)
                
                if training_data is None or len(training_data) < 50:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                for horizon in horizons:
                    horizon_results = {}
                    
                    # Prepare features and targets for this horizon
                    features, targets = self._prepare_features_targets(training_data, horizon)
                    
                    if len(features) < 30:
                        logger.warning(f"Insufficient samples for {symbol} {horizon}")
                        continue
                    
                    # Split data (time series aware)
                    split_idx = int(len(features) * 0.8)
                    X_train, X_test = features[:split_idx], features[split_idx:]
                    y_train, y_test = targets[:split_idx], targets[split_idx:]
                    
                    # Scale features
                    X_train_scaled = self.feature_scaler.fit_transform(X_train)
                    X_test_scaled = self.feature_scaler.transform(X_test)
                    
                    best_model = None
                    best_score = float('-inf')
                    
                    for model_type in model_types:
                        try:
                            # Train model
                            model = self._create_model(model_type)
                            model.fit(X_train_scaled, y_train)
                            
                            # Evaluate
                            predictions = model.predict(X_test_scaled)
                            score = r2_score(y_test, predictions)
                            mse = mean_squared_error(y_test, predictions)
                            mae = mean_absolute_error(y_test, predictions)
                            
                            horizon_results[model_type.value] = {
                                'r2_score': score,
                                'mse': mse,
                                'mae': mae,
                                'model': model
                            }
                            
                            if score > best_score:
                                best_score = score
                                best_model = model
                            
                            logger.info(f"  {model_type.value}: R² = {score:.3f}, MSE = {mse:.6f}")
                            
                        except Exception as e:
                            logger.error(f"Error training {model_type} for {symbol}: {str(e)}")
                    
                    # Store best model
                    if best_model is not None:
                        if symbol not in self.price_models:
                            self.price_models[symbol] = {}
                        self.price_models[symbol][horizon] = best_model
                        
                        # Save model to disk
                        self._save_model(best_model, f"price_{symbol}_{horizon.value}")
                    
                    symbol_results[horizon.value] = horizon_results
                
                training_results[symbol] = symbol_results
            
            # Train ensemble models
            ensemble_results = await self._train_ensemble_models(symbols, horizons)
            training_results['ensemble'] = ensemble_results
            
            logger.info("Price prediction model training completed")
            
            return {
                'training_results': training_results,
                'models_trained': sum(len(self.price_models.get(s, {})) for s in symbols),
                'symbols_processed': len([s for s in symbols if s in self.price_models]),
                'horizons_covered': [h.value for h in horizons],
                'model_types_used': [m.value for m in model_types],
                'feature_count': len(self.technical_indicators) + len(self.macro_features),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error training price prediction models: {str(e)}")
            return {'error': str(e)}
    
    async def predict_price_movements(
        self,
        symbols: List[str],
        horizons: List[PredictionHorizon] = None,
        include_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Predict short-term price movements using trained ML models.
        """
        try:
            logger.info(f"Generating price predictions for {len(symbols)} symbols")
            
            if horizons is None:
                horizons = [PredictionHorizon.SHORT_TERM, PredictionHorizon.MEDIUM_TERM]
            
            predictions = {}
            
            for symbol in symbols:
                if symbol not in self.price_models:
                    logger.warning(f"No trained models for {symbol}")
                    continue
                
                symbol_predictions = {}
                
                # Get current features
                current_data = await self._get_current_features(symbol)
                
                if current_data is None:
                    logger.warning(f"Could not get current data for {symbol}")
                    continue
                
                for horizon in horizons:
                    if horizon not in self.price_models[symbol]:
                        continue
                    
                    try:
                        model = self.price_models[symbol][horizon]
                        
                        # Prepare features
                        features = self._extract_features(current_data)
                        features_scaled = self.feature_scaler.transform([features])
                        
                        # Make prediction
                        predicted_return = model.predict(features_scaled)[0]
                        
                        # Calculate confidence and probabilities
                        confidence = 0.5  # Default
                        prob_up = 0.5
                        prob_down = 0.5
                        
                        if include_confidence:
                            confidence_data = await self._calculate_prediction_confidence(
                                model, features_scaled, symbol, horizon
                            )
                            confidence = confidence_data['confidence']
                            prob_up = confidence_data['prob_up']
                            prob_down = confidence_data['prob_down']
                        
                        # Get key factors
                        key_factors = await self._get_key_factors(model, features, symbol)
                        
                        prediction = PredictionResult(
                            symbol=symbol,
                            horizon=horizon,
                            predicted_return=predicted_return,
                            confidence=confidence,
                            probability_up=prob_up,
                            probability_down=prob_down,
                            key_factors=key_factors,
                            model_used=ModelType.ENSEMBLE,  # Could determine actual type
                            timestamp=datetime.now()
                        )
                        
                        symbol_predictions[horizon.value] = {
                            'predicted_return': predicted_return,
                            'confidence': confidence,
                            'probability_up': prob_up,
                            'probability_down': prob_down,
                            'key_factors': key_factors,
                            'signal_strength': abs(predicted_return) * confidence
                        }
                        
                        logger.info(f"  {symbol} {horizon.value}: {predicted_return:+.2%} (conf: {confidence:.2f})")
                        
                    except Exception as e:
                        logger.error(f"Error predicting {symbol} {horizon}: {str(e)}")
                
                if symbol_predictions:
                    predictions[symbol] = symbol_predictions
            
            # Generate portfolio-level insights
            portfolio_insights = await self._generate_portfolio_insights(predictions)
            
            logger.info("Price movement predictions completed")
            
            return {
                'predictions': predictions,
                'portfolio_insights': portfolio_insights,
                'prediction_count': sum(len(p) for p in predictions.values()),
                'symbols_predicted': list(predictions.keys()),
                'horizons_covered': [h.value for h in horizons],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting price movements: {str(e)}")
            return {'error': str(e)}
    
    async def identify_market_regimes(
        self,
        lookback_days: int = 252,
        retrain_model: bool = False
    ) -> Dict[str, Any]:
        """
        Use clustering algorithms to identify market regimes more accurately.
        """
        try:
            logger.info("Identifying market regimes using clustering algorithms")
            
            # Get market data for regime analysis
            regime_data = await self._prepare_regime_data(lookback_days)
            
            if regime_data is None or len(regime_data) < 30:
                logger.error("Insufficient data for regime analysis")
                return {'error': 'Insufficient data for regime analysis'}
            
            # Train or load regime model
            if retrain_model or self.regime_model is None:
                self.regime_model = await self._train_regime_model(regime_data)
            
            # Get current regime features
            current_features = self._extract_regime_features(regime_data.tail(1))
            
            # Predict current regime
            current_regime_idx = self.regime_model.predict([current_features])[0]
            regime_probs = await self._get_regime_probabilities(current_features)
            
            # Map to regime enum
            regime_mapping = {
                0: MarketRegime.BULL_TRENDING,
                1: MarketRegime.BEAR_TRENDING,
                2: MarketRegime.BULL_VOLATILE,
                3: MarketRegime.BEAR_VOLATILE,
                4: MarketRegime.SIDEWAYS_LOW_VOL,
                5: MarketRegime.SIDEWAYS_HIGH_VOL,
                6: MarketRegime.CRISIS,
                7: MarketRegime.RECOVERY
            }
            
            current_regime = regime_mapping.get(current_regime_idx, MarketRegime.SIDEWAYS_LOW_VOL)
            
            # Calculate regime duration
            regime_duration = await self._calculate_regime_duration(regime_data, current_regime_idx)
            
            # Predict next regime probabilities
            next_regime_probs = await self._predict_regime_transitions(current_features, regime_probs)
            
            # Get key indicators
            key_indicators = self._get_regime_indicators(current_features)
            
            # Regime stability analysis
            stability_analysis = await self._analyze_regime_stability(regime_data)
            
            # Generate regime-based recommendations
            regime_recommendations = await self._generate_regime_recommendations(current_regime, key_indicators)
            
            result = MarketRegimeResult(
                current_regime=current_regime,
                regime_probability=regime_probs[current_regime_idx],
                regime_duration=regime_duration,
                next_regime_probability={
                    regime: prob for regime, prob in zip(regime_mapping.values(), next_regime_probs)
                },
                key_indicators=key_indicators,
                timestamp=datetime.now()
            )
            
            logger.info(f"Current market regime: {current_regime.value} (confidence: {regime_probs[current_regime_idx]:.2f})")
            
            return {
                'current_regime': current_regime.value,
                'regime_probability': regime_probs[current_regime_idx],
                'regime_duration_days': regime_duration,
                'next_regime_probabilities': {
                    regime.value: prob for regime, prob in result.next_regime_probability.items()
                },
                'key_indicators': key_indicators,
                'stability_analysis': stability_analysis,
                'regime_recommendations': regime_recommendations,
                'regime_history': await self._get_regime_history(regime_data),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error identifying market regimes: {str(e)}")
            return {'error': str(e)}
    
    async def reinforcement_learning_optimization(
        self,
        portfolio_weights: Dict[str, float],
        market_state: Dict[str, Any],
        learning_mode: bool = False,
        episodes: int = 1000
    ) -> Dict[str, Any]:
        """
        Implement reinforcement learning for dynamic portfolio optimization.
        """
        try:
            logger.info("Running reinforcement learning portfolio optimization")
            
            # Initialize or load RL agent
            if self.rl_agent is None or learning_mode:
                self.rl_agent = await self._initialize_rl_agent(portfolio_weights.keys())
            
            # Prepare state representation
            state_vector = await self._prepare_rl_state(portfolio_weights, market_state)
            
            if learning_mode:
                # Training mode
                logger.info(f"Training RL agent for {episodes} episodes")
                training_results = await self._train_rl_agent(episodes, portfolio_weights.keys())
                
                return {
                    'mode': 'training',
                    'training_results': training_results,
                    'episodes_completed': episodes,
                    'final_reward': training_results.get('final_reward', 0),
                    'convergence_achieved': training_results.get('converged', False),
                    'timestamp': datetime.now().isoformat()
                }
            
            else:
                # Inference mode
                action_vector = await self._get_rl_action(state_vector)
                
                # Interpret action
                rl_recommendation = await self._interpret_rl_action(
                    action_vector, portfolio_weights, market_state
                )
                
                # Calculate expected outcomes
                expected_outcomes = await self._calculate_rl_outcomes(
                    rl_recommendation, portfolio_weights, market_state
                )
                
                logger.info(f"RL recommendation: {rl_recommendation.action.value} (confidence: {rl_recommendation.confidence:.2f})")
                
                return {
                    'mode': 'inference',
                    'recommendation': {
                        'action': rl_recommendation.action.value,
                        'confidence': rl_recommendation.confidence,
                        'expected_return': rl_recommendation.expected_return,
                        'risk_adjustment': rl_recommendation.risk_adjustment,
                        'allocation_weights': rl_recommendation.allocation_weights,
                        'reasoning': rl_recommendation.reasoning
                    },
                    'expected_outcomes': expected_outcomes,
                    'state_analysis': await self._analyze_rl_state(state_vector),
                    'action_space_exploration': await self._explore_action_space(state_vector),
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Error in RL optimization: {str(e)}")
            return {'error': str(e)}
    
    async def ensemble_ml_prediction(
        self,
        symbols: List[str],
        prediction_horizon: PredictionHorizon = PredictionHorizon.MEDIUM_TERM,
        include_regime_analysis: bool = True,
        include_rl_insights: bool = False
    ) -> Dict[str, Any]:
        """
        Comprehensive ML prediction combining multiple approaches.
        """
        try:
            logger.info(f"Running ensemble ML prediction for {len(symbols)} symbols")
            
            # Get price predictions
            price_predictions = await self.predict_price_movements(
                symbols, [prediction_horizon], include_confidence=True
            )
            
            # Get regime analysis
            regime_analysis = {}
            if include_regime_analysis:
                regime_analysis = await self.identify_market_regimes()
            
            # Get RL insights
            rl_insights = {}
            if include_rl_insights:
                # Create dummy portfolio weights for RL analysis
                equal_weights = {symbol: 1.0/len(symbols) for symbol in symbols}
                market_state = {
                    'volatility': 0.15,
                    'correlation': 0.5,
                    'momentum': 0.0
                }
                rl_insights = await self.reinforcement_learning_optimization(
                    equal_weights, market_state, learning_mode=False
                )
            
            # Combine insights
            ensemble_signals = await self._combine_ml_signals(
                price_predictions, regime_analysis, rl_insights, symbols
            )
            
            # Generate final recommendations
            final_recommendations = await self._generate_ensemble_recommendations(
                ensemble_signals, symbols, prediction_horizon
            )
            
            # Calculate portfolio-level metrics
            portfolio_metrics = await self._calculate_ensemble_metrics(
                final_recommendations, regime_analysis
            )
            
            logger.info("Ensemble ML prediction completed")
            
            return {
                'ensemble_signals': ensemble_signals,
                'final_recommendations': final_recommendations,
                'portfolio_metrics': portfolio_metrics,
                'component_analyses': {
                    'price_predictions': price_predictions.get('predictions', {}),
                    'regime_analysis': regime_analysis,
                    'rl_insights': rl_insights
                },
                'prediction_horizon': prediction_horizon.value,
                'symbols_analyzed': symbols,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble ML prediction: {str(e)}")
            return {'error': str(e)}
    
    # Helper Methods
    
    async def _prepare_training_data(self, symbol: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """Prepare training data for a symbol."""
        try:
            # Get price data
            data = yf.download(symbol, period=f"{lookback_days + 100}d", progress=False)
            
            if data.empty:
                return None
            
            # Calculate technical indicators
            data = self._add_technical_indicators(data)
            
            # Add macro features (simplified)
            data = await self._add_macro_features(data)
            
            # Clean data
            data = data.dropna()
            
            return data
            
        except Exception as e:
            logger.error(f"Error preparing training data for {symbol}: {str(e)}")
            return None
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data."""
        try:
            # Simple Moving Averages
            data['sma_5'] = data['Close'].rolling(5).mean()
            data['sma_20'] = data['Close'].rolling(20).mean()
            data['sma_50'] = data['Close'].rolling(50).mean()
            
            # Exponential Moving Averages
            data['ema_12'] = data['Close'].ewm(span=12).mean()
            data['ema_26'] = data['Close'].ewm(span=26).mean()
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            data['macd'] = data['ema_12'] - data['ema_26']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = data['Close'].rolling(bb_period).std()
            data['bollinger_upper'] = data['sma_20'] + (bb_std * 2)
            data['bollinger_lower'] = data['sma_20'] - (bb_std * 2)
            
            # Volume indicators
            data['volume_sma'] = data['Volume'].rolling(20).mean()
            
            # Price ratios
            data['high_low_pct'] = (data['High'] - data['Low']) / data['Close']
            data['open_close_pct'] = (data['Close'] - data['Open']) / data['Open']
            
            return data
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return data
    
    async def _add_macro_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add macro economic features (simplified version)."""
        try:
            # For now, add synthetic macro features
            # In production, these would come from economic data APIs
            
            data['vix'] = 20 + 10 * np.random.random(len(data))  # Synthetic VIX
            data['dxy'] = 100 + 5 * np.random.random(len(data))  # Synthetic Dollar Index
            data['treasury_10y'] = 3 + 2 * np.random.random(len(data))  # Synthetic 10Y yield
            data['oil_price'] = 70 + 20 * np.random.random(len(data))  # Synthetic oil
            data['gold_price'] = 1800 + 200 * np.random.random(len(data))  # Synthetic gold
            
            return data
            
        except Exception as e:
            logger.error(f"Error adding macro features: {str(e)}")
            return data
    
    def _prepare_features_targets(self, data: pd.DataFrame, horizon: PredictionHorizon) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for ML training."""
        try:
            # Extract features
            feature_columns = self.technical_indicators + self.macro_features
            features = data[feature_columns].dropna()
            
            # Calculate targets (future returns)
            horizon_days = int(horizon.value.split('_')[0])
            returns = data['Close'].pct_change(horizon_days).shift(-horizon_days)
            
            # Align features and targets
            valid_indices = features.index.intersection(returns.dropna().index)
            features = features.loc[valid_indices].values
            targets = returns.loc[valid_indices].values
            
            return features, targets
            
        except Exception as e:
            logger.error(f"Error preparing features and targets: {str(e)}")
            return np.array([]), np.array([])
    
    def _create_model(self, model_type: ModelType):
        """Create ML model based on type."""
        try:
            if model_type == ModelType.RANDOM_FOREST:
                return RandomForestRegressor(**self.model_configs[model_type])
            
            elif model_type == ModelType.GRADIENT_BOOSTING:
                return GradientBoostingRegressor(**self.model_configs[model_type])
            
            elif model_type == ModelType.NEURAL_NETWORK:
                return MLPRegressor(**self.model_configs[model_type])
            
            elif model_type == ModelType.SVM:
                return SVR(**self.model_configs[model_type])
            
            elif model_type == ModelType.XGBOOST and ADVANCED_ML_AVAILABLE:
                return xgb.XGBRegressor(random_state=42)
            
            elif model_type == ModelType.LIGHTGBM and ADVANCED_ML_AVAILABLE:
                return lgb.LGBMRegressor(random_state=42, verbose=-1)
            
            else:
                # Default to Random Forest
                return RandomForestRegressor(**self.model_configs[ModelType.RANDOM_FOREST])
            
        except Exception as e:
            logger.error(f"Error creating model {model_type}: {str(e)}")
            return RandomForestRegressor(**self.model_configs[ModelType.RANDOM_FOREST])
    
    async def _train_ensemble_models(self, symbols: List[str], horizons: List[PredictionHorizon]) -> Dict[str, Any]:
        """Train ensemble models combining multiple base models."""
        try:
            ensemble_results = {}
            
            for horizon in horizons:
                # Collect predictions from individual models
                model_predictions = {}
                
                for symbol in symbols:
                    if symbol in self.price_models and horizon in self.price_models[symbol]:
                        # Get model performance data
                        # This would be expanded with actual ensemble training
                        ensemble_results[f"{symbol}_{horizon.value}"] = {
                            'ensemble_score': 0.75,  # Placeholder
                            'component_models': ['random_forest', 'gradient_boosting', 'neural_network'],
                            'weights': [0.4, 0.4, 0.2]  # Model weights in ensemble
                        }
            
            return ensemble_results
            
        except Exception as e:
            logger.error(f"Error training ensemble models: {str(e)}")
            return {}
    
    def _save_model(self, model, model_name: str):
        """Save trained model to disk."""
        try:
            model_path = self.models_dir / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved: {model_path}")
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {str(e)}")
    
    def _load_model(self, model_name: str):
        """Load trained model from disk."""
        try:
            model_path = self.models_dir / f"{model_name}.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            return None
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None
    
    async def _get_current_features(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get current feature data for prediction."""
        try:
            # Get recent data
            data = yf.download(symbol, period="60d", progress=False)
            
            if data.empty:
                return None
            
            # Add features
            data = self._add_technical_indicators(data)
            data = await self._add_macro_features(data)
            
            return data.tail(1)
            
        except Exception as e:
            logger.error(f"Error getting current features for {symbol}: {str(e)}")
            return None
    
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract feature vector from data."""
        try:
            feature_columns = self.technical_indicators + self.macro_features
            features = data[feature_columns].iloc[-1].values
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return np.array([])
    
    async def _calculate_prediction_confidence(
        self, 
        model, 
        features: np.ndarray, 
        symbol: str, 
        horizon: PredictionHorizon
    ) -> Dict[str, float]:
        """Calculate prediction confidence and probabilities."""
        try:
            # For tree-based models, use prediction variance
            if hasattr(model, 'estimators_'):
                # Random Forest or Gradient Boosting
                predictions = [tree.predict(features)[0] for tree in model.estimators_]
                std_dev = np.std(predictions)
                confidence = max(0.1, 1.0 - min(1.0, std_dev * 10))  # Normalize std to confidence
            else:
                confidence = 0.6  # Default confidence
            
            # Calculate directional probabilities
            prediction = model.predict(features)[0]
            
            if prediction > 0.01:  # > 1% positive
                prob_up = 0.6 + confidence * 0.2
                prob_down = 1.0 - prob_up
            elif prediction < -0.01:  # < -1% negative
                prob_down = 0.6 + confidence * 0.2
                prob_up = 1.0 - prob_down
            else:  # Near neutral
                prob_up = 0.5
                prob_down = 0.5
            
            return {
                'confidence': confidence,
                'prob_up': prob_up,
                'prob_down': prob_down
            }
            
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {str(e)}")
            return {'confidence': 0.5, 'prob_up': 0.5, 'prob_down': 0.5}
    
    async def _get_key_factors(self, model, features: np.ndarray, symbol: str) -> List[str]:
        """Get key factors driving the prediction."""
        try:
            # For tree-based models, use feature importance
            if hasattr(model, 'feature_importances_'):
                feature_names = self.technical_indicators + self.macro_features
                importances = model.feature_importances_
                
                # Get top 3 features
                top_indices = np.argsort(importances)[-3:][::-1]
                key_factors = [feature_names[i] for i in top_indices]
                
                return key_factors
            else:
                return ['technical_analysis', 'market_momentum', 'volatility_regime']
            
        except Exception as e:
            logger.error(f"Error getting key factors: {str(e)}")
            return ['price_momentum', 'volume_trend', 'market_sentiment']
    
    async def _generate_portfolio_insights(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate portfolio-level insights from individual predictions."""
        try:
            if not predictions:
                return {}
            
            # Aggregate insights
            total_symbols = len(predictions)
            positive_signals = 0
            negative_signals = 0
            high_confidence_signals = 0
            
            avg_predicted_return = 0
            avg_confidence = 0
            
            for symbol, symbol_preds in predictions.items():
                for horizon, pred_data in symbol_preds.items():
                    predicted_return = pred_data['predicted_return']
                    confidence = pred_data['confidence']
                    
                    if predicted_return > 0.01:
                        positive_signals += 1
                    elif predicted_return < -0.01:
                        negative_signals += 1
                    
                    if confidence > 0.7:
                        high_confidence_signals += 1
                    
                    avg_predicted_return += predicted_return
                    avg_confidence += confidence
            
            total_predictions = sum(len(p) for p in predictions.values())
            avg_predicted_return /= total_predictions if total_predictions > 0 else 1
            avg_confidence /= total_predictions if total_predictions > 0 else 1
            
            # Market sentiment
            if positive_signals > negative_signals * 1.5:
                market_sentiment = "bullish"
            elif negative_signals > positive_signals * 1.5:
                market_sentiment = "bearish"
            else:
                market_sentiment = "neutral"
            
            return {
                'market_sentiment': market_sentiment,
                'avg_predicted_return': avg_predicted_return,
                'avg_confidence': avg_confidence,
                'signal_distribution': {
                    'positive': positive_signals,
                    'negative': negative_signals,
                    'neutral': total_predictions - positive_signals - negative_signals
                },
                'high_confidence_signals': high_confidence_signals,
                'prediction_quality': 'high' if avg_confidence > 0.7 else 'medium' if avg_confidence > 0.5 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error generating portfolio insights: {str(e)}")
            return {}
    
    async def _prepare_regime_data(self, lookback_days: int) -> Optional[pd.DataFrame]:
        """Prepare data for regime analysis."""
        try:
            # Get market data (SPY as proxy)
            spy_data = yf.download("SPY", period=f"{lookback_days + 50}d", progress=False)
            
            if spy_data.empty:
                return None
            
            # Calculate regime features
            regime_data = pd.DataFrame(index=spy_data.index)
            
            # Returns and volatility
            regime_data['market_return'] = spy_data['Close'].pct_change()
            regime_data['market_volatility'] = regime_data['market_return'].rolling(20).std()
            
            # Try to get bonds data for correlation
            try:
                bond_data = yf.download("TLT", period=f"{lookback_days + 50}d", progress=False)
                bond_returns = bond_data['Close'].pct_change()
                regime_data['correlation_spy_bonds'] = regime_data['market_return'].rolling(60).corr(bond_returns)
            except:
                regime_data['correlation_spy_bonds'] = 0.0  # Default if bonds data unavailable
            
            # Synthetic features (in production, these would be real)
            regime_data['vix_level'] = 20 + 10 * np.random.random(len(regime_data))
            regime_data['yield_curve_slope'] = 1 + 2 * np.random.random(len(regime_data))
            regime_data['sector_rotation'] = np.random.random(len(regime_data))
            
            return regime_data.dropna()
            
        except Exception as e:
            logger.error(f"Error preparing regime data: {str(e)}")
            return None
    
    async def _train_regime_model(self, data: pd.DataFrame):
        """Train clustering model for regime identification."""
        try:
            # Prepare features
            features = data[self.regime_features].values
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Use KMeans clustering with 8 regimes
            kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
            kmeans.fit(features_scaled)
            
            # Store scaler with model
            kmeans.scaler = scaler
            
            return kmeans
            
        except Exception as e:
            logger.error(f"Error training regime model: {str(e)}")
            # Return dummy model
            from sklearn.cluster import KMeans
            return KMeans(n_clusters=8, random_state=42)
    
    def _extract_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract regime features from data."""
        try:
            features = data[self.regime_features].iloc[-1].values
            return features
        except Exception as e:
            logger.error(f"Error extracting regime features: {str(e)}")
            return np.array([0.0] * len(self.regime_features))
    
    async def _get_regime_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Get regime probabilities."""
        try:
            if hasattr(self.regime_model, 'scaler'):
                features_scaled = self.regime_model.scaler.transform([features])
            else:
                features_scaled = [features]
            
            # For KMeans, calculate distances to centroids as proxy for probabilities
            distances = self.regime_model.transform(features_scaled)[0]
            
            # Convert distances to probabilities (closer = higher probability)
            probabilities = 1.0 / (1.0 + distances)
            probabilities = probabilities / probabilities.sum()
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error getting regime probabilities: {str(e)}")
            return np.array([1.0/8] * 8)  # Equal probabilities
    
    async def _calculate_regime_duration(self, data: pd.DataFrame, current_regime: int) -> int:
        """Calculate how long we've been in current regime."""
        try:
            # This would normally track regime history
            # For now, return a reasonable estimate
            return np.random.randint(5, 30)  # 5-30 days
        except Exception as e:
            logger.error(f"Error calculating regime duration: {str(e)}")
            return 15  # Default
    
    async def _predict_regime_transitions(self, features: np.ndarray, current_probs: np.ndarray) -> np.ndarray:
        """Predict regime transition probabilities."""
        try:
            # Simple transition model - in practice this would be more sophisticated
            # Add some noise to current probabilities to simulate transitions
            noise = np.random.normal(0, 0.1, len(current_probs))
            next_probs = current_probs + noise
            next_probs = np.clip(next_probs, 0, 1)
            next_probs = next_probs / next_probs.sum()
            
            return next_probs
            
        except Exception as e:
            logger.error(f"Error predicting regime transitions: {str(e)}")
            return current_probs
    
    def _get_regime_indicators(self, features: np.ndarray) -> Dict[str, float]:
        """Get key regime indicators."""
        try:
            feature_names = self.regime_features
            indicators = {}
            
            for i, name in enumerate(feature_names):
                if i < len(features):
                    indicators[name] = float(features[i])
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error getting regime indicators: {str(e)}")
            return {}
    
    async def _analyze_regime_stability(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze regime stability."""
        try:
            return {
                'stability_score': 0.75,  # Placeholder
                'volatility_trend': 'increasing',
                'correlation_trend': 'stable',
                'regime_persistence': 0.8
            }
        except Exception as e:
            logger.error(f"Error analyzing regime stability: {str(e)}")
            return {}
    
    async def _generate_regime_recommendations(self, regime: MarketRegime, indicators: Dict[str, float]) -> List[str]:
        """Generate recommendations based on current regime."""
        try:
            recommendations = []
            
            if regime == MarketRegime.BULL_TRENDING:
                recommendations = [
                    "Increase equity allocation",
                    "Reduce hedge positions",
                    "Consider momentum strategies"
                ]
            elif regime == MarketRegime.BEAR_TRENDING:
                recommendations = [
                    "Increase defensive positions",
                    "Consider short strategies",
                    "Raise cash levels"
                ]
            elif regime == MarketRegime.CRISIS:
                recommendations = [
                    "Maximize cash and bonds",
                    "Activate tail hedges",
                    "Reduce leverage"
                ]
            else:
                recommendations = [
                    "Maintain balanced allocation",
                    "Monitor for regime changes",
                    "Use mean reversion strategies"
                ]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating regime recommendations: {str(e)}")
            return ["Monitor market conditions", "Maintain diversification"]
    
    async def _get_regime_history(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get historical regime information."""
        try:
            # Simplified regime history
            history = []
            for i in range(min(5, len(data)//20)):  # Last 5 regime periods
                start_date = data.index[-(i+1)*20]
                end_date = data.index[-i*20] if i > 0 else data.index[-1]
                
                history.append({
                    'regime': f'regime_{i%8}',
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'duration_days': (end_date - start_date).days,
                    'avg_return': np.random.uniform(-0.1, 0.1)  # Placeholder
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting regime history: {str(e)}")
            return []
    
    # Reinforcement Learning Methods (Simplified Implementation)
    
    async def _initialize_rl_agent(self, symbols: List[str]):
        """Initialize RL agent."""
        try:
            # Simplified RL agent - in practice would use more sophisticated RL libraries
            class SimpleRLAgent:
                def __init__(self, symbols):
                    self.symbols = symbols
                    self.q_table = {}  # State-action Q-table
                    self.learning_rate = 0.1
                    self.discount_factor = 0.95
                    self.exploration_rate = 0.1
                
                def get_action(self, state):
                    # Epsilon-greedy action selection
                    if np.random.random() < self.exploration_rate:
                        return np.random.choice(['buy', 'sell', 'hold', 'rebalance'])
                    else:
                        # Get best action from Q-table
                        state_str = str(state)
                        if state_str in self.q_table:
                            return max(self.q_table[state_str], key=self.q_table[state_str].get)
                        else:
                            return 'hold'  # Default action
                
                def update_q_value(self, state, action, reward, next_state):
                    # Q-learning update
                    state_str = str(state)
                    next_state_str = str(next_state)
                    
                    if state_str not in self.q_table:
                        self.q_table[state_str] = {'buy': 0, 'sell': 0, 'hold': 0, 'rebalance': 0}
                    
                    if next_state_str in self.q_table:
                        max_next_q = max(self.q_table[next_state_str].values())
                    else:
                        max_next_q = 0
                    
                    current_q = self.q_table[state_str][action]
                    new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
                    self.q_table[state_str][action] = new_q
            
            return SimpleRLAgent(symbols)
            
        except Exception as e:
            logger.error(f"Error initializing RL agent: {str(e)}")
            return None
    
    async def _prepare_rl_state(self, portfolio_weights: Dict[str, float], market_state: Dict[str, Any]) -> np.ndarray:
        """Prepare state vector for RL agent."""
        try:
            # Combine portfolio and market state into vector
            state_vector = []
            
            # Portfolio weights
            for symbol in sorted(portfolio_weights.keys()):
                state_vector.append(portfolio_weights.get(symbol, 0))
            
            # Market state
            state_vector.append(market_state.get('volatility', 0.15))
            state_vector.append(market_state.get('correlation', 0.5))
            state_vector.append(market_state.get('momentum', 0.0))
            
            return np.array(state_vector)
            
        except Exception as e:
            logger.error(f"Error preparing RL state: {str(e)}")
            return np.array([0.5] * (len(portfolio_weights) + 3))
    
    async def _train_rl_agent(self, episodes: int, symbols: List[str]) -> Dict[str, Any]:
        """Train RL agent."""
        try:
            # Simplified training simulation
            total_reward = 0
            rewards_history = []
            
            for episode in range(episodes):
                # Simulate market environment
                state = np.random.random(len(symbols) + 3)
                action = self.rl_agent.get_action(state)
                
                # Simulate reward (random for demo)
                reward = np.random.normal(0, 0.1)
                total_reward += reward
                rewards_history.append(reward)
                
                # Update Q-table
                next_state = np.random.random(len(symbols) + 3)
                self.rl_agent.update_q_value(state, action, reward, next_state)
            
            # Check convergence
            recent_rewards = rewards_history[-100:]
            converged = len(recent_rewards) == 100 and np.std(recent_rewards) < 0.05
            
            return {
                'total_reward': total_reward,
                'avg_reward': total_reward / episodes,
                'final_reward': rewards_history[-1] if rewards_history else 0,
                'converged': converged,
                'q_table_size': len(self.rl_agent.q_table)
            }
            
        except Exception as e:
            logger.error(f"Error training RL agent: {str(e)}")
            return {'total_reward': 0, 'converged': False}
    
    async def _get_rl_action(self, state_vector: np.ndarray) -> str:
        """Get action from RL agent."""
        try:
            if self.rl_agent:
                return self.rl_agent.get_action(state_vector)
            else:
                return 'hold'
        except Exception as e:
            logger.error(f"Error getting RL action: {str(e)}")
            return 'hold'
    
    async def _interpret_rl_action(
        self, 
        action: str, 
        portfolio_weights: Dict[str, float], 
        market_state: Dict[str, Any]
    ) -> RLRecommendation:
        """Interpret RL action into portfolio recommendation."""
        try:
            confidence = 0.7  # Placeholder
            expected_return = 0.05  # Placeholder
            risk_adjustment = 0.0
            
            if action == 'buy':
                # Increase equity allocation
                new_weights = portfolio_weights.copy()
                for symbol in new_weights:
                    if 'SPY' in symbol or 'AAPL' in symbol:  # Equity-like
                        new_weights[symbol] *= 1.1
                reasoning = ["Market conditions favor equity exposure", "Momentum signals positive"]
                
            elif action == 'sell':
                # Reduce risk
                new_weights = portfolio_weights.copy()
                for symbol in new_weights:
                    new_weights[symbol] *= 0.9
                reasoning = ["Risk reduction recommended", "Market volatility increasing"]
                
            elif action == 'rebalance':
                # Return to target weights
                new_weights = {symbol: 1.0/len(portfolio_weights) for symbol in portfolio_weights}
                reasoning = ["Portfolio drift detected", "Rebalancing to target allocation"]
                
            else:  # hold
                new_weights = portfolio_weights.copy()
                reasoning = ["Maintain current allocation", "No strong signals detected"]
            
            # Normalize weights
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                new_weights = {k: v/total_weight for k, v in new_weights.items()}
            
            return RLRecommendation(
                action=RLAction(action),
                confidence=confidence,
                expected_return=expected_return,
                risk_adjustment=risk_adjustment,
                allocation_weights=new_weights,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error interpreting RL action: {str(e)}")
            return RLRecommendation(
                action=RLAction.HOLD,
                confidence=0.5,
                expected_return=0.0,
                risk_adjustment=0.0,
                allocation_weights=portfolio_weights,
                reasoning=["Error in action interpretation"],
                timestamp=datetime.now()
            )
    
    async def _calculate_rl_outcomes(
        self, 
        recommendation: RLRecommendation, 
        portfolio_weights: Dict[str, float], 
        market_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate expected outcomes of RL recommendation."""
        try:
            return {
                'expected_sharpe_ratio': 1.2,  # Placeholder
                'expected_volatility': 0.15,
                'expected_max_drawdown': 0.10,
                'risk_adjusted_return': recommendation.expected_return / 0.15,
                'portfolio_efficiency_score': 0.8
            }
        except Exception as e:
            logger.error(f"Error calculating RL outcomes: {str(e)}")
            return {}
    
    async def _analyze_rl_state(self, state_vector: np.ndarray) -> Dict[str, Any]:
        """Analyze current RL state."""
        try:
            return {
                'state_complexity': len(state_vector),
                'portfolio_concentration': np.max(state_vector[:-3]),  # Exclude market state
                'market_stress_level': state_vector[-3],  # Volatility
                'correlation_regime': 'high' if state_vector[-2] > 0.7 else 'low',
                'momentum_signal': 'positive' if state_vector[-1] > 0 else 'negative'
            }
        except Exception as e:
            logger.error(f"Error analyzing RL state: {str(e)}")
            return {}
    
    async def _explore_action_space(self, state_vector: np.ndarray) -> Dict[str, Any]:
        """Explore possible actions and their expected outcomes."""
        try:
            actions = ['buy', 'sell', 'hold', 'rebalance']
            action_analysis = {}
            
            for action in actions:
                # Simulate action outcomes
                expected_return = np.random.uniform(-0.05, 0.10)
                risk_score = np.random.uniform(0.1, 0.3)
                
                action_analysis[action] = {
                    'expected_return': expected_return,
                    'risk_score': risk_score,
                    'sharpe_estimate': expected_return / risk_score,
                    'confidence': 0.6
                }
            
            return action_analysis
            
        except Exception as e:
            logger.error(f"Error exploring action space: {str(e)}")
            return {}
    
    # Ensemble Methods
    
    async def _combine_ml_signals(
        self, 
        price_predictions: Dict[str, Any], 
        regime_analysis: Dict[str, Any], 
        rl_insights: Dict[str, Any], 
        symbols: List[str]
    ) -> Dict[str, Any]:
        """Combine signals from different ML approaches."""
        try:
            combined_signals = {}
            
            for symbol in symbols:
                signal = {
                    'composite_score': 0.0,
                    'confidence': 0.0,
                    'signal_components': {},
                    'recommendation': 'hold'
                }
                
                # Price prediction component
                if symbol in price_predictions.get('predictions', {}):
                    pred_data = price_predictions['predictions'][symbol]
                    # Use first available horizon
                    first_horizon = list(pred_data.keys())[0]
                    pred_return = pred_data[first_horizon]['predicted_return']
                    pred_conf = pred_data[first_horizon]['confidence']
                    
                    signal['signal_components']['price_prediction'] = {
                        'return': pred_return,
                        'confidence': pred_conf,
                        'weight': 0.4
                    }
                    signal['composite_score'] += pred_return * pred_conf * 0.4
                    signal['confidence'] += pred_conf * 0.4
                
                # Regime component
                if regime_analysis and 'current_regime' in regime_analysis:
                    regime = regime_analysis['current_regime']
                    regime_weight = 0.3
                    
                    # Map regime to signal
                    if 'bull' in regime:
                        regime_signal = 0.05
                    elif 'bear' in regime:
                        regime_signal = -0.05
                    else:
                        regime_signal = 0.0
                    
                    signal['signal_components']['regime_analysis'] = {
                        'regime': regime,
                        'signal': regime_signal,
                        'weight': regime_weight
                    }
                    signal['composite_score'] += regime_signal * regime_weight
                    signal['confidence'] += 0.7 * regime_weight
                
                # RL component
                if rl_insights and 'recommendation' in rl_insights:
                    rl_conf = rl_insights['recommendation'].get('confidence', 0.5)
                    rl_return = rl_insights['recommendation'].get('expected_return', 0.0)
                    rl_weight = 0.3
                    
                    signal['signal_components']['rl_insights'] = {
                        'expected_return': rl_return,
                        'confidence': rl_conf,
                        'weight': rl_weight
                    }
                    signal['composite_score'] += rl_return * rl_conf * rl_weight
                    signal['confidence'] += rl_conf * rl_weight
                
                # Determine recommendation
                if signal['composite_score'] > 0.02 and signal['confidence'] > 0.6:
                    signal['recommendation'] = 'buy'
                elif signal['composite_score'] < -0.02 and signal['confidence'] > 0.6:
                    signal['recommendation'] = 'sell'
                else:
                    signal['recommendation'] = 'hold'
                
                combined_signals[symbol] = signal
            
            return combined_signals
            
        except Exception as e:
            logger.error(f"Error combining ML signals: {str(e)}")
            return {}
    
    async def _generate_ensemble_recommendations(
        self, 
        ensemble_signals: Dict[str, Any], 
        symbols: List[str], 
        horizon: PredictionHorizon
    ) -> Dict[str, Any]:
        """Generate final recommendations from ensemble signals."""
        try:
            recommendations = {}
            
            for symbol in symbols:
                if symbol in ensemble_signals:
                    signal = ensemble_signals[symbol]
                    
                    recommendations[symbol] = {
                        'action': signal['recommendation'],
                        'confidence': signal['confidence'],
                        'expected_return': signal['composite_score'],
                        'signal_strength': abs(signal['composite_score']) * signal['confidence'],
                        'horizon': horizon.value,
                        'key_drivers': list(signal['signal_components'].keys()),
                        'risk_assessment': 'low' if signal['confidence'] > 0.7 else 'medium'
                    }
            
            # Portfolio-level recommendations
            buy_signals = sum(1 for r in recommendations.values() if r['action'] == 'buy')
            sell_signals = sum(1 for r in recommendations.values() if r['action'] == 'sell')
            
            portfolio_recommendation = {
                'overall_sentiment': 'bullish' if buy_signals > sell_signals else 'bearish' if sell_signals > buy_signals else 'neutral',
                'signal_distribution': {
                    'buy': buy_signals,
                    'sell': sell_signals,
                    'hold': len(recommendations) - buy_signals - sell_signals
                },
                'recommended_action': 'increase_risk' if buy_signals > len(symbols) * 0.6 else 'reduce_risk' if sell_signals > len(symbols) * 0.6 else 'maintain'
            }
            
            return {
                'individual_recommendations': recommendations,
                'portfolio_recommendation': portfolio_recommendation
            }
            
        except Exception as e:
            logger.error(f"Error generating ensemble recommendations: {str(e)}")
            return {}
    
    async def _calculate_ensemble_metrics(
        self, 
        recommendations: Dict[str, Any], 
        regime_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate ensemble-level metrics."""
        try:
            individual_recs = recommendations.get('individual_recommendations', {})
            
            if not individual_recs:
                return {}
            
            # Calculate metrics
            avg_confidence = np.mean([r['confidence'] for r in individual_recs.values()])
            avg_expected_return = np.mean([r['expected_return'] for r in individual_recs.values()])
            signal_strength = np.mean([r['signal_strength'] for r in individual_recs.values()])
            
            # Risk assessment
            high_conf_signals = sum(1 for r in individual_recs.values() if r['confidence'] > 0.7)
            total_signals = len(individual_recs)
            
            quality_score = high_conf_signals / total_signals if total_signals > 0 else 0
            
            return {
                'average_confidence': avg_confidence,
                'average_expected_return': avg_expected_return,
                'average_signal_strength': signal_strength,
                'prediction_quality_score': quality_score,
                'high_confidence_ratio': high_conf_signals / total_signals if total_signals > 0 else 0,
                'market_regime_alignment': regime_analysis.get('current_regime', 'unknown'),
                'ensemble_performance_estimate': 'high' if quality_score > 0.7 else 'medium' if quality_score > 0.4 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error calculating ensemble metrics: {str(e)}")
            return {}

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all trained models."""
        try:
            status = {
                'price_prediction_models': {},
                'regime_model_trained': self.regime_model is not None,
                'rl_agent_initialized': self.rl_agent is not None,
                'total_models': 0
            }
            
            for symbol, models in self.price_models.items():
                status['price_prediction_models'][symbol] = list(models.keys())
                status['total_models'] += len(models)
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting model status: {str(e)}")
            return {'error': str(e)}
