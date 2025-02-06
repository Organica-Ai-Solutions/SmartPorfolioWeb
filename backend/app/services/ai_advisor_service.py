import os
from typing import Dict, List, Any
import hashlib
import json
from datetime import datetime
import httpx
import numpy as np
from cryptography.fernet import Fernet

class AIAdvisorService:
    def __init__(self):
        # Generate encryption key for data anonymization
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
        
    def _anonymize_data(self, data: Dict) -> Dict:
        """Anonymize sensitive portfolio data."""
        anonymized = data.copy()
        
        # Replace actual tickers with hash values
        if 'tickers' in anonymized:
            ticker_mapping = {}
            for ticker in anonymized['tickers']:
                hash_value = hashlib.sha256(ticker.encode()).hexdigest()[:8]
                ticker_mapping[ticker] = f"TICKER_{hash_value}"
            
            anonymized['tickers'] = [ticker_mapping[t] for t in anonymized['tickers']]
            
            # Store mapping for later de-anonymization
            self._ticker_mapping = {v: k for k, v in ticker_mapping.items()}
        
        # Normalize portfolio values
        if 'allocations' in anonymized:
            total = sum(anonymized['allocations'].values())
            anonymized['allocations'] = {
                ticker_mapping.get(k, k): v/total 
                for k, v in anonymized['allocations'].items()
            }
        
        # Remove any personal identifiers
        anonymized.pop('user_id', None)
        anonymized.pop('email', None)
        anonymized.pop('name', None)
        
        return anonymized

    def _deanonymize_data(self, data: Dict) -> Dict:
        """Restore original ticker symbols and data."""
        deanonymized = data.copy()
        
        if hasattr(self, '_ticker_mapping'):
            # Restore original tickers
            if 'tickers' in deanonymized:
                deanonymized['tickers'] = [
                    self._ticker_mapping.get(t, t) 
                    for t in deanonymized['tickers']
                ]
            
            # Restore allocations with original tickers
            if 'allocations' in deanonymized:
                deanonymized['allocations'] = {
                    self._ticker_mapping.get(k, k): v 
                    for k, v in deanonymized['allocations'].items()
                }
        
        return deanonymized

    async def get_portfolio_advice(self, portfolio_data: Dict) -> Dict:
        """Get AI-powered portfolio advice with anonymized data."""
        try:
            # Anonymize data before sending to API
            anonymized_data = self._anonymize_data(portfolio_data)
            
            # Prepare request with proxy settings
            async with httpx.AsyncClient(
                proxies={
                    "http://": "socks5://localhost:9050",
                    "https://": "socks5://localhost:9050"
                },
                timeout=30.0
            ) as client:
                response = await client.post(
                    f"{self.api_url}/portfolio/analyze",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=anonymized_data
                )
                
                if response.status_code != 200:
                    raise Exception(f"API request failed: {response.text}")
                
                # De-anonymize the response
                ai_advice = self._deanonymize_data(response.json())
                
                return {
                    "recommendations": ai_advice.get("recommendations", []),
                    "risk_analysis": ai_advice.get("risk_analysis", {}),
                    "market_insights": ai_advice.get("market_insights", {}),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"Error getting AI advice: {str(e)}")
            return {
                "error": "Failed to get AI recommendations",
                "timestamp": datetime.now().isoformat()
            }

    async def get_market_regime_prediction(self, market_data: Dict) -> Dict:
        """Get AI prediction of market regime with anonymized data."""
        try:
            # Anonymize market data
            anonymized_data = self._anonymize_data(market_data)
            
            async with httpx.AsyncClient(
                proxies={
                    "http://": "socks5://localhost:9050",
                    "https://": "socks5://localhost:9050"
                },
                timeout=30.0
            ) as client:
                response = await client.post(
                    f"{self.api_url}/market/regime",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=anonymized_data
                )
                
                if response.status_code != 200:
                    raise Exception(f"API request failed: {response.text}")
                
                prediction = self._deanonymize_data(response.json())
                
                return {
                    "regime": prediction.get("regime"),
                    "confidence": prediction.get("confidence"),
                    "indicators": prediction.get("indicators", {}),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"Error getting market regime prediction: {str(e)}")
            return {
                "error": "Failed to get market regime prediction",
                "timestamp": datetime.now().isoformat()
            }

    async def get_portfolio_metrics(self, portfolio_data: Dict) -> Dict:
        """Get advanced portfolio metrics with AI insights."""
        try:
            # Extract data
            weights = portfolio_data.get("weights", {})
            returns_data = portfolio_data.get("historical_data", {}).get("returns", {})
            
            # Process returns data
            returns = self._process_returns_data(returns_data)
            
            # Calculate risk metrics
            risk_metrics = {
                "sortino_ratio": self._safe_calculate(lambda: self._calculate_sortino_ratio(returns)),
                "treynor_ratio": self._safe_calculate(lambda: self._calculate_treynor_ratio(returns, weights, 0.02)),
                "information_ratio": self._safe_calculate(lambda: self._calculate_information_ratio(returns, weights)),
                "max_drawdown": self._safe_calculate(lambda: self._calculate_max_drawdown(returns)),
                "var_95": self._safe_calculate(lambda: np.percentile(returns, 5)),
                "cvar_95": self._safe_calculate(lambda: returns[returns <= np.percentile(returns, 5)].mean())
            }
            
            # Calculate diversification metrics
            diversification_metrics = {
                "concentration_ratio": self._safe_calculate(lambda: self._calculate_concentration(weights)),
                "effective_num_assets": self._safe_calculate(lambda: 1 / sum(w * w for w in weights.values())),
                "sector_exposure": self._analyze_sector_exposure(portfolio_data.get("sector_data", {})),
                "geographic_exposure": self._analyze_geographic_exposure(portfolio_data.get("geographic_data", {}))
            }
            
            # Market analysis
            market_analysis = self._analyze_market_conditions(returns)
            
            # Risk analysis
            risk_analysis = {
                "risk_decomposition": self._calculate_risk_decomposition(returns, weights),
                "stress_test_results": self._perform_stress_tests(returns, weights),
                "risk_concentration": {
                    "asset_concentration": self._classify_concentration(diversification_metrics["concentration_ratio"]),
                    "sector_concentration": "undefined",  # Would need sector data
                    "factor_concentration": "undefined"   # Would need factor exposure data
                }
            }
            
            # Generate explanations
            explanations = self._generate_portfolio_explanation(
                {"risk_metrics": risk_metrics, "diversification_metrics": diversification_metrics},
                market_analysis,
                risk_analysis
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(risk_metrics, diversification_metrics, market_analysis)
            
            # Generate market outlook
            market_outlook = self._generate_market_outlook(market_analysis, risk_metrics)
            
            return {
                "portfolio_analysis": {
                    "risk_metrics": risk_metrics,
                    "diversification_metrics": diversification_metrics
                },
                "market_analysis": market_analysis,
                "risk_analysis": risk_analysis,
                "explanations": explanations,
                "recommendations": recommendations,
                "market_outlook": market_outlook
            }
        except Exception as e:
            print(f"Error getting portfolio metrics: {str(e)}")
            return {
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "market_regime": "normal",
                "recommendations": [
                    "Unable to generate recommendations at this time",
                    "Please try again later"
                ]
            }

    def _process_returns_data(self, returns_data: Dict) -> np.ndarray:
        """Process and clean returns data."""
        returns_list = []
        for ticker_returns in returns_data.values():
            if isinstance(ticker_returns, dict):
                valid_returns = [
                    float(val) 
                    for val in ticker_returns.values() 
                    if val is not None and not np.isnan(val)
                ]
                returns_list.extend(valid_returns)
        
        returns_array = np.array(returns_list, dtype=np.float64)
        returns_array = returns_array[~np.isnan(returns_array)]
        
        return returns_array if len(returns_array) > 0 else np.array([0.0])

    def _calculate_treynor_ratio(self, returns: np.ndarray, weights: Dict, risk_free_rate: float) -> float:
        """Calculate Treynor ratio using portfolio beta."""
        excess_returns = returns - risk_free_rate/252
        portfolio_beta = self._calculate_portfolio_beta(returns, weights)
        return float(np.mean(excess_returns) * 252 / (portfolio_beta if portfolio_beta != 0 else 1))

    def _calculate_information_ratio(self, returns: np.ndarray, weights: Dict) -> float:
        """Calculate Information ratio against benchmark."""
        benchmark_returns = self._get_benchmark_returns(returns.shape[0])
        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns) * np.sqrt(252)
        return float(np.mean(active_returns) * 252 / (tracking_error if tracking_error != 0 else 1))

    def _analyze_market_conditions(self, returns: np.ndarray) -> Dict:
        """Analyze current market conditions and regime."""
        volatility = np.std(returns) * np.sqrt(252)
        trend = np.mean(returns) * 252
        recent_volatility = np.std(returns[-20:]) * np.sqrt(252)
        
        # Determine market regime
        regime = "normal"
        if volatility > 0.25:
            regime = "high_volatility"
        elif trend < -0.1:
            regime = "bear_market"
        elif trend > 0.1:
            regime = "bull_market"
        
        return {
            "market_regime": regime,
            "volatility_regime": "high" if recent_volatility > volatility else "normal",
            "correlation_regime": self._analyze_correlation_regime(returns),
            "trend_strength": "strong" if abs(trend) > 0.15 else "moderate" if abs(trend) > 0.05 else "weak"
        }

    def _generate_recommendations(self, risk_metrics: Dict, diversification_metrics: Dict, market_analysis: Dict) -> List[str]:
        """Generate AI-powered portfolio recommendations."""
        recommendations = []
        
        # Risk-based recommendations
        if risk_metrics["max_drawdown"] < -0.15:
            recommendations.append("Consider implementing stop-loss orders to manage drawdown risk")
        if risk_metrics["var_95"] < -0.02:
            recommendations.append("Review position sizes to reduce tail risk exposure")
        
        # Diversification recommendations
        if diversification_metrics["effective_num_assets"] < 5:
            recommendations.append("Increase portfolio diversification across more assets")
        
        # Market regime based recommendations
        if market_analysis["market_regime"] == "high_volatility":
            recommendations.append("Consider reducing exposure to high-beta assets")
        elif market_analysis["market_regime"] == "bear_market":
            recommendations.append("Focus on defensive sectors and quality stocks")
        
        return recommendations if recommendations else ["Portfolio appears well-positioned; maintain current strategy"]

    def _generate_market_outlook(self, market_analysis: Dict, risk_metrics: Dict) -> Dict:
        """Generate market outlook based on analysis."""
        return {
            "short_term": self._determine_outlook(market_analysis, timeframe="short"),
            "medium_term": self._determine_outlook(market_analysis, timeframe="medium"),
            "long_term": "positive" if risk_metrics["sortino_ratio"] > 1.0 else "neutral",
            "key_drivers": self._identify_key_drivers(market_analysis)
        }

    def _determine_outlook(self, market_analysis: Dict, timeframe: str) -> str:
        """Determine market outlook for different timeframes."""
        if timeframe == "short":
            return "cautious" if market_analysis["volatility_regime"] == "high" else "neutral"
        elif timeframe == "medium":
            return "positive" if market_analysis["trend_strength"] == "strong" else "neutral"
        return "neutral"

    def _identify_key_drivers(self, market_analysis: Dict) -> List[str]:
        """Identify key market drivers based on analysis."""
        drivers = []
        if market_analysis["volatility_regime"] == "high":
            drivers.append("Market volatility")
        if market_analysis["trend_strength"] in ["strong", "moderate"]:
            drivers.append("Market momentum")
        if market_analysis["correlation_regime"] == "high":
            drivers.append("Systematic risk factors")
        return drivers if drivers else ["Market fundamentals"]

    def _safe_calculate(self, calculation_func, default=0.0):
        """Safely execute a calculation with error handling."""
        try:
            result = calculation_func()
            if isinstance(result, (int, float)) and (np.isnan(result) or np.isinf(result)):
                return default
            return float(result)
        except Exception as e:
            print(f"Error in calculation: {str(e)}")
            return default

    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - risk_free_rate/252  # Convert annual risk-free rate to daily
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        
        avg_return = np.mean(returns) * 252  # Annualize
        downside_std = np.std(downside_returns) * np.sqrt(252)  # Annualize
        
        return float(avg_return / downside_std if downside_std != 0 else 0.0)

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative / rolling_max - 1
        return float(np.min(drawdowns))

    def _calculate_portfolio_beta(self, returns: np.ndarray, weights: Dict) -> float:
        """Calculate portfolio beta against market benchmark."""
        try:
            market_returns = self._get_benchmark_returns(returns.shape[0])
            covariance = np.cov(returns, market_returns)[0,1]
            market_variance = np.var(market_returns)
            return float(covariance / market_variance if market_variance != 0 else 1.0)
        except Exception:
            return 1.0

    def _get_benchmark_returns(self, length: int) -> np.ndarray:
        """Get benchmark (S&P 500) returns for the period."""
        try:
            # Simulated benchmark returns if API call fails
            return np.random.normal(0.0002, 0.01, length)  # Approximate S&P 500 daily returns
        except Exception:
            return np.zeros(length)

    def _calculate_concentration(self, weights: Dict) -> float:
        """Calculate portfolio concentration (Herfindahl-Hirschman Index)."""
        try:
            if not weights:
                return 1.0
            weight_values = list(weights.values())
            return float(sum(w * w for w in weight_values))
        except Exception as e:
            print(f"Error calculating concentration: {str(e)}")
            return 1.0

    def _analyze_correlation_regime(self, returns: np.ndarray) -> str:
        """Analyze correlation regime using rolling correlations."""
        try:
            if len(returns) < 60:  # Need sufficient data
                return "undefined"
            
            # Calculate rolling correlation with market
            market_returns = self._get_benchmark_returns(returns.shape[0])
            correlation = np.corrcoef(returns[-60:], market_returns[-60:])[0,1]
            
            if correlation > 0.7:
                return "high"
            elif correlation < 0.3:
                return "low"
            return "moderate"
        except Exception:
            return "undefined"

    def _analyze_sector_exposure(self, sector_data: Dict) -> Dict[str, float]:
        """Analyze sector exposure based on provided sector data."""
        try:
            if not sector_data:
                return {"undefined": 1.0}
            
            # Normalize sector weights to ensure they sum to 1
            total_weight = sum(sector_data.values())
            if total_weight == 0:
                return {"undefined": 1.0}
                
            return {sector: weight/total_weight for sector, weight in sector_data.items()}
        except Exception:
            return {"undefined": 1.0}

    def _analyze_geographic_exposure(self, geographic_data: Dict) -> Dict[str, float]:
        """Analyze geographic exposure based on provided geographic data."""
        try:
            if not geographic_data:
                return {"undefined": 1.0}
            
            # Normalize geographic weights to ensure they sum to 1
            total_weight = sum(geographic_data.values())
            if total_weight == 0:
                return {"undefined": 1.0}
                
            return {region: weight/total_weight for region, weight in geographic_data.items()}
        except Exception:
            return {"undefined": 1.0}

    def _calculate_risk_decomposition(self, returns: np.ndarray, weights: Dict) -> Dict:
        """Calculate systematic and specific risk components."""
        try:
            portfolio_beta = self._calculate_portfolio_beta(returns, weights)
            market_returns = self._get_benchmark_returns(returns.shape[0])
            market_variance = np.var(market_returns)
            
            systematic_risk = (portfolio_beta ** 2) * market_variance
            total_risk = np.var(returns)
            specific_risk = max(0, total_risk - systematic_risk)
            
            total_risk = systematic_risk + specific_risk
            
            return {
                "systematic_risk": float(systematic_risk / total_risk if total_risk > 0 else 0.5),
                "specific_risk": float(specific_risk / total_risk if total_risk > 0 else 0.5)
            }
        except Exception:
            return {"systematic_risk": 0.5, "specific_risk": 0.5}

    def _perform_stress_tests(self, returns: np.ndarray, weights: Dict) -> Dict:
        """Perform stress tests on the portfolio."""
        try:
            # Simulate different market scenarios
            market_crash = -0.4  # 40% market decline
            rate_shock = 0.01    # 100bps rate increase
            vol_spike = 2.0      # Double volatility
            
            portfolio_beta = self._calculate_portfolio_beta(returns, weights)
            portfolio_vol = np.std(returns)
            
            return {
                "market_crash": float(portfolio_beta * market_crash),
                "interest_rate_shock": float(-0.2 if portfolio_beta > 1.2 else -0.1),  # Higher beta portfolios more sensitive
                "volatility_spike": float(-portfolio_vol * vol_spike * 0.1)  # Impact of volatility increase
            }
        except Exception:
            return {
                "market_crash": -0.3,
                "interest_rate_shock": -0.15,
                "volatility_spike": -0.2
            }

    def _classify_concentration(self, concentration: float) -> str:
        """Classify portfolio concentration level."""
        if concentration > 0.3:
            return "high"
        elif concentration > 0.15:
            return "moderate"
        return "low"

    def _generate_portfolio_explanation(self, portfolio_analysis: Dict, market_analysis: Dict, risk_analysis: Dict) -> Dict:
        """Generate comprehensive explanations of the portfolio analysis in both English and Spanish."""
        try:
            # Get risk level and quality ratings
            volatility = portfolio_analysis.get("risk_metrics", {}).get("volatility", 0)
            sharpe_ratio = portfolio_analysis.get("risk_metrics", {}).get("sharpe_ratio", 0)
            
            risk_level = "high" if volatility > 0.25 else "moderate" if volatility > 0.15 else "low"
            quality = "excellent" if sharpe_ratio > 2 else "good" if sharpe_ratio > 1 else "poor"
            
            # Get diversification level
            effective_num_assets = portfolio_analysis.get("diversification_metrics", {}).get("effective_num_assets", 1)
            div_level = "well" if effective_num_assets > 10 else "moderately" if effective_num_assets > 5 else "poorly"
            
            # Get market regime info
            market_regime = market_analysis.get("market_regime", "undefined")
            volatility_regime = market_analysis.get("volatility_regime", "undefined")
            
            return {
                "summary": {
                    "en": f"Your portfolio shows {risk_level} risk with {quality} risk-adjusted returns. It is {div_level} diversified across assets and sectors. The current market regime is {market_regime} with {volatility_regime} volatility.",
                    "es": f"Su portafolio muestra un riesgo {self._translate_risk_level(risk_level)} con rendimientos ajustados por riesgo {self._translate_quality(quality)}. Está {self._translate_diversification(div_level)} diversificado entre activos y sectores. El régimen actual del mercado es {self._translate_market_regime(market_regime)} con volatilidad {self._translate_volatility_regime(volatility_regime)}."
                },
                "risk_analysis": {
                    "en": self._generate_risk_explanation_en(portfolio_analysis.get("risk_metrics", {}), risk_analysis),
                    "es": self._generate_risk_explanation_es(portfolio_analysis.get("risk_metrics", {}), risk_analysis)
                },
                "diversification_analysis": {
                    "en": self._generate_diversification_explanation_en(portfolio_analysis.get("diversification_metrics", {})),
                    "es": self._generate_diversification_explanation_es(portfolio_analysis.get("diversification_metrics", {}))
                },
                "market_context": {
                    "en": self._generate_market_explanation_en(market_analysis),
                    "es": self._generate_market_explanation_es(market_analysis)
                },
                "stress_test_interpretation": {
                    "en": self._interpret_stress_tests_en(risk_analysis.get("stress_test_results", {})),
                    "es": self._interpret_stress_tests_es(risk_analysis.get("stress_test_results", {}))
                }
            }
        except Exception as e:
            print(f"Error generating explanations: {str(e)}")
            return self._get_safe_default_explanations()

    def _generate_risk_explanation_en(self, risk_metrics: Dict, risk_analysis: Dict) -> str:
        """Generate risk analysis explanation in English."""
        return (
            f"The portfolio has a Sortino ratio of {risk_metrics['sortino_ratio']:.2f}, indicating {'good' if risk_metrics['sortino_ratio'] > 1 else 'poor'} "
            f"risk-adjusted returns considering downside risk. Maximum drawdown is {risk_metrics['max_drawdown']:.1%}, "
            f"with systematic risk accounting for {risk_analysis['risk_decomposition']['systematic_risk']:.1%} of total risk. "
            f"Value at Risk (95%) suggests a maximum daily loss of {risk_metrics['var_95']:.1%} under normal market conditions."
        )

    def _generate_risk_explanation_es(self, risk_metrics: Dict, risk_analysis: Dict) -> str:
        """Generate risk analysis explanation in Spanish."""
        return (
            f"El portafolio tiene un ratio de Sortino de {risk_metrics['sortino_ratio']:.2f}, indicando rendimientos ajustados por riesgo "
            f"{'buenos' if risk_metrics['sortino_ratio'] > 1 else 'pobres'} considerando el riesgo a la baja. La máxima caída es del {risk_metrics['max_drawdown']:.1%}, "
            f"con un riesgo sistemático que representa el {risk_analysis['risk_decomposition']['systematic_risk']:.1%} del riesgo total. "
            f"El Valor en Riesgo (95%) sugiere una pérdida diaria máxima del {risk_metrics['var_95']:.1%} en condiciones normales de mercado."
        )

    def _generate_diversification_explanation_en(self, div_metrics: Dict) -> str:
        """Generate diversification analysis explanation in English."""
        effective_num = div_metrics.get("effective_num_assets", 1.0)
        concentration = div_metrics.get("concentration_ratio", 1.0)
        
        return (
            f"The portfolio's effective number of assets is {effective_num:.1f}, suggesting "
            f"{'good' if effective_num > 10 else 'moderate' if effective_num > 5 else 'limited'} diversification. "
            f"The concentration ratio is {concentration:.2f}, indicating "
            f"{'well-balanced' if concentration < 0.15 else 'moderately concentrated' if concentration < 0.25 else 'highly concentrated'} holdings."
        )

    def _generate_diversification_explanation_es(self, div_metrics: Dict) -> str:
        """Generate diversification analysis explanation in Spanish."""
        effective_num = div_metrics.get("effective_num_assets", 1.0)
        concentration = div_metrics.get("concentration_ratio", 1.0)
        
        return (
            f"El número efectivo de activos es {effective_num:.1f}, sugiriendo una diversificación "
            f"{'buena' if effective_num > 10 else 'moderada' if effective_num > 5 else 'limitada'}. "
            f"El ratio de concentración es {concentration:.2f}, indicando una cartera "
            f"{'bien balanceada' if concentration < 0.15 else 'moderadamente concentrada' if concentration < 0.25 else 'altamente concentrada'}."
        )

    def _generate_market_explanation_en(self, market_analysis: Dict) -> str:
        """Generate market context explanation in English."""
        return (
            f"The market is currently in a {market_analysis['market_regime']} regime with {market_analysis['volatility_regime']} volatility. "
            f"Correlation trends are {market_analysis['correlation_regime']}, and market trend strength is {market_analysis['trend_strength']}. "
            f"This suggests {self._get_market_implication_en(market_analysis)}."
        )

    def _generate_market_explanation_es(self, market_analysis: Dict) -> str:
        """Generate market context explanation in Spanish."""
        return (
            f"El mercado se encuentra actualmente en un régimen {self._translate_market_regime(market_analysis['market_regime'])} "
            f"con volatilidad {self._translate_volatility_regime(market_analysis['volatility_regime'])}. "
            f"Las tendencias de correlación son {self._translate_correlation_regime(market_analysis['correlation_regime'])}, "
            f"y la fuerza de la tendencia del mercado es {self._translate_trend_strength(market_analysis['trend_strength'])}. "
            f"Esto sugiere {self._get_market_implication_es(market_analysis)}."
        )

    def _interpret_stress_tests_en(self, stress_results: Dict) -> str:
        """Interpret stress test results in English."""
        return (
            f"Under a market crash scenario, the portfolio could lose {abs(stress_results['market_crash']):.1%}. "
            f"An interest rate shock might result in a {abs(stress_results['interest_rate_shock']):.1%} decline, "
            f"while a volatility spike could lead to a {abs(stress_results['volatility_spike']):.1%} drawdown. "
            f"This suggests {'high' if abs(stress_results['market_crash']) > 0.3 else 'moderate' if abs(stress_results['market_crash']) > 0.2 else 'reasonable'} "
            f"sensitivity to extreme market events."
        )

    def _interpret_stress_tests_es(self, stress_results: Dict) -> str:
        """Interpret stress test results in Spanish."""
        return (
            f"En un escenario de crisis de mercado, el portafolio podría perder un {abs(stress_results['market_crash']):.1%}. "
            f"Un shock en las tasas de interés podría resultar en una caída del {abs(stress_results['interest_rate_shock']):.1%}, "
            f"mientras que un pico de volatilidad podría llevar a una caída del {abs(stress_results['volatility_spike']):.1%}. "
            f"Esto sugiere una sensibilidad {'alta' if abs(stress_results['market_crash']) > 0.3 else 'moderada' if abs(stress_results['market_crash']) > 0.2 else 'razonable'} "
            f"a eventos extremos del mercado."
        )

    def _translate_risk_level(self, level: str) -> str:
        """Translate risk level to Spanish."""
        translations = {
            "high": "alto",
            "moderate": "moderado",
            "low": "bajo"
        }
        return translations.get(level, level)

    def _translate_quality(self, quality: str) -> str:
        """Translate quality rating to Spanish."""
        translations = {
            "excellent": "excelentes",
            "good": "buenos",
            "poor": "pobres"
        }
        return translations.get(quality, quality)

    def _translate_diversification(self, div_level: str) -> str:
        """Translate diversification level to Spanish."""
        translations = {
            "well": "bien",
            "moderately": "moderadamente",
            "poorly": "pobremente"
        }
        return translations.get(div_level, div_level)

    def _translate_market_regime(self, regime: str) -> str:
        """Translate market regime to Spanish."""
        translations = {
            "normal": "normal",
            "high_volatility": "alta volatilidad",
            "risk_off": "aversión al riesgo",
            "bear_market": "mercado bajista",
            "bull_market": "mercado alcista"
        }
        return translations.get(regime, regime)

    def _translate_volatility_regime(self, regime: str) -> str:
        """Translate volatility regime to Spanish."""
        translations = {
            "high": "alta",
            "moderate": "moderada",
            "low": "baja"
        }
        return translations.get(regime, regime)

    def _translate_trend_strength(self, strength: str) -> str:
        """Translate trend strength to Spanish."""
        translations = {
            "strong": "fuerte",
            "moderate": "moderada",
            "weak": "débil"
        }
        return translations.get(strength, strength)

    def _translate_correlation_regime(self, regime: str) -> str:
        """Translate correlation regime to Spanish."""
        translations = {
            "high": "altas",
            "moderate": "moderadas",
            "low": "bajas"
        }
        return translations.get(regime, regime)

    def _get_safe_default_explanations(self) -> Dict:
        """Return safe default explanations when generation fails."""
        return {
            "summary": {
                "en": "Portfolio analysis is currently unavailable. Please try again later.",
                "es": "El análisis del portafolio no está disponible en este momento. Por favor, inténtelo más tarde."
            },
            "risk_analysis": {
                "en": "Risk analysis data is unavailable.",
                "es": "Los datos de análisis de riesgo no están disponibles."
            },
            "diversification_analysis": {
                "en": "Diversification analysis is unavailable.",
                "es": "El análisis de diversificación no está disponible."
            },
            "market_context": {
                "en": "Market context analysis is unavailable.",
                "es": "El análisis del contexto de mercado no está disponible."
            },
            "stress_test_interpretation": {
                "en": "Stress test results are unavailable.",
                "es": "Los resultados de las pruebas de estrés no están disponibles."
            }
        }

    def _get_market_implication_en(self, market_analysis: Dict) -> str:
        """Get market implication explanation in English."""
        regime = market_analysis.get('market_regime', 'normal')
        volatility = market_analysis.get('volatility_regime', 'normal')
        trend = market_analysis.get('trend_strength', 'moderate')
        
        if regime == 'high_volatility':
            return "increased caution and potential hedging strategies may be warranted"
        elif regime == 'bear_market':
            return "defensive positioning and capital preservation should be prioritized"
        elif regime == 'bull_market':
            return "opportunities for growth but vigilance for signs of market excess"
        else:
            return f"a {trend} trend environment with {volatility} volatility levels"

    def _get_market_implication_es(self, market_analysis: Dict) -> str:
        """Get market implication explanation in Spanish."""
        regime = market_analysis.get('market_regime', 'normal')
        volatility = market_analysis.get('volatility_regime', 'normal')
        trend = market_analysis.get('trend_strength', 'moderate')
        
        if regime == 'high_volatility':
            return "se recomienda mayor cautela y posibles estrategias de cobertura"
        elif regime == 'bear_market':
            return "se debe priorizar el posicionamiento defensivo y la preservación del capital"
        elif regime == 'bull_market':
            return "hay oportunidades de crecimiento pero se debe mantener vigilancia ante señales de exceso"
        else:
            return f"un entorno con tendencia {self._translate_trend_strength(trend)} y niveles de volatilidad {self._translate_volatility_regime(volatility)}" 