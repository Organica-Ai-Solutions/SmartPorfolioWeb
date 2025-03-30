# Portfolio Optimization Guide

This guide explains the portfolio optimization techniques used in SmartPortfolio, including examples and theoretical background.

## Overview

SmartPortfolio uses several advanced portfolio optimization techniques:
- Modern Portfolio Theory (MPT)
- Monte Carlo Simulation
- Efficient Frontier Analysis
- Risk-adjusted Return Optimization

## Libraries Used

- **PyPortfolioOpt**: For efficient frontier optimization and portfolio allocation
- **FinQuant**: For portfolio analysis and Monte Carlo simulations
- **yfinance**: For fetching historical market data
- **matplotlib**: For visualization

## Real-World Use Cases

### 1. Conservative Retirement Portfolio
```python
# Example: Building a conservative portfolio for retirement
conservative_assets = ['BND', 'VTI', 'VXUS', 'VTIP']  # Bonds, US Stocks, Int'l Stocks, TIPS
risk_aversion = 25  # Higher number = more conservative

# Create portfolio with risk constraints
ef = EfficientFrontier(mu, S)
ef.add_constraint(lambda w: w[0] + w[3] >= 0.5)  # At least 50% in bonds + TIPS
weights = ef.efficient_risk(target_volatility=0.1)  # Target low volatility
```

### 2. Growth-Focused Tech Portfolio
```python
# Example: High-growth technology sector portfolio
tech_assets = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSM']
start_date = datetime(2020, 1, 1)

# Build portfolio with sector constraints
ef = EfficientFrontier(mu, S)
ef.add_sector_constraints(sectors, max_sector_weights={'technology': 0.6})
weights = ef.max_sharpe()  # Optimize for highest risk-adjusted return
```

### 3. ESG-Focused Portfolio
```python
# Example: Environmental, Social, and Governance focused portfolio
esg_assets = ['ICLN', 'ESGV', 'VSGX', 'ESGU']
constraints = {
    'min_weight': 0.05,  # Minimum 5% in each asset
    'max_weight': 0.4    # Maximum 40% in any asset
}

ef = EfficientFrontier(mu, S)
ef.add_constraint(lambda w: sum(w) == 1)  # Fully invested
weights = ef.efficient_return(target_return=0.12)  # Target 12% return
```

## Advanced Visualization Examples

### 1. Portfolio Performance Comparison
```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_portfolio_comparison(portfolios, returns):
    """
    Compare performance of different portfolio strategies
    
    portfolios: dict of portfolio weights
    returns: DataFrame of asset returns
    """
    plt.figure(figsize=(12, 6))
    for name, weights in portfolios.items():
        performance = (returns * weights).sum(axis=1).cumsum()
        plt.plot(performance, label=name)
    
    plt.title('Portfolio Strategy Comparison')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
portfolios = {
    'Conservative': conservative_weights,
    'Growth': growth_weights,
    'ESG': esg_weights
}
plot_portfolio_comparison(portfolios, returns_data)
```

### 2. Risk-Return Scatter Plot
```python
def plot_risk_return_scatter(returns, weights, labels):
    """
    Create risk-return scatter plot with portfolio labels
    """
    risk = []
    ret = []
    
    for w in weights:
        portfolio_return = np.sum(returns.mean() * w) * 252
        portfolio_risk = np.sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w)))
        risk.append(portfolio_risk)
        ret.append(portfolio_return)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(risk, ret, c='b', marker='o')
    
    for i, label in enumerate(labels):
        plt.annotate(label, (risk[i], ret[i]))
    
    plt.xlabel('Annual Risk (Volatility)')
    plt.ylabel('Annual Expected Return')
    plt.title('Risk-Return Profile of Different Portfolios')
    plt.grid(True)
    plt.show()
```

### 3. Correlation Heatmap
```python
def plot_correlation_heatmap(returns):
    """
    Create correlation heatmap for portfolio assets
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        returns.corr(),
        annot=True,
        cmap='RdYlBu',
        center=0,
        vmin=-1,
        vmax=1
    )
    plt.title('Asset Correlation Heatmap')
    plt.show()
```

## Optimization Techniques

### 1. Modern Portfolio Theory

Modern Portfolio Theory, developed by Harry Markowitz, helps find the optimal portfolio allocation that:
- Maximizes expected return for a given level of risk
- Minimizes risk for a given level of expected return

```python
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(prices)
S = risk_models.sample_cov(prices)

# Optimize for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
```

### 2. Monte Carlo Simulation

Monte Carlo simulation helps understand potential portfolio outcomes by:
- Generating thousands of possible portfolio allocations
- Analyzing the risk-return characteristics of each allocation
- Finding the optimal portfolio based on various metrics

```python
# Perform Monte Carlo simulation with 5000 iterations
opt_weights, opt_results = pf.mc_optimisation(num_trials=5000)

# Results include:
# - Expected annual return
# - Annual volatility
# - Sharpe ratio
# - Portfolio weights
```

### 3. Efficient Frontier Analysis

The efficient frontier represents the set of optimal portfolios that offer:
- The highest expected return for a defined level of risk
- The lowest risk for a defined expected return

```python
# Plot the efficient frontier
ef = EfficientFrontier(returns, covariance)
ef.plot_efficient_frontier()

# Find minimum volatility portfolio
min_vol_weights = ef.min_volatility()

# Find maximum Sharpe ratio portfolio
max_sharpe_weights = ef.max_sharpe_ratio()
```

## Advanced Portfolio Constraints

### 1. Sector Constraints
```python
# Limit exposure to any single sector
sector_mapper = {
    'AAPL': 'technology',
    'JPM': 'financial',
    'XOM': 'energy',
    # ... more mappings
}

ef.add_sector_constraints(
    sector_mapper,
    sector_lower={'financial': 0.1},  # Min 10% in financials
    sector_upper={'technology': 0.3}   # Max 30% in technology
)
```

### 2. Risk Constraints
```python
# Add risk constraints to the portfolio
ef.add_constraint(lambda w: w.std() <= 0.15)  # Max portfolio volatility
ef.add_constraint(lambda w: w.max() <= 0.25)  # Max weight in any asset
```

### 3. Transaction Cost Optimization
```python
# Include transaction costs in optimization
def transaction_cost(w, current_weights):
    return 0.001 * np.abs(w - current_weights).sum()  # 0.1% transaction cost

ef.add_objective(transaction_cost, current_weights)
```

## Rebalancing Strategies

### 1. Threshold Rebalancing
```python
def needs_rebalancing(current_weights, target_weights, threshold=0.05):
    """
    Check if portfolio needs rebalancing based on threshold
    """
    return np.any(np.abs(current_weights - target_weights) > threshold)

# Example usage
if needs_rebalancing(current_weights, target_weights):
    new_weights = ef.optimize()
    # Execute trades to achieve new_weights
```

### 2. Calendar Rebalancing
```python
def quarterly_rebalance(portfolio, date):
    """
    Perform quarterly portfolio rebalancing
    """
    if date.month in [3, 6, 9, 12] and date.day == 1:
        return True
    return False
```

## Performance Analysis

### 1. Risk Metrics
```python
def calculate_risk_metrics(returns):
    """
    Calculate comprehensive risk metrics
    """
    metrics = {
        'Volatility': returns.std() * np.sqrt(252),
        'Sharpe': calculate_sharpe_ratio(returns),
        'Sortino': calculate_sortino_ratio(returns),
        'Max Drawdown': calculate_max_drawdown(returns),
        'VaR_95': calculate_var(returns, 0.95),
        'CVaR_95': calculate_cvar(returns, 0.95)
    }
    return metrics
```

## Performance Metrics

### 1. Sharpe Ratio
- Measures risk-adjusted return
- Higher is better
- Formula: (Portfolio Return - Risk-free Rate) / Portfolio Volatility

### 2. Volatility
- Measures portfolio risk
- Lower is generally better
- Calculated as the standard deviation of returns

### 3. Maximum Drawdown
- Measures the largest peak-to-trough decline
- Important for risk management
- Helps understand worst-case scenarios

## Example Usage

See `portfolio_optimization_example.ipynb` for a complete working example. Key steps:

1. Data Preparation:
```python
names = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
start_date = datetime(2020, 1, 1)
pf = build_portfolio(names=names, start_date=start_date)
```

2. Portfolio Analysis:
```python
# Get cumulative returns
returns = pf.comp_cumulative_returns()

# Perform Monte Carlo optimization
weights, results = pf.mc_optimisation(num_trials=5000)
```

3. Visualization:
```python
# Plot efficient frontier
pf.ef.plot_efficient_frontier()

# Plot Monte Carlo results
pf.mc_plot_results()
```

## Best Practices

1. **Data Quality**
   - Use sufficient historical data (typically 3-5 years)
   - Handle missing data appropriately
   - Consider market conditions and outliers

2. **Risk Management**
   - Don't rely solely on historical data
   - Consider multiple risk metrics
   - Include transaction costs in optimization

3. **Portfolio Rebalancing**
   - Set appropriate rebalancing frequency
   - Consider tax implications
   - Monitor tracking error

## Common Pitfalls

1. **Overfitting**
   - Using too short historical periods
   - Over-optimizing based on past performance
   - Not considering future market conditions

2. **Concentration Risk**
   - Not setting proper weight constraints
   - Allowing excessive allocation to single assets
   - Ignoring sector/geographic concentration

3. **Implementation Issues**
   - Not considering transaction costs
   - Ignoring liquidity constraints
   - Frequent rebalancing leading to high costs

## References

1. Markowitz, H. (1952). Portfolio Selection
2. Sharpe, W. F. (1964). Capital Asset Prices
3. PyPortfolioOpt Documentation
4. FinQuant Documentation

## Further Reading

- [PyPortfolioOpt Documentation](https://pyportfolioopt.readthedocs.io/)
- [Modern Portfolio Theory](https://www.investopedia.com/terms/m/modernportfoliotheory.asp)
- [Efficient Frontier](https://www.investopedia.com/terms/e/efficientfrontier.asp) 