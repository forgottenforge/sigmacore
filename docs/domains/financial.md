# Financial Optimization Guide

## Overview
The Financial domain optimizes trading strategies and portfolios. It balances **Returns** (Performance) against **Market Risk** (Stability).

## Metrics

### Performance
- **CAGR**: Compound Annual Growth Rate.
- **Sharpe Ratio**: Risk-adjusted return.
- **Sortino Ratio**: Return relative to downside risk.

### Stability ($\sigma_c$)
In finance, stability is the inverse of volatility or drawdown risk.
$$ \sigma_c = \frac{1}{1 + \text{MaxDrawdown}} $$

## Market Data
Sigma-C uses `yfinance` to fetch real-time or historical market data.

```python
from sigma_c.adapters.financial import FinancialAdapter

adapter = FinancialAdapter()
data = adapter.fetch_data(tickers=['AAPL', 'GOOGL'], period='1y')
```

## Strategy Optimization
You can optimize parameters like moving average windows, stop-loss thresholds, and leverage.

```python
from sigma_c.optimization.financial import BalancedFinancialOptimizer

# Define a strategy factory
def moving_average_strategy(short_window, long_window):
    # ... implementation ...
    return strategy

# Optimize
optimizer = BalancedFinancialOptimizer()
result = optimizer.optimize_strategy(
    strategy_factory=moving_average_strategy,
    param_space={
        'short_window': [10, 20, 50],
        'long_window': [100, 200]
    }
)
```
