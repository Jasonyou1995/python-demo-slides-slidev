---
theme: seriph
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Python & Statistics for Financial Computing
  Comprehensive introduction to Python programming, statistics, and probability
  for financial applications.
drawings:
  persist: false
transition: slide-left
title: Python & Statistics for Financial Computing
mdc: true
---

# Python & Statistics for Financial Computing

A Comprehensive Guide for Finance Students

<div class="pt-12">
  <span @click="$slidev.nav.next" class="px-2 py-1 rounded cursor-pointer" hover="bg-white bg-opacity-10">
    Press Space for next page <carbon:arrow-right class="inline"/>
  </span>
</div>

---

layout: image-right
image: https://source.unsplash.com/collection/139386/600x600?finance

---

# Why Python for Finance?

<v-clicks>

## 🚀 Industry Standard

- **Quantitative Analysis**
- **Algorithmic Trading**
- **Risk Management**
- **Financial Modeling**

## 📊 Data Science Power

```python
# Financial data analysis pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load financial data
stock_data = pd.read_csv('financial_data.csv')
returns = stock_data.pct_change()
```

</v-clicks>

::right::

<v-clicks>

## 🎯 Key Advantages

- Rich ecosystem (pandas, numpy, scipy)
- Excellent visualization (matplotlib, seaborn)
- Machine learning integration
- Open source and free

</v-clicks>

---

## layout: two-cols

# Python Fundamentals Crash Course

<v-clicks>

## Variables & Data Types

```python
# Basic data types
price = 145.67           # float
ticker = "AAPL"          # string
volume = 1000000         # integer
is_trading = True        # boolean
```

## Financial Data Structures

```python
# Lists for price series
prices = [100, 102, 105, 103, 108]
returns = []

# Calculate daily returns
for i in range(1, len(prices)):
    ret = (prices[i] - prices[i-1]) / prices[i-1]
    returns.append(ret)
```

</v-clicks>

::right::

<v-clicks>

## Functions for Financial Calculations

```python
def calculate_returns(prices):
    """Calculate daily returns from price series"""
    returns = []
    for i in range(1, len(prices)):
        daily_return = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(daily_return)
    return returns

# Usage
stock_prices = [100, 102, 105, 103, 108]
daily_returns = calculate_returns(stock_prices)
```

## List Comprehensions

```python
# Compact way to calculate returns
returns = [(prices[i] - prices[i-1]) / prices[i-1]
          for i in range(1, len(prices))]
```

</v-clicks>

---

## layout: two-cols

# Essential Python Libraries

<v-clicks>

## 📦 NumPy - Numerical Computing

```python
import numpy as np

# Array operations
returns = np.array([0.02, -0.01, 0.03, 0.015])
mean_return = np.mean(returns)
std_dev = np.std(returns)

# Matrix operations for portfolios
weights = np.array([0.6, 0.4])
returns = np.array([0.08, 0.12])
portfolio_return = np.dot(weights, returns)
```

## 🐼 Pandas - Data Manipulation

```python
import pandas as pd

# Time series data
dates = pd.date_range('2023-01-01', periods=5)
df = pd.DataFrame({
    'AAPL': [150, 152, 149, 155, 158],
    'GOOGL': [2800, 2820, 2790, 2850, 2870]
}, index=dates)
```

</v-clicks>

::right::

<v-clicks>

## 📊 Matplotlib - Visualization

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(df.index, df['AAPL'], label='AAPL')
plt.plot(df.index, df['GOOGL'], label='GOOGL')
plt.title('Stock Price Movement')
plt.legend()
plt.show()
```

## 🔬 SciPy - Scientific Computing

```python
from scipy import stats
from scipy.optimize import minimize

# Statistical functions
t_stat, p_value = stats.ttest_1samp(returns, 0)

# Optimization for portfolio
def portfolio_variance(weights):
    return weights @ covariance_matrix @ weights.T
```

</v-clicks>

---

## class: px-20

# Probability Theory Fundamentals

<div class="grid grid-cols-2 gap-10">
<div>

## 📈 Random Variables

```python
import numpy as np
import matplotlib.pyplot as plt

# Discrete random variable - Binomial
n, p = 10, 0.5
binomial_rv = np.random.binomial(n, p, 1000)

# Continuous random variable - Normal
mu, sigma = 0, 1
normal_rv = np.random.normal(mu, sigma, 1000)
```

</div>
<div>

## 🎲 Probability Distributions

**Discrete:**

- Binomial - Success/failure outcomes
- Poisson - Rare events counting

**Continuous:**

- Normal - Natural phenomena
- Lognormal - Stock prices
- Exponential - Waiting times

</div>
</div>

<br>

### Probability Density Function (PDF)

<div class="text-center">
$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$
</div>

---

## layout: two-cols

# Descriptive Statistics

<v-clicks>

## Central Tendency

```python
returns = np.array([0.02, -0.01, 0.03, 0.015, -0.005])

mean = np.mean(returns)          # Average
median = np.median(returns)      # Middle value
mode = stats.mode(returns)       # Most frequent
```

## Dispersion Measures

```python
variance = np.var(returns)       # Spread from mean
std_dev = np.std(returns)        # Risk measure
range_val = np.ptp(returns)      # Total spread
iqr = stats.iqr(returns)         # Middle 50% spread
```

## Shape Measures

```python
skewness = stats.skew(returns)   # Distribution asymmetry
kurtosis = stats.kurtosis(returns) # Tail thickness
```

</v-clicks>

::right::

<v-clicks>

## Financial Application

```python
def analyze_returns(returns):
    """Comprehensive return analysis"""
    analysis = {
        'mean_return': np.mean(returns),
        'volatility': np.std(returns),
        'sharpe_ratio': np.mean(returns) / np.std(returns),
        'max_drawdown': np.min(returns),
        'positive_days': np.sum(returns > 0) / len(returns)
    }
    return analysis

# Real-world example
daily_returns = np.random.normal(0.001, 0.02, 252)  # 1 year of trading
results = analyze_returns(daily_returns)
```

## Key Financial Metrics

- **Volatility** = Standard deviation of returns
- **Sharpe Ratio** = Return per unit of risk
- **Maximum Drawdown** = Worst peak-to-trough decline

</v-clicks>

---

layout: image-right
image: https://source.unsplash.com/collection/928423/600x600?chart

---

# Probability Distributions in Finance

<v-clicks>

## 📊 Normal Distribution

Most common assumption for returns

```python
from scipy.stats import norm

# Probability calculations
prob_positive = 1 - norm.cdf(0, loc=mean, scale=std_dev)
var_95 = norm.ppf(0.05, mean, std_dev)  # Value at Risk
```

## 📈 Lognormal Distribution

Models stock prices (can't be negative)

```python
# If returns are normal, prices are lognormal
log_returns = np.log(1 + returns)
price_future = price_today * np.exp(mean_log_return)
```

## 📉 Student's t-Distribution

Fat tails - more realistic for financial crises

```python
from scipy.stats import t

# Fitting t-distribution to returns
df, loc, scale = t.fit(returns)
```

</v-clicks>

---

## layout: two-cols

# Statistical Inference

<v-clicks>

## Hypothesis Testing

```python
from scipy.stats import ttest_1samp, norm

# Test if mean return is significantly different from 0
t_stat, p_value = ttest_1samp(returns, 0)

if p_value < 0.05:
    print("Returns are statistically significant")
else:
    print("Returns are not statistically significant")
```

## Confidence Intervals

```python
def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean - h, mean + h

ci_low, ci_high = confidence_interval(returns)
```

</v-clicks>

::right::

<v-clicks>

## Correlation Analysis

```python
# Portfolio correlation
stocks = ['AAPL', 'GOOGL', 'MSFT']
correlation_matrix = returns_df[stocks].corr()

# Test correlation significance
from scipy.stats import pearsonr
corr, p_value = pearsonr(returns_AAPL, returns_GOOGL)
```

## Regression Analysis

```python
from scipy.stats import linregress

# CAPM model: Stock vs Market returns
slope, intercept, r_value, p_value, std_err = \
    linregress(market_returns, stock_returns)

beta = slope  # Systematic risk measure
alpha = intercept  # Excess return
```

</v-clicks>

---

## class: px-20

# Monte Carlo Simulation

## 🎯 Financial Applications

<div class="grid grid-cols-2 gap-8">
<div>

### Stock Price Simulation

```python
def simulate_gbm(S0, mu, sigma, T, dt, n_simulations):
    """Geometric Brownian Motion for stock prices"""
    n_steps = int(T/dt)
    prices = np.zeros((n_simulations, n_steps))
    prices[:, 0] = S0

    for t in range(1, n_steps):
        Z = np.random.standard_normal(n_simulations)
        prices[:, t] = prices[:, t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt +
            sigma * np.sqrt(dt) * Z
        )
    return prices

# Parameters
S0, mu, sigma = 100, 0.08, 0.2  # Initial price, drift, volatility
T, dt, n_sim = 1, 1/252, 1000   # 1 year, daily steps, 1000 simulations
```

</div>
<div>

### Option Pricing

```python
def monte_carlo_option_price(S0, K, T, r, sigma, n_simulations):
    """European call option pricing"""
    # Simulate terminal stock prices
    Z = np.random.standard_normal(n_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T +
                     sigma * np.sqrt(T) * Z)

    # Calculate payoffs
    payoffs = np.maximum(ST - K, 0)

    # Discount expected payoff
    option_price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.std(payoffs) / np.sqrt(n_simulations)

    return option_price, std_error

# Example usage
price, error = monte_carlo_option_price(100, 105, 1, 0.05, 0.2, 100000)
```

</div>
</div>

---

## layout: two-cols

# Risk Management Metrics

<v-clicks>

## Value at Risk (VaR)

```python
def calculate_var(returns, confidence=0.95):
    """Historical VaR"""
    return np.percentile(returns, (1 - confidence) * 100)

# Parametric VaR (Normal assumption)
def parametric_var(returns, confidence=0.95):
    mu, sigma = np.mean(returns), np.std(returns)
    z_score = norm.ppf(1 - confidence)
    return mu + z_score * sigma
```

## Conditional VaR (Expected Shortfall)

```python
def expected_shortfall(returns, confidence=0.95):
    """Average loss beyond VaR"""
    var = calculate_var(returns, confidence)
    losses_beyond_var = returns[returns <= var]
    return np.mean(losses_beyond_var)
```

</v-clicks>

::right::

<v-clicks>

## Portfolio Risk

```python
def portfolio_risk(weights, cov_matrix):
    """Calculate portfolio volatility"""
    return np.sqrt(weights.T @ cov_matrix @ weights)

def diversify_portfolio(returns_data, target_return):
    """Optimize portfolio weights"""
    n_assets = returns_data.shape[1]
    mean_returns = returns_data.mean()
    cov_matrix = returns_data.cov()

    def objective(weights):
        return portfolio_risk(weights, cov_matrix)

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: w @ mean_returns - target_return}
    ]
    bounds = [(0, 1) for _ in range(n_assets)]

    result = minimize(objective, n_assets * [1/n_assets],
                     bounds=bounds, constraints=constraints)
    return result.x
```

</v-clicks>

---

layout: center
class: text-center

---

# Time Series Analysis

## Essential Techniques for Financial Data

<v-clicks>

### Stationarity Testing

```python
from statsmodels.tsa.stattools import adfuller

# Augmented Dickey-Fuller test
result = adfuller(stock_prices)
p_value = result[1]
is_stationary = p_value < 0.05
```

### Autocorrelation

```python
from statsmodels.tsa.stattools import acf, pacf

# Check for patterns in returns
autocorr = acf(returns, nlags=10)
partial_autocorr = pacf(returns, nlags=10)
```

### ARIMA Modeling

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(returns, order=(1,0,1))
fitted_model = model.fit()
forecast = fitted_model.forecast(steps=10)
```

</v-clicks>

---

## layout: two-cols

# Complete Financial Analysis Example

<v-clicks>

## Data Preparation

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download stock data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
data = yf.download(tickers, start='2020-01-01',
                   end='2023-01-01')['Adj Close']

# Calculate returns
returns = data.pct_change().dropna()
```

## Portfolio Analysis

```python
# Equal weight portfolio
weights = np.array([0.25, 0.25, 0.25, 0.25])
portfolio_returns = returns.dot(weights)

# Risk metrics
annual_return = portfolio_returns.mean() * 252
annual_volatility = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = annual_return / annual_volatility
```

</v-clicks>

::right::

<v-clicks>

## Visualization

```python
plt.figure(figsize=(12, 8))

# Cumulative returns
cumulative_returns = (1 + returns).cumprod()
plt.subplot(2, 1, 1)
for ticker in tickers:
    plt.plot(cumulative_returns.index,
             cumulative_returns[ticker], label=ticker)
plt.legend()
plt.title('Cumulative Returns')

# Correlation heatmap
plt.subplot(2, 1, 2)
import seaborn as sns
sns.heatmap(returns.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

plt.tight_layout()
plt.show()
```

## Risk Analysis

```python
# VaR calculation
var_95 = calculate_var(portfolio_returns, 0.95)
es_95 = expected_shortfall(portfolio_returns, 0.95)

print(f"95% VaR: {var_95:.4f}")
print(f"Expected Shortfall: {es_95:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
```

</v-clicks>

---

layout: center
class: text-center

---

# Next Steps & Resources

<v-clicks>

## 🎯 Advanced Topics to Explore

- **Machine Learning for Finance** (scikit-learn, tensorflow)
- **High-Frequency Trading** strategies
- **Options Pricing Models** (Black-Scholes, Binomial)
- **Risk Parity & Modern Portfolio Theory**
- **Blockchain & Cryptocurrency Analysis**

## 📚 Recommended Libraries

```python
# Quantitative finance
import quantlib    # Advanced derivatives pricing
import zipline     # Backtesting engine
import pyfolio    # Portfolio analysis

# Machine learning
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf

# Big data
import dask        # Parallel computing
import pyspark     # Distributed computing
```

## 🚀 Practice Projects

1. Build a trading strategy backtester
2. Create portfolio optimization tool
3. Develop risk management dashboard
4. Implement machine learning price predictor

</v-clicks>

---

## layout: fact

# Thank You!

## Questions?

<div class="mt-10">

**Key Takeaways:**

- Python is powerful for financial computing
- Statistics and probability are fundamental
- Practice with real financial data
- Build projects and portfolios

</div>

<div class="mt-8">

[Documentation](https://python.org) | [GitHub Repository](https://github.com) | [Course Materials](#)

</div>
