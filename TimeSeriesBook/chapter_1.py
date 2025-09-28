# Exercise that splits a time series into trend, seasonal, and residual parts
# In additive decomposition:
#   Observed_t = Trend_t + Seasonal_t + Residual_t
#   Residual_t = Observed_t − (Trend_t + Seasonal_t)

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. Download daily stock prices (adjusted close prices)
ticker = "AAPL"
data = yf.download(ticker, start="2015-01-01", interval="1d", auto_adjust=True, progress=False)

# 2. Extract the adjusted close as the price series
#    This is P_t (the observed stock price at time t)
px = data["Close"]

# 3. Convert prices to daily returns
#    r_t = (P_t / P_{t-1}) - 1
#    Returns are closer to stationary than raw prices, making seasonality easier to see
returns = px.pct_change().dropna()

# 4. Choose the seasonality period:
#    - If period = 5 → weekly cycle (Mon–Fri)
#    - If period = 12 → yearly cycle of months (Jan–Dec) if using monthly data
period = 12

# 5. Decompose using the classical moving-average method
#    seasonal_decompose splits the observed series into three parts:
#    Trend_t   = centered moving average of r_t (window = period)
#    Seasonal_t = average pattern repeating every "period" steps
#                 (e.g., mean return for each month if period=12)
#    Residual_t = r_t − (Trend_t + Seasonal_t)
decomp = seasonal_decompose(
    returns,
    model="additive",      # Additive model: r_t = Trend_t + Seasonal_t + Residual_t
    period=period,         # Periodicity of the seasonal cycle
    extrapolate_trend="freq"  # Extend trend at the edges so it's defined everywhere
)

# 6. Plot decomposition:
#    Top panel    = Observed returns r_t
#    Second panel = Trend_t (slow-moving average)
#    Third panel  = Seasonal_t (repeating 12-month cycle)
#    Bottom panel = Residual_t (remaining noise) = r_t - Trend_t - Seasonal_t
decomp.plot()
plt.suptitle(f"{ticker} — Seasonal Decomposition (period={period})", fontsize=14)
plt.show()