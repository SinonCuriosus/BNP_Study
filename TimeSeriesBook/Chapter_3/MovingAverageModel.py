import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

# 1) Download prices
px = yf.download("AAPL", start="2018-01-01", progress=False)["Adj Close"].dropna()

# 2) Make "price changes" (first differences)
dP = px.diff().dropna()   # ΔP_t = P_t - P_{t-1}

# 3) Fit a pure MA(1) to ΔP_t  (ARIMA(order=(0,0,1)) == MA(1); no AR, no I)
ma1 = sm.tsa.ARIMA(dP, order=(0,0,1)).fit()
print(ma1.summary())

# 4) Forecast next 5 deltas, then map back to PRICE levels
steps = 5
fc = ma1.get_forecast(steps=steps).predicted_mean  # forecasts for ΔP
# Rebuild prices: P_{t+h} = P_t + Σ ΔP_{future}
price_path = px.iloc[-1] + fc.cumsum()
print("\nLast observed price:", px.iloc[-1])
print("Price level forecasts:\n", price_path)
