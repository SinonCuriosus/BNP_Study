# --- Random Walk with Drift on a Real Stock (AAPL) ---
# pip install yfinance statsmodels matplotlib pandas numpy

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# 1) Load daily prices (Adj Close)
ticker = "ADBE"
data = yf.download(ticker, period="1y", auto_adjust=True)
price = data["Close"].dropna()  # already adjusted if auto_adjust=True

# Ensure we grab a Series, not a 1-col DataFrame
price = data["Close"]              # NOT data[["Close"]]
logp  = np.log(price)
rets  = logp.diff().dropna()       # Series

# Force scalar floats for printing
mu_daily    = float(rets.mean())
sigma_daily = float(rets.std())

mu_annual    = mu_daily * 252
sigma_annual = sigma_daily * np.sqrt(252)

print(f"μ_daily       = {mu_daily:.6f}")
print(f"μ_annual      = {mu_annual:.2%}")
print(f"σ_daily       = {sigma_daily:.6f}")
print(f"σ_annual      = {sigma_annual:.2%}")

# 4) ADF tests: price (nonstationary) vs returns (stationary)
def adf(name, series):
    stat, pval, lags, nobs, crit, _ = adfuller(series.dropna(), autolag="AIC")
    print(f"\nADF — {name}")
    print(f"  statistic = {stat: .4f}, p-value = {pval: .4f}, used lags = {lags}, n = {int(nobs)}")
    for k, v in crit.items():
        print(f"  critical {k}: {v: .4f}")

adf("PRICE (log level)", logp)
adf("RETURNS (diff log price)", rets)

# 5) Quick plots (price shows trend; returns oscillate around mean)
plt.figure(figsize=(10,4))
plt.plot(price)
plt.title(f"{ticker} Price — Nonstationary (Random Walk with Drift)")
plt.xlabel("Date"); plt.ylabel("Price"); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,4))
plt.plot(rets)
plt.axhline(mu_daily, linestyle="--")
plt.title(f"{ticker} Log Returns — Stationary Around Mean (μ ≈ {mu_daily:.5f})")
plt.xlabel("Date"); plt.ylabel("Log Return"); plt.tight_layout(); plt.show()

# 6) Optional: check RW-with-drift reconstruction error
#    P_t ≈ P_0 * exp( Σ(μ + ε_t) ), so cumulative sum of returns should track log price
recon_logp = logp.iloc[0] + rets.cumsum()
tracking_err = (logp.reindex_like(recon_logp) - recon_logp).dropna()
print(f"\nTracking error (log-price minus cumulative returns) — mean: {tracking_err.mean():.6e}, std: {tracking_err.std():.6e}")
