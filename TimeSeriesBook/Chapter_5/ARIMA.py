# arima_demo.py
# ------------------------------------------------------------
# Pedagogic ARIMA walkthrough on real stock data.
# - ARIMA(p, d, q) models *non-stationary* series by differencing (d)
#   and modeling the stationary remainder with ARMA(p, q).
# - We'll use log(Price) and find d to achieve stationarity.
# ------------------------------------------------------------

# pip installs (uncomment if needed):
# pip install yfinance statsmodels matplotlib pandas numpy

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# -----------------------------
# 0) Config
# -----------------------------
TICKER = "AAPL"         # <- change me
START  = "2015-01-01"
END    = None           # None = today
INTERVAL = "1d"         # daily
CSV_FALLBACK = None     # e.g., "AAPL_prices.csv" (must have a 'Date' and 'Close' column)

# Grid for (p, q) search around small values (pedagogic, not exhaustive)
P_GRID = range(0, 4)
Q_GRID = range(0, 4)

# -----------------------------
# 1) Fetch prices (with simple rate-limit handling)
# -----------------------------
def fetch_prices(ticker, start, end=None, interval="1d", retries=3, sleep_s=5):
    for attempt in range(retries):
        try:
            data = yf.download(
                ticker, start=start, end=end, interval=interval,
                auto_adjust=True, progress=False
            )
            if not data.empty:
                return data
        except Exception as e:
            print(f"[yfinance] Attempt {attempt+1}/{retries} failed: {e}")
        time.sleep(sleep_s)
    return pd.DataFrame()

px_df = fetch_prices(TICKER, START, END, INTERVAL)

if px_df.empty:
    if CSV_FALLBACK is None:
        raise RuntimeError(
            "Could not download data (rate limited?). "
            "Set CSV_FALLBACK to a local CSV with Date,Close columns."
        )
    print("[INFO] Falling back to local CSV:", CSV_FALLBACK)
    px_df = pd.read_csv(CSV_FALLBACK, parse_dates=["Date"]).set_index("Date")

# Use Adjusted Close if available; else Close
if "Adj Close" in px_df.columns:
    price = px_df["Adj Close"].dropna().copy()
else:
    price = px_df["Close"].dropna().copy()

price = price.asfreq("B").ffill()  # business-day freq, forward-fill gaps
print(f"\nLoaded {len(price)} observations for {TICKER} "
      f"from {price.index.min().date()} to {price.index.max().date()}.")

# -----------------------------
# 2) Visualize raw price & explain non-stationarity
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(price, label=f"{TICKER} Price")
plt.title(f"{TICKER} Price – Typically Non-Stationary (Trends)")
plt.legend(); plt.tight_layout(); plt.show()

print("""
ARIMA intuition:
- Many financial price series trend (non-stationary mean/variance).
- ARIMA addresses that by differencing d times to remove trend (make stationary),
  then fits ARMA(p, q) on the differenced series.
- We'll use log(price) so differences approximate returns.
""")

# -----------------------------
# 3) Stationarity check (ADF) and differencing plan
# -----------------------------
def adf_report(series, name="Series"):
    res = adfuller(series.dropna(), autolag='AIC')
    ADF, pval, usedlag, nobs, crit, icbest = res
    print(f"ADF test on '{name}':")
    print(f"  ADF Statistic = {ADF: .4f}")
    print(f"  p-value       = {pval: .4f}")
    print(f"  Critical Values = {crit}")
    if pval < 0.05:
        print("  => Likely stationary (reject H0 of unit root).\n")
    else:
        print("  => Likely NON-stationary (fail to reject H0).\n")

logp = np.log(price)

print("ADF on raw PRICE:")
adf_report(price, "Price")

print("ADF on LOG(PRICE):")
adf_report(logp, "log(Price)")

# First difference (this is ~log-returns)
d1 = logp.diff().dropna()

print("ADF on DIFF(LOG(PRICE)):")
adf_report(d1, "diff(log(Price))")

plt.figure(figsize=(10, 4))
plt.plot(d1, label="diff(log(Price)) ≈ log-returns")
plt.axhline(0, ls="--", alpha=0.6)
plt.title("First Difference of Log Price (Often Stationary)")
plt.legend(); plt.tight_layout(); plt.show()

# Quick ACF/PACF glance on stationary series
fig = plt.figure(figsize=(10, 4))
plot_acf(d1.dropna(), lags=40)
plt.title("ACF of diff(log(Price))")
plt.tight_layout(); plt.show()

fig = plt.figure(figsize=(10, 4))
plot_pacf(d1.dropna(), lags=40, method="ywm")
plt.title("PACF of diff(log(Price))")
plt.tight_layout(); plt.show()

# -----------------------------
# 4) Choose d, then (p, q) by tiny grid-search using AIC
# -----------------------------
# We saw above that diff(log(price)) tends to be stationary, so d=1 is a good default.
d = 1

def select_arima_order(y_log, d=1, p_grid=range(0,4), q_grid=range(0,4)):
    """
    Brute-force AIC search over small p, q for ARIMA(p,d,q) on log-price.
    Returns (p, d, q, best_aic).
    """
    best_aic, best_cfg = np.inf, None
    for p in p_grid:
        for q in q_grid:
            if p == 0 and q == 0:
                # ARIMA(0,d,0) is white noise on differenced; allowed but often trivial.
                pass
            try:
                model = ARIMA(y_log, order=(p, d, q))
                res = model.fit()
                if res.aic < best_aic:
                    best_aic, best_cfg = res.aic, (p, d, q)
            except Exception:
                continue
    return (*best_cfg, best_aic)

print("Selecting (p, q) via small AIC search...")
p_sel, d_sel, q_sel, best_aic = select_arima_order(logp, d=d, p_grid=P_GRID, q_grid=Q_GRID)
print(f"Selected order: ARIMA({p_sel}, {d_sel}, {q_sel}) with AIC={best_aic:.2f}\n")

# -----------------------------
# 5) Fit final ARIMA and check residuals
# -----------------------------
final_model = ARIMA(logp, order=(p_sel, d_sel, q_sel))
final_res = final_model.fit()
print(final_res.summary())

resid = final_res.resid
plt.figure(figsize=(10, 4))
plt.plot(resid, label="Residuals")
plt.axhline(0, ls="--", alpha=0.6)
plt.title("Model Residuals (Should Look Like White Noise)")
plt.legend(); plt.tight_layout(); plt.show()

fig = plt.figure(figsize=(10, 4))
plot_acf(resid.dropna(), lags=40)
plt.title("Residual ACF (We Want Small Auto-Correlation)")
plt.tight_layout(); plt.show()

# -----------------------------
# 6) Forecast and back-transform to PRICE
# -----------------------------
H = 30  # forecast horizon (days)
fc = final_res.get_forecast(steps=H)
fc_mean_log = fc.predicted_mean          # forecasts on log-scale
fc_ci_log = fc.conf_int()                # log-scale intervals

# Back-transform (because we modeled log(price) with differencing):
# statsmodels ARIMA on logp with d=1 internally handles differencing.
# Predicted 'levels' are on the original scale of 'endog' (log-price).
# Use exponent to get price forecasts.
fc_mean_price = np.exp(fc_mean_log)
fc_ci_price = np.exp(fc_ci_log)

# Build a neat frame
fc_df = pd.DataFrame({
    "fc_price": fc_mean_price,
    "fc_low":   fc_ci_price.iloc[:, 0],
    "fc_high":  fc_ci_price.iloc[:, 1],
})

# Plot historical prices + forecast
plt.figure(figsize=(11, 5))
plt.plot(price, label="History")
plt.plot(fc_df.index, fc_df["fc_price"], label="Forecast", linewidth=2)
plt.fill_between(fc_df.index, fc_df["fc_low"], fc_df["fc_high"], alpha=0.2, label="95% CI")
plt.title(f"{TICKER}: ARIMA({p_sel},{d_sel},{q_sel}) {H}-Day Forecast (Back-Transformed to Price)")
plt.legend(); plt.tight_layout(); plt.show()

print("""
Interpretation:
- We modeled log-prices with d=1 (first difference) to achieve stationarity.
- The ARIMA captured serial dependence on the differenced series.
- Forecasts were produced on the (log) modeling scale, then exponentiated to price.
- The shaded band is the 95% confidence interval on price.
""")

# -----------------------------
# 7) (Bonus) Simple walk-forward backtest on last N days
#    to see stability of forecasts over time.
# -----------------------------
def walk_forward_backtest(y_log, order, steps_ahead=1, n_last=60):
    """
    Refit ARIMA each day on expanding window and forecast 1-step ahead.
    Returns a DataFrame with actual & predicted (on price scale).
    """
    idx = y_log.index
    start_i = len(y_log) - n_last
    preds, dates = [], []

    for i in range(start_i, len(y_log) - steps_ahead):
        train_log = y_log.iloc[:i]
        try:
            res = ARIMA(train_log, order=order).fit()
            pred_log = res.get_forecast(steps=steps_ahead).predicted_mean.iloc[-1]
            preds.append(np.exp(pred_log))  # back to price
            dates.append(idx[i + steps_ahead])
        except Exception:
            preds.append(np.nan)
            dates.append(idx[i + steps_ahead])

    # Align with actual prices
    actual = np.exp(y_log.reindex(dates))
    out = pd.DataFrame({"actual_price": actual, "pred_price": preds}, index=dates)
    return out

print("Running a tiny walk-forward demo (1-step ahead over last 60 days)...")
wf = walk_forward_backtest(logp, (p_sel, d_sel, q_sel), steps_ahead=1, n_last=60)

plt.figure(figsize=(10, 4))
plt.plot(wf.index, wf["actual_price"], label="Actual")
plt.plot(wf.index, wf["pred_price"], label="1-step Forecast")
plt.title(f"{TICKER}: Walk-Forward 1-Step Forecasts (last ~60 days)")
plt.legend(); plt.tight_layout(); plt.show()

mae = (wf["actual_price"] - wf["pred_price"]).abs().mean()
mape = ( (wf["actual_price"] - wf["pred_price"]).abs() / wf["actual_price"] ).mean() * 100
print(f"Walk-forward MAE:  {mae:,.2f}")
print(f"Walk-forward MAPE: {mape:,.2f}%")

# -----------------------------
# 8) What about SARIMA?
# -----------------------------
print("""
NOTE on SARIMA (seasonal ARIMA):
- If your data show *seasonal* patterns (e.g., monthly seasonality),
  use SARIMA(p, d, q) × (P, D, Q, s).
- Example workflow:
  * Resample to monthly, test stationarity, difference seasonally if needed (D),
    then fit SARIMA with period s=12.
- The steps (stationarity check, AIC selection, residual diagnostics, forecast)
  are identical in spirit—just with seasonal components added.
""")

print("Done ✅")
