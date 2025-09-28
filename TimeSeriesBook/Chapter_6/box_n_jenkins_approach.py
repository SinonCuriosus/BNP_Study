# --- Cell 1: Setup & Download ---

import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from itertools import product

# Helper: ADF p-value
def adf_p(s):
    return adfuller(s.dropna(), autolag="AIC")[1]

# Helper: pick the right close series (works for flat/MultiIndex, with or without auto_adjust)
def get_close_series(df, ticker):
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        if ('Adj Close', ticker) in cols:
            s = df[('Adj Close', ticker)]
        elif ('Close', ticker) in cols:
            s = df[('Close', ticker)]
        elif 'Adj Close' in cols.get_level_values(0):
            s = df.xs('Adj Close', level=0, axis=1).squeeze()
        elif 'Close' in cols.get_level_values(0):
            s = df.xs('Close', level=0, axis=1).squeeze()
        else:
            raise KeyError("Neither 'Adj Close' nor 'Close' found.")
    else:
        if 'Adj Close' in cols:
            s = df['Adj Close']
        elif 'Close' in cols:
            s = df['Close']
        else:
            raise KeyError("Neither 'Adj Close' nor 'Close' found.")
    return s

ticker = "ADBE"

# auto_adjust=True avoids Adj Close confusion (Close becomes adjusted)
raw = yf.download(ticker, start="2015-01-01", auto_adjust=True, progress=False)
if raw.empty:
    raise ValueError("No data returned. Check ticker or connection.")

price = get_close_series(raw, ticker).asfreq("B").ffill().rename(ticker)
y = np.log(price)  # log-price (for ARIMA on levels)

print(price.head(), "\n")
print("Data points:", len(price), "| First date:", price.index.min().date(), "| Last date:", price.index.max().date())
# --- Cell 2: Visualize log-price ---

# plt.figure(figsize=(10,3))
# plt.plot(y, lw=1)
# plt.title(f"{ticker} — Log Price")
# plt.xlabel("Date"); plt.ylabel("log(Price)")
# plt.tight_layout(); plt.show()


# --- Cell 3: Stationarity tests (ADF) & differencing ---

p0 = adf_p(y)
y_d1 = y.diff().dropna()
p1 = adf_p(y_d1)

print(f"ADF p-value (log-level, d=0): {p0:.4f}  -> expect > 0.05 (non-stationary)")
print(f"ADF p-value (1st diff, d=1): {p1:.4f}  -> expect < 0.05 (stationary)")

# Seasonal period for daily stocks (weekly trading seasonality)
s = 5
# optional seasonal differencing if you *clearly* see weekly seasonal unit root
y_d1D1 = y_d1.diff(s).dropna()
p_seas = adf_p(y_d1D1)

print(f"ADF p-value (d=1, seasonal diff D=1, s=5): {p_seas:.4f}  -> only use D=1 if this is much better")

# Decision for the rest of the workflow:
d = 1
D = 0  # set to 1 only if you see strong seasonal nonstationarity remaining
print(f"Using d={d}, D={D}, s={s}")

# --- Cell 4: ACF/PACF on stationary series ---

stationary = y.diff().dropna()  # our chosen d=1 series

# plt.figure(figsize=(10,3))
# plot_acf(stationary, lags=40)
# plt.title("ACF — diff(log(Price))")
# plt.tight_layout(); plt.show()

# plt.figure(figsize=(10,3))
# plot_pacf(stationary, lags=40, method="ywm")
# plt.title("PACF — diff(log(Price))")
# plt.tight_layout(); plt.show()

print("""
Interpretation guide:
- AR(p): PACF cuts off near p, ACF tails.
- MA(q): ACF cuts off near q, PACF tails.
- Seasonal AR(P): PACF spikes at lags s, 2s, ...
- Seasonal MA(Q): ACF spikes at lags s, 2s, ...
We'll still confirm with a small order grid next.
""")

# --- Cell 5: Fit small grid and choose by AIC/BIC ---

def fit_sarima(endog, order, seasonal_order):
    return SARIMAX(endog, order=order, seasonal_order=seasonal_order,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

candidates = []
for p, q in product([0,1,2], [0,1,2]):
    for P, Q in product([0,1], [0,1]):
        try:
            res = fit_sarima(y, (p,1,q), (P,0,Q,5))  # using our chosen d=1, D=0, s=5
            candidates.append(((p,1,q,P,0,Q,5), res.aic, res.bic))
        except Exception as e:
            # print("Skip", (p,1,q,P,0,Q,5), "->", e)
            pass

if not candidates:
    raise RuntimeError("No model converged. Reduce the grid or confirm the data.")

best_aic = min(candidates, key=lambda x: x[1])
order = best_aic[0]
print("Best by AIC:", order, "AIC:", best_aic[1], "BIC:", best_aic[2])

# Keep best model for diagnostics
(p,d,q,P,D,Q,s) = order
best_model = fit_sarima(y, (p,d,q), (P,D,Q,s))
print(best_model.summary())

# --- Cell 6: Diagnostics (residuals should be white noise) ---

# best_model.plot_diagnostics(figsize=(10,8))
# plt.tight_layout(); plt.show()

resid = best_model.resid.dropna()
lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
print("\nLjung–Box (want high p-values to FAIL to reject autocorrelation):\n")
print(lb)

# --- Cell 7: Validation split (~1y test) ---

split = y.index[-252]
y_train = y.loc[:split]
y_test  = y.loc[split + pd.Timedelta(days=1):]

val_model = SARIMAX(y_train, order=(p,d,q), seasonal_order=(P,D,Q,s),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

pred = val_model.get_prediction(start=y_test.index[0], end=y_test.index[-1], dynamic=False)
y_hat = pred.predicted_mean

rmse = np.sqrt(np.mean((y_test - y_hat)**2))
mae  = np.mean(np.abs(y_test - y_hat))
print(f"Validation metrics — RMSE: {rmse:.6f} | MAE: {mae:.6f}")

# plt.figure(figsize=(10,4))
# plt.plot(y_train[-252:], label="Train (log)")
# plt.plot(y_test, label="Test (log)")
# plt.plot(y_hat, label="1-step forecast (log)")
# plt.title(f"{ticker} — Out-of-sample 1-step ahead (log scale)")
# plt.legend(); plt.tight_layout(); plt.show()

# --- Cell 8: Refit on all data & forecast future price ---

final = SARIMAX(y, order=(p,d,q), seasonal_order=(P,D,Q,s),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

steps = 30
fc = final.get_forecast(steps=steps)
mean_fc = fc.predicted_mean
conf = fc.conf_int(alpha=0.05)

# Back-transform from log to price
price_fc = np.exp(mean_fc)
price_lo = np.exp(conf.iloc[:,0])
price_hi = np.exp(conf.iloc[:,1])

# plt.figure(figsize=(10,4))
# plt.plot(price[-252:], label="Recent price")
# plt.plot(price_fc, label=f"{steps}-day forecast")
# plt.fill_between(price_fc.index, price_lo, price_hi, alpha=0.2, label="95% CI")
# plt.title(f"{ticker} — {steps}-Day Price Forecast (SARIMA)")
# plt.legend(); plt.tight_layout(); plt.show()


# --- Cell 9 (Optional): Work with returns and ARMA ---

r = y.diff().dropna()  # log-returns
print("ADF p (returns):", adf_p(r), " -> usually stationary, so d=0")

# Quick ACF/PACF on returns
plt.figure(figsize=(10,3)); plot_acf(r, lags=40); plt.title("ACF — log-returns"); plt.tight_layout(); plt.show()
plt.figure(figsize=(10,3)); plot_pacf(r, lags=40, method="ywm"); plt.title("PACF — log-returns"); plt.tight_layout(); plt.show()

# Tiny grid for ARMA(p,q) with seasonal OFF
arma_cands = []
for p_, q_ in product([0,1,2], [0,1,2]):
    try:
        res = SARIMAX(r, order=(p_,0,q_), seasonal_order=(0,0,0,0),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        arma_cands.append(((p_,0,q_), res.aic))
    except: pass

best_arma = min(arma_cands, key=lambda x: x[1])
print("Best ARMA by AIC:", best_arma)
