# Time Series Models: MA, AR, and ARMA

This document explains the **Moving Average (MA)**, **Autoregressive (AR)**, and **Autoregressive Moving Average (ARMA)** models, with stock returns as examples.  
These are fundamental building blocks in time series analysis.

---

## 📌 1. Moving Average (MA) Model

### General Form (MA(q))
`X_t = μ + ε_t + θ_1 ε_(t-1) + θ_2 ε_(t-2) + ... + θ_q ε_(t-q)`

Where:
- `X_t` = value of the series at time *t* (e.g., stock return)  
- `μ` = average return  
- `ε_t` = today’s random shock (white noise)  
- `θ_i` = coefficients measuring the effect of past shocks  
- `q` = order of the model  

### Intuition
- Today’s return depends on **today’s shock** and **lingering effects of past shocks** weitghed trough θ.

### Example (MA(1))
`R_t = μ + ε_t + θ_1 ε_(t-1)`

- μ = 0.001 (0.1% daily return)  
- θ₁ = 0.5  
- Yesterday’s shock: +0.02 (+2%)  
- Today’s shock: -0.01 (–1%)  

**Calculation:**  
`R_t = 0.001 + (-0.01) + 0.5(0.02) = 0.001`  

✅ Despite today’s negative news, yesterday’s positive surprise still carries weight,  
leaving a near-zero return.

---

## 📌 2. Autoregressive (AR) Model

### General Form (AR(p))
`X_t = μ + φ_1 X_(t-1) + φ_2 X_(t-2) + ... + φ_p X_(t-p) + ε_t`

Where:
- `X_t` = value of the series at time *t*  
- `μ` = average return  
- `p` = order of the model  
- `ε_t` = white noise (shock today)
- `φᵢ` = fixed parameter estimated from the time series (not random). Measures how strongly past values influence the present.
  Interpretation:
  - **φ > 0** → persistence / momentum (high yesterday → high today).
  - **φ < 0** → mean reversion (high yesterday → low today).
  - **φ ≈ 0** → little to no influence from the past
  Note:
  `φᵢ` may use several ways to be calculated:
  - **OLS (demeaned) closed-form**:

  $$
  \hat{\phi}=\frac{\sum_{t=2}^{T}(r_t-\bar{r})(r_{t-1}-\bar{r})}
  {\sum_{t=2}^{T}(r_{t-1}-\bar{r})^2}
  $$

### Intuition
- Today’s return depends on **past values of the series**.  
- Memory is in the *series itself*, not just in shocks.  
- Positive φ → persistence (momentum).  
- Negative φ → mean reversion.  

### Example (AR(1))
`X_t = μ + φ_1 X_(t-1) + ε_t`

- μ = 0  
- φ₁ = 0.3  
- Yesterday’s return = +0.02 (+2%)  
- Today’s shock ε_t = -0.005 (-0.5%)  

**Calculation:**  
`X_t = 0 + 0.3(0.02) - 0.005 = 0.001`  

✅ Even with a negative shock today, yesterday’s positive return persistence  
results in a small positive outcome.

---

## 📌 3. Autoregressive Moving Average (ARMA) Model

### General Form (ARMA(p, q))
`X_t = μ + φ_1 X_(t-1) + ... + φ_p X_(t-p) + ε_t + θ_1 ε_(t-1) + ... + θ_q ε_(t-q)`

Where:
- AR part (`φ`): captures persistence from past values  
- MA part (`θ`): captures effects of past shocks  
- Together: ARMA balances **past values + past shocks**  

- ACF, Autocorrelation Function: How to figure out what order should we use for MA;

### Intuition
- Today’s return is shaped by:
  - its own past (AR)  
  - past surprises/news (MA)  

---

## ✅ Key Takeaways
- **MA model:** Today = past shocks echoing forward  
- **AR model:** Today = past values influencing today  
- **ARMA model:** Combination of both effects  

---

## 📌 4. Autoregressive **Integrated** Moving Average (ARIMA) Model

### General Form (ARIMA(p, d, q))
`Δ^d X_t = μ + φ_1 Δ^d X_(t-1) + ... + φ_p Δ^d X_(t-p) + ε_t + θ_1 ε_(t-1) + ... + θ_q ε_(t-q)`

Where:
- `Δ` = first difference operator, `Δ X_t = X_t − X_(t−1)`; `Δ^d` = apply `Δ` d times  
- `d` (**integration order**) = how many differences to remove trend / unit root  
- AR part (`φ`) = persistence in the differenced series  
- MA part (`θ`) = echo of past shocks in the differenced series  
- Together: **ARIMA = ARMA on the differenced series**

**Identification (quick):**
- Pick the smallest `d` that makes the series look stationary (often `d=1` for prices).  
- On `Δ^d X_t`, use ACF/PACF to choose `p,q` (AIC/BIC to compare).

### Intuition
- If levels drift (trend/random walk), **difference first** → stable series.  
- Model `Δ^d X_t` with ARMA:  
  - AR: today’s **change** depends on past **changes**  
  - MA: today’s **change** adjusts for past **shocks**  
- Forecast `Δ^d X_{t+1}`, then **integrate back**: add it to the last level to get `X_{t+1}`.
- Used for prices VS returns which usually is used with ARMA.

### Example (ARIMA(1, 1, 1))
`Δ X_t = μ + φ_1 Δ X_(t−1) + ε_t + θ_1 ε_(t−1)`

---

## ✅ Key Takeaways
- **Prices → ARIMA (often `d=1`)**; **returns → ARMA (`d=0`)**.  
- Choose **lowest `d`** that yields stationarity; then pick `p,q` via ACF/PACF + AIC/BIC.  
- After differencing: AR roots outside unit circle (**stationary/causal**); MA roots outside (**invertible**).

---

## 📌 5. Seasonal ARIMA (SARIMA) Model

### General Form (SARIMA(p, d, q) × (P, D, Q)_s)
On the **seasonally differenced** series:
`(1 − L)^d (1 − L^s)^D X_t = μ + [AR terms φ, seasonal Φ at lag s] + ε_t + [MA terms θ, seasonal Θ at lag s]`

Where:
- `L` = lag operator (`L X_t = X_(t−1)`), `s` = season length (e.g., 5 trading days, 12 months)
- `d` = non-seasonal differences; `D` = **seasonal** differences
- **AR(p)** & **MA(q)** act at lag 1; **Seasonal AR(P)** & **Seasonal MA(Q)** act at lag `s`

**Compact backshift form:**
`Φ(L^s) φ(L) (1 − L)^d (1 − L^s)^D X_t = μ + Θ(L^s) θ(L) ε_t`

---

### Intuition
- Remove **trend** with `(1 − L)^d`
- Remove **seasonal pattern** with `(1 − L^s)^D`
- Model the stationary remainder with ARMA at **lag 1** and **lag s**
  - AR parts: persistence in changes
  - MA parts: shock corrections (at 1 and at s)

---

### Identification (quick)
- Pick `D`/`s` by visible seasonal cycles (e.g., **12** for monthly, **5** for weekday).
- Use **seasonal ACF/PACF**:
  - Seasonal spikes at lags `s, 2s, ...` → consider `P`/`Q`
  - Non-seasonal tail behavior → choose `p`/`q`
- Compare models with AIC/BIC.

---

### Example (monthly prices): SARIMA(1, 1, 1) × (1, 1, 1)_12
`(1 − L)(1 − L^12) X_t = μ + φ_1 (1 − L)(1 − L^12) X_(t−1) + ε_t + θ_1 ε_(t−1) + Φ_1 (1 − L)(1 − L^12) X_(t−12) + Θ_1 ε_(t−12)`

Reading it:
- Difference once (trend) and once seasonally (annual pattern),
- Then apply ARMA terms at lag 1 and lag 12.

---

## ✅ Key Takeaways
- **SARIMA = ARIMA + seasonal structure** at period `s`.
- Choose the **smallest** `d` and `D` that yield stationarity (avoid over-differencing).
- Use ACF/PACF at **regular** and **seasonal** lags to pick `(p, q)` and `(P, Q)`.
- Prices with clear seasonality → **SARIMA** (often `d=1`, `D=1`). Returns may need `D=0`.
