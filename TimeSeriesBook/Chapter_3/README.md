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
- `φᵢ` = fixed parameter estimated from the time series (not random).
- Measures how strongly past values influence the present.
- Interpretation:
  - **φ > 0** → persistence / momentum (high yesterday → high today).
  - **φ < 0** → mean reversion (high yesterday → low today).
  - **φ ≈ 0** → little to no influence from the past

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

### Intuition
- Today’s return is shaped by:
  - its own past (AR)  
  - past surprises/news (MA)  

---

## ✅ Key Takeaways
- **MA model:** Today = past shocks echoing forward  
- **AR model:** Today = past values influencing today  
- **ARMA model:** Combination of both effects  
