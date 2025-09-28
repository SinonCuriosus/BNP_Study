# Chapter 3: Autoregressive Moving Average Models

####### Moving Average (MA) Model - Stock Returns Example #######
# --------------------------------------------------

# We use stock *returns* (not prices) because returns are more stationary.

# MA(1) Model:
# R_t = μ + ε_t + θ_1 * ε_(t-1)

# Where:
# - R_t = today's return
# - μ = average return (e.g., 0.1% daily)
# - ε_t = today's shock (unexpected news)
# - θ_1 = weight on yesterday's shock

# Intuition:
# - Today’s return is due to today’s shock AND some leftover effect from yesterday’s shock.
# - If θ_1 > 0: yesterday’s shock carries over positively.
# - If θ_1 < 0: yesterday’s shock tends to be corrected (mean reversion).

# MA(2) Model:
# R_t = μ + ε_t + θ_1 * ε_(t-1) + θ_2 * ε_(t-2)
# => includes shocks from the last 2 days.

# Concrete Example (MA(1)):
# - μ = 0.001 (0.1% daily return)
# - θ_1 = 0.5
# - ε_(t-1) = +0.02 (yesterday’s +2% shock)
# - ε_t = -0.01 (today’s -1% shock)

# Calculation:
# R_t = 0.001 + (-0.01) + 0.5 * (0.02)
# R_t = 0.001 - 0.01 + 0.01 = 0.001

# So, despite today’s negative news (-1%), yesterday’s positive news (+2% with 0.5 weight)
# balances it out, leaving a near-zero return today.

# Key takeaway:
# - The MA model is about the *memory of shocks*.
# - Stock returns today depend on both today’s and past surprises.