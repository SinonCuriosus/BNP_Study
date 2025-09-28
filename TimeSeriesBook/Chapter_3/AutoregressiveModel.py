# Autoregressive (AR) Model - Stock Returns Example
# --------------------------------------------------

# General AR(p) Model:
# X_t = μ + φ_1 * X_(t-1) + φ_2 * X_(t-2) + ... + φ_p * X_(t-p) + ε_t

# Where:
# - X_t = today's value (e.g., stock return)
# - μ = average return
# - φ_i = autoregressive coefficients
# - p = order of the AR model
# - ε_t = today’s random shock (white noise)

# Intuition:
# - Today’s return depends on its *own past values* plus a random shock.
# - The series has "memory" in its past values, not just in shocks.

# Example: AR(1)
# X_t = μ + φ_1 * X_(t-1) + ε_t

# Interpretation of φ_1:
# - If φ_1 > 0 (e.g., 0.8): momentum effect → positive yesterday → likely positive today.
# - If φ_1 < 0 (e.g., -0.8): mean reversion → positive yesterday → likely negative today.
# - If φ_1 ≈ 0: the process is close to white noise (no memory).

# Stock Return Example:
# - μ = 0
# - φ_1 = 0.3
# - Yesterday’s return = +2% (0.02)
# - Today’s shock ε_t = -0.005 (-0.5%)

# Calculation:
# X_t = 0 + 0.3 * (0.02) + (-0.005)
# X_t = 0.006 - 0.005 = 0.001 (+0.1%)

# So even with a negative shock today (-0.5%), the positive carryover from yesterday
# (0.6%) results in a small positive return.

# Key takeaway:
# - The AR model captures persistence or mean reversion in stock returns.
# - AR looks at *past values*, while MA looks at *past shocks*.
